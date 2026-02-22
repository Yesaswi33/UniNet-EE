from flask import Flask, render_template, request, redirect, url_for, session
import os
import uuid
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import random
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")  # Ignore all warnings

import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Activation, GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras import backend as K
import keras

# PyTorch imports
import torch
import torch.nn as nn
from torchvision import models, transforms
import cv2

# -------------------- FLASK APP SETUP --------------------
app = Flask(__name__)
app.secret_key = "yesaswi_madabattula"
app.permanent_session_lifetime = timedelta(minutes=30)

# Local in-memory storage (for demo)
users = {}
user_history = {}
otp_store = {}

# Upload folder setup
UPLOAD_FOLDER = os.path.join("static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# -------------------- CNN ARCHITECTURE --------------------
class Architecture:
    @staticmethod
    def build(width, height, depth, classes, finalAct="softmax"):
        inputShape = (height, width, depth)
        chanDim = -1
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
            chanDim = 1

        initializer = keras.initializers.he_uniform(seed=200)
        model = Sequential()

        # BLOCK 1
        model.add(Conv2D(32, (3,3), padding='same', input_shape=inputShape, kernel_initializer=initializer))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Activation("relu"))
        model.add(Conv2D(32, (3,3), padding='same', kernel_initializer=initializer))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Activation("relu"))
        model.add(MaxPooling2D((2,2)))

        # BLOCK 2
        model.add(Conv2D(64, (3,3), padding='same', kernel_initializer=initializer))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Activation("relu"))
        model.add(Conv2D(64, (3,3), padding='same', kernel_initializer=initializer))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Activation("relu"))
        model.add(MaxPooling2D((2,2)))

        # BLOCK 3
        model.add(Conv2D(128, (3,3), padding='same', kernel_initializer=initializer))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Activation("relu"))
        model.add(Conv2D(128, (3,3), padding='same', kernel_initializer=initializer))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Activation("relu"))
        model.add(MaxPooling2D((2,2)))

        # BLOCK 4
        model.add(Conv2D(256, (5,5), padding='same', kernel_initializer=initializer))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Activation("relu"))
        model.add(Conv2D(256, (5,5), padding='same', kernel_initializer=initializer))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Activation("relu"))
        model.add(MaxPooling2D((2,2)))

        # Classifier
        model.add(GlobalAveragePooling2D())
        model.add(Dense(512, activation="relu", kernel_initializer=initializer))
        model.add(Dropout(0.5))
        model.add(Dense(128, activation="relu", kernel_initializer=initializer))
        model.add(Dropout(0.5))
        model.add(Dense(classes, activation=finalAct))

        return model

# -------------------- LOAD WEATHER MODEL --------------------
img_dims = (250, 250, 3)
weather_model = Architecture.build(width=img_dims[0], height=img_dims[1], depth=img_dims[2], classes=4)
MODEL_PATH = "/Users/yesaswimadabattula/Documents/major_project/saved_models/final_model.h5"
weather_model.load_weights(MODEL_PATH)
weather_classes = {0: 'Cloudy', 1: 'Rain', 2: 'Shine', 3: 'Sunrise'}

# -------------------- LOAD DAY/NIGHT PYTORCH MODELS --------------------
def load_simple_model():
    def conv_bn_relu(ni, nf, stride=2, bn=True, act=True):
        layers = [nn.Conv2d(ni, nf, kernel_size=3, stride=2, padding=1, bias=not bn)]
        if bn: layers.append(nn.BatchNorm2d(nf))
        if act: layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    class Flatten(nn.Module):
        def forward(self, x): return x.squeeze()

    simple_model = nn.Sequential(
        conv_bn_relu(1, 8),
        conv_bn_relu(8, 16),
        conv_bn_relu(16, 32),
        conv_bn_relu(32, 8),
        conv_bn_relu(8, 2, bn=False, act=False),
        nn.AdaptiveAvgPool2d(1),
        Flatten()
    )
    simple_model.load_state_dict(torch.load('/Users/yesaswimadabattula/Documents/major_project/saved_models/simple_best_model.pth', map_location=torch.device('cpu')))
    simple_model.eval()
    return simple_model

def load_mbv2():
    mbv2 = models.mobilenet_v2(pretrained=True)
    in_features = mbv2.classifier[1].in_features
    mbv2.classifier[1] = torch.nn.Linear(in_features, 2)
    mbv2.load_state_dict(torch.load('/Users/yesaswimadabattula/Documents/major_project/saved_models/mbv2_best_model.pth', map_location=torch.device('cpu')))
    mbv2.eval()
    return mbv2

# Load once
simple_model = load_simple_model()
mbv2_model = load_mbv2()

# Transforms
simple_tfms = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.42718], [0.22672])])
mbv2_mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
mbv2_std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)

# -------------------- HELPER --------------------
def is_logged_in(): return 'user_email' in session

# -------------------- HOME --------------------
@app.route('/')
def home():
    logged_in = is_logged_in()
    user_email = session['user_email'] if logged_in else None
    return render_template("index.html", logged_in=logged_in, user_email=user_email)

# -------------------- REGISTER / LOGIN / LOGOUT --------------------
@app.route('/register', methods=['GET','POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']; email = request.form['email']; password = request.form['password']
        if email in users: return "User already exists"
        users[email] = {'name': name,'password': password}
        return redirect(url_for('login'))
    return render_template("register.html")

@app.route('/login', methods=['GET','POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']; password = request.form['password']
        user = users.get(email)
        if user and user['password'] == password:
            session['user_email'] = email; session.permanent = True
            return redirect(url_for('home'))
        return "Invalid credentials"
    return render_template("login.html")

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for("home"))

# -------------------- FORGOT / RESET PASSWORD --------------------
SMTP_SERVER = "smtp.gmail.com"; SMTP_PORT = 587; SMTP_EMAIL = "madabattulayesaswi@gmail.com"; SMTP_PASSWORD = "uvzy hkft dept wbrd"

@app.route("/forgot-password", methods=["GET","POST"])
def forgot_password():
    if request.method=="POST":
        email = request.form.get("email")
        if email not in users: return render_template("forgot_password.html", error="Email not registered!")
        otp = random.randint(100000, 999999)
        otp_store[email] = {'code': otp,'expires': datetime.now() + timedelta(minutes=10)}
        try:
            subject = "UrbanVision AI Password Reset OTP"
            body = f"Hello,\n\nYour OTP to reset your UrbanVision AI password is: {otp}\nIt expires in 10 minutes.\n\nIgnore if not requested."
            server = smtplib.SMTP(SMTP_SERVER,SMTP_PORT); server.starttls(); server.login(SMTP_EMAIL,SMTP_PASSWORD)
            msg = MIMEMultipart(); msg['From']=SMTP_EMAIL; msg['To']=email; msg['Subject']=subject
            msg.attach(MIMEText(body,'plain')); server.send_message(msg); server.quit()
            return redirect(url_for("reset_password", email=email))
        except Exception as e:
            print("Error sending OTP:",e)
            return render_template("forgot_password.html", error="Failed to send OTP. Try again later.")
    return render_template("forgot_password.html")

@app.route("/reset-password", methods=["GET","POST"])
def reset_password():
    email = request.args.get("email")
    if not email or email not in users: return redirect(url_for("forgot_password"))
    if request.method=="POST":
        otp_input = request.form.get("otp"); password=request.form.get("password"); confirm_password=request.form.get("confirm_password")
        otp_data = otp_store.get(email)
        if not otp_data: return render_template("reset_password.html", email=email,error="OTP expired. Request new.")
        if datetime.now()>otp_data['expires']: otp_store.pop(email,None); return render_template("reset_password.html",email=email,error="OTP expired. Request new.")
        if str(otp_input)!=str(otp_data['code']): return render_template("reset_password.html",email=email,error="Invalid OTP.")
        if password!=confirm_password: return render_template("reset_password.html",email=email,error="Passwords do not match.")
        if len(password)<6: return render_template("reset_password.html",email=email,error="Password must be at least 6 characters.")
        users[email]['password']=password; otp_store.pop(email,None)
        return render_template("login.html", message="Password reset successful! Please login.")
    return render_template("reset_password.html", email=email)

# -------------------- STATIC PAGES --------------------
@app.route("/about")
def about():
    logged_in = is_logged_in(); user_email = session['user_email'] if logged_in else None
    return render_template("about.html", logged_in=logged_in,user_email=user_email)

TEAM_EMAILS = ["madabattulayesaswi@gmail.com","dileepvayilapalli@gmail.com","chandrasekhar20634@gmail.com","ramyasreeveerla117@gmail.com"]

@app.route("/contact", methods=["GET","POST"])
def contact():
    logged_in = is_logged_in(); user_email = session['user_email'] if logged_in else None
    if request.method=="POST":
        name=request.form.get("name"); email=request.form.get("email"); user_message=request.form.get("message")
        subject=f"UrbanVision AI Feedback from {name}"; body=f"Name:{name}\nEmail:{email}\n\nMessage:\n{user_message}"
        try:
            server=smtplib.SMTP(SMTP_SERVER,SMTP_PORT); server.starttls(); server.login(SMTP_EMAIL,SMTP_PASSWORD)
            for recipient in TEAM_EMAILS:
                msg=MIMEMultipart(); msg['From']=SMTP_EMAIL; msg['To']=recipient; msg['Subject']=subject; msg.attach(MIMEText(body,'plain')); server.send_message(msg)
            server.quit()
            return redirect(url_for("contact",status="success"))
        except Exception as e:
            print("Error sending email:",e); return redirect(url_for("contact",status="error"))
    status=request.args.get("status"); message=""
    if status=="success": message="Your message has been sent successfully!"
    elif status=="error": message="There was an error sending your message. Please try again later."
    return render_template("contact.html", message=message, logged_in=logged_in,user_email=user_email)

# -------------------- UPLOAD ROUTE (WEATHER + DAY/NIGHT) --------------------
@app.route("/upload", methods=["GET","POST"])
def upload():
    if not is_logged_in():
        return redirect(url_for("login"))

    logged_in = True
    email = session['user_email']
    user_email = email
    uploaded_filename = None
    weather_result = None
    daynight_result = None

    if request.method == "POST":
        file = request.files.get("file")
        if file and file.filename != "":
            unique_name = str(uuid.uuid4()) + "_" + file.filename
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], unique_name)
            file.save(file_path)
            user_history.setdefault(email, []).append(unique_name)
            uploaded_filename = unique_name

            # -------- WEATHER PREDICTION (Keras) --------
            img = load_img(file_path, target_size=(250,250))
            img_array = img_to_array(img)/255.0
            img_array = np.expand_dims(img_array, axis=0)
            pred = weather_model.predict(img_array)
            pred_class = np.argmax(pred)
            weather_result = weather_classes[pred_class]

            # -------- DAY/NIGHT PREDICTION (PyTorch) --------
            bgr_img = cv2.imread(file_path)
            bgr_resized = cv2.resize(bgr_img, (500,500))
            hsv = cv2.cvtColor(bgr_resized, cv2.COLOR_BGR2HSV)
            v_channel = hsv[:,:,2]  # Extract V channel

            # ---- Simple 5-layer model ----
            v_tensor = transforms.ToTensor()(v_channel)             # [H, W] -> [1, H, W]
            v_tensor = transforms.Normalize([0.42718],[0.22672])(v_tensor)
            v_tensor = v_tensor.unsqueeze(0)                        # add batch dim -> [1,1,H,W]
            simple_out = simple_model(v_tensor)
            simple_out = simple_out.view(1,2)
            simple_pred = torch.argmax(simple_out)
            simple_label = 'Day' if simple_pred==1 else 'Night'

            # ---- MobileNetV2 ----
            rgb_img = cv2.cvtColor(bgr_resized, cv2.COLOR_BGR2RGB)
            rgb_img = torch.tensor(rgb_img, dtype=torch.float32).permute(2,0,1)/255.0
            rgb_img = (rgb_img - mbv2_mean)/mbv2_std
            rgb_img = rgb_img.unsqueeze(0)  # add batch dim
            mbv2_out = mbv2_model(rgb_img)
            mbv2_out = mbv2_out.view(1,2)
            mbv2_pred = torch.argmax(mbv2_out)
            mbv2_label = 'Day' if mbv2_pred==0 else 'Night'

            # ---- Choose MobileNetV2 as final ----
            daynight_result = mbv2_label

    return render_template("upload.html",
                           logged_in=logged_in,
                           user_email=user_email,
                           uploaded_filename=uploaded_filename,
                           weather_result=weather_result,
                           daynight_result=daynight_result)


# -------------------- PROFILE --------------------
@app.route("/profile")
def profile():
    if not is_logged_in(): return redirect(url_for("login"))
    email=session['user_email']; user=users[email]; history=user_history.get(email,[])
    return render_template("profile.html", user=user, history=history)

# -------------------- ADMIN --------------------
@app.route("/admin/login", methods=["GET","POST"])
def admin_login():
    if request.method=="POST":
        email=request.form.get("email"); password=request.form.get("password")
        if email=="admin@example.com" and password=="admin123": session["admin_logged_in"]=True; return redirect(url_for("admin_dashboard"))
        return "Invalid admin credentials"
    return render_template("admin_login.html")

@app.route("/admin/dashboard")
def admin_dashboard():
    if not session.get("admin_logged_in"): return redirect(url_for("admin_login"))
    return render_template("admin_dashboard.html")

# -------------------- RUN APP --------------------
if __name__=="__main__":
    app.run(debug=True)
