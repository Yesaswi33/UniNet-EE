from flask import Flask, render_template, request, redirect, url_for, session
import os
import uuid
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import random
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

import cv2
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import YOLO
import segmentation_models_pytorch as smp
import torchvision.models as models

# -------------------- FLASK APP SETUP --------------------
app = Flask(__name__)
app.secret_key = "yesaswi_madabattula"
app.permanent_session_lifetime = timedelta(minutes=30)

users = {}
user_history = {}
otp_store = {}

UPLOAD_FOLDER = os.path.join("static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

device = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------- SMTP CONFIG --------------------
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
SMTP_EMAIL = "madabattulayesaswi@gmail.com"
SMTP_PASSWORD = "uvzy hkft dept wbrd"  # App password

TEAM_EMAILS = [
    "madabattulayesaswi@gmail.com",
    "dileepvayilapalli@gmail.com",
    "chandrasekhar20634@gmail.com",
    "ramyasreeveerla117@gmail.com"
]

# -------------------- LOAD MODELS --------------------

# 1️⃣ YOLO Detection Model
yolo_model = YOLO("/Users/yesaswimadabattula/Documents/PROJECTS/major_project/saved_models/bdd_project_best.pt")

# 2️⃣ Segmentation Model
seg_model = smp.DeepLabV3(
    encoder_name="mobilenet_v2",
    encoder_weights="imagenet",
    in_channels=3,
    classes=19
)
seg_model.load_state_dict(torch.load(
    "/Users/yesaswimadabattula/Documents/PROJECTS/major_project/saved_models/seg_model.pth",
    map_location=device
))
seg_model.to(device)
seg_model.eval()

# 3️⃣ Classification Model
state_dict = torch.load(
    "/Users/yesaswimadabattula/Documents/PROJECTS/major_project/saved_models/class_model.pth",
    map_location=device
)

num_classes = state_dict["classifier.1.weight"].shape[0]

model_cls = models.efficientnet_b0(pretrained=False)
model_cls.classifier[1] = nn.Linear(1280, num_classes)
model_cls.load_state_dict(state_dict)
model_cls.to(device)
model_cls.eval()

idx_to_class = {i: f"Class_{i}" for i in range(num_classes)}

# -------------------- HELPER --------------------
def is_logged_in():
    return 'user_email' in session

# -------------------- HOME --------------------
@app.route('/')
def home():
    logged_in = is_logged_in()
    user_email = session['user_email'] if logged_in else None
    return render_template("index.html", logged_in=logged_in, user_email=user_email)

# -------------------- ABOUT --------------------
@app.route('/about')
def about():
    logged_in = is_logged_in()
    user_email = session['user_email'] if logged_in else None
    return render_template("about.html", logged_in=logged_in, user_email=user_email)

# -------------------- CONTACT (NEWLY ADDED) --------------------
@app.route("/contact", methods=["GET", "POST"])
def contact():
    logged_in = is_logged_in()
    user_email = session['user_email'] if logged_in else None

    if request.method == "POST":
        name = request.form.get("name")
        email = request.form.get("email")
        user_message = request.form.get("message")

        subject = f"UrbanVision AI Feedback from {name}"
        body = f"""
        Name: {name}
        Email: {email}

        Message:
        {user_message}
        """

        try:
            server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
            server.starttls()
            server.login(SMTP_EMAIL, SMTP_PASSWORD)

            for recipient in TEAM_EMAILS:
                msg = MIMEMultipart()
                msg['From'] = SMTP_EMAIL
                msg['To'] = recipient
                msg['Subject'] = subject
                msg.attach(MIMEText(body, 'plain'))
                server.send_message(msg)

            server.quit()

            return redirect(url_for("contact", status="success"))

        except Exception as e:
            print("Error sending email:", e)
            return redirect(url_for("contact", status="error"))

    status = request.args.get("status")
    message = ""

    if status == "success":
        message = "Your message has been sent successfully!"
    elif status == "error":
        message = "There was an error sending your message. Please try again later."

    return render_template("contact.html",
                           message=message,
                           logged_in=logged_in,
                           user_email=user_email)

# -------------------- REGISTER --------------------
@app.route('/register', methods=['GET','POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']
        if email in users:
            return "User already exists"
        users[email] = {'name': name,'password': password}
        return redirect(url_for('login'))
    return render_template("register.html")

# -------------------- LOGIN --------------------
@app.route('/login', methods=['GET','POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        user = users.get(email)
        if user and user['password'] == password:
            session['user_email'] = email
            session.permanent = True
            return redirect(url_for('home'))
        return "Invalid credentials"
    return render_template("login.html")

# -------------------- LOGOUT --------------------
@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for("home"))

# -------------------- ADMIN LOGIN --------------------
@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    logged_in = is_logged_in()
    user_email = session['user_email'] if logged_in else None

    if request.method == 'POST':
        admin_email = request.form.get('email')
        admin_password = request.form.get('password')

        # Simple static admin credentials (you can replace with DB check)
        if admin_email == "admin@gmail.com" and admin_password == "admin123":
            session['admin_logged_in'] = True
            return redirect(url_for('admin_dashboard'))
        else:
            flash("Invalid Admin Credentials", "error")
            return redirect(url_for('admin_login'))

    return render_template(
        "admin_login.html",
        logged_in=logged_in,
        user_email=user_email
    )
    
    

# -------------------- ADMIN DASHBOARD --------------------
@app.route('/admin/dashboard')
def admin_dashboard():
    if not session.get('admin_logged_in'):
        return redirect(url_for('admin_login'))

    logged_in = is_logged_in()
    user_email = session['user_email'] if logged_in else None

    return render_template(
        "admin_dashboard.html",
        logged_in=logged_in,
        user_email=user_email
    )
    
# -------------------- UPLOAD ROUTE --------------------
@app.route("/upload", methods=["GET","POST"])
def upload():
    if not is_logged_in():
        return redirect(url_for("login"))

    logged_in = True
    user_email = session['user_email']

    uploaded_filename = None
    detection_img = None
    segmentation_img = None
    classification_result = None
    combined_img = None

    if request.method == "POST":
        file = request.files.get("file")

        if file and file.filename != "":
            unique_name = str(uuid.uuid4()) + "_" + file.filename
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], unique_name)
            file.save(file_path)

            user_history.setdefault(user_email, []).append(unique_name)
            uploaded_filename = unique_name

            # Detection
            det_results = yolo_model.predict(file_path, conf=0.25)
            det_img = det_results[0].plot()
            det_output_path = os.path.join(app.config["UPLOAD_FOLDER"], "det_" + unique_name)
            cv2.imwrite(det_output_path, det_img)
            detection_img = "det_" + unique_name

            # Segmentation
            image = cv2.imread(file_path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_seg = cv2.resize(image_rgb, (256, 256))
            tensor_seg = torch.tensor(image_seg).permute(2,0,1).unsqueeze(0).float()/255.0
            tensor_seg = tensor_seg.to(device)

            with torch.no_grad():
                output = seg_model(tensor_seg)
                pred_mask = torch.argmax(output, dim=1).squeeze().cpu().numpy()

            mask_colored = cv2.applyColorMap((pred_mask*10).astype(np.uint8), cv2.COLORMAP_JET)
            mask_colored = cv2.resize(mask_colored, (image.shape[1], image.shape[0]))
            seg_overlay = cv2.addWeighted(image_rgb, 0.7, mask_colored, 0.3, 0)
            seg_overlay = cv2.cvtColor(seg_overlay, cv2.COLOR_RGB2BGR)

            seg_output_path = os.path.join(app.config["UPLOAD_FOLDER"], "seg_" + unique_name)
            cv2.imwrite(seg_output_path, seg_overlay)
            segmentation_img = "seg_" + unique_name

            # Classification
            image_cls = cv2.resize(image_rgb, (224,224))
            tensor_cls = torch.tensor(image_cls).permute(2,0,1).unsqueeze(0).float()/255.0
            tensor_cls = tensor_cls.to(device)

            with torch.no_grad():
                output_cls = model_cls(tensor_cls)
                probabilities = F.softmax(output_cls, dim=1)
                confidence, predicted = torch.max(probabilities, 1)

            predicted_class = idx_to_class.get(predicted.item(), "Unknown")
            confidence_score = confidence.item()
            classification_result = f"{predicted_class} ({confidence_score:.2f})"

            # Combined
            combined = cv2.addWeighted(det_img, 0.6, seg_overlay, 0.4, 0)
            combined_output_path = os.path.join(app.config["UPLOAD_FOLDER"], "combined_" + unique_name)
            cv2.imwrite(combined_output_path, combined)
            combined_img = "combined_" + unique_name

    return render_template("upload.html",
                           logged_in=logged_in,
                           user_email=user_email,
                           uploaded_filename=uploaded_filename,
                           detection_img=detection_img,
                           segmentation_img=segmentation_img,
                           classification_result=classification_result,
                           combined_img=combined_img)

# -------------------- PROFILE --------------------
@app.route("/profile")
def profile():
    if not is_logged_in():
        return redirect(url_for("login"))
    email = session['user_email']
    user = users[email]
    history = user_history.get(email,[])
    return render_template("profile.html", user=user, history=history)

# -------------------- RUN --------------------
if __name__=="__main__":
    app.run(debug=True)