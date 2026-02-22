# 🌆 UrbanVision AI

## Unified Multi-Task AI System for Urban & Environmental Scene Understanding

UrbanVision AI is a **hybrid multi-task computer vision platform** that performs:

- 🧠 Image Classification  
- 🚗 Object Detection  
- 🏙️ Semantic Segmentation  

All from a **single uploaded image**.

It also includes a **full-stack web application** with authentication, user history, OTP-based password recovery, admin dashboard, and AI-powered image analysis.

This project demonstrates the integration of:

> 🤖 Artificial Intelligence + 🌐 Web Development + 🏗️ System Design

---

# 🚀 Key Highlights

- 🧠 Hybrid AI Model → Classification + Detection + Segmentation  
- ⚡ Single Image → Multiple Insights  
- 👤 User Authentication System  
- 🔐 Secure Password Reset with OTP (Email)  
- 📁 File Upload & AI Analysis  
- 🕒 User History Tracking  
- 👤 Profile Dashboard  
- 🛡️ Session & Cookie-Based Login  
- 📩 Contact Form with Email Integration  
- 👨‍💻 Admin Dashboard  
- 🌐 Responsive Web UI  
- 🧩 Modular Flask Architecture  

---

# 📌 Project Architecture


UrbanVision-AI/
│
├── app.py # Main Flask backend
├── uploads/ # Uploaded files
├── static/
│ ├── css/
│ │ └── style.css # Main styles
│ └── team_images/ # Team photos
│
├── templates/
│ ├── index.html
│ ├── login.html
│ ├── register.html
│ ├── forgot_password.html
│ ├── reset_password.html
│ ├── upload.html
│ ├── profile.html
│ ├── about.html
│ ├── contact.html
│ ├── admin_login.html
│ ├── admin_dashboard.html
│
└── README.md # Project documentation


---

# 🧠 AI Capabilities

UrbanVision AI performs three core computer vision tasks:

---

## 1️⃣ Image Classification

Identifies scene attributes such as:

- 🌞 Day / Night  
- 🌧️ Weather (Clear, Rainy, Foggy)  
- 🛣️ Road Type (Highway, City, Residential)  
- 🌳 Environment Category  

---

## 2️⃣ Object Detection

Detects objects like:

- 🚗 Vehicles  
- 🚶 Pedestrians  
- 🚴 Cyclists  
- 🚦 Traffic Lights  
- 🛑 Traffic Signs  

---

## 3️⃣ Semantic Segmentation

Performs pixel-wise segmentation of:

- 🛣️ Roads  
- 🏢 Buildings  
- ☁️ Sky  
- 🚗 Vehicles  
- 🚶 Sidewalks  
- 🟡 Lane Markings  

---

# 💻 Web Application Features

## 👤 User Authentication

- Register new users  
- Login with email & password  
- Session-based authentication (cookies)  
- Logout functionality  

---

## 🔐 Forgot Password (OTP System)

- Email-based OTP verification  
- OTP expiry (10 minutes)  
- Secure password reset  

---

## 📁 File Upload & Analysis

- Users upload images/files  
- Files stored securely with unique IDs  
- User-specific upload history tracked  

---

## 🕒 User History

Each user has a personal history of uploaded files stored and displayed in the profile page.

---

## 👤 User Profile Page

Displays:

- Name  
- Email  
- Upload History  

---

## 📩 Contact Form

- Sends emails to team members  
- Prevents duplicate form submission  
- Uses POST → Redirect → GET pattern  

---

## 👨‍💻 Admin Panel

- Admin login  
- Dashboard with system stats  
- Recent AI analysis logs  
- Quick action panel  

---

# 🍪 Session & Cookie System

- Flask session used for authentication  
- Session stores user email  
- Optional session timeout configuration  
- Navbar dynamically changes based on login status  

### Dynamic Navbar Example

| User Status        | Navbar Options |
|-------------------|----------------|
| Guest              | Home, About, Contact, Login, Register |
| Logged-in User     | Home, Upload, Profile, History, Logout |
| Admin              | Dashboard, System Controls |

---

# 🔄 Dynamic Navbar System

Each page dynamically displays navigation links based on the user's session state.

---

# 🛠️ Technologies Used

## Backend

- Python 🐍  
- Flask 🌐  
- SMTP (Email Service)  
- Session & Cookies  

## Frontend

- HTML5  
- CSS3  
- JavaScript  
- Jinja2 Templates  

## AI / ML (Conceptual Integration)

- CNN (Convolutional Neural Networks)  
- Multi-head Attention  
- Semantic Segmentation  
- Object Detection Models  

## Security

- OTP Verification  
- Session Management  
- Unique File Naming  
- Email Authentication  

---

# ⚙️ How to Run the Project

1️⃣ Install Dependencies
```bash
pip install flask
2️⃣ Run the Application
python app.py
3️⃣ Open in Browser
http://127.0.0.1:5000/
🧪 Demo Flow (User Journey)

Register a new account

Login to the system

Upload an image

View upload history in profile

Test forgot password (OTP email)

Explore About & Contact pages

Login as admin and view dashboard

🌍 Real-World Applications

🚦 Smart Traffic Monitoring

🏙️ Smart City Planning

👮 Public Safety & Surveillance

🌦️ Environmental Analysis

🤖 AI Research & Education

📊 Urban Data Analytics

🧩 Unique Features of UrbanVision AI

Unified AI pipeline (multi-task learning)

Full-stack AI web platform

Real-time user interaction

Modular & scalable design

Educational + practical implementation

📈 Future Enhancements

✅ Database integration (MySQL / MongoDB)

✅ Real AI model inference (YOLO, UNet, ResNet)

✅ Google OAuth Login

✅ Role-based access control

✅ Real-time analytics dashboard

✅ Cloud deployment (AWS / Azure / GCP)

✅ REST API for AI services

✅ Mobile app integration

👨‍💻 Developed By

UrbanVision AI Team

Venkata Padma Yesaswi Madabattula (Team Leader)

Vayilapalli Dileep

K. Chandra Sekhar

V. Ramya Sree

Supervisor: Dr. P. Satheesh
MVGR College of Engineering, Vizianagaram
Department of Data Engineering

⭐ Final Note

UrbanVision AI is not just a project —

It is a fusion of:

🤖 Artificial Intelligence
🌐 Web Engineering
🏙️ Smart City Vision
