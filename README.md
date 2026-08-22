# 🚀 Smart Data Analyzer v2.0

Transform raw CSV data into actionable insights instantly with AI-powered analytics.

## 🏗️ Architecture (Separated Frontend/Backend)

- **Frontend**: Static HTML/JS/CSS hosted on **Netlify**.
- **Backend**: Flask API with Machine Learning hosted on **Render**.

---

## 🌍 Deployment Guide

### **1. Backend (Render)**

- Connect your GitHub repo to **Render.com**.
- Create a new **Web Service**.
- Select the repository.
- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `gunicorn app:app`
- Copy your Render URL (e.g., `https://your-app.onrender.com`).

### **2. Frontend (Netlify)**

- Open `frontend/js/script.js`.
- Update the `API_BASE_URL` on line 4 with your Render URL.
- Deploy the **`frontend`** folder only to **Netlify**.

---

## ✨ Features

- **AI Auto-Predictor**: Random Forest model with actual vs. predicted visualization.
- **Smart Insights**: Automated outlier detection and trend analysis.
- **Correlation Heatmap**: Visual link between variables.
- **Data Cleaner**: Automatically handles missing values and data types.
