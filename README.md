# Fake Job Detection System using Python & NLP

## 📌 Project Overview
The Fake Job Detection System is a full-stack machine learning web application that detects whether a given job posting is **real or fraudulent** using **Natural Language Processing (NLP)** techniques and **Machine Learning algorithms**.  
This system helps users avoid job scams by analyzing job descriptions and related text data.

---

## 🎯 Objectives
- Detect fake job postings using NLP
- Build a machine learning classification model
- Deploy the model using a Flask-based web interface
- Provide a simple and user-friendly frontend

---

## 🛠️ Technologies Used

### Backend
- Python
- Flask
- Scikit-learn
- Pandas, NumPy
- Joblib

### NLP
- NLTK
- TF-IDF Vectorizer
- Text Cleaning & Stopword Removal

### Frontend
- HTML
- CSS
- Bootstrap (optional)

---

## 📂 Project Structure

fake-job-detection/
│
├── dataset/
│ └── fake_job_postings.csv
│
├── model/
│ ├── fake_job_model.pkl
│ └── tfidf_vectorizer.pkl
│
├── static/
│ └── style.css
│
├── templates/
│ └── index.html
│
├── app.py
├── train_model.py
├── preprocess.py
├── requirements.txt
└── README.md

---

## 📊 Dataset
The dataset contains job postings with various features such as:
- Job title
- Job description
- Requirements
- Company profile
- Fraudulent label (0 = Real, 1 = Fake)

Source: Public Kaggle Dataset (Fake Job Postings)

---

## ⚙️ Installation & Setup

### Step 1: Clone the Repository
```bash
git clone https://github.com/your-username/fake-job-detection.git
cd fake-job-detection
```
### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```
### Step 3: Train the Model
```bash
python train_model.py
```

### Step 4: Run the Application
```bash
python app.py
```
### Step 5: Open in Browser
```bash
http://127.0.0.1:5000/
