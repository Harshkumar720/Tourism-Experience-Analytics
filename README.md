# 🌍 Tourism Experience Analytics

A complete AI-powered tourism analytics platform built using Machine Learning, Streamlit, and Recommendation Systems.

This project predicts:

* ⭐ Tourist attraction ratings
* 🧳 Tourist visit modes
* 🎯 Personalized attraction recommendations

The application provides an interactive dashboard with predictive analytics, recommendation systems, and tourism insights using real-world tourism experience data.

---

# 📌 Project Overview

Tourism industries generate massive amounts of visitor interaction data. This project uses Machine Learning models to analyze tourism behavior and provide intelligent predictions and recommendations.

The system combines:

* Regression Modeling
* Classification Modeling
* Recommendation Systems
* Streamlit Dashboard Deployment
* Interactive Data Visualization

---

# 🚀 Features

## ⭐ Attraction Rating Prediction

Predicts the rating a tourist may give to an attraction based on:

* Continent
* Region
* Country
* City
* Attraction Type
* Visit Year
* Visit Month
* Visit Mode

### Models Used

* Linear Regression
* Random Forest Regressor
* XGBoost Regressor

---

## 🧳 Visit Mode Prediction

Predicts the type of tourist visit:

* Business
* Family
* Couples
* Friends
* Solo

### Models Used

* Logistic Regression
* Random Forest Classifier
* XGBoost Classifier

---

## 🎯 Tourism Recommendation System

Recommends tourist attractions using collaborative filtering.

### Recommendation Features

* Similar attraction recommendation
* User-item interaction analysis
* Personalized tourism suggestions

---

# 🛠️ Technologies Used

## Programming Language

* Python

## Machine Learning Libraries

* Scikit-learn
* XGBoost
* Pandas
* NumPy

## Visualization

* Plotly
* Matplotlib
* Seaborn

## Deployment

* Streamlit

## Model Serialization

* Joblib

---

# 📂 Project Structure

```bash
Tourism-Experience-Analytics/
│
├── app/
│   └── streamlit_app.py
│
├── data/
│   ├── raw/
│   └── processed/
│       └── final_processed_data.csv
│
├── models/
│   ├── rating_model.pkl
│   ├── rating_preprocessor.pkl
│   ├── visit_mode_model.pkl
│   ├── visit_mode_preprocessor.pkl
│   ├── visit_mode_label_encoder.pkl
│   ├── recommender.pkl
│   ├── item_similarity_df.pkl
│   ├── user_item_matrix.pkl
│   ├── regression_metrics.json
│   ├── classification_metrics.json
│   ├── feature_importance.csv
│   └── model_comparison.csv
│
├── notebooks/
│   ├── 01_data_loading.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_preprocessing.ipynb
│   ├── 04_regression.ipynb
│   ├── 05_classification.ipynb
│   └── 06_recommendation.ipynb
│
├── requirements.txt
└── README.md
```

---

# ⚙️ Installation

## 1️⃣ Clone Repository

```bash
git clone https://github.com/your-username/Tourism-Experience-Analytics.git
```

---

## 2️⃣ Navigate to Project Folder

```bash
cd Tourism-Experience-Analytics
```

---

## 3️⃣ Create Virtual Environment (Optional)

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / Mac

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 4️⃣ Install Requirements

```bash
pip install -r requirements.txt
```

---

# ▶️ Run Streamlit Application

```bash
streamlit run app/streamlit_app.py
```

---

# 📊 Machine Learning Workflow

## 1. Data Collection

Tourism interaction datasets are collected and merged.

## 2. Data Cleaning

* Missing value handling
* Duplicate removal
* Feature formatting

## 3. Feature Engineering

* Label Encoding
* One-Hot Encoding
* Numerical scaling

## 4. Model Training

Regression and classification models are trained.

## 5. Evaluation

Models are evaluated using:

### Regression Metrics

* MAE
* RMSE
* R² Score

### Classification Metrics

* Accuracy
* Precision
* Recall
* F1 Score

## 6. Deployment

Best models are deployed using Streamlit.

---

# 📈 Dashboard Features

The Streamlit dashboard includes:

* Interactive UI
* Dynamic dropdowns
* Real-time predictions
* Recommendation engine
* Performance metrics
* Tourism analytics visualizations
* Responsive layout

---

# 📌 Input Features

## Rating Prediction Inputs

| Feature         | Description                  |
| --------------- | ---------------------------- |
| Continent       | Tourist continent            |
| Region          | Tourist region               |
| Country         | Tourist country              |
| City            | Tourist city                 |
| Attraction Type | Type of attraction           |
| Visit Mode      | Business / Family / Solo etc |
| Visit Year      | Year of visit                |
| Visit Month     | Month of visit               |

---

# 📌 Output Predictions

## Rating Prediction

Predicts tourist rating between:

```bash
1 ⭐ to 5 ⭐
```

## Visit Mode Prediction

Predicts:

```bash
Business
Family
Couples
Friends
Solo
```

---

# 🧠 Recommendation System

The recommendation engine uses collaborative filtering techniques.

### Recommendation Process

* Builds user-item interaction matrix
* Computes similarity scores
* Recommends related attractions

---

# 📦 Saved Models

| File                         | Purpose                      |
| ---------------------------- | ---------------------------- |
| rating_model.pkl             | Rating prediction model      |
| visit_mode_model.pkl         | Visit mode classifier        |
| recommender.pkl              | Recommendation system        |
| visit_mode_label_encoder.pkl | Label decoding               |
| item_similarity_df.pkl       | Similarity matrix            |
| user_item_matrix.pkl         | User-item interaction matrix |

---

# 📷 Future Improvements

Planned improvements:

* Deep Learning integration
* Real-time tourism APIs
* Cloud deployment
* User authentication
* Sentiment analysis
* NLP-based recommendation system
* Multi-language support
* Mobile responsive optimization

---

# 🧪 Example Use Cases

* Smart tourism platforms
* Travel recommendation systems
* Tourism behavior analytics
* Tourism market research
* Visitor experience analysis
* Personalized tourism applications

---

# 👨‍💻 Author

## Harsh Kumar

B.Tech Student | Machine Learning Enthusiast | AI Developer

---

# 📄 License

This project is developed for educational and learning purposes.

---

# ⭐ Acknowledgements

Special thanks to:

* Scikit-learn
* XGBoost
* Streamlit
* Open-source ML community

---

# 📬 Contact

For suggestions, improvements, or collaborations:

* GitHub: [https://github.com/](https://github.com/)
* LinkedIn: [https://linkedin.com/](https://linkedin.com/)

---

# 🌟 Final Note

This project demonstrates an end-to-end Machine Learning workflow including:

* Data preprocessing
* Model training
* Evaluation
* Recommendation systems
* Deployment using Streamlit

It combines practical Machine Learning concepts with a real-world tourism analytics use case.

