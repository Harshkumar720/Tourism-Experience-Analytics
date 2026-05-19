# 🌍 Tourism Experience Analytics

A complete end-to-end Machine Learning project that analyzes tourism behavior and provides:

* ⭐ Attraction Rating Prediction
* 🧳 Visit Mode Classification
* 🎯 Personalized Attraction Recommendations
* 📊 Interactive Business Dashboard

Built using Python, Scikit-learn, XGBoost, and Streamlit.

---

## 📌 Project Overview

### Problem Statement

Tourism platforms collect large volumes of user reviews, ratings, and trip information. However, they often struggle to:

* Predict how users will rate attractions.
* Understand the type of trip a user is taking.
* Recommend attractions tailored to user preferences.

### Objective

This project uses machine learning to:

1. Predict the rating a user is likely to give to an attraction.
2. Classify the user's visit mode.
3. Recommend attractions based on historical behavior.
4. Visualize insights through a professional Streamlit dashboard.

### Real-World Applications

* Travel booking websites
* Tourism recommendation systems
* Destination marketing platforms
* Personalized travel planning tools

---

# 🚀 Key Features

### ⭐ Rating Prediction

Predicts attraction ratings on a scale of 1 to 5.

### 🧳 Visit Mode Classification

Classifies trip type into:

* Business
* Family
* Couples
* Friends
* Solo

### 🎯 Recommendation System

Provides personalized attraction recommendations using item-based collaborative filtering.

### 📊 Interactive Dashboard

Includes:

* KPI cards
* Prediction pages
* Recommendation page
* Data Explorer
* Model Performance analytics

---

# 🛠️ Technology Stack

| Category                | Tools Used                  |
| ----------------------- | --------------------------- |
| Programming Language    | Python                      |
| Data Analysis           | Pandas, NumPy               |
| Visualization           | Matplotlib, Seaborn, Plotly |
| Machine Learning        | Scikit-learn, XGBoost       |
| Recommendation System   | Cosine Similarity           |
| Model Serialization     | Joblib                      |
| Web App                 | Streamlit                   |
| Development Environment | Jupyter Notebook, VS Code   |
| Version Control         | Git, GitHub                 |

---

# 📂 Project Structure

```text
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
│   ├── model_comparison.csv
│   └── feature_importance.csv
│
├── notebooks/
│   ├── 01_data_loading.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_preprocessing.ipynb
│   ├── 04_regression.ipynb
│   ├── 05_classification.ipynb
│   └── 06_recommendation.ipynb
│
├── main.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

# 📊 Dataset Information

The project uses tourism-related datasets containing:

* User demographics
* Attraction details
* Ratings
* Visit modes
* Geographic information

### Important Features

* `UserId`
* `AttractionId`
* `Rating`
* `VisitMode`
* `Continent`
* `Region`
* `Country`
* `CityName`
* `AttractionType`
* `VisitYear`
* `VisitMonth`

---

# 🔄 Project Workflow

1. Data Loading
2. Exploratory Data Analysis (EDA)
3. Data Preprocessing
4. Regression Modeling
5. Classification Modeling
6. Recommendation System
7. Streamlit Dashboard Development
8. Model Performance Visualization

---

# 🤖 Machine Learning Models

## Rating Prediction (Regression)

Models trained:

* Linear Regression
* Random Forest Regressor
* XGBoost Regressor

**Best Model:** XGBoost Regressor

### Metrics

* R² Score: 0.1259
* RMSE: 0.9073
* MAE: 0.7103

---

## Visit Mode Classification

Models trained:

* Logistic Regression
* Random Forest Classifier
* XGBoost Classifier

**Best Model:** Random Forest Classifier

### Metrics

* Accuracy: 0.4790
* Precision: 0.4898
* Recall: 0.4790
* F1 Score: 0.4600 (approx.)

---

## Recommendation System

Method used:

* Item-Based Collaborative Filtering
* Cosine Similarity

Saved artifacts:

* `item_similarity_df.pkl`
* `user_item_matrix.pkl`
* `recommender.pkl`

---

# 💾 Saved Model Files

| File                           | Purpose                               |
| ------------------------------ | ------------------------------------- |
| `rating_model.pkl`             | Best regression model                 |
| `rating_preprocessor.pkl`      | Regression preprocessing pipeline     |
| `visit_mode_model.pkl`         | Best classification model             |
| `visit_mode_preprocessor.pkl`  | Classification preprocessing pipeline |
| `visit_mode_label_encoder.pkl` | Label decoding                        |
| `recommender.pkl`              | Recommendation logic                  |
| `item_similarity_df.pkl`       | Item similarity matrix                |
| `user_item_matrix.pkl`         | User-item interaction matrix          |

---

# 📈 Streamlit Dashboard Pages

## 🏠 Home

* Project overview
* KPI cards
* Business insights

## ⭐ Rating Prediction

Predict expected attraction rating.

## 🧳 Visit Mode Prediction

Predict trip type.

## 🎯 Recommendations

Generate personalized attraction recommendations.

## 📂 Data Explorer

Browse and analyze processed data.

## 📊 Model Performance

Visualize regression and classification metrics.

## ℹ️ About

Project details and author information.

---

# ⚙️ Installation

## 1. Clone Repository

```bash
git clone https://github.com/your-username/Tourism-Experience-Analytics.git
cd Tourism-Experience-Analytics
```

## 2. Create Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### macOS/Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Running the Application

## Option 1

```bash
python main.py
```

## Option 2

```bash
streamlit run app/streamlit_app.py
```

---

# 📦 Requirements

Main libraries used:

* streamlit
* pandas
* numpy
* scikit-learn
* xgboost
* plotly
* matplotlib
* seaborn
* joblib

---

# 📊 Output Files Generated

* Trained model `.pkl` files
* Metric `.json` files
* Comparison `.csv` files
* Feature importance `.csv`
* Streamlit dashboard visualizations

---

# 🧠 Business Insights

* Personalized recommendations improve user engagement.
* Visit mode prediction enables targeted marketing.
* High-rated attractions can be promoted as premium experiences.
* Geographic insights help optimize regional campaigns.

---

# 🔮 Future Improvements

* Hyperparameter tuning with Optuna.
* Deep learning-based recommendation models.
* User authentication.
* Cloud deployment.
* Real-time data integration.

---

# 👨‍💻 Author

**Harsh Kumar**

B.Tech Project: Tourism Experience Analytics

---

# ⭐ GitHub Repository

If you found this project useful, consider giving it a star.

---

# 📄 License

This project is for educational and portfolio purposes.

