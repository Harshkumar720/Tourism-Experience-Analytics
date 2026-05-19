🌍 Tourism Experience Analytics

An end-to-end Machine Learning project that predicts tourist attraction ratings, classifies visit modes, and generates personalized attraction recommendations through an interactive Streamlit dashboard.

📌 Project Overview
📖 Problem Statement

Tourism platforms collect large amounts of user reviews, ratings, and travel preferences, but often fail to fully leverage this data to personalize traveler experiences.

This project addresses three core business problems:

Predicting Attraction Ratings — Estimate the rating a user is likely to give to an attraction.
Classifying Visit Modes — Predict whether a trip is for Business, Family, Couples, Friends, or Solo travel.
Generating Recommendations — Recommend similar attractions based on collaborative filtering.
🎯 Objective

Build a complete data science solution that:

Cleans and preprocesses tourism datasets.
Performs exploratory data analysis (EDA).
Trains regression and classification models.
Builds a recommendation engine.
Deploys all functionality using Streamlit.
🌐 Real-World Use Cases
Personalized recommendations on travel websites.
Dynamic rating prediction before booking.
Travel segmentation for targeted marketing.
Destination intelligence dashboards.
✨ Key Features
⭐ Attraction Rating Prediction
🧳 Visit Mode Classification
🎯 Personalized Attraction Recommendations
📊 Exploratory Data Analysis Dashboard
🏆 Model Performance Tracking
📈 Feature Importance Visualization
🌙 Professional Dark-Themed UI
🛠️ Technology Stack
Programming Language
Python 3.10+
Data Analysis
Pandas
NumPy
Visualization
Matplotlib
Plotly
Seaborn
Machine Learning
Scikit-learn
XGBoost
Deployment
Streamlit
Model Serialization
Joblib
📂 Project Structure
Tourism-Experience-Analytics/
│
├── app/
│   └── streamlit_app.py
│
├── data/
│   ├── raw/
│   │   ├── transaction.csv
│   │   ├── user.csv
│   │   ├── item.csv
│   │   ├── city.csv
│   │   ├── country.csv
│   │   ├── region.csv
│   │   ├── continent.csv
│   │   ├── type.csv
│   │   └── visit_mode.csv
│   │
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
📊 Dataset Information
Raw Data Files
File	Description
transaction.csv	User transactions and ratings
user.csv	User demographic information
item.csv	Attraction details
city.csv	City metadata
country.csv	Country metadata
region.csv	Regional metadata
continent.csv	Continent metadata
type.csv	Attraction type mapping
visit_mode.csv	Visit mode labels
Key Columns Used
Rating
VisitMode
Attraction
AttractionType
Continent
Region
Country
CityName
VisitYear
VisitMonth
📓 Notebook Workflow
1️⃣ Data Loading
Load all raw CSV files.
Inspect shapes and columns.
Merge datasets.
2️⃣ Exploratory Data Analysis
Missing value analysis.
Distribution plots.
Correlation analysis.
Business insights.
3️⃣ Data Preprocessing
Handle missing values.
Feature engineering.
Encoding.
Save final_processed_data.csv.
4️⃣ Regression Modeling
Predict Rating.
Compare Linear Regression, Random Forest, and XGBoost.
Save best model.
5️⃣ Classification Modeling
Predict VisitMode.
Compare Logistic Regression, Random Forest, and XGBoost.
Save best model and label encoder.
6️⃣ Recommendation System
Build user-item matrix.
Compute cosine similarity.
Save recommendation artifacts.
🤖 Machine Learning Models
⭐ Regression Task
Target Variable

Rating

Models Evaluated
Linear Regression
Random Forest Regressor
XGBoost Regressor
Best Model

XGBoost Regressor

Metrics
R² Score: 0.1259
RMSE: 0.9073
MAE: 0.7103
🧳 Classification Task
Target Variable

VisitMode

Models Evaluated
Logistic Regression
Random Forest Classifier
XGBoost Classifier
Best Model

Random Forest Classifier

Metrics
Accuracy: 0.4790
Precision: 0.4898
Recall: 0.4790
F1 Score: 0.4530 (approx.)
🎯 Recommendation System
Approach

Item-based collaborative filtering using cosine similarity.

Output

Top-N similar attractions for a selected attraction.

💾 Saved Model Artifacts
File	Purpose
rating_model.pkl	Best regression model
rating_preprocessor.pkl	Regression preprocessing pipeline
visit_mode_model.pkl	Best classification model
visit_mode_preprocessor.pkl	Classification preprocessing pipeline
visit_mode_label_encoder.pkl	Decodes numeric labels
recommender.pkl	Recommendation helper object
item_similarity_df.pkl	Item-item similarity matrix
user_item_matrix.pkl	User-item matrix
📈 Metrics Files
File	Description
regression_metrics.json	Regression KPI metrics
classification_metrics.json	Classification KPI metrics
model_comparison.csv	Combined model comparison
feature_importance.csv	Top feature importances
🖥️ Streamlit Dashboard Modules
🏠 Home

Overview dashboard with KPIs and charts.

⭐ Rating Prediction

Predict expected attraction rating.

🧳 Visit Mode Prediction

Predict trip type.

🎯 Recommendation System

Suggest similar attractions.

📊 Data Explorer

Explore dataset interactively.

🏆 Model Performance

Display metrics, comparisons, and feature importance.

ℹ️ About

Project summary and author information.

🚀 Installation Guide
1. Clone Repository
git clone https://github.com/your-username/Tourism-Experience-Analytics.git
cd Tourism-Experience-Analytics
2. Create Virtual Environment
Windows
python -m venv venv
venv\Scripts\activate
macOS/Linux
python3 -m venv venv
source venv/bin/activate
3. Install Dependencies
pip install -r requirements.txt
▶️ Run the Application
Option 1
python main.py
Option 2
streamlit run app/streamlit_app.py
📦 requirements.txt
pandas
numpy
matplotlib
seaborn
plotly
scikit-learn
xgboost
streamlit
joblib
🔄 Execution Workflow
Run 01_data_loading.ipynb
Run 02_eda.ipynb
Run 03_preprocessing.ipynb
Run 04_regression.ipynb
Run 05_classification.ipynb
Run 06_recommendation.ipynb
Launch Streamlit using python main.py
📷 Dashboard Screenshots

Add screenshots here after uploading to GitHub.

Home Dashboard
Rating Prediction
Visit Mode Prediction
Recommendation System
Model Performance
📌 Business Insights
Personalized recommendations can increase user engagement.
Visit mode prediction enables better segmentation.
Highly rated attractions can be promoted as premium experiences.
Geographic trends help optimize marketing strategies.
🔮 Future Enhancements
Deploy on Streamlit Community Cloud.
Add user authentication.
Integrate real-time APIs.
Use deep learning models.
Add sentiment analysis of reviews.
🧪 Example Predictions
Rating Prediction

Input: Paris, Museum, Family, June 2023
Output: ⭐ 4.47 / 5

Visit Mode Prediction

Input: Tokyo, Historical Site
Output: 🧳 Family

Recommendation

Input: Eiffel Tower
Output: Louvre Museum, Arc de Triomphe, Notre-Dame Cathedral

👨‍💻 Author

Harsh Kumar

B.Tech Student | Machine Learning Enthusiast | Data Science Practitioner

GitHub: https://github.com/your-username
LinkedIn: https://www.linkedin.com/in/your-linkedin-profile
📜 License

This project is developed for academic and portfolio purposes.

⭐ Support

If you found this project useful:

⭐ Star the repository
🍴 Fork the project
📝 Share feedback
🙏 Acknowledgements
Scikit-learn
XGBoost
Streamlit
Plotly
Open-source Python community

Built with Python, Machine Learning, and Streamlit to deliver a complete Tourism Experience Analytics solution.
