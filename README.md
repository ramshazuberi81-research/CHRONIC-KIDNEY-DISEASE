 🧪 Chronic Kidney Disease (CKD) Detection Using Machine Learning

👩‍⚕️ Author: Ramsha
📍 Field: Clinical Epidemiology | AI & Data Science Learner

 📋 PROJECT OVERVIEW
 This project uses Python-based data analysis and AI to identify key predictors 
 of Chronic Kidney Disease (CKD). The dataset includes various clinical and 
 biochemical indicators to help classify patients as CKD or non-CKD.
 The goal is to detect patterns, visualize important factors, and 
 build a predictive model for early detection.

 🎯 OBJECTIVES
 ✅ Clean and preprocess clinical data
 ✅ Explore variables using descriptive statistics
 ✅ Visualize distributions and relationships
 ✅ Perform hypothesis testing (t-test, correlation)
✅ Build a Logistic Regression model for CKD prediction
✅ Interpret results using accuracy and clinical relevance

 🧰 TOOLS & LIBRARIES
 - Python (Jupyter Notebook)
 - Pandas → Data manipulation
- NumPy → Numerical operations
- Matplotlib & Seaborn → Visualization
 - SciPy → Statistical tests (t-test, ANOVA)
- Scikit-learn → Machine Learning models
- Statsmodels → Regression & odds ratio interpretation

 📂 DATASET DESCRIPTION
 The dataset contains patient-level clinical and laboratory information:
 Columns include:
 age          → Age in years
 bp           → Blood pressure (mmHg)
 sg           → Urine specific gravity
   al           → Albumin in urine
  bgr          → Blood glucose random (mg/dL)
  bu           → Blood urea (mg/dL)
  sc           → Serum creatinine (mg/dL)
   hemo         → Hemoglobin (g/dL)
  htn          → Hypertension (yes/no)
  dm           → Diabetes Mellitus (yes/no)
   classification → Target variable (ckd / notckd)

 🧩 WORKFLOW
 Step 1️⃣: Load the dataset
import pandas as pd
data = pd.read_csv("chronic_kidney_disease.csv")

Step 2️⃣: Explore and clean data
print(data.info())
print(data.describe())

Step 3️⃣: Handle missing values
data = data.dropna()

 Step 4️⃣: Visualize relationships
import seaborn as sns
import matplotlib.pyplot as plt

sns.boxplot(x='classification', y='sc', data=data)
plt.title('Serum Creatinine in CKD vs Non-CKD')
plt.show()

 Step 5️⃣: Statistical test (t-test)
from scipy.stats import ttest_ind
ckd = data[data['classification'] == 'ckd']['sc'].dropna()
non_ckd = data[data['classification'] == 'notckd']['sc'].dropna()
t_stat, p_val = ttest_ind(ckd, non_ckd)
print("T-statistic:", t_stat)
print("P-value:", p_val)

 Step 6️⃣: Machine Learning Model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

X = data[['age','bp','bgr','sc','hemo']]
y = data['classification'].map({'ckd':1, 'notckd':0})

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Model Accuracy:", accuracy_score(y_test, y_pred)
Step 7️⃣: Interpret results
- If P-value < 0.05, the difference in serum creatinine is significant.
 - High serum creatinine and blood urea indicate impaired kidney function.
- Logistic Regression accuracy shows how well the model predicts CKD.

 📊 RESULTS
 - Creatinine and Blood Urea levels are higher in CKD patients.
 - Logistic Regression achieved ~90% accuracy.
 - Results align with clinical understanding of CKD risk factors.

 📚 CLINICAL INSIGHT
 - High serum creatinine → poor kidney filtration
 - Low hemoglobin → anemia due to CKD
 - Hypertension & Diabetes → major CKD risk factors

 🧠 FUTURE SCOPE
- Use advanced ML models (Random Forest, XGBoost)
 - Add ROC Curve & Confusion Matrix for model evaluation
 - Build a simple web app for CKD prediction

