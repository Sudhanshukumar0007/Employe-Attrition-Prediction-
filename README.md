# Employee Attrition Prediction using Logistic Regression

This project aims to predict whether an employee is likely to leave the company (attrition) using logistic regression. The model is built using a dataset from IBM HR Analytics and provides valuable insights into employee behavior.

---

## 📁 Dataset

- *Name*: WA_Fn-UseC_-HR-Employee-Attrition.csv
- *Source*: IBM HR Analytics Employee Attrition & Performance dataset
- *Target Variable*: Attrition (Yes/No)

---

## 📌 Objectives

- Predict employee attrition using logistic regression.
- Perform data preprocessing (cleaning, encoding, scaling).
- Visualize feature relationships with attrition.
- Evaluate model performance using various metrics.

---

## 📦 Libraries Used

- pandas - for data manipulation
- numpy - for numerical operations
- seaborn and matplotlib - for data visualization
- sklearn - for model building and evaluation

---

## 🔄 Workflow

### 1. Upload Dataset
```python
from google.colab import files
uploaded = files.upload()
2. Import Libraries
3. Data Preprocessing
Label encoding for categorical variables

Dropping non-informative columns

Feature scaling using StandardScaler

4. Model Training
python
Copy
Edit
model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)
5. Model Evaluation
Accuracy Score

Classification Report

Confusion Matrix

ROC Curve and AUC Score

Precision-Recall Curve

📊 Visualizations
Confusion Matrix Heatmap

ROC Curve

Precision-Recall Curve

Class Distribution

KDE Plots for features like Age, MonthlyIncome, DistanceFromHome, YearsAtCompany with respect to attrition

✅ Results
Evaluated using accuracy, precision, recall, F1-score, AUC
![WhatsApp Image 2025-05-27 at 10 40 27 AM](https://github.com/user-attachments/assets/f22bd98b-f5e1-4839-92ab-9842648![WhatsApp Image 2025-05-27 at 10 40 27 AM (1)](https://github.com/user-attachments/assets/4b5114ce-3260-4f9a-9e6a-6cb21ca29da1)
b1f94)
![WhatsApp Image 2025-05-27 at 10 40 28 AM](https://github.com/user-attachments/assets/5015b7df-3bc2-4199-a90d-7398b1baf526)
![WhatsApp Image 2025-05-27 at 10 40 28 AM (1)](https://github.com/user-attachments/assets/8058d786-49e5-4792-aeac-32282f7767da)
![WhatsApp Image 2025-05-27 at 10 40 29 AM](https://github.com/user-attachments/assets/c11f61e5-8aef-4f71-b700-a7904251d0b1)
![WhatsApp Image 2025-05-27 at 10 40 29 AM (1)](https://github.com/user-attachments/assets/6fc95b43-c906-431e-bf35-263af6260ca5)

Balanced class weights were used to address class imbalance.

📚 Future Work
Try other classification models like Random Forest, SVM, or XGBoost.

Perform hyperparameter tuning.

Use SMOTE for handling class imbalance.

Deploy model using Flask/Streamlit.

🙌 Acknowledgements
Dataset by IBM Watson Analytics

Implemented on Google Colab using Python

📌 Author
Sudhanshu Kumar
