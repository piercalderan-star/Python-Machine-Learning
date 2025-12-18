#https://www.kaggle.com/code/devraai/synthetic-medical-symptoms-data-analysis-model

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

import matplotlib

#matplotlib.use('Agg')  # Use Agg backend for matplotlib

import matplotlib.pyplot as plt


import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, label_binarize
from sklearn.inspection import permutation_importance

import os

# Set plotting style
sns.set(style='whitegrid')

# Load the synthetic medical symptoms and diagnosis dataset
data_path = 'synthetic_medical_symptoms_dataset.csv'
df = pd.read_csv(data_path, encoding='ascii', delimiter=',')
print('Dataset loaded successfully.')

# Display first few rows of the dataframe
df.head()

# Check for missing values and data types
print('Data types:')
print(df.dtypes)

print('\nMissing values in each column:')
print(df.isnull().sum())

# Since there are no date columns provided explicitly in this dataset,
# we do not perform date type inference. If future data includes dates, 
# methods such as pd.to_datetime() will be crucial.

# Convert any potential numeric string columns to proper numeric types if necessary
numeric_columns = ['age', 'fever', 'cough', 'fatigue', 'headache', 'muscle_pain', 
                   'nausea', 'vomiting', 'diarrhea', 'skin_rash', 'loss_smell', 
                   'loss_taste', 'systolic_bp', 'diastolic_bp', 'heart_rate', 
                   'temperature_c', 'oxygen_saturation', 'wbc_count', 'hemoglobin', 
                   'platelet_count', 'crp_level', 'glucose_level']

for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# After conversion, check for any new missing values
print('\nMissing values after conversion:')
print(df[numeric_columns].isnull().sum())

# For this analysis we assume missing values would be filled or dropped as appropriate.
# A simple strategy could be to fill missing numeric values with the median or drop the rows.
df.dropna(inplace=True)
print('\nShape of dataframe after dropping missing values:', df.shape)

# Plot histograms for numeric features
plt.figure(figsize=(12, 8))
for i, col in enumerate(numeric_columns):
    plt.subplot(5, 5, i+1)
    sns.histplot(df[col], kde=True, bins=20)
    plt.title(col)
    plt.tight_layout()
plt.show()

# Count plot for categorical variable 'diagnosis'
plt.figure(figsize=(8, 4))
sns.countplot(data=df, x='diagnosis', order=df['diagnosis'].value_counts().index)
plt.title('Diagnosis Distribution')
plt.xticks(rotation=45)
plt.show()

# Heatmap of correlation for numeric features if there are 4 or more numeric columns
numeric_df = df.select_dtypes(include=[np.number])
if numeric_df.shape[1] >= 4:
    plt.figure(figsize=(12, 10))
    corr = numeric_df.corr()
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', square=True)
    plt.title('Correlation Heatmap')
    plt.show()

# Pair Plot for a subset of features (for performance reasons)
sample_cols = ['age', 'fever', 'heart_rate', 'temperature_c', 'oxygen_saturation']
sns.pairplot(df[sample_cols])
plt.show()

# Box Plot to visualize distribution of age across diagnosis
plt.figure(figsize=(10, 6))
sns.boxplot(x='diagnosis', y='age', data=df, order=df['diagnosis'].value_counts().index)
plt.title('Age Distribution by Diagnosis')
plt.xticks(rotation=45)
plt.show()


# For prediction, we will try to build a classification model to predict diagnosis based on the symptoms and measurements.

# Prepare features and target variable
X = df.drop(['diagnosis', 'gender'], axis=1)  # dropping gender for simplicity in numeric model; it could be encoded if needed
y = df['diagnosis']

# Encode target labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

# Initialize and train a Random Forest Classifier
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train, y_train)

# Predict on test set
y_pred = rf_clf.predict(X_test)

# Calculate and print accuracy score
acc = accuracy_score(y_test, y_pred)
print('Accuracy Score:', acc)

# Confusion Matrix visualization
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Permutation Importance
perm_importance = permutation_importance(rf_clf, X_test, y_test, n_repeats=10, random_state=42)
sorted_idx = perm_importance.importances_mean.argsort()

plt.figure(figsize=(10, 8))
plt.barh(range(len(sorted_idx)), perm_importance.importances_mean[sorted_idx], align='center')
plt.yticks(range(len(sorted_idx)), np.array(X.columns)[sorted_idx])
plt.xlabel('Permutation Importance Mean')
plt.title('Feature Importance (Permutation Importance)')
plt.show()

# Note: Generating ROC curves for multiclass classification requires a One-vs-Rest approach.
# Let's perform ROC plotting for one class as an example.
from sklearn.preprocessing import label_binarize

# Binarize the output
classes = np.unique(y_encoded)
y_test_binarized = label_binarize(y_test, classes=classes)
y_score = rf_clf.predict_proba(X_test)

# Compute ROC curve for class 0 as an example
# Receiver Operating Characteristic = caratteristica operativa del ricevitore
fpr, tpr, _ = roc_curve(y_test_binarized[:, 0], y_score[:, 0])
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for class: ' + le.inverse_transform([0])[0])
plt.legend(loc='lower right')
plt.show()
















