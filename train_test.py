import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
from imblearn.over_sampling import SMOTE
import joblib

# Load datasets
df_train = pd.read_csv('loan_sanction_train.csv')
df_train = df_train.convert_dtypes()
df_test = pd.read_csv('loan_sanction_test.csv')
df_test = df_test.convert_dtypes()

# Exploratory Data Analysis (on training data)
print("Training Dataset Info:")
print(df_train.info())
print("\nTraining Dataset Description:")
print(df_train.describe())
print("\nFirst 5 Rows of Training Data:")
print(df_train.head())

plt.figure(figsize=(15, 10))
plt.subplot(2, 2, 1)
sns.countplot(x='Loan_Status', data=df_train)
plt.title('Loan Status Distribution')
plt.subplot(2, 2, 2)
sns.histplot(df_train['ApplicantIncome'], bins=30, kde=True)
plt.title('Applicant Income Distribution')
plt.subplot(2, 2, 3)
sns.histplot(df_train['LoanAmount'].dropna(), bins=30, kde=True)
plt.title('Loan Amount Distribution')
plt.subplot(2, 2, 4)
sns.countplot(x='Credit_History', data=df_train)
plt.title('Credit History Distribution')
plt.tight_layout()
plt.savefig('univariate_analysis.png')
plt.close()

plt.figure(figsize=(15, 10))
plt.subplot(2, 2, 1)
sns.boxplot(x='Loan_Status', y='ApplicantIncome', data=df_train)
plt.title('Loan Status vs Applicant Income')
plt.subplot(2, 2, 2)
sns.boxplot(x='Loan_Status', y='LoanAmount', data=df_train)
plt.title('Loan Status vs Loan Amount')
plt.subplot(2, 2, 3)
sns.countplot(x='Credit_History', hue='Loan_Status', data=df_train)
plt.title('Credit History vs Loan Status')
plt.subplot(2, 2, 4)
sns.countplot(x='Property_Area', hue='Loan_Status', data=df_train)
plt.title('Property Area vs Loan Status')
plt.tight_layout()
plt.savefig('bivariate_analysis.png')
plt.close()

print("\nMissing Values in Training Data:")
print(df_train.isnull().sum())
print("\nMissing Values in Test Data:")
print(df_test.isnull().sum())

# Imputation
cat_columns = ['Gender', 'Married', 'Dependents', 'Self_Employed', 'Credit_History']
num_columns = ['LoanAmount', 'Loan_Amount_Term']

for col in cat_columns:
    df_train[col] = df_train[col].fillna(df_train[col].mode()[0])
    df_test[col] = df_test[col].fillna(df_train[col].mode()[0])  # Use training mode for consistency
for col in num_columns:
    df_train[col] = df_train[col].fillna(df_train[col].median())
    df_test[col] = df_test[col].fillna(df_train[col].median())  # Use training median for consistency

print("\nMissing Values in Training Data After Imputation:")
print(df_train.isnull().sum())
print("\nMissing Values in Test Data After Imputation:")
print(df_test.isnull().sum())

# Outlier treatment
def treat_outliers(df, column):
    df[column] = df[column].astype('float64')
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)
    return df

for col in ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount']:
    df_train = treat_outliers(df_train, col)
    df_test = treat_outliers(df_test, col)

# Encoding
label_encoders = {}
cat_columns = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area', 'Credit_History', 'Loan_Status']
for col in cat_columns:
    le = LabelEncoder()
    if col in df_train.columns:  # Loan_Status is not in df_test
        df_train[col] = le.fit_transform(df_train[col].astype(str))
        label_encoders[col] = le
    if col != 'Loan_Status':  # Apply to test data (no Loan_Status)
        df_test[col] = le.transform(df_test[col].astype(str))

# Scaling
scaler = StandardScaler()
num_columns = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']
df_train[num_columns] = scaler.fit_transform(df_train[num_columns])
df_test[num_columns] = scaler.transform(df_test[num_columns])

# Define features and target
X = df_train.drop(['Loan_ID', 'Loan_Status'], axis=1)
y = df_train['Loan_Status']
X_test_df = df_test.drop(['Loan_ID'], axis=1)

# Train-validation split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training and evaluation
def evaluate_model(y_true, y_pred, y_prob=None):
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, pos_label=1),
        'Recall': recall_score(y_true, y_pred, pos_label=1),
        'F1 Score': f1_score(y_true, y_pred, pos_label=1)
    }
    if y_prob is not None:
        metrics['ROC AUC'] = roc_auc_score(y_true, y_prob[:, 1])
    return metrics

models = {
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'AdaBoost': AdaBoostClassifier(random_state=42)
}

# Part 1: Without oversampling
results_part1 = {}
best_f1_part1 = 0
best_model_part1 = None
best_model_name_part1 = None
scaler_part1 = scaler

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val) if hasattr(model, 'predict_proba') else None
    results_part1[name] = evaluate_model(y_val, y_pred, y_prob)
    print(f"\nClassification Report for {name} (Validation):")
    print(classification_report(y_val, y_pred))
    if results_part1[name]['F1 Score'] > best_f1_part1:
        best_f1_part1 = results_part1[name]['F1 Score']
        best_model_part1 = model
        best_model_name_part1 = name

# Save artifacts for Part 1
joblib.dump(best_model_part1, 'best_model_part1.pkl')
joblib.dump(scaler_part1, 'scaler_part1.pkl')
joblib.dump(label_encoders, 'label_encoders_part1.pkl')
print(f"\nBest model for Part 1 (without oversampling): {best_model_name_part1} saved as 'best_model_part1.pkl'")
print("Scaler saved as 'scaler_part1.pkl'")
print("Label encoders saved as 'label_encoders_part1.pkl'")

# Feature engineering
df_train['Total_Income'] = df_train['ApplicantIncome'] + df_train['CoapplicantIncome']
df_train['EMI'] = df_train['LoanAmount'] / df_train['Loan_Amount_Term']
df_train['Income_to_Loan_Ratio'] = df_train['Total_Income'] / df_train['LoanAmount']
df_test['Total_Income'] = df_test['ApplicantIncome'] + df_test['CoapplicantIncome']
df_test['EMI'] = df_test['LoanAmount'] / df_test['Loan_Amount_Term']
df_test['Income_to_Loan_Ratio'] = df_test['Total_Income'] / df_test['LoanAmount']

# Update features
X = df_train.drop(['Loan_ID', 'Loan_Status'], axis=1)
X_test_df = df_test.drop(['Loan_ID'], axis=1)

# Scale new features
num_columns_extended = num_columns + ['Total_Income', 'EMI', 'Income_to_Loan_Ratio']
X[num_columns_extended] = scaler.fit_transform(X[num_columns_extended])
X_test_df[num_columns_extended] = scaler.transform(X_test_df[num_columns_extended])

# Convert to float64
X = X.astype('float64')
X_test_df = X_test_df.astype('float64')

# Part 2: With oversampling
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

X_train_res, X_val_res, y_train_res, y_val_res = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

results_part2 = {}
best_f1_part2 = 0
best_model_part2 = None
best_model_name_part2 = None
scaler_part2 = scaler

for name, model in models.items():
    model.fit(X_train_res, y_train_res)
    y_pred = model.predict(X_val_res)
    y_prob = model.predict_proba(X_val_res) if hasattr(model, 'predict_proba') else None
    results_part2[name] = evaluate_model(y_val_res, y_pred, y_prob)
    print(f"\nClassification Report for {name} (After Oversampling, Validation):")
    print(classification_report(y_val_res, y_pred))
    if results_part2[name]['F1 Score'] > best_f1_part2:
        best_f1_part2 = results_part2[name]['F1 Score']
        best_model_part2 = model
        best_model_name_part2 = name

# Save artifacts for Part 2
joblib.dump(best_model_part2, 'best_model_part2.pkl')
joblib.dump(scaler_part2, 'scaler_part2.pkl')
joblib.dump(label_encoders, 'label_encoders_part2.pkl')
print(f"\nBest model for Part 2 (with oversampling): {best_model_name_part2} saved as 'best_model_part2.pkl'")
print("Scaler saved as 'scaler_part2.pkl'")
print("Label encoders saved as 'label_encoders_part2.pkl'")

# Predict on test data using the best model from Part 2
test_predictions = best_model_part2.predict(X_test_df)
test_predictions_labels = ['Y' if pred == 1 else 'N' for pred in test_predictions]
submission = pd.DataFrame({
    'Loan_ID': df_test['Loan_ID'],
    'Loan_Status': test_predictions_labels
})
submission.to_csv('test_predictions.csv', index=False)
print("\nTest predictions saved to 'test_predictions.csv'")

# Results comparison
print("\nResults Comparison:")
print("\nPart 1 (Without Oversampling, Validation):")
for model, metrics in results_part1.items():
    print(f"{model}:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")

print("\nPart 2 (With Oversampling, Validation):")
for model, metrics in results_part2.items():
    print(f"{model}:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")

results_df = pd.DataFrame({
    'Model': list(results_part1.keys()) + list(results_part2.keys()),
    'Part': ['Part 1'] * len(results_part1) + ['Part 2'] * len(results_part2),
    'Accuracy': [results_part1[m]['Accuracy'] for m in results_part1] + [results_part2[m]['Accuracy'] for m in results_part2],
    'Precision': [results_part1[m]['Precision'] for m in results_part1] + [results_part2[m]['Precision'] for m in results_part2],
    'Recall': [results_part1[m]['Recall'] for m in results_part1] + [results_part2[m]['Recall'] for m in results_part2],
    'F1 Score': [results_part1[m]['F1 Score'] for m in results_part1] + [results_part2[m]['F1 Score'] for m in results_part2],
    'ROC AUC': [results_part1[m].get('ROC AUC', np.nan) for m in results_part1] + 
               [results_part2[m].get('ROC AUC', np.nan) for m in results_part2]
})
results_df.to_csv('model_results.csv', index=False)