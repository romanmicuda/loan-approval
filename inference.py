import pandas as pd
import numpy as np
import joblib

model = joblib.load('best_model_part2.pkl')
scaler = joblib.load('scaler_part2.pkl')
label_encoders = joblib.load('label_encoders_part2.pkl')

def treat_outliers(df, column):
    df[column] = df[column].astype('float64')
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)
    return df

new_data = pd.DataFrame({
    'Gender': ['Male'], 'Married': ['Yes'], 'Dependents': ['1'], 'Education': ['Graduate'],
    'Self_Employed': ['No'], 'ApplicantIncome': [4583], 'CoapplicantIncome': [1508.0],
    'LoanAmount': [128], 'Loan_Amount_Term': [360], 'Credit_History': [1], 'Property_Area': ['Rural']
})

cat_columns = ['Gender', 'Married', 'Dependents', 'Self_Employed', 'Credit_History']
num_columns = ['LoanAmount', 'Loan_Amount_Term']
for col in cat_columns:
    new_data[col] = new_data[col].fillna(new_data[col].mode()[0])
for col in num_columns:
    new_data[col] = new_data[col].fillna(new_data[col].median())

new_data = treat_outliers(new_data, 'ApplicantIncome')
new_data = treat_outliers(new_data, 'CoapplicantIncome')
new_data = treat_outliers(new_data, 'LoanAmount')

cat_columns = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area', 'Credit_History']
for col in cat_columns:
    try:
        new_data[col] = label_encoders[col].transform(new_data[col].astype(str))
    except ValueError as e:
        print(f"Error encoding {col}: {e}. Ensure input values match training data.")
        exit(1)

new_data['Total_Income'] = new_data['ApplicantIncome'] + new_data['CoapplicantIncome']
new_data['EMI'] = new_data['LoanAmount'] / new_data['Loan_Amount_Term']
new_data['Income_to_Loan_Ratio'] = new_data['Total_Income'] / new_data['LoanAmount']

scale_columns = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 
                 'Total_Income', 'EMI', 'Income_to_Loan_Ratio']
new_data[scale_columns] = scaler.transform(new_data[scale_columns])

new_data = new_data.astype('float64')

prediction = model.predict(new_data)
print("Predicted Loan Status:", "Approved" if prediction[0] == 1 else "Not Approved")
