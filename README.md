# Loan Approval Prediction Project

This project focuses on predicting loan approval status using machine learning models. The dataset contains information about loan applicants, including demographic details, income, loan amount, and credit history. The goal is to build and evaluate models to classify loan applications as approved or not approved.

## Key Features of the Project

1. **Data Preprocessing**:
   - Missing values were imputed for both training and test datasets.
   - Outliers in numerical columns were treated using the Interquartile Range (IQR) method.
   - New features were engineered, such as `Total_Income`, `EMI`, and `Income_to_Loan_Ratio`.

2. **Model Training**:
   - Multiple machine learning models were trained and evaluated:
     - Decision Tree
     - Random Forest
     - Gradient Boosting
     - AdaBoost
   - Models were trained in two parts:
     - **Part 1**: Without oversampling.
     - **Part 2**: With oversampling to handle class imbalance.

3. **Evaluation Metrics**:
   - Models were evaluated using the following metrics:
     - Accuracy
     - Precision
     - Recall
     - F1 Score
     - ROC AUC

4. **Best Models**:
   - **Part 1 (Without Oversampling)**:
     - Best Model: **AdaBoost**
     - Saved as: `best_model_part1.pkl`
     - Performance:
       - Accuracy: 79.67%
       - Precision: 76.70%
       - Recall: 98.75%
       - F1 Score: 86.34%
       - ROC AUC: 70.93%
   - **Part 2 (With Oversampling)**:
     - Best Model: **Gradient Boosting**
     - Saved as: `best_model_part2.pkl`
     - Performance:
       - Accuracy: 80.47%
       - Precision: 71.00%
       - Recall: 94.67%
       - F1 Score: 81.14%
       - ROC AUC: 85.72%

5. **Artifacts**:
   - Scalers and label encoders were saved for both parts:
     - `scaler_part1.pkl`, `label_encoders_part1.pkl`
     - `scaler_part2.pkl`, `label_encoders_part2.pkl`
   - Test predictions were saved to `test_predictions.csv`.

6. **Results Comparison**:
   - Models trained with oversampling generally performed better in terms of recall and ROC AUC.
   - AdaBoost performed best without oversampling, while Gradient Boosting was the best model with oversampling.

## How to Run the Project

1. **Dependencies**:
   - Install required Python packages using:
     ```bash
     pip install -r requirements.txt
     ```

2. **Training**:
   - Run the training script:
     ```bash
     python train_test.py
     ```

3. **Inference**:
   - Use the saved models for predictions:
     ```bash
     python inference.py
     ```

4. **Results**:
   - Model performance metrics are saved in `model_results.csv`.
   - Test predictions are saved in `test_predictions.csv`.

## Conclusion

The project demonstrates the effectiveness of machine learning models in predicting loan approval status. Gradient Boosting and AdaBoost emerged as the best models for different scenarios, providing high accuracy and recall. The saved models and artifacts can be used for further predictions and analysis.