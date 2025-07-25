import joblib
import pandas as pd
from flask import Flask, request, render_template

# Initialize the flask app
app = Flask(__name__)

# Load the model
model = joblib.load('churn_model.joblib')

# Load the column names from the training data (we'll need this)
# You should save this from your notebook, or define it manually
# For now, I will manually define the columns based on our notebook
TRAIN_COLUMNS = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges', 'gender_Male', 'Partner_Yes', 'Dependents_Yes', 'PhoneService_Yes', 'MultipleLines_No phone service', 'MultipleLines_Yes', 'InternetService_Fiber optic', 'InternetService_No', 'OnlineSecurity_No internet service', 'OnlineSecurity_Yes', 'OnlineBackup_No internet service', 'OnlineBackup_Yes', 'DeviceProtection_No internet service', 'DeviceProtection_Yes', 'TechSupport_No internet service', 'TechSupport_Yes', 'StreamingTV_No internet service', 'StreamingTV_Yes', 'StreamingMovies_No internet service', 'StreamingMovies_Yes', 'Contract_One year', 'Contract_Two year', 'PaperlessBilling_Yes', 'PaymentMethod_Credit card (automatic)', 'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the form data
    form_features = [float(x) for x in request.form.values()]

    # Create a DataFrame from the form data
    # The order of features in the form must match the order here
    input_data = pd.DataFrame([form_features], columns=[
        'tenure', 'MonthlyCharges', 'TotalCharges', 'Partner_Yes', 'Dependents_Yes',
        'SeniorCitizen',
        'PhoneService_Yes', 'PaperlessBilling_Yes', 'MultipleLines_Yes',
        'OnlineSecurity_Yes', 'OnlineBackup_Yes', 'DeviceProtection_Yes',
        'TechSupport_Yes', 'StreamingTV_Yes', 'StreamingMovies_Yes',
        'Contract_One year', 'Contract_Two year', 'InternetService_Fiber optic',
        'PaymentMethod_Electronic check'
    ])

    # One-hot encode the input data to match the training columns
    # This is a simplified way to ensure all columns are present
    final_features = pd.DataFrame(columns=TRAIN_COLUMNS)
    final_features = pd.concat([final_features, input_data])
    final_features = final_features.fillna(0) # Fill any missing columns with 0

    final_features = final_features[TRAIN_COLUMNS]

    # Make prediction
    prediction_proba = model.predict_proba(final_features)[:, 1]
    prediction = (prediction_proba > 0.5).astype(int)

    # Prepare the output
    if prediction[0] == 1:
        output_text = f"High Risk of Churn (Probability: {prediction_proba[0]:.2f})"
        output_class = "high-risk"
    else:
        output_text = f"Low Risk of Churn (Probability: {prediction_proba[0]:.2f})"
        output_class = "low-risk"

    return render_template('index.html', prediction_text=output_text, prediction_class=output_class)


if __name__ == "__main__":
    app.run(debug=True)