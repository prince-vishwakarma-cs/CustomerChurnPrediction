from flask import Flask,request,render_template,jsonify
import pickle
import pandas as pd

app= Flask(__name__)

model=pickle.load(open('ab_model.pkl', 'rb'))
transformer=pickle.load(open('column_transformations.pkl', 'rb'))
label_encoder=pickle.load(open('label_encoder.pkl', 'rb'))

@app.route("/")
def index():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    data = request.form
    gender = data['gender']
    senior_citizen = data['SeniorCitizen']
    partner = data['Partner']
    dependents = data['Dependents']
    tenure = int(data['tenure'])
    phone_service = data['PhoneService']
    multiple_lines = data['MultipleLines']
    internet_service = data['InternetService']
    online_security = data['OnlineSecurity']
    online_backup = data['OnlineBackup']
    device_protection = data['DeviceProtection']
    tech_support = data['TechSupport']
    streaming_tv = data['StreamingTV']
    streaming_movies = data['StreamingMovies']
    contract = data['Contract']
    paperless_billing = data['PaperlessBilling']
    payment_method = data['PaymentMethod']
    monthly_charges = float(data['MonthlyCharges'])
    total_charges = float(data['TotalCharges'])
    
    test_df = pd.DataFrame({
    'gender': [gender],
    'SeniorCitizen': [senior_citizen],
    'Partner': [partner],
    'Dependents': [dependents],
    'tenure': [tenure],
    'PhoneService': [phone_service],
    'MultipleLines': [multiple_lines],
    'InternetService': [internet_service],
    'OnlineSecurity': [online_security],
    'OnlineBackup': [online_backup],
    'DeviceProtection': [device_protection],
    'TechSupport': [tech_support],
    'StreamingTV': [streaming_tv],
    'StreamingMovies': [streaming_movies],
    'Contract': [contract],
    'PaperlessBilling': [paperless_billing],
    'PaymentMethod': [payment_method],
    'MonthlyCharges': [monthly_charges],
    'TotalCharges': [total_charges],
    })

    
    test_df_transformed=pd.DataFrame(transformer.transform(test_df),columns = transformer.get_feature_names_out())
    
    encoded_prediction = model.predict(test_df_transformed)[0]
    decoded_prediction = label_encoder.inverse_transform([encoded_prediction])[0]
    
    return render_template('result.html', prediction=decoded_prediction)


if __name__ == '__main__':
    app.run(debug=True)


