from flask import Flask,render_template,request
import joblib
import pandas as pd
import mysql.connector
import os
from dotenv import load_dotenv
import shap

app = Flask(__name__)
load_dotenv()

## Database connection
db = mysql.connector.connect(
    host=os.getenv("DB_HOST"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"),
    database=os.getenv("DB_NAME"),
    port=int(os.getenv("DB_PORT"))
)

cursor = db.cursor()

model = joblib.load("models/random_forest.pkl")
scaler = joblib.load("models/scaler.pkl")
feature_columns = joblib.load("models/feature_columns.pkl")
explainer = shap.TreeExplainer(model)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict",methods = ['POST'])
def predict():
    person_age = int(request.form['age'])
    person_name = request.form['person_name'].strip()
    person_income = float(request.form['income'])
    loan_amount = float(request.form['Loan_amount'])
    interest_rate = float(request.form['int_rate'])
    credit_history = float(request.form['person_cred_hist_length'])
    credit_score = int(request.form['credit_score'])

    previous_loan_defaults = request.form['previous_loan_defaults']
    person_education = request.form['person_education']
    person_home_ownership = request.form['home_ownership']
    loan_intent = request.form['loan_intent']

    loan_percent_amount = loan_amount / person_income
    # ----------Validation----------

    if person_age < 18 or person_age > 100 :
        return render_template(
            "index.html",
            prediction = "❌ Age must be between 18 and 100."
        )
    
    if len(person_name) < 3 or len(person_name) > 50:
        return render_template(
            "index.html",
            prediction = "❌ Name must be 3–50 characters."
        )
    
    if person_income <= 0 :
        return render_template(
            "index.html",
            prediction = "❌ Income must be greater than 0."
        )
    
    if loan_amount <= 0 :
        return render_template(
            "index.html",
            prediction = "❌ Enter a valid loan amount."
        )

    if interest_rate < 0 or interest_rate > 50 :
        return render_template(
            "index.html",
            prediction = "❌ Interest rate must be between 0% and 50%."
        )
    
    if credit_history <= 0 :
        return render_template(
            "index.html",
            prediction = "❌ Enter a valid credit history length."
        )
    
    if credit_score < 300 or credit_score >850 :
        return render_template(
            "index.html",
            prediction = "❌ Credit score must be between 300 and 850."
        )
    
    if person_education == "" :
        return render_template(
            "index.html",
            prediction = "❌ Please select your education."
        )
    
    if person_home_ownership == "" :
        return render_template(
            "index.html",
            prediction = "❌ Please select your ownership."
        )
    
    if loan_intent == "" :
        return render_template(
            "index.html",
            prediction = "❌ Please select your loan intent."
        )
    
    if previous_loan_defaults not in ["Yes" , "No"] :
        return render_template(
            "index.html",
            prediction="❌ Please choose Yes or No."
        )

    # ----------Preprocessing----------
    data = {
    "person_age": person_age,
    "person_income": person_income,
    "loan_amnt": loan_amount,
    "loan_int_rate": interest_rate,
    "loan_percent_income": loan_amount / person_income,
    "cb_person_cred_hist_length": credit_history,
    "credit_score": credit_score,
    "previous_loan_defaults_on_file": previous_loan_defaults,
    "person_education": person_education,
    "person_home_ownership": person_home_ownership,
    "loan_intent": loan_intent
    }
    


    df = pd.DataFrame([data])

    previous_loan_defaults_encoded = 1 if previous_loan_defaults == "Yes" else 0
    df['previous_loan_defaults_on_file'] = previous_loan_defaults_encoded

    categorical_cols = [
    "person_education",
    "person_home_ownership",
    "loan_intent"
    ]
    
    df = pd.get_dummies(
        df,
        columns=categorical_cols,
        dtype=int,
    )

    df = df.reindex(columns=feature_columns, fill_value=0)

    numerical_cols = [
    "person_age",
    "person_income",
    "loan_amnt",
    "loan_int_rate",
    "loan_percent_income",
    "cb_person_cred_hist_length",
    "credit_score"
    ]
    df[numerical_cols] = scaler.transform(df[numerical_cols])

# ---------------------Explainable AI-------------------#

    shap_values = explainer(df)

    prediction = model.predict(df)
    if prediction[0] == 1:
        result = "Loan Approved ✅"
        probability = round(model.predict_proba(df)[0][1] * 100,2)
        shap_values = shap_values.values[0][:,1]
        shap_values_df = pd.DataFrame({
            "Features" : df.columns,
            "SHAP" : shap_values
        })
        shap_values_df['abs'] = shap_values_df['SHAP'].abs()
        shap_values_df = shap_values_df.sort_values(
            by="abs",
            ascending=False
        )
        shap_values_df = shap_values_df[
            shap_values_df["Features"] != "loan_int_rate"
        ]
        high_influence = shap_values_df.head(3)
        features = high_influence['Features'].tolist()
        score = high_influence['SHAP'].round(2).tolist()
        
    else :
        result = "Loan Rejected ❌"
        probability = round(model.predict_proba(df)[0][1] * 100,2)
        shap_values = shap_values.values[0][:,1]
        shap_values_df = pd.DataFrame({
            "Features" : df.columns,
            "SHAP" : shap_values
        })
        shap_values_df['abs'] = shap_values_df['SHAP'].abs()
        shap_values_df = shap_values_df.sort_values(
            by="abs",
            ascending=False
        )
        shap_values_df = shap_values_df[
            shap_values_df["Features"] != "loan_int_rate"
        ]
        high_influence = shap_values_df.head(3)
        features = high_influence['Features'].tolist()  
        score = high_influence['SHAP'].round(2).tolist() 
    
    features_name = {
        'person_age' : 'Person Age',
        'person_income' : 'Person Income',
        'loan_amnt' : 'Loan Amount',
        'loan_int_rate' : 'Interest Rate',
        'loan_percent_income' : 'Loan-Income Ratio',
        'cb_person_cred_hist_length' : 'Credit History',
        'credit_score' : 'Credit Score',
        'previous_loan_defaults_on_file' : 'Previous Loan Defaults',

        "person_education_Bachelor" : "Education",
        "person_education_Master" : "Education",
        "person_education_Doctorate" : "Education",
        "person_education_High School" : "Education",

        'person_home_ownership_OWN' : 'Ownership',
        'person_home_ownership_RENT' : 'Ownership',
        'person_home_ownership_OTHER' : 'Ownership',

        'loan_intent_EDUCATION' : 'Intent',
        'loan_intent_HOMEIMPROVEMENT' : 'Intent',
        'loan_intent_MEDICAL' : 'Intent',
        'loan_intent_PERSONAL' : 'Intent',
        'loan_intent_VENTURE' : 'Intent'
    }
    features = [features_name[f] for f in features]
    message = []
    
    for f,s in zip(features,score) :
        if f == "Person Age" :
            if s < 0 :
                message.append("Your age had a slight negative impact on this loan application.")
            else :
                message.append("Your age falls within a favorable range for this loan application.t")
            
        elif f == "Person Income":
            if s < 0 :
                message.append(f"Your annual income of ₹{person_income} is lower than ideal, which reduced your approval chances.")
            else :
                message.append(f"Your annual income of ₹{person_income} strengthens your loan approval chances.")
            
        elif f == "Loan Amount" :
            if s < 0 :
                message.append(f"Your requested loan amount is relatively high, which reduced your approval chances.")
            else :
                message.append(f"Your requested loan amount is considered manageable based on your financial profile.")
        
        elif f == "Loan-Income Ratio" :
            ratio = loan_amount / person_income * 100
            if s < 0 :
                message.append(f"Your requested loan is {loan_percent_amount}% of your annual income, indicating a high repayment burden.")
            else :
                message.append(f"Your requested loan is only {loan_percent_amount}% of your annual income, indicating a healthy repayment capacity.")

        elif f == "Credit History" :
            if s < 0 :
                message.append(f"Your credit history of {credit_history} years is relatively short, which reduced your approval chances.")
            else :
                message.append(f"Your credit history of {credit_history} years demonstrates stable borrowing experience.")

        elif f == "Credit Score" :
            if s < 0 :
                message.append(f"Your credit score of {credit_score} reduced your approval chances.")
            else :
                message.append(f"Your credit score of {credit_score} significantly improved your approval chances.")

        elif f == "Previous Loan Defaults" :
            if s < 0 :
                message.append(f"Your previous loan default history significantly reduced your approval chances.")
            else :
                message.append(f"Your clean repayment history improved your loan approval chances.")
            
        elif f == "Education":
            if s < 0:
                 message.append(f"Your education level ({person_education}) had a slight negative influence on this prediction.")
            else:
                  message.append(f"Your education level ({person_education}) had a slight positive influence on this prediction.")
            
        elif f == "Ownership":
            if s < 0:
                message.append(f"Your home ownership status ({person_home_ownership}) had a slight negative influence on this prediction.")
            else:
                message.append(f"Your home ownership status ({person_home_ownership}) had a slight positive influence on this prediction.")
                
        elif f == "Intent":
   
         if s < 0:
            message.append(f"The purpose of your loan ({loan_intent}) had a slight negative influence on this prediction.")
         else:
            message.append(f"The purpose of your loan ({loan_intent}) had a slight positive influence on this prediction.")        




# --------------------Database Query--------------------#

    query = """
    INSERT INTO loan_predictions(
    person_age,
    person_name,
    person_income,
    loan_amount,
    interest_rate,
    loan_percent_income,
    credit_history,
    credit_score,
    previous_loan_defaults,
    person_education,
    home_ownership,
    loan_intent,
    prediction
    )
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """

    values = (
    person_age,
    person_name,
    person_income,
    loan_amount,
    interest_rate,
    loan_percent_amount,
    credit_history,
    credit_score,
    "Yes" if previous_loan_defaults == 1 else "No",
    person_education,
    person_home_ownership,
    loan_intent,
    result
    )

    cursor.execute(query, values) 
    db.commit()

    print(features)
    print(message)
    print(len(features))
    print(len(message))
    return render_template("index.html",prediction=result,person_name = person_name,probability = probability,features = features,scores=score * 100,message = message)
 
    

if __name__ == "__main__":
    app.run()