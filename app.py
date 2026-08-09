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

@app.route("/predict", methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return render_template("predict.html")

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

    if person_income <= 0 :
        return render_template(
        "predict.html",
        prediction = "❌ Income must be greater than 0."
        )
    loan_percent_amount = loan_amount / person_income
    # ----------Validation----------

    if person_age < 18 or person_age > 100 :
        return render_template(
            "predict.html",
            prediction = "❌ Age must be between 18 and 100."
        )
    
    if len(person_name) < 3 or len(person_name) > 50:
        return render_template(
            "predict.html",
            prediction = "❌ Name must be 3–50 characters."
        )

    
    if loan_amount <= 0 :
        return render_template(
            "predict.html",
            prediction = "❌ Enter a valid loan amount."
        )

    if interest_rate < 0 or interest_rate > 50 :
        return render_template(
            "predict.html",
            prediction = "❌ Interest rate must be between 0% and 50%."
        )
    
    if credit_history <= 0 :
        return render_template(
            "predict.html",
            prediction = "❌ Enter a valid credit history length."
        )
    
    if credit_score < 300 or credit_score >850 :
        return render_template(
            "predict.html",
            prediction = "❌ Credit score must be between 300 and 850."
        )
    
    if person_education == "" :
        return render_template(
            "predict.html",
            prediction = "❌ Please select your education."
        )
    
    if person_home_ownership == "" :
        return render_template(
            "predict.html",
            prediction = "❌ Please select your ownership."
        )
    
    if loan_intent == "" :
        return render_template(
            "predict.html",
            prediction = "❌ Please select your loan intent."
        )
    
    if previous_loan_defaults not in ["Yes" , "No"] :
        return render_template(
            "predict.html",
            prediction="❌ Please choose Yes or No."
        )

    # ----------Preprocessing----------
    data = {
    "person_age": person_age,
    "person_income": person_income,
    "loan_amnt": loan_amount,
    "loan_int_rate": interest_rate,
    "loan_percent_income": loan_percent_amount,
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
                message.append("For this prediction your age reduces the model's confidence in approval.")
            else :
                message.append("For this prediction your age increases the model's confidence in approval.")
            
        elif f == "Person Income":
            if s < 0 :
                message.append(f"For this prediction your annual income of ₹{person_income} ,reduces the model's confidence in approval.")
            else :
                message.append(f"Your annual income of ₹{person_income} ,increases the model's confidence in approval.")
            
        elif f == "Loan Amount" :
            if s < 0 :
                message.append(f"For this prediction your loan amount of ₹{loan_amount} ,reduces the model's confidence in approval.")
            else :
                message.append(f"For this prediction your loan amount of ₹{loan_amount} ,increases the model's confidence in approval.")
        
        elif f == "Loan-Income Ratio":
            if s < 0:
              message.append(f"For this prediction your loan income ration  of {loan_percent_amount} ,reduces the model's confidence in approval.")
            else:
                message.append(f"For this prediction your loan income ratio of {loan_percent_amount} ,increases the model's confidence in approval.")

        elif f == "Credit History" :
            if s < 0 :
                message.append(f"For this prediction your credit history of {credit_history} ,reduces the model's confidence in approval.")
            else :
                message.append(f"For this prediction your credit history of {credit_history} ,increases the model's confidence in approval.")

        elif f == "Credit Score" :
            if s < 0 :
                message.append(f"For this prediction your credit score of {credit_score} ,reduces the model's confidence in approval.")
            else :
                message.append(f"For this prediction your credit score of {credit_score} ,increases the model's confidence in approval.")

        elif f == "Previous Loan Defaults" :
            if s < 0 :
                message.append(f"For this prediction your previous loan defaults of {previous_loan_defaults} ,reduces the model's confidence in approval.")
            else :
                message.append(f"For this prediction your previous loan defaults of {previous_loan_defaults} ,increases the model's confidence in approval.")
            
        elif f == "Education":
            if s < 0:
                 message.append(f"For this prediction your education is ({person_education}) ,which reduces the model's confidence in approval.")
            else:
                  message.append(f"For this prediction your education is ({person_education}) ,which increases the model's confidence in approval.")
            
        elif f == "Ownership":
            if s < 0:
                message.append(f"For this prediction your home ownership is ({person_home_ownership}) ,which reduces the model's confidence in approval.")
            else:
                message.append(f"For this prediction your home ownership is ({person_home_ownership}) ,which increases the model's confidence in approval.")
                
        elif f == "Intent":
         if s < 0:
            message.append(f"For this prediction your loan intent is ({loan_intent}) ,which reduces the model's confidence in approval.")
         else:
            message.append(f"For this prediction your loan intent is ({loan_intent}) ,which increases the model's confidence in approval.")        




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
    previous_loan_defaults,
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
    return render_template("predict.html",prediction=result,person_name = person_name,probability = probability,features = features,message = message)
 
    

if __name__ == "__main__":
    app.run()