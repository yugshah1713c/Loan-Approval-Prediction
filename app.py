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

    previous_loan_defaults = 1 if previous_loan_defaults == "Yes" else 0
    df['previous_loan_defaults_on_file'] = previous_loan_defaults

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
        high_influence = shap_values_df.head(3)
        features = high_influence['Features'].tolist()
        score = high_influence['SHAP'].round(2).tolist()
        
    else :
        result = "Loan Rejected ❌"
        probability = round(model.predict_proba(df)[0][1] * 100,2)
        shap_values = shap_values.values[0][:,0]
        shap_values_df = pd.DataFrame({
            "Features" : df.columns,
            "SHAP" : shap_values
        })
        shap_values_df['abs'] = shap_values_df['SHAP'].abs()
        shap_values_df = shap_values_df.sort_values(
            by="abs",
            ascending=False
        )
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

        "person_education_Bachelor" : "Education (Bachelor's)",
        "person_education_Master" : "Education (Master's)",
        "person_education_Doctorate" : "Education (Doctorate)",
        "person_education_High School" : "Education (High School)",

        'person_home_ownership_OWN' : 'Ownership (Own)',
        'person_home_ownership_RENT' : 'Ownership (Rent)',
        'person_home_ownership_OTHER' : 'Ownership (Other)',

        'loan_intent_EDUCATION' : 'Intent (Education)',
        'loan_intent_HOMEIMPROVEMENT' : 'Intent (Home Improvement)',
        'loan_intent_MEDICAL' : 'Intent (Medical)',
        'loan_intent_PERSONAL' : 'Intent (Personal)',
        'loan_intent_VENTURE' : 'Intent (Venture)'
    }

    features = [features_name[f] for f in features]
    message = []
    
    for f,s in zip(features,score) :
        if f == "Person Age" :
            if s < 0 :
                message.append("Your age has negative impact")
            else :
                message.append("Your age has positively impact")
            
        elif f == "Person Income":
            if s < 0 :
                message.append(f"Your  income is {person_income:,.0f}, which negatively influenced this prediction")
            else :
                message.append(f"Your income is {person_income:,.0f}, which positively influenced this prediction")
            
        elif f == "Loan Amount" :
            if s < 0 :
                message.append(f"Your loan amount is {loan_amount:,.0f}, which negatively influenced this prediction")
            else :
                message.append(f"Your loan amount is {loan_amount:,0f}, which positevely influenced this prediction")
        
        elif f == "Interest Rate" :
            if s < 0 :
                message.append(f"Your interest rate is {interest_rate}, which negatively influenced this prediction")
            else :
                message.append(f"Your interest rate is {interest_rate}, which positively influenced this prediction")

        elif f == "Loan-Income Ratio" :
            ratio = loan_amount / person_income * 100
            if s < 0 :
                message.append(f"The requested loan amount is {ratio:,.0f}% of your annual income, which negatively influenced this prediction.")
            else :
                message.append(f"The requested loan amount is {ratio:,.0f}% of your annual income, which positively influenced this prediction.")

        elif f == "Credit History" :
            if s < 0 :
                message.append(f"Your credit history is {credit_history}, which negatively influenced this prediction.")
            else :
                message.append(f"The credit history is {credit_history}, which negatively influenced this prediction.")

        elif f == "Credit Score" :
            if s < 0 :
                message.append(f"Your credit score is {credit_score}, which negatively influenced this prediction.")
            else :
                message.append(f"The credit score is {credit_score}, which positively influenced this prediction.")

        elif f == "Previous Loan Defaults" :
            if s < 0 :
                previous_loan_defaults = "Yes"
                message.append(f"Your Previous Loan Defaults is {previous_loan_defaults}, which negatively influenced this prediction.")
            else :
                previous_loan_defaults = "No"
                message.append(f"The Previous Loan Defaults is {previous_loan_defaults}, which positively influenced this prediction.")
            
        elif f =="Education (Bachelor's)" :
            if s < 0 :
                message.append(f"Your education is {person_education}, which negatively influenced this prediction.")
            else :
                message.append(f"The education is {person_education}, which positively influenced this prediction.")

        elif f =="Education (Master's)" :
            if s < 0 :
                message.append(f"Your education is {person_education}, which negatively influenced this prediction.")
            else :
                message.append(f"The education is {person_education}, which positively influenced this prediction.")

        elif f =="Education (Doctorate)" :
            if s < 0 :
                message.append(f"Your education is {person_education}, which negatively influenced this prediction.")
            else :
                message.append(f"The education is {person_education}, which positively influenced this prediction.")

        elif f =="Education (High School)" :
            if s < 0 :
                message.append(f"Your education is {person_education}, which negatively influenced this prediction.")
            else :
                message.append(f"The education is {person_education}, which positively influenced this prediction.")
            
        elif f == "Ownership (Own)" :
            if s < 0 :
                message.append(f"Your ownership is {person_home_ownership}, which negatively influenced this prediction.")
            else :
                message.append(f"The ownership is {person_home_ownership}, which positively influenced this prediction.")
                
        elif f == "Ownership (Rent)" :
            if s < 0 :
                message.append(f"Your ownership is {person_home_ownership}, which negatively influenced this prediction.")
            else :
                message.append(f"The ownership is {person_home_ownership}, which positively influenced this prediction.")


        elif f == "Ownership (Other)" :
            if s < 0 :
                message.append(f"Your ownership is {person_home_ownership}, which negatively influenced this prediction.")
            else :
                message.append(f"The ownership is {person_home_ownership}, which positively influenced this prediction.")

        elif f == "Intent (Education)" :
            if s < 0 :
                message.append(f"Your intent is {loan_intent}, which negatively influenced this prediction.")
            else :
                message.append(f"The intent is {loan_intent}, which positively influenced this prediction.")
        
        elif f == "Intent (Home Improvement)" :
            if s < 0 :
                message.append(f"Your intent is {loan_intent}, which negatively influenced this prediction.")
            else :
                message.append(f"The intent is {loan_intent}, which positively influenced this prediction.")

        elif f == "Intent (Medical)" :
            if s < 0 :
                message.append(f"Your intent is {loan_intent}, which negatively influenced this prediction.")
            else :
                message.append(f"The intent is {loan_intent}, which positively influenced this prediction.")

        elif f == "Intent (Personal)" :
            if s < 0 :
                message.append(f"Your intent is {loan_intent}, which negatively influenced this prediction.")
            else :
                message.append(f"The intent is {loan_intent}, which positively influenced this prediction.")

        elif f == "Intent (Venture)" :
            if s < 0 :
                message.append(f"Your intent is {loan_intent}, which negatively influenced this prediction.")
            else :
                message.append(f"The intent is {loan_intent}, which positively influenced this prediction.")



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

    return render_template("index.html",prediction=result,person_name = person_name,probability = probability,features = features * 100,scores=score * 100,message = message)

    

if __name__ == "__main__":
    app.run()