from flask import Flask,render_template,request
import joblib
import pandas as pd
import mysql.connector
import os
from dotenv import load_dotenv

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

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict",methods = ['POST'])
def predict():
    print("predict route called")
    person_age = int(request.form['age'])
    person_name = request.form['person_name'].strip()
    person_income = float(request.form['income'])
    loan_amount = float(request.form['Loan_amount'])
    interest_rate = float(request.form['int_rate'])
    loan_percent_amount = float(request.form['percent_amount'])
    credit_history = float(request.form['person_cred_hist_length'])
    credit_score = int(request.form['credit_score'])

    previous_loan_defaults = request.form['previous_loan_defaults']
    person_education = request.form['person_education']
    person_home_ownership = request.form['home_ownership']
    loan_intent = request.form['loan_intent']

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
    
    if loan_percent_amount < 0 or loan_percent_amount > 1:
       return render_template(
        "index.html",
        prediction="❌ Loan percent income must be between 0 and 1."
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
    "loan_percent_income": loan_percent_amount,
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

    print(df)
    prediction = model.predict(df)
    if prediction[0] == 1:
        result = "Loan Approved ✅"
        probability = round(model.predict_proba(df)[0][1] * 100,2)
    else :
        result = "Loan Rejected ❌"
        probability = round(model.predict_proba(df)[0][1] * 100,2)

# --------------------Database Query--------------------

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

    return render_template("index.html",prediction=result,person_name = person_name,probability = probability)

    

if __name__ == "__main__":
    app.run(debug=True)