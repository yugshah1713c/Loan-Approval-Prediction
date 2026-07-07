from flask import Flask,render_template,request
import joblib
import pandas as pd

app = Flask(__name__)

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
    else :
        result = "Loan Rejected ❌"

    return render_template("index.html",prediction=result)
if __name__ == "__main__":
    app.run(debug=True)