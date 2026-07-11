const age = document.querySelector("#age")
const ageError = document.querySelector("#ageError")

const name = document.querySelector("#name")
const nameError = document.querySelector("#nameError")

const income = document.querySelector("#income")
const incomeError = document.querySelector("#incomeError")

const loan_amt = document.querySelector("#loan_amt")
const loan_amtError = document.querySelector("#loan_amtError")

const int_rate = document.querySelector("#int_rate")
const int_rateError = document.querySelector("#int_rateError")

const credit_history = document.querySelector("#credit_history")
const credit_historyError = document.querySelector("#credit_historyError")

const credit_score = document.querySelector("#credit_score")
const credit_scoreError = document.querySelector("#credit_scoreError")

const education = document.querySelector("#education")
const ownership = document.querySelector("#ownership")
const loan_intent = document.querySelector("#loan_intent")

const predictionForm = document.querySelector("#pred-form")
const button = document.querySelector("#predictionBtn")
const btnText = document.querySelector("#btntext")
const spinner = document.querySelector("#spinner")

age.addEventListener("input",  () => {
    if(age.value <18 || age.value > 100)
    {
        age.style.border = "2px solid red";
        ageError.textContent = "❌ Age must be between 18 and 100.";
    }else{
        age.style.border = "2px solid lightgreen";
        ageError.textContent = "";  
    }
});

name.addEventListener("input", () => {

    const nameLength = name.value.trim().length;

    if (nameLength < 3 || nameLength > 50) {
        name.style.border = "2px solid red";
        nameError.textContent = "❌ Name must be 3–50 characters.";
    } else {
        name.style.border = "2px solid lightgreen";
        nameError.textContent = "";
    }

});

income.addEventListener("input",() => {
    if(income.value <= 0)
    {
        income.style.border = "2px solid red";
        incomeError.textContent = "❌ Income must be greater than 0."
    }else{
        income.style.border = "2px solid lightgreen";
        incomeError.textContent = ""
    }
})

loan_amt.addEventListener("input",() =>{
    if(loan_amt.value <= 0)
    {
        loan_amt.style.border = "2px solid red";
        loan_amtError.textContent = "❌ Income must be greater than 0."
    }else{
        loan_amt.style.border = "2px solid lightgreen";
        loan_amtError.textContent = ""
    }
})

int_rate.addEventListener("input",() => {
    if(int_rate.value <= 0 || int_rate.value > 50)
    {
        int_rate.style.border = "2px solid red";
        int_rateError.textContent = "❌ Interest rate must be between 0% and 50%."
    }else{
        int_rate.style.border = "2px solid lightgreen";
        int_rateError.textContent = ""
    }
})

credit_history.addEventListener("input",() => {
    if(credit_history.value < 0){
        credit_history.style.border = "2px solid red";
        credit_historyError.textContent = "❌ Enter a valid credit history length."
    }else{
        credit_history.style.border = "2px solid lightgreen";
        credit_historyError.textContent = ""
    }
})

credit_score.addEventListener("input",() => {
    if(credit_score.value<300 || credit_score.value > 850)
    {
        credit_score.style.border = "2px solid red";
        credit_scoreError.textContent = "❌ Credit score must be between 300 and 850."
    }else{
        credit_score.style.border = "2px solid lightgreen";
        credit_scoreError.textContent = ""
    }
})

education.addEventListener("change",() => {
    if(education.value != "")
    {
        education.style.border = "2px solid lightgreen"
    }
})

ownership.addEventListener("change",() => {
    if(ownership.value != "")
    {
        ownership.style.border = "2px solid lightgreen"
    }
})

loan_intent.addEventListener("change",() => {
    if(loan_intent.value != "")
    {
        loan_intent.style.border = "2px solid lightgreen"
    }
})


predictionForm.addEventListener("submit",() => {
    button.disabled = true

    btnText.textContent = "Predicting..."
    
    spinner.classList.remove("spin-hidden")
    spinner.classList.add("spin")
})
