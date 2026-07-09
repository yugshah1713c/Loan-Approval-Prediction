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

const loan_per_inc = document.querySelector("#loan_per_inc")
const loan_per_incError = document.querySelector("#loan_per_incError")

const credit_history = document.querySelector("#credit_history")
const credit_historyError = document.querySelector("#credit_historyError")

const credit_score = document.querySelector("#credit_score")
const credit_scoreError = document.querySelector("#credit_scoreError")

age.addEventListener("input", function () {
    if(age.value <18 || age.value > 100)
    {
        ageError.textContent = "❌ Age must be between 18 and 100.";
    }else{
        ageError.textContent = "";  
    }
});

// name.addEventListener("input",() => {
//     if(name.)
// })

income.addEventListener("input",() => {
    if(income.value <= 0)
    {
        incomeError.textContent = "❌ Income must be greater than 0."
    }else{
        incomeError.textContent = ""
    }
})

// loan_amt.addEventListener("input",() =>{
//     if(income.value <= 0)
//     {
//         loan_amtError.textContent = "❌ Income must be greater than 0."
//     }else{
//         loan_amtError.textContent = ""
//     }
// })

int_rate.addEventListener("input",() => {
    if(int_rate.value <= 0 || int_rate.value > 50)
    {
        int_rateError.textContent = "❌ Interest rate must be between 0% and 50%."
    }else{
        int_rateError.textContent = ""
    }
})

loan_per_inc.addEventListener("input",() => {
    if(loan_per_inc.value < 0 || loan_per_inc.value > 1)
    {
        loan_per_incError.textContent = "❌ Loan percent income must be between 0 and 1."
    }else{
        loan_per_incError.textContent = ""
    }
})

credit_history.addEventListener("input",() => {
    if(credit_history.value < 0){
        credit_historyError.textContent = "❌ Enter a valid credit history length."
    }else{
        credit_historyError.textContent = ""
    }
})

credit_score.addEventListener("input",() => {
    if(credit_score.value<300 || credit_score.value > 850)
    {
        credit_scoreError.textContent = "❌ Credit score must be between 300 and 850."
    }else{
        credit_scoreError.textContent = ""
    }
})

