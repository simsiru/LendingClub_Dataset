import numpy as np
from flask import Flask, request, jsonify
import joblib
import pandas as pd


grade_labels = ['A','B','C','D','E','F','G']

grade_clf = joblib.load('./grade_clf_xgb.pkl')
sub_grade_clf = joblib.load('./sub_grade_clf_xgb.pkl')
int_rate_reg = joblib.load('./int_rate_reg_xgb.pkl')
loan_clf = joblib.load('./loan_clf_lgbm.pkl')

api_features = [
    'initial_list_status', 'total_rec_prncp', 'last_fico_range_high',
    'disbursement_method','funded_amnt', 'funded_amnt_inv', 'term',
    'fico_range_low', 'loan_amnt', 'installment', 'grade', 'total_rec_int', 
    'last_pymnt_d','title','fico_range_high','emp_length'
]

loan_clf_cols = ['title', 'fico_range_low', 'emp_length']

grade_clf_cols = [
    'loan_amnt', 'funded_amnt', 'funded_amnt_inv', 'term', 'installment', 
    'fico_range_low', 'initial_list_status', 'total_rec_prncp', 
    'total_rec_int', 'last_pymnt_d', 'last_fico_range_high', 
    'disbursement_method'
]

sub_grade_clf_cols = [
    'funded_amnt', 'funded_amnt_inv', 'term', 'installment', 
    'grade', 'fico_range_low', 'total_rec_int'
]

int_rate_reg_cols = [
    'loan_amnt', 'installment', 'grade', 'total_rec_int', 'last_pymnt_d'
]


def data_preprocessing(df: pd.DataFrame) -> None:
    df = df.replace('NaN', np.nan)

    titles = ['credit_card_refinancing', 'debt_consolidation', 'wedding_loan',
       'debt_free', 'home_improvement', 'other', 'business',
       'moving_and_relocation', 'car_financing', 'major_purchase',
       'debt_consolidation_loan', 'medical_expenses', 'consolidation',
       'vacation', 'home_buying', 'freedom', 'wedding',
       'credit_card_payoff', 'home_improvement_loan', 'credit_card_loan',
       'consolidation_loan', 'credit_card_consolidation', 'credit_cards',
       'personal_loan', 'consolidate', 'my_loan', 'credit_consolidation',
       'debt_loan', 'personal', 'small_business_loan',
       'credit_card_refinance', 'debt', 'payoff', 'home',
       'credit_card_debt', 'credit_card', 'debt_payoff', 'refinance',
       'bill_consolidation', 'green_loan', 'pay_off', 'loan',
       'pay_off_credit_cards', 'cc_consolidation', 'credit_card_refi',
       'credit_card_pay_off', 'car_loan']

    df['title'] = df['title'].apply(
        lambda x: x if x in titles else 'other')

    cat_features = [
        'initial_list_status', 'disbursement_method', 
        'term', 'grade', 'title', 'emp_length'
    ]

    df = df.astype({col:'category' for col in cat_features})

    return df


app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        df = pd.DataFrame(
            data=request.json['instances'],
            columns=api_features
        )

        df = data_preprocessing(df)

        df['accepted'] = loan_clf.predict(df[loan_clf_cols])

        accepted_df = df[df['accepted']==1]

        if accepted_df.empty:
            grades = []
            sub_grades = []
            interest_rates = []
        else:
            accepted_df['grade'] = grade_clf.predict(
                accepted_df[grade_clf_cols])
            accepted_df['grade'] = accepted_df['grade'].astype('category')
            accepted_df['grade'] = accepted_df['grade'].apply(
                lambda x: grade_labels[x])

            grades = accepted_df['grade'].values.tolist()

            sub_grades = sub_grade_clf.predict(
                accepted_df[sub_grade_clf_cols]).tolist()

            sub_grades = [accepted_df['grade'].iloc[i]+str(sg+1) \
                for i, sg in enumerate(sub_grades)]

            interest_rates = int_rate_reg.predict(
                accepted_df[int_rate_reg_cols]).tolist()

        data = {
            "loan_classification": df['accepted'].values.tolist(),
            "grade_prediction": grades,
            "sub_grade_prediction": sub_grades,
            "interest_rate_prediction": interest_rates
        }

        return jsonify(data)

    return "OK"


if __name__ == "__main__":
    app.run(debug=True)