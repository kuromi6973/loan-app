from flask import Flask, render_template_string, request
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
import shap
import matplotlib.pyplot as plt
import io
import base64
import warnings
import os
from ucimlrepo import fetch_ucirepo 
import joblib

warnings.warn("目前使用的是模擬資料，建議替換為真實樣本以提升模型實用性。", UserWarning)

app = Flask(__name__)

class LoanEvaluator:
    def __init__(self):
        self.model = joblib.load('model.pkl')
        self.feature_names = [
            'income', 'age', 'employment_years', 'debt_ratio',
            'credit_card_repayment', 'credit_score', 'overdue_count'
        ]
        self.accuracy = 0.0
        self.auc = 0.0
        self.confusion_matrix = [[0, 0], [0, 0]]
        self.feature_importance_plot = ""
        # 如果只做推論，不需要再呼叫 self._train_model()
    
    def _load_real_data(self, path=None):
        if path is None:
            path = self.data_path  # 預設用 __init__ 設定的路徑
        df = pd.read_csv(path)
        # 重新命名欄位
        df = df.rename(columns={
            'LIMIT_BAL': 'income',
            'AGE': 'age',
            # 其他欄位依需求對應
            # ...
            'default payment next month': 'target'
        })
        # 只保留你需要的欄位
        df = df[['income', 'age', 'employment_years', 'debt_ratio', 'credit_card_repayment', 'credit_score', 'overdue_count', 'target']]
        X = df[self.feature_names].values
        y = df['target'].values
        return X, y
    
    def _train_model(self):
        """訓練模型並計算評估指標"""
        X, y = self._load_real_data()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # 訓練模型
        self.model.fit(X_train, y_train)
        
        # 計算評估指標
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        self.accuracy = accuracy_score(y_test, y_pred)
        self.auc = roc_auc_score(y_test, y_pred_proba)
        self.confusion_matrix = confusion_matrix(y_test, y_pred)
        
        # 計算SHAP值
        explainer = shap.TreeExplainer(self.model)
        self.shap_values = explainer.shap_values(X_test)
        
        # 生成特徵重要性圖
        self._generate_feature_importance_plot()
        
        # 打印測試集準確率
        print("測試集準確率：", self.accuracy)
        
        joblib.dump(self.model, "model.pkl")
        print("✅ 模型已儲存為 model.pkl")
    
    def _generate_feature_importance_plot(self):
        """生成特徵重要性圖"""
        plt.figure(figsize=(10, 6))
        shap.summary_plot(self.shap_values, feature_names=self.feature_names, show=False)
        plt.tight_layout()
        
        # 將圖表轉換為base64編碼
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        self.feature_importance_plot = base64.b64encode(img.getvalue()).decode()
        plt.close()
    
    def calculate_credit_score(self, income, age, employment_years, debt_ratio, credit_card_repayment, credit_score, overdue_count):
        """根據業界常見信用評分原則計算信用分數 (模仿 FICO)"""
        # 權重設計（參考 FICO）：
        # 付款歷史(信用卡還款/逾期)35%、負債30%、收入10%、信用歷史(年資/年齡)15%、信用組合(聯徵分數)10%
        # 分數滿分100
        score = 0
        # 付款歷史（信用卡還款紀錄、逾期次數）
        payment_history = (credit_card_repayment / 12) * 25 + (1 - min(overdue_count, 5)/5) * 10  # 35分
        # 負債比率
        debt_score = (1 - min(debt_ratio, 1)) * 30  # 30分
        # 收入
        income_score = min(income / 1_000_000, 1) * 10  # 10分
        # 信用歷史（年資、年齡）
        history_score = min(employment_years / 20, 1) * 7 + min((age-18)/47, 1) * 8  # 15分
        # 聯徵分數
        credit_score_score = (credit_score - 300) / 550 * 10  # 10分
        score = payment_history + debt_score + income_score + history_score + credit_score_score
        return int(round(score))
    
    def calculate_monthly_payment(self, loan_amount, interest_rate, years):
        """計算每月還款金額"""
        monthly_rate = interest_rate / 12
        num_payments = years * 12
        monthly_payment = loan_amount * (monthly_rate * (1 + monthly_rate)**num_payments) / ((1 + monthly_rate)**num_payments - 1)
        return monthly_payment
    
    def calculate_apr(self, loan_amount, interest_rate, years, processing_fee):
        """計算總費用年百分率 (APR)"""
        # 計算總利息支出
        monthly_payment = self.calculate_monthly_payment(loan_amount, interest_rate, years)
        total_payment = monthly_payment * years * 12
        total_interest = total_payment - loan_amount
        
        # 計算總費用（利息 + 手續費）
        total_cost = total_interest + processing_fee
        
        # 計算 APR
        # 使用簡化的 APR 計算公式：(總費用 / 貸款金額) / 貸款年限
        apr = (total_cost / loan_amount) / years
        
        return apr
    
    def get_personal_advice(self, income, age, employment_years, debt_ratio):
        advice = []
        if income < 400000:
            advice.append("建議提升收入，有助於提高信用分數與可貸金額。")
        if debt_ratio > 0.4:
            advice.append("建議降低負債比率，減少現有債務或增加收入。")
        if employment_years < 3:
            advice.append("建議累積更多工作年資，提升穩定性。")
        if age < 25 or age > 60:
            advice.append("年齡在 25-60 歲區間較易獲得較佳條件。")
        if not advice:
            advice.append("您的條件良好，請持續保持良好財務紀律！")
        return advice

    def evaluate_loan(self, income, age, employment_years, debt_ratio, loan_amount, loan_years, credit_card_repayment=0, credit_score=650, overdue_count=0):
        """評估貸款申請（優化版）"""
        # 新增參數以便計算信用分數
        credit_score_val = self.calculate_credit_score(
            income, age, employment_years, debt_ratio, credit_card_repayment, credit_score, overdue_count
        )
        # 風險分級與利率規則（更貼近業界）
        if credit_score_val >= 80:
            interest_rate = 0.045
            processing_fee = 2000
            max_amount = min(income * 12, 6000000)
            approval = "核准"
            risk_level = "低風險"
            reasons = ["信用紀錄極佳，財務穩健"]
        elif credit_score_val >= 70:
            interest_rate = 0.06
            processing_fee = 4000
            max_amount = min(income * 10, 4000000)
            approval = "核准"
            risk_level = "中低風險"
            reasons = ["信用紀錄良好，財務狀況穩定"]
        elif credit_score_val >= 60:
            interest_rate = 0.08
            processing_fee = 6000
            max_amount = min(income * 8, 2500000)
            approval = "核准"
            risk_level = "中風險"
            reasons = ["信用紀錄尚可，部分指標需加強"]
        elif credit_score_val >= 50:
            interest_rate = 0.11
            processing_fee = 9000
            max_amount = min(income * 5, 1000000)
            approval = "有條件核准"
            risk_level = "高風險"
            reasons = ["信用紀錄偏弱，建議改善財務習慣"]
        else:
            interest_rate = 0.15
            processing_fee = 12000
            max_amount = 0
            approval = "拒絕"
            risk_level = "極高風險"
            reasons = ["信用紀錄不佳，無法核貸"]
        if loan_amount <= max_amount and approval != "拒絕":
            monthly_payment = self.calculate_monthly_payment(loan_amount, interest_rate, loan_years)
            total_payment = monthly_payment * loan_years * 12
            total_interest = total_payment - loan_amount
            apr = self.calculate_apr(loan_amount, interest_rate, loan_years, processing_fee)
        else:
            monthly_payment = 0
            total_payment = 0
            total_interest = 0
            apr = 0
            approval = "拒絕"
            reasons.append("申請金額超過可貸額度")
        personal_advice = self.get_personal_advice(income, age, employment_years, debt_ratio)
        return {
            "信用評分": credit_score_val,
            "風險等級": risk_level,
            "核准狀態": approval,
            "可貸金額": f"{max_amount:,}",
            "年利率": f"{interest_rate*100:.1f}%",
            "手續費": f"{processing_fee:,}",
            "總費用年百分率(APR)": f"{apr*100:.2f}%",
            "每月還款": f"{round(monthly_payment, 2):,}" if monthly_payment > 0 else "不適用",
            "總還款金額": f"{round(total_payment, 2):,}" if total_payment > 0 else "不適用",
            "總利息支出": f"{round(total_interest, 2):,}" if total_interest > 0 else "不適用",
            "貸款期限": f"{loan_years}年",
            "決策原因": reasons,
            "個人化建議": personal_advice
        }

# 創建全局的貸款評估器實例
evaluator = LoanEvaluator()

# HTML 模板
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>貸款評估系統</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body { padding: 20px; }
        .result-card { margin-top: 20px; }
        .risk-low { color: green; }
        .risk-medium { color: orange; }
        .risk-high { color: red; }
        .risk-very-high { color: darkred; }
        .score-bar { height: 24px; background: #eee; border-radius: 12px; overflow: hidden; }
        .score-bar-inner { height: 100%; transition: width 0.5s; }
        .score-bar-low { background: #4caf50; }
        .score-bar-medium { background: #ffc107; }
        .score-bar-high { background: #ff5722; }
        .score-bar-very-high { background: #b71c1c; }
        .nav-tabs { margin-bottom: 20px; }
        .tab-content { padding: 20px; }
        .apr-info {
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            margin: 20px 0;
        }
        .privacy-alert {
            background: #e3f2fd;
            color: #1565c0;
            padding: 10px 20px;
            border-radius: 6px;
            margin-bottom: 20px;
            font-size: 1.1em;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="privacy-alert">
            <b>隱私提醒：</b> 您的個人資料僅用於本系統評估，並受到嚴格保護，請安心填寫。
        </div>
        <h1 class="mb-4">貸款評估系統</h1>
        
        <div class="apr-info">
            <h5>什麼是總費用年百分率(APR)？</h5>
            <p>總費用年百分率(APR)是申貸過程中產生的所有費用，包含申貸額度、開辦費、帳管費及利息等平均分攤到實際還款期數後計算出來的總費用比率。這個數值可以幫助您更準確地比較不同貸款方案的真實成本。</p>
            <ul>
                <li>APR 計算包含：利息、手續費、帳管費等所有相關費用。</li>
                <li>本系統 APR 計算公式：(總費用/貸款金額)/貸款年限。</li>
            </ul>
        </div>
        
        <ul class="nav nav-tabs" id="myTab" role="tablist">
            <li class="nav-item">
                <a class="nav-link active" id="evaluation-tab" data-bs-toggle="tab" href="#evaluation" role="tab">貸款評估</a>
            </li>
            <li class="nav-item">
                <a class="nav-link" id="model-tab" data-bs-toggle="tab" href="#model" role="tab">模型說明</a>
            </li>
        </ul>
        
        <div class="tab-content" id="myTabContent">
            <div class="tab-pane fade show active" id="evaluation" role="tabpanel">
                <div class="card">
                    <div class="card-body">
                        <h5 class="card-title">輸入申請資料</h5>
                        <form method="POST">
                            <div class="row">
                                <div class="col-md-6">
                                    <h6 class="mt-3">基本資料</h6>
                                    <div class="mb-3">
                                        <label class="form-label">年收入</label>
                                        <input type="number" class="form-control" name="income" required min="0" step="1000" value="{{ request.form.income }}">
                                    </div>
                                    <div class="mb-3">
                                        <label class="form-label">年齡</label>
                                        <input type="number" class="form-control" name="age" required min="20" max="80" value="{{ request.form.age }}">
                                    </div>
                                    <div class="mb-3">
                                        <label class="form-label">工作年資</label>
                                        <input type="number" class="form-control" name="employment_years" required min="0" step="0.5" value="{{ request.form.employment_years }}">
                                    </div>
                                    <div class="mb-3">
                                        <label class="form-label">負債比率 (0-1之間)</label>
                                        <input type="number" class="form-control" name="debt_ratio" required min="0" max="1" step="0.01" value="{{ request.form.debt_ratio }}">
                                    </div>
                                    <div class="mb-3">
                                        <label class="form-label">信用卡還款紀錄 (0~12)</label>
                                        <input type="number" class="form-control" name="credit_card_repayment" required min="0" max="12" step="1" value="{{ request.form.credit_card_repayment }}">
                                    </div>
                                    <div class="mb-3">
                                        <label class="form-label">聯徵分數 (300~850)</label>
                                        <input type="number" class="form-control" name="credit_score" required min="300" max="850" step="1" value="{{ request.form.credit_score }}">
                                    </div>
                                    <div class="mb-3">
                                        <label class="form-label">歷史逾期次數</label>
                                        <input type="number" class="form-control" name="overdue_count" required min="0" max="20" step="1" value="{{ request.form.overdue_count }}">
                                    </div>
                                </div>
                                <div class="col-md-6">
                                    <h6 class="mt-3">貸款資料</h6>
                                    <div class="mb-3">
                                        <label class="form-label">申請貸款金額</label>
                                        <input type="number" class="form-control" name="loan_amount" required min="10000" step="10000" value="{{ request.form.loan_amount }}">
                                    </div>
                                    <div class="mb-3">
                                        <label class="form-label">貸款期限 (年)</label>
                                        <input type="number" class="form-control" name="loan_years" required min="1" max="30" value="{{ request.form.loan_years }}">
                                    </div>
                                </div>
                            </div>
                            <button type="submit" class="btn btn-primary mt-3">評估貸款</button>
                        </form>
                    </div>
                </div>

                {% if result %}
                <div class="card result-card">
                    <div class="card-body">
                        <h5 class="card-title">評估結果</h5>
                        <div class="mb-3">
                            <label>信用分數：</label>
                            <div class="score-bar">
                                <div class="score-bar-inner {% if result['風險等級']=='低風險' %}score-bar-low{% elif result['風險等級']=='中風險' %}score-bar-medium{% elif result['風險等級']=='高風險' %}score-bar-high{% else %}score-bar-very-high{% endif %}\" style=\"width: {{ result['信用評分'] }}%;\"></div>
                            </div>
                            <span>{{ result['信用評分'] }} 分</span>
                        </div>
                        <table class="table">
                            <tbody>
                                {% for key, value in result.items() %}
                                    {% if key != '決策原因' and key != '個人化建議' %}
                                    <tr>
                                        <th>{{ key }}</th>
                                        <td class="{% if key == '風險等級' %}
                                            {% if value == '低風險' %}risk-low
                                            {% elif value == '中風險' %}risk-medium
                                            {% elif value == '高風險' %}risk-high
                                            {% else %}risk-very-high
                                            {% endif %}
                                        {% endif %}">{{ value }}</td>
                                    </tr>
                                    {% endif %}
                                {% endfor %}
                            </tbody>
                        </table>
                        <div class="mb-3">
                            <b>風險等級對應利率：</b>
                            <ul>
                                <li><span class="risk-low">低風險</span>：5% ~ 6%</li>
                                <li><span class="risk-medium">中風險</span>：7% ~ 9%</li>
                                <li><span class="risk-high">高風險</span>：10% ~ 13%</li>
                                <li><span class="risk-very-high">極高風險</span>：14% 以上</li>
                            </ul>
                        </div>
                        <h6 class="mt-4">決策原因說明：</h6>
                        <ul>
                            {% for reason in result.決策原因 %}
                            <li>{{ reason }}</li>
                            {% endfor %}
                        </ul>
                        <h6 class="mt-4">個人化建議：</h6>
                        <ul>
                            {% for adv in result.個人化建議 %}
                            <li>{{ adv }}</li>
                            {% endfor %}
                        </ul>
                    </div>
                </div>
                {% endif %}
            </div>
            
            <div class="tab-pane fade" id="model" role="tabpanel">
                <div class="card">
                    <div class="card-body">
                        <h5 class="card-title">模型說明</h5>
                        <h6>模型評估指標</h6>
                        <ul>
                            <li>準確率：{{ "%.2f"|format(evaluator.accuracy * 100) }}%</li>
                            <li>AUC：{{ "%.2f"|format(evaluator.auc * 100) }}%</li>
                            <li>資料更新頻率：每次系統重啟時自動產生新模擬資料</li>
                        </ul>
                        
                        <h6>信用評分模型依據</h6>
                        <ul>
                            <li>還款歷史（如信用卡還款紀錄、逾期次數）</li>
                            <li>負債狀況（負債比率）</li>
                            <li>信用紀錄長度（工作年資、聯徵分數）</li>
                            <li>年收入</li>
                        </ul>
                        
                        <h6>信用評分意義</h6>
                        <ul>
                            <li>80分以上：低風險，享有最低利率</li>
                            <li>60-79分：中風險，利率中等</li>
                            <li>40-59分：高風險，利率較高</li>
                            <li>40分以下：極高風險，通常無法核貸</li>
                        </ul>
                        
                        <h6>混淆矩陣</h6>
                        <table class="table table-bordered">
                            <tr>
                                <th></th>
                                <th>預測拒絕</th>
                                <th>預測核准</th>
                            </tr>
                            <tr>
                                <th>實際拒絕</th>
                                <td>{{ evaluator.confusion_matrix[0][0] }}</td>
                                <td>{{ evaluator.confusion_matrix[0][1] }}</td>
                            </tr>
                            <tr>
                                <th>實際核准</th>
                                <td>{{ evaluator.confusion_matrix[1][0] }}</td>
                                <td>{{ evaluator.confusion_matrix[1][1] }}</td>
                            </tr>
                        </table>
                        
                        <h6>特徵重要性</h6>
                        <img src="data:image/png;base64,{{ evaluator.feature_importance_plot }}" class="img-fluid">
                        
                        <h6>評分標準說明</h6>
                        <ul>
                            <li>收入（最高30分）：年收入越高，分數越高</li>
                            <li>年齡（最高20分）：25-55歲最佳，20-60歲次之</li>
                            <li>工作年資（最高20分）：年資越長，分數越高</li>
                            <li>負債比率（最高30分）：比率越低，分數越高</li>
                            <li>信用卡還款紀錄、聯徵分數、逾期次數等亦納入評分</li>
                        </ul>
                        
                        <h6>風險等級判定</h6>
                        <ul>
                            <li>80分以上：低風險，利率5%</li>
                            <li>60-79分：中風險，利率8%</li>
                            <li>40-59分：高風險，利率12%</li>
                            <li>40分以下：極高風險，拒絕貸款</li>
                        </ul>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
"""

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        try:
            income = float(request.form['income'])
            age = float(request.form['age'])
            employment_years = float(request.form['employment_years'])
            debt_ratio = float(request.form['debt_ratio'])
            credit_card_repayment = int(request.form['credit_card_repayment'])
            credit_score = int(request.form['credit_score'])
            overdue_count = int(request.form['overdue_count'])
            loan_amount = float(request.form['loan_amount'])
            loan_years = int(request.form['loan_years'])
            result = evaluator.evaluate_loan(
                income, age, employment_years, debt_ratio, loan_amount, loan_years,
                credit_card_repayment, credit_score, overdue_count
            )
        except ValueError as e:
            result = {"錯誤": "請輸入有效的數值"}
    
    return render_template_string(HTML_TEMPLATE, result=result, evaluator=evaluator)

if __name__ == '__main__':
    app.run(debug=True, port=5001)
