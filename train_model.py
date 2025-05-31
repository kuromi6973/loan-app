from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
import joblib
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np

# 載入資料集
X, y = load_iris(return_X_y=True)

# 建立並訓練模型
model = RandomForestClassifier()
model.fit(X, y)

# 預測
y_pred = model.predict(X)

# 儲存模型
joblib.dump(model, 'model.pkl')
print("✅ 模型已成功儲存為 model.pkl")

# 自動找出所有類別
classes = sorted(set(y) | set(y_pred))
cm = confusion_matrix(y, y_pred, labels=classes)

# 自訂類別名稱
label_names = [f'類別{c}' for c in classes]

# 建立混淆矩陣 DataFrame
cm_df = pd.DataFrame(cm, index=[f'實際{l}' for l in label_names],
                         columns=[f'預測{l}' for l in label_names])

# 存成 CSV 檔
cm_df.to_csv("confusion_matrix.csv", encoding='utf-8-sig')

# 輸出為 Excel
cm_df.to_excel("confusion_matrix.xlsx", index=True)

