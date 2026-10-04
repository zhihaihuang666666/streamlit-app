# 导入包
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
import shap
import matplotlib.pyplot as plt
import streamlit.components.v1 as components


## ===================== 加载模型 =====================##
#model = joblib.load("C:/Users/HZH/Desktop/CMM二修/streamlit.app/RF/rf_model.pkl") # 本地部署
model = joblib.load("rf_model.pkl") # Streamlit部署

# 查看模型训练时的特征
MODEL_FEATURES = list(model.feature_names_in_)
print("模型训练时的特征名：", MODEL_FEATURES)
print("模型训练时的特征数量：", len(MODEL_FEATURES))


## ===================== 特征列表与配置 =====================##
# 页面显示顺序：每两个变量显示在同一行
FEATURES = [
    "Gender", "Age", 
    "Hypertension", "Lung disease",
    "Memory problem", "SBP",
    "Weight", "WHtR",
    "HDLC", "LDLC",
    "FBG", "HbA1c"
]
# 分类变量
CATEGORICAL_FEATURES = ["Gender", "Hypertension", "Lung disease", "Memory problem"]
# 数值变量
NUMERICAL_FEATURES = [f for f in FEATURES if f not in CATEGORICAL_FEATURES]
# 页面显示名称
FEATURE_NAMES = {
    "Age": "Age (years)",
    "Gender": "Gender",
    "Hypertension": "Hypertension",
    "Lung disease": "Lung disease",
    "Memory problem": "Memory problem",
    "SBP": "SBP (mmHg)",
    "Weight": "Weight (kg)",
    "WHtR": "Waist-to-height ratio",
    "HDLC": "HDL-C (mg/dL)",
    "LDLC": "LDL-C (mg/dL)",
    "FBG": "FBG (mg/dL)",
    "HbA1c": "HbA1c (%)"
}


## ===================== Streamlit页面配置 =====================##
st.set_page_config(page_title="CMM Prediction Model", layout="wide")
st.title("🫀 CMM Prediction Model")


## ===================== 单样本预测 =====================##
input_data = {}
col1, col2 = st.columns(2)

for i, feature in enumerate(FEATURES):
    with col1 if i % 2 == 0 else col2:
        feature_name = FEATURE_NAMES.get(feature, feature)

        if feature in CATEGORICAL_FEATURES:
            if feature == "Gender":
                val = st.selectbox(
                    feature_name,
                    options=[0, 1],
                    format_func=lambda x: "Male" if x == 1 else "Female",
                    key=feature,
                    index=1
                )
            else:
                val = st.selectbox(
                    feature_name,
                    options=[0, 1],
                    format_func=lambda x: "Yes" if x == 1 else "No",
                    key=feature,
                    index=1
                )

        else:
            if feature == "Age":
                val = st.number_input(feature_name, min_value=50.0, max_value=150.0, value=50.0, step=1.0)

            elif feature == "SBP":
                val = st.number_input(feature_name, min_value=50.0, max_value=200.0, value=100.0, step=1.0)

            elif feature == "Weight":
                val = st.number_input(feature_name, min_value=30.0, max_value=200.0, value=60.0, step=0.1)

            elif feature == "WHtR":
                val = st.number_input(feature_name, min_value=0.00, max_value=1.00, value=0.50, step=0.01)

            elif feature == "HDLC":
                val = st.number_input(feature_name, min_value=10.0, max_value=200.0, value=50.0, step=0.1)

            elif feature == "LDLC":
                val = st.number_input(feature_name, min_value=20.0, max_value=300.0, value=100.0, step=0.1)

            elif feature == "FBG":
                val = st.number_input(feature_name, min_value=50.0, max_value=600.0, value=100.0, step=0.1)

            elif feature == "HbA1c":
                val = st.number_input(feature_name, min_value=3.0, max_value=20.0, value=5.5, step=0.1)

        input_data[feature] = val


## ===================== 预测按钮与逻辑 =====================##
if st.button("Predict CMM"):
    try:
        # 按页面显示顺序构造输入数据
        df_input = pd.DataFrame([input_data], columns=FEATURES)

        # 恢复模型训练时的变量顺序
        df_input = df_input[MODEL_FEATURES]

        # 字符变量转为数值
        for col in df_input.columns:
            if df_input[col].dtype == object:
                le = LabelEncoder()
                df_input[col] = le.fit_transform(df_input[col].astype(str))

        # 随机森林不需要标准化
        X_scaled = df_input

        # 模型预测
        y_pred = model.predict(X_scaled)[0]
        y_proba = model.predict_proba(X_scaled)[0][1]

        # 显示预测结果
        st.success(f"👉🏻 CMM Probability: {(y_proba * 100):.1f}%")


        ## ===================== SHAP分析 =====================##
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_scaled)
        sample_index = 0
        # 兼容不同SHAP版本
        if isinstance(shap_values, list):
            shap_values_cmm = shap_values[1][sample_index]
            expected_value_cmm = explainer.expected_value[1]

        elif np.asarray(shap_values).ndim == 3:
            shap_values_cmm = shap_values[sample_index, :, 1]
            expected_value_cmm = np.asarray(explainer.expected_value)[1]
        else:
            shap_values_cmm = shap_values[sample_index]
            expected_value = np.asarray(explainer.expected_value)
            expected_value_cmm = float(expected_value) if expected_value.ndim == 0 else expected_value[-1]


        #### SHAP Force Plot ####
        st.subheader("📊 Force Plot")
        force_plot_html = shap.force_plot(
            expected_value_cmm,
            shap_values_cmm,
            features=df_input.iloc[sample_index],
            feature_names=df_input.columns.tolist(),
            matplotlib=False,
            contribution_threshold=0
        )

        shap_html = f"<head>{shap.getjs()}</head><body>{force_plot_html.html()}</body>"
        components.html(shap_html, height=280, width="100%")

    except Exception as e:
        st.error(f"Prediction process error: {str(e)}")


## 终端运行：
## streamlit run "C:\Users\HZH\Desktop\CMM二修\streamlit.app\RF\prediction.py"
