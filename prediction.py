# 导入包
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
import shap
import matplotlib.pyplot as plt
# 导入必要的组件以显示SHAP的HTML可视化
import streamlit.components.v1 as components


## ===================== 加载模型 =====================##
#MODEL_PATH = "C:/Users/HZH/Desktop/CHARLS心脏代谢共病/streamlit.app/RF/rf_model.pkl"
model = joblib.load("rf_model.pkl")

# 查看特征 - 这会显示模型期望的特征顺序
model_feature_names = model.feature_names_in_
print("模型训练时的特征名：", model_feature_names)
print("模型训练时的特征数量：", len(model_feature_names))

## ===================== 特征列表与配置 =====================##
# 修正特征顺序，使其与模型训练时的顺序一致
FEATURES = model_feature_names  # 使用模型的特征顺序

# 特征类型配置：区分分类特征（二元）和数值特征
CATEGORICAL_FEATURES = ["Dyslipidaemia"]
NUMERICAL_FEATURES = [f for f in FEATURES if f not in CATEGORICAL_FEATURES]

# 特征映射（提升用户体验）
FEATURE_NAMES = {
    "Age": "Age(years)",
    "FI": "Frailty Index",
    "Weight": "Weight(kg)",
    "Waist": "Waist(cm)",
    "SBP": "SBP(mmHg)",
    "FBG": "FBG(mg/dL)",
    "HbA1c": "HbA1c(%)",
    "TyG": "TyG",   
    "Dyslipidaemia": "Dyslipidaemia",
}

## ===================== Streamlit 页面配置 =====================##
# 设置Streamlit页面配置：页面标题和宽屏布局
st.set_page_config(page_title="CMM Prediction Model", layout="wide")
# 设置应用程序主标题
st.title("🫀 CMM Prediction Model (Random Forest Model )")

## ===================== 单样本预测 =============================##
st.header("🔹 Predict CMM")

# 创建空字典用于存储用户输入的所有特征值
input_data = {} 
col1, col2 = st.columns(2)

# 遍历特征生成输入组件
for i, feature in enumerate(FEATURES):
    # 按奇偶分配到不同列
    with col1 if i % 2 == 0 else col2:
        feature_name = FEATURE_NAMES.get(feature, feature)
        
        if feature in CATEGORICAL_FEATURES:
            # 分类特征使用选择框（0=否，1=是）
            val = st.selectbox(
                f"{feature_name}",
                options=[0, 1],
                format_func=lambda x: "Yes" if x == 1 else "No",
                key=feature
            )
        else:
            # 数值特征使用数字输入，并设置合理范围
            if feature == "Age":
                val = st.number_input(f"{feature_name}", min_value=45, max_value=120, value=60, step=1)
            elif feature == "FI":
                val = st.number_input(f"{feature_name}", min_value=0.00, max_value=1.00, value=0.50, step=0.01) 
            elif feature == "Weight":
                # 将所有参数统一为浮点数，与step=0.5保持一致
                val = st.number_input(f"{feature_name}", min_value=30.0, max_value=150.0, value=60.0, step=0.1)  
            elif feature == "Waist":
                # 将所有参数统一为浮点数，与step=0.5保持一致
                val = st.number_input(f"{feature_name}", min_value=3.0, max_value=150.0, value=70.0, step=0.1)   
            elif feature == "SBP":
                val = st.number_input(f"{feature_name}", min_value=60, max_value=220, value=120, step=1)
            elif feature == "FBG":
                val = st.number_input(f"{feature_name}", min_value=50.0, max_value=260.0, value=110.0, step=0.1)
            elif feature == "HbA1c":
                val = st.number_input(f"{feature_name}", min_value=3.0, max_value=15.0, value=5.0, step=0.1)
            elif feature == "TyG":
                val = st.number_input(f"{feature_name}", min_value=5.0, max_value=12.0, value=8.0, step=0.1)
        # 将用户输入的特征值存储到input_data字典中，键为特征名，值为用户输入值
        input_data[feature] = val

# 预测按钮与逻辑
if st.button("Predict CMM"):
    try:
        # 构造输入DataFrame，确保列顺序与模型期望一致
        df_input = pd.DataFrame([input_data], columns=FEATURES)
        
        # 处理可能的分类特征（如果有字符串类型）
        for col in df_input.columns:
            if df_input[col].dtype == object:
                le = LabelEncoder()
                df_input[col] = le.fit_transform(df_input[col].astype(str))
        
        # 标准化数值特征（注意：实际部署应使用训练时的scaler，此处为简化处理）
        #scaler = StandardScaler()
        #X_scaled = scaler.fit_transform(df_input)
        X_scaled = df_input
        
        # 模型预测
        # model.predict返回预测类别数组，[0]表示取第一个样本的预测结果
        y_pred = model.predict(X_scaled)[0]
        # model.predict_proba返回概率预测数组，[0][1]表示第一个样本预测为类别1的概率
        y_proba = model.predict_proba(X_scaled)[0][1]
        
        # 显示结果
        st.success(f"CMM Probability: {y_proba:.3f}")
        
     ## ===================== SHAP分析 =====================##
        # 初始化SHAP解释器
        explainer = shap.TreeExplainer(model)
        # 计算SHAP值
        shap_values = explainer.shap_values(X_scaled)
        # 解释第n+1个样本（索引从0开始）。注意：只能为0
        sample_index = 0   
        # 设定要显示的特征数量 
        top_n = 9 
        
        ####  SHAP Force Plot ####
        st.subheader("📊 SHAP Force Plot")         
        # 创建force_plot
        force_plot_html = shap.force_plot(
            explainer.expected_value[1],        
            shap_values[sample_index, :top_n, 1],  
            features=df_input.iloc[sample_index, :top_n],   
            feature_names=df_input.columns,  
            matplotlib=False,                 
            contribution_threshold=0 )
        # 将SHAP的force_plot转换为HTML并在Streamlit中显示
        shap_html = f"<head>{shap.getjs()}</head><body>{force_plot_html.html()}</body>"
        components.html(shap_html, height=1000,width='100%')

        ####  SHAP Waterfall Plot ####
        st.subheader("💧 SHAP Waterfall Plot")
        # 创建waterfall_plot
        exp = shap.Explanation(
            values=shap_values[sample_index, :, 1],  # 类别1的SHAP值
            base_values=explainer.expected_value[1], # 类别1的基准值
            data=df_input.iloc[sample_index].values, # 当前样本的原始特征值
            feature_names=df_input.columns           # 特征名称
            )

        # 创建瀑布图
        shap.plots.waterfall(exp, max_display=10, show=False) # max_display控制显示的特征数量
        plt.tight_layout() # 调整布局，防止标签重叠
        # 在Streamlit中显示Matplotlib图表
        st.pyplot(plt.gcf(), width=1200,dpi=500)  # 禁用自动扩展use_container_width=False

    except Exception as e:
        st.error(f"Prediction process error:{str(e)}")


##打开终端win+R,再运行streamlit run "C:\Users\HZH\Desktop\CHARLS心脏代谢共病\streamlit.app\RF\prediction.py"##
