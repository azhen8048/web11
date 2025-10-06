import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from matplotlib import font_manager

# 加载保存的随机森林模型
model = joblib.load('rf_model.pkl')

# 特征范围定义（根据提供的特征范围和数据类型）
feature_ranges = {
    "Ventilation_Status": {"type": "categorical", "options": [0, 1, 2, 3, 4, 5], "default": 0},
    "Charlson_Comorbidity_Index": {"type": "numerical", "min": 0, "max": 30, "default": 2},
    "SBP": {"type": "numerical", "min": 10, "max": 400, "default": 120},
    "Resp_Rate": {"type": "numerical", "min": 5, "max": 300, "default": 18},
    "Creatinine": {"type": "numerical", "min": 0.1, "max": 50.0, "default": 1.0},
    "Weight": {"type": "numerical", "min": 1.0, "max": 300.0, "default": 70.0},
    "Chloride": {"type": "numerical", "min": 10, "max": 200, "default": 100},
    "INR": {"type": "numerical", "min": 0.1, "max": 10.0, "default": 1.0},
    "Urine_output": {"type": "numerical", "min": 0, "max": 10000, "default": 1500},
    "Heart_Rate": {"type": "numerical", "min": 0, "max": 500, "default": 80},
    "Spo2": {"type": "numerical", "min": 50, "max": 100, "default": 97},
}

# Streamlit 界面
st.title("Prediction Model")

# 动态生成输入项
st.header("Enter the following feature values:")
feature_values = []
for feature, properties in feature_ranges.items():
    display_name = feature.replace('_', ' ') 
    if properties["type"] == "numerical":
        value = st.number_input(
            label=f"{display_name} ({properties['min']} - {properties['max']})",
            min_value=float(properties["min"]),
            max_value=float(properties["max"]),
            value=float(properties["default"]),
        )
    elif properties["type"] == "categorical":
        value = st.selectbox(
            label=f"{display_name} (Select a value)",
            options=properties["options"],
        )
    feature_values.append(value)

# 转换为模型输入格式
features = np.array([feature_values])

# 预测与 SHAP 可视化
if st.button("Predict"):
    # 模型预测
    predicted_class = model.predict(features)[0]
    predicted_proba = model.predict_proba(features)[0]

    # 提取预测的类别概率
    probability = predicted_proba[predicted_class] * 100

    # 显示预测结果，使用 Matplotlib 渲染指定字体
    text = f"Based on feature values, predicted possibility of Outcome is {probability:.2f}%"
    fig, ax = plt.subplots(figsize=(8, 1))

    # 设置Times New Roman斜体加粗
    try:
        prop = font_manager.FontProperties(
            family='Times New Roman',
            style='italic',
            weight='bold',
            size=16
        )
        ax.text(
            0.5, 0.5, text,
            fontproperties=prop,
            ha='center', va='center',
            transform=ax.transAxes
        )
    except:
        ax.text(
            0.5, 0.5, text,
            fontsize=16,
            ha='center', va='center',
            style='italic',
            weight='bold',
            family='serif',
            transform=ax.transAxes
        )

    ax.axis('off')
    plt.savefig("prediction_text.png", bbox_inches='tight', dpi=300, transparent=True)
    st.image("prediction_text.png")

    # 计算 SHAP 值
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(pd.DataFrame([feature_values], columns=feature_ranges.keys()))

    # 生成 SHAP 力图
    class_index = predicted_class
    shap_fig = shap.force_plot(
        explainer.expected_value[class_index],
        shap_values[:, :, class_index],
        pd.DataFrame([feature_values], columns=feature_ranges.keys()),
        matplotlib=True,
    )
    # 保存并显示 SHAP 图
    plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)
    st.image("shap_force_plot.png")