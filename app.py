"""
银行客户信用风险评估系统 — Flask 后端
模型：EXP-1 CatBoost (AUC=0.8422, KS=0.542)
数据集：Give Me Some Credit (GMSC)
"""

import os
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# =============================================================================
# 模型加载：自动在多个位置寻找 catboost_model.pkl
# 把 pkl 文件放在以下任意一个位置都能被识别到：
#   1. flask_app/ 目录下（与 app.py 同级）← 推荐
#   2. flask_app/model/ 目录下
#   3. 通过环境变量 MODEL_PATH 指定绝对路径
# =============================================================================
def find_model_path() -> str:
    env_path = os.environ.get("MODEL_PATH", "")
    if env_path and os.path.exists(env_path):
        return env_path

    base = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(base, "catboost_model.pkl"),
        os.path.join(base, "models", "catboost_model.pkl"),
        os.path.join(base, "..", "catboost_model.pkl"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return os.path.normpath(p)
    return ""


model = None

def load_model():
    global model
    path = find_model_path()
    if not path:
        print("\n❌ 找不到 catboost_model.pkl，请先按说明导出模型。")
        return
    model = joblib.load(path)
    print(f"✅ 模型加载成功：{path}")

# ── 特征列（与训练时完全一致，顺序不能变）──────────────────────────────────────
FEATURE_COLS = [
    "RevolvingUtilizationOfUnsecuredLines",
    "age",
    "NumberOfTime30-59DaysPastDueNotWorse",
    "DebtRatio",
    "MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans",
    "NumberOfTimes90DaysLate",
    "NumberRealEstateLoansOrLines",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfDependents",
]

# ── 风险等级判定（基于 KS 最优阈值 ~0.27 + 业务分层）──────────────────────────
def get_risk_level(proba: float):
    if proba < 0.10:
        return "低风险", "low"
    elif proba < 0.27:
        return "中等风险", "medium"
    elif proba < 0.50:
        return "较高风险", "high"
    else:
        return "高风险", "critical"


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "模型未加载，请联系管理员"}), 503

    try:
        data = request.get_json()

        # ── 解析并校验输入 ────────────────────────────────────────────────────
        values = {}
        for col in FEATURE_COLS:
            raw = data.get(col)
            if raw is None or str(raw).strip() == "":
                return jsonify({"error": f"缺少字段：{col}"}), 400
            try:
                values[col] = float(raw)
            except ValueError:
                return jsonify({"error": f"字段 {col} 必须为数字"}), 400

        # ── 构建 DataFrame（保持列顺序）──────────────────────────────────────
        X = pd.DataFrame([values], columns=FEATURE_COLS)

        # ── 缺失值填充（与训练预处理保持一致：用中位数）────────────────────────
        MEDIANS = {
            "RevolvingUtilizationOfUnsecuredLines": 0.154,
            "age": 52.0,
            "NumberOfTime30-59DaysPastDueNotWorse": 0.0,
            "DebtRatio": 0.366,
            "MonthlyIncome": 5400.0,
            "NumberOfOpenCreditLinesAndLoans": 8.0,
            "NumberOfTimes90DaysLate": 0.0,
            "NumberRealEstateLoansOrLines": 1.0,
            "NumberOfTime60-89DaysPastDueNotWorse": 0.0,
            "NumberOfDependents": 0.0,
        }
        for col, med in MEDIANS.items():
            if pd.isna(X[col]).any():
                X[col] = X[col].fillna(med)

        # ── 预测 ──────────────────────────────────────────────────────────────
        proba = float(model.predict_proba(X)[0, 1])
        original_proba = proba  # 保留原始概率供展示

        # ── 资产修正（四种组合，行为金融学锚定效应）──────────────────────────
        owns_house = bool(data.get("owns_house", False))
        owns_car   = bool(data.get("owns_car",   False))
        asset_corrections = []

        if owns_house and owns_car:
            proba = proba * 0.78
            asset_corrections.append("持有全款房产及车辆，资产锚定效应强，风险评估下调")
        elif owns_house and not owns_car:
            proba = proba * 0.82
            asset_corrections.append("持有全款房产，不动产锚定效应显著，风险评估下调")
        elif not owns_house and owns_car:
            proba = proba * 1.05
            asset_corrections.append("无固定房产，居住锚定较弱，风险评估小幅上调")
        # 无房无车：不修正，使用模型原始输出

        proba = round(min(max(proba, 0.0), 1.0), 4)
        risk_label, risk_code = get_risk_level(proba)

        # ── 风险因素分析 ──────────────────────────────────────────────────────
        tips = []
        tips.extend(asset_corrections)

        if values["RevolvingUtilizationOfUnsecuredLines"] > 0.5:
            tips.append("循环信贷使用率偏高（>50%），是违约的重要预警信号")
        if values["NumberOfTimes90DaysLate"] > 0:
            tips.append(f"存在 {int(values['NumberOfTimes90DaysLate'])} 次严重逾期（≥90天），风险显著上升")
        if values["NumberOfTime30-59DaysPastDueNotWorse"] > 2:
            tips.append("近期逾期次数较多（30-59天），建议关注还款能力")
        if values["DebtRatio"] > 0.5:
            tips.append("债务收入比偏高，偿债压力较大")
        if values["age"] < 25:
            tips.append("客户年龄较低，信用历史较短")
        if not tips:
            tips.append("各项指标处于正常范围，综合风险较低")

        return jsonify({
            "probability":          round(proba, 4),
            "probability_pct":      f"{proba * 100:.1f}%",
            "original_probability": round(original_proba, 4),
            "asset_corrected":      len(asset_corrections) > 0,
            "risk_level":           risk_label,
            "risk_code":            risk_code,
            "tips":                 tips,
        })

    except Exception as e:
        return jsonify({"error": f"预测异常：{str(e)}"}), 500

load_model()

if __name__ == "__main__":
    app.run(debug=True, port=5000)