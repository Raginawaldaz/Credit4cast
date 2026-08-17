# 银行客户信用风险评估系统

> 基于机器学习（CatBoost）的银行客户信用风险评估系统 —— 本科毕业设计原型系统
>
> Bank Customer Credit Risk Assessment System — powered by CatBoost

基于 **Flask** 的轻量 Web 应用：输入客户特征，调用 **CatBoost** 模型输出违约概率与风险等级，并给出简要风险提示。

## ✨ 功能特性

- 🧠 **CatBoost 梯度提升模型**：训练指标 AUC ≈ 0.8422，KS ≈ 0.542
- 📊 **10 维客户特征输入**（Give Me Some Credit 风格），自动缺失值填充
- 🏷️ **四级风险分层**：低风险 / 中等风险 / 较高风险 / 高风险
- 💰 **资产锚定修正**（可选）：支持补充全款房产 / 车辆持有情况，基于行为金融学锚定效应微调违约概率
- 💡 **风险因素分析**：自动生成可解释的风险提示（逾期次数、信贷使用率、负债比等）
- 🖥️ **简洁单页界面**：表单录入 + 实时结果仪表盘
- 🔌 **JSON REST API**：`POST /predict` 可直接集成到其他系统

## 🧠 模型说明

| 项目 | 说明 |
|------|------|
| 模型 | EXP-1 CatBoost 梯度提升模型 |
| 数据背景 | Give Me Some Credit (GMSC) 风格特征 |
| 评估指标 | AUC ≈ 0.8422，KS ≈ 0.542 |
| 输出 | 两年内违约概率 `p ∈ [0, 1]` |

**风险等级划分**（基于 KS 最优阈值 ≈ 0.27 与业务分层）：

| 违约概率 | 风险等级 | risk_code |
|----------|----------|-----------|
| < 10% | 低风险 | `low` |
| 10% ~ 27% | 中等风险 | `medium` |
| 27% ~ 50% | 较高风险 | `high` |
| ≥ 50% | 高风险 | `critical` |

**资产修正规则**（可选补充，仅影响展示概率，锚定效应设计）：

| 资产情况 | 概率调整 |
|----------|----------|
| 全款房产 + 车辆 | × 0.78（下调） |
| 仅全款房产 | × 0.82（下调） |
| 仅全款车辆 | × 1.05（小幅上调） |
| 均无 | 不修正 |

## 📋 环境要求

- Python 3.9+（建议与训练环境一致）
- 依赖见 [`requirements.txt`](requirements.txt)：Flask、CatBoost、pandas、numpy、joblib、scikit-learn

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 放置模型文件

将导出的 **`catboost_model.pkl`** 放到以下任一位置（应用启动时自动查找）：

1. 与 `app.py` 同目录（推荐）
2. `models/catboost_model.pkl`（本仓库已自带）
3. 通过环境变量 **`MODEL_PATH`** 指定绝对路径

```bash
# 方式三示例（Windows）
set MODEL_PATH=D:\path\to\catboost_model.pkl
```

### 3. 启动应用

```bash
python app.py
```

浏览器访问 <http://127.0.0.1:5000>

## 🔌 API 接口

### `GET /` — 网页表单

返回前端评估页面。

### `POST /predict` — JSON 预测

请求体为 JSON，需包含与训练一致的 10 个数值字段（字段名由后端 `FEATURE_COLS` 定义），可选布尔字段 `owns_house`、`owns_car` 用于资产修正。

**请求示例：**

```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "RevolvingUtilizationOfUnsecuredLines": 0.35,
    "age": 45,
    "NumberOfTime30-59DaysPastDueNotWorse": 1,
    "DebtRatio": 0.366,
    "MonthlyIncome": 8000,
    "NumberOfOpenCreditLinesAndLoans": 8,
    "NumberOfTimes90DaysLate": 0,
    "NumberRealEstateLoansOrLines": 1,
    "NumberOfTime60-89DaysPastDueNotWorse": 0,
    "NumberOfDependents": 1,
    "owns_house": true,
    "owns_car": false
  }'
```

**响应示例：**

```json
{
  "probability": 0.0969,
  "probability_pct": "9.7%",
  "original_probability": 0.1182,
  "asset_corrected": true,
  "risk_level": "低风险",
  "risk_code": "low",
  "tips": [
    "持有全款房产，不动产锚定效应显著，风险评估下调",
    "循环信贷使用率偏高（>50%），是违约的重要预警信号"
  ]
}
```

**错误码：**

| 状态码 | 含义 |
|--------|------|
| 400 | 请求缺少字段或字段非数字 |
| 500 | 预测过程异常 |
| 503 | 模型未加载（请检查 `catboost_model.pkl` 是否就位） |

## 📁 项目结构

```
.
├── app.py                 # Flask 应用与预测逻辑（模型加载 / 特征校验 / 资产修正）
├── models/
│   └── catboost_model.pkl # 训练好的 CatBoost 模型
├── templates/
│   └── index.html         # 前端页面（表单 + 结果仪表盘）
├── static/
│   └── mina.bin           # 静态资源（彩蛋用图片）
├── requirements.txt       # Python 依赖
├── LICENSE                # Apache License 2.0
└── README.md              # 本文档
```

## 📄 许可证

本项目采用 **Apache License 2.0** 开源协议，详见 [LICENSE](LICENSE)。

## ⚠️ 免责声明

- **本项目为本科毕业设计原型系统，仅供学习交流与演示，不构成任何信贷或投资建议。**
- 实际业务部署需满足金融监管合规、审计与模型治理（Model Governance）要求，模型需经过独立验证与审批后方可上线。
- 资产修正规则为演示性业务设计，不代表真实风险评估准则。

---

# Bank Customer Credit Risk Assessment System

> A lightweight machine-learning-based credit risk assessment system — bachelor thesis prototype
>
> 基于 CatBoost 的银行客户信用风险评估系统

A **Flask**-based lightweight web application: input customer features, call the **CatBoost** model to output the probability of default (PD) and a risk rating, along with brief risk explanations.

## ✨ Features

- 🧠 **CatBoost gradient boosting model**: AUC ≈ 0.8422, KS ≈ 0.542
- 📊 **10-dimensional customer features** (Give Me Some Credit style) with automatic missing-value imputation
- 🏷️ **Four-tier risk rating**: Low / Medium / High / Critical
- 💰 **Asset anchoring adjustment** (optional): flag paid-off house / car ownership to fine-tune the PD based on behavioral-finance anchoring effects
- 💡 **Risk factor analysis**: auto-generated interpretable warnings (delinquency counts, revolving utilization, debt ratio, etc.)
- 🖥️ **Clean single-page UI**: form input + real-time result dashboard
- 🔌 **JSON REST API**: `POST /predict` can be integrated into other systems directly

## 🧠 Model Details

| Item | Description |
|------|-------------|
| Model | EXP-1 CatBoost gradient boosting model |
| Data basis | Give Me Some Credit (GMSC) style features |
| Metrics | AUC ≈ 0.8422, KS ≈ 0.542 |
| Output | Probability of default within 2 years `p ∈ [0, 1]` |

**Risk rating tiers** (based on the KS-optimal threshold ≈ 0.27 plus business segmentation):

| PD range | Risk level | risk_code |
|----------|------------|-----------|
| < 10% | Low risk | `low` |
| 10% – 27% | Medium risk | `medium` |
| 27% – 50% | High risk | `high` |
| ≥ 50% | Critical risk | `critical` |

**Asset adjustment rules** (optional; affects the displayed probability only — anchoring-effect design):

| Asset situation | Probability adjustment |
|-----------------|------------------------|
| Paid-off house + car | × 0.78 (down) |
| Paid-off house only | × 0.82 (down) |
| Paid-off car only | × 1.05 (slightly up) |
| Neither | No adjustment |

## 📋 Requirements

- Python 3.9+ (preferably matching the training environment)
- Dependencies in [`requirements.txt`](requirements.txt): Flask, CatBoost, pandas, numpy, joblib, scikit-learn

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Place the model file

Put the exported **`catboost_model.pkl`** in any of the following locations (auto-detected at startup):

1. Same directory as `app.py` (recommended)
2. `models/catboost_model.pkl` (already included in this repo)
3. Or point to an absolute path via the **`MODEL_PATH`** environment variable

```bash
# Example (Windows)
set MODEL_PATH=D:\path\to\catboost_model.pkl
```

### 3. Run the app

```bash
python app.py
```

Open <http://127.0.0.1:5000> in your browser.

## 🔌 API Reference

### `GET /` — Web form

Serves the front-end assessment page.

### `POST /predict` — JSON prediction

The request body is JSON and must include the 10 numeric fields consistent with training (field names are defined by `FEATURE_COLS` in the backend). Optional boolean fields `owns_house` and `owns_car` enable asset adjustment.

**Example request:**

```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "RevolvingUtilizationOfUnsecuredLines": 0.35,
    "age": 45,
    "NumberOfTime30-59DaysPastDueNotWorse": 1,
    "DebtRatio": 0.366,
    "MonthlyIncome": 8000,
    "NumberOfOpenCreditLinesAndLoans": 8,
    "NumberOfTimes90DaysLate": 0,
    "NumberRealEstateLoansOrLines": 1,
    "NumberOfTime60-89DaysPastDueNotWorse": 0,
    "NumberOfDependents": 1,
    "owns_house": true,
    "owns_car": false
  }'
```

**Example response:**

```json
{
  "probability": 0.0969,
  "probability_pct": "9.7%",
  "original_probability": 0.1182,
  "asset_corrected": true,
  "risk_level": "低风险",
  "risk_code": "low",
  "tips": [
    "持有全款房产，不动产锚定效应显著，风险评估下调",
    "循环信贷使用率偏高（>50%），是违约的重要预警信号"
  ]
}
```

**Error codes:**

| Status | Meaning |
|--------|---------|
| 400 | Missing field(s) or non-numeric values |
| 500 | Unexpected error during prediction |
| 503 | Model not loaded (check whether `catboost_model.pkl` is in place) |

## 📁 Project Structure

```
.
├── app.py                 # Flask app & prediction logic (model loading / validation / asset adjustment)
├── models/
│   └── catboost_model.pkl # Trained CatBoost model
├── templates/
│   └── index.html         # Front-end page (form + result dashboard)
├── static/
│   └── mina.bin           # Static asset (image used by an easter egg)
├── requirements.txt       # Python dependencies
├── LICENSE                # Apache License 2.0
└── README.md              # This document
```

## 📄 License

This project is licensed under the **Apache License 2.0** — see [LICENSE](LICENSE).

## ⚠️ Disclaimer

- **This project is a bachelor thesis prototype intended for learning, exchange and demonstration only. It does not constitute any credit or investment advice.**
- Any production deployment must meet financial regulatory compliance, audit, and model governance requirements; the model must pass independent validation and approval before going live.
- The asset adjustment rules are demonstrative business designs and do not represent real-world risk assessment criteria.
