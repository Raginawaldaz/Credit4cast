# 银行客户信用风险评估系统

基于 **Flask** 的轻量 Web 应用：输入客户特征，调用 **CatBoost** 模型输出违约概率与风险等级，并给出简要风险提示。

## 技术说明

 **模型**：CatBoost（训练指标参考：AUC ≈ 0.84，KS ≈ 0.54）
 **数据背景**：Give Me Some Credit（GMSC）风格特征
 **接口**：`GET /` 网页表单；`POST /predict` JSON 预测

## 环境要求

 Python 3.9+（建议与训练环境相近）
 依赖见 `requirements.txt`

## 安装与运行

```bash
cd BankCreditRiskAPP
pip install -r requirements.txt
```

将导出的 **`catboost_model.pkl`** 放到以下任一位置（应用启动时会自动查找）：

1. 与 `app.py` 同目录（推荐）
2. `models/catboost_model.pkl`
3. 或通过环境变量 **`MODEL_PATH`** 指定绝对路径

启动：

```bash
python app.py
```

浏览器访问：<http://127.0.0.1:5000>

## API 示例（`/predict`）

请求体为 JSON，需包含与训练一致的数值字段（字段名与顺序由后端 `FEATURE_COLS` 定义），可选 `owns_house`、`owns_car`（布尔）用于资产相关修正。

若模型未加载，接口返回 **503**。

## 项目结构（简要）

```
app.py              # Flask 应用与预测逻辑
templates/index.html   # 前端页面
requirements.txt    # Python 依赖
```

## 免责声明

*本系统仅用于学习或演示，不构成任何信贷或投资建议；实际业务需满足合规、审计与模型治理要求。
*此项目为本人本科毕业设计，版权归本人所有，仅供学习交流。
