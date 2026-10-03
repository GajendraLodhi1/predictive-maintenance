import nbformat as nbf
import json

def make_nb(cells_list, out_name):
    nb = nbf.v4.new_notebook()
    for ctype, ctext in cells_list:
        if ctype == 'md':
            nb.cells.append(nbf.v4.new_markdown_cell(ctext))
        else:
            nb.cells.append(nbf.v4.new_code_cell(ctext))
    with open(out_name, 'w') as f:
        nbf.write(nb, f)

nb1_cells = [
    ('md', '# 01. Data Understanding\n\nThis notebook covers Step 1 to Step 4.'),
    ('code', 'import pandas as pd\nimport numpy as np\nimport matplotlib.pyplot as plt\nimport seaborn as sns\nplt.style.use("seaborn-v0_8-whitegrid")'),
    ('code', 'columns = ["engine_id", "cycle", "op_setting_1", "op_setting_2", "op_setting_3"] + [f"sensor_{i}" for i in range(1, 22)]\ndf_train = pd.read_csv("../data/train_FD001.txt", sep="\\s+", header=None, names=columns)\nprint(df_train.shape)\ndf_train.head()'),
    ('code', 'df_train.info()'),
    ('code', 'df_train.describe().T'),
    ('code', 'print(df_train.isnull().sum())\nprint(df_train.nunique())'),
    ('code', 'print(f"Number of engines: {df_train[\'engine_id\'].nunique()}")\nconstant_cols = [c for c in df_train.columns if df_train[c].nunique() == 1]\nprint(f"Constant cols: {constant_cols}")')
]

nb2_cells = [
    ('md', '# 02. EDA and Degradation Analysis'),
    ('code', 'import pandas as pd\nimport matplotlib.pyplot as plt\nplt.style.use("seaborn-v0_8-whitegrid")\n\ncolumns = ["engine_id", "cycle", "op_setting_1", "op_setting_2", "op_setting_3"] + [f"sensor_{i}" for i in range(1, 22)]\ndf_train = pd.read_csv("../data/train_FD001.txt", sep="\\s+", header=None, names=columns)'),
    ('code', 'engine_1 = df_train[df_train["engine_id"] == 1]\nplt.figure(figsize=(10, 4))\nplt.plot(engine_1["cycle"], engine_1["sensor_2"])\nplt.xlabel("Operating Cycle")\nplt.ylabel("Sensor 2")\nplt.title("Sensor 2 vs Operating Cycle (Engine 1)")\nplt.savefig("../figures/sensor_degradation.png", bbox_inches="tight")\nplt.show()'),
    ('code', 'fig, axes = plt.subplots(7, 3, figsize=(15, 20))\naxes = axes.flatten()\nsensors = [f"sensor_{i}" for i in range(1, 22)]\nfor i, sensor in enumerate(sensors):\n    axes[i].plot(engine_1["cycle"], engine_1[sensor])\n    axes[i].set_title(sensor)\nplt.tight_layout()\nplt.show()'),
    ('code', 'variances = df_train.var(numeric_only=True)\nuseless_sensors = [col for col in df_train.columns if df_train[col].var() < 1e-10]\nprint(f"Sensors to remove: {useless_sensors}")')
]

nb3_cells = [
    ('md', '# 03. RUL Feature Engineering'),
    ('code', 'import pandas as pd\nimport matplotlib.pyplot as plt\nimport seaborn as sns\ncolumns = ["engine_id", "cycle", "op_setting_1", "op_setting_2", "op_setting_3"] + [f"sensor_{i}" for i in range(1, 22)]\ndf_train = pd.read_csv("../data/train_FD001.txt", sep="\\s+", header=None, names=columns)\nuseless = ["sensor_1", "sensor_10", "sensor_18", "sensor_19", "op_setting_3"]\ndf_train = df_train.drop(columns=useless)'),
    ('code', 'df_train["RUL"] = df_train.groupby("engine_id")["cycle"].transform("max") - df_train["cycle"]\ndf_train[["engine_id", "cycle", "RUL"]].head()'),
    ('code', 'plt.figure(figsize=(8, 4))\nplt.hist(df_train["RUL"], bins=50, color="skyblue", edgecolor="black")\nplt.xlabel("Remaining Useful Life")\nplt.ylabel("Frequency")\nplt.title("RUL Distribution")\nplt.savefig("../figures/rul_distribution.png", bbox_inches="tight")\nplt.show()'),
    ('code', 'plt.figure(figsize=(12, 10))\nsns.heatmap(df_train.corr(), annot=False, cmap="coolwarm")\nplt.title("Correlation Matrix")\nplt.savefig("../figures/correlation_matrix.png", bbox_inches="tight")\nplt.show()'),
    ('code', 'def add_features(df, window=5):\n    df_feat = df.copy()\n    sensor_cols = [c for c in df.columns if "sensor" in c]\n    for col in sensor_cols:\n        df_feat[f"{col}_roll_mean"] = df_feat.groupby("engine_id")[col].transform(lambda x: x.rolling(window, min_periods=1).mean())\n        df_feat[f"{col}_roll_std"] = df_feat.groupby("engine_id")[col].transform(lambda x: x.rolling(window, min_periods=1).std().fillna(0))\n    return df_feat\ndf_train_feat = add_features(df_train)\ndf_train_feat.to_csv("../data/train_engineered.csv", index=False)')
]

nb4_cells = [
    ('md', '# 04. Modeling and Evaluation'),
    ('code', 'import pandas as pd\nimport numpy as np\nimport matplotlib.pyplot as plt\nimport seaborn as sns\nfrom sklearn.linear_model import LinearRegression\nfrom sklearn.ensemble import RandomForestRegressor\nfrom xgboost import XGBRegressor\nfrom sklearn.metrics import mean_squared_error\ndf_train = pd.read_csv("../data/train_engineered.csv")'),
    ('code', 'engines = df_train["engine_id"].unique()\nnp.random.seed(42)\ntrain_engines = np.random.choice(engines, size=int(0.8 * len(engines)), replace=False)\ntrain_set = df_train[df_train["engine_id"].isin(train_engines)]\nval_set = df_train[~df_train["engine_id"].isin(train_engines)]\nfeatures = [c for c in train_set.columns if c not in ["engine_id", "cycle", "RUL"]]\ntarget = "RUL"\nX_train, y_train = train_set[features], train_set[target]\nX_val, y_val = val_set[features], val_set[target]'),
    ('code', 'lr = LinearRegression()\nlr.fit(X_train, y_train)\nrmse_lr = np.sqrt(mean_squared_error(y_val, lr.predict(X_val)))\nprint(f"Linear Regression RMSE: {rmse_lr:.2f}")'),
    ('code', 'rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)\nrf.fit(X_train, y_train)\nrmse_rf = np.sqrt(mean_squared_error(y_val, rf.predict(X_val)))\nprint(f"Random Forest RMSE: {rmse_rf:.2f}")'),
    ('code', 'xgb = XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42, n_jobs=-1)\nxgb.fit(X_train, y_train)\npreds_xgb = xgb.predict(X_val)\nrmse_xgb = np.sqrt(mean_squared_error(y_val, preds_xgb))\nprint(f"XGBoost RMSE: {rmse_xgb:.2f}")'),
    ('code', 'plt.figure(figsize=(10, 5))\nplt.scatter(y_val, preds_xgb, alpha=0.3, color="blue")\nplt.plot([0, y_val.max()], [0, y_val.max()], "r--")\nplt.xlabel("Actual RUL")\nplt.ylabel("Predicted RUL")\nplt.title("Actual vs Predicted RUL (XGBoost)")\nplt.savefig("../figures/actual_vs_predicted.png", bbox_inches="tight")\nplt.show()'),
    ('code', 'importance = pd.DataFrame({"Feature": features, "Importance": rf.feature_importances_}).sort_values("Importance", ascending=False)\nplt.figure(figsize=(10, 8))\nsns.barplot(x="Importance", y="Feature", data=importance.head(20), palette="viridis")\nplt.title("Top 20 Features Importance (Random Forest)")\nplt.savefig("../figures/feature_importance.png", bbox_inches="tight")\nplt.show()')
]

if __name__ == "__main__":
    make_nb(nb1_cells, "../notebooks/01_data_understanding.ipynb")
    make_nb(nb2_cells, "../notebooks/02_eda_and_degradation_analysis.ipynb")
    make_nb(nb3_cells, "../notebooks/03_rul_feature_engineering.ipynb")
    make_nb(nb4_cells, "../notebooks/04_modeling_and_evaluation.ipynb")
    print("Success")
