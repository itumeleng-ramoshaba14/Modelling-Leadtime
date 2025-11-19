import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# ============ 1️⃣ LOAD DATA ====================
Orders = pd.read_excel("C:/Users/emihl/OneDrive/Documentos/327/05_Project_DT_C.xlsx", sheet_name="Orders")
DistanceMatrix = pd.read_excel("C:/Users/emihl/OneDrive/Documentos/327/05_Project_DT_C.xlsx", sheet_name="DistanceMatrix")
DeliveryLogs = pd.read_excel("C:/Users/emihl/OneDrive/Documentos/327/05_Project_DT_C.xlsx", sheet_name="DeliveryLogs")

# ============ 2️⃣ PREPARE DISTANCE MATRIX =======
distances = []
for i in range(len(DistanceMatrix)):
    from_city = DistanceMatrix.iloc[i, 0]
    for to_city in DistanceMatrix.columns[1:]:
        distance = DistanceMatrix.at[i, to_city]
        if pd.notna(distance):
            distances.append({'From': from_city, 'To': to_city, 'Distance': distance})
distances_df = pd.DataFrame(distances)

# ============ 3️⃣ PROCESS DELIVERY LOGS =========
DeliveryLogs['Time'] = pd.to_datetime(DeliveryLogs['Time'])
lead_time_data = DeliveryLogs.pivot_table(
    index='ID', columns='Status', values='Time', aggfunc='first'
).reset_index()
lead_time_data['LeadTime'] = (lead_time_data['Delivered'] - lead_time_data['Ordered']).dt.days

# ============ 4️⃣ MERGE DATA ====================
OrdersWithLeadTime = Orders.merge(lead_time_data[['ID','LeadTime']], on='ID', how='left')
merged_df = OrdersWithLeadTime.merge(distances_df, on=['From','To'], how='left')

# ============ 5️⃣ SPLIT DATA 70/30 ============
size70 = round(len(merged_df) * 0.7)
df70 = merged_df.iloc[:size70].copy()
df30 = merged_df[size70:].copy()

# ============ 6️⃣ CLEAN DATA ===================
df70['LeadTime'] = df70['LeadTime'].fillna(0)
df70['OrderVolume'] = df70['OrderVolume'].fillna(0)
df70['Distance'] = df70['Distance'].fillna(0)

# ================= 7️⃣ LINEAR REGRESSION ========
target = "LeadTime"
numeric_cols = df70.select_dtypes(include=['float64','int64']).columns.drop(target)
formula = f"{target} ~ {' + '.join(numeric_cols)} + C(DestinationType)"
Lm = smf.ols(formula=formula, data=df70).fit()
print(Lm.summary())

# Predict on 30% test data
Lm_Predict = Lm.predict(df30)

# ================= 8️⃣ RANDOM FOREST ===========
df_train = df70.copy()
df_test = df30.copy()

# Drop ID if present
for df in [df_train, df_test]:
    if 'ID' in df.columns:
        df.drop(columns=['ID'], inplace=True)

# Encode categorical variables consistently
label_encoders = {}
for col in ['From','To','DestinationType']:
    le = LabelEncoder()
    combined = pd.concat([df_train[col], df_test[col]], axis=0)
    le.fit(combined)
    df_train[col] = le.transform(df_train[col])
    df_test[col] = le.transform(df_test[col])
    label_encoders[col] = le

# Features and target
X_train = df_train.drop(columns=['LeadTime'])
y_train = df_train['LeadTime']
X_test = df_test.drop(columns=['LeadTime'])
y_test = df_test['LeadTime']

# Train Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Predict
y_pred_rf = rf.predict(X_test)

# Evaluate Random Forest
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)
AbsError_rf = abs(y_test - y_pred_rf) / y_test.replace(0, np.nan)
AverageError_rf = round(np.nanmean(AbsError_rf) * 100)

print("========== RANDOM FOREST RESULTS (70/30) ==========")
print(f"R² Score: {r2_rf:.4f}")
print(f"MSE: {mse_rf:.4f}")
print(f"Average % Error: {AverageError_rf}%")

# ============ 9️⃣ MODEL COMPARISON ============
try:
    mse_lr = mean_squared_error(y_test, Lm_Predict)
    r2_lr = r2_score(y_test, Lm_Predict)
    AbsError_lr = abs(y_test - Lm_Predict) / y_test.replace(0, np.nan)
    AverageError_lr = round(np.nanmean(AbsError_lr) * 100)

    print("\n========== MODEL COMPARISON ==========")
    print(f"Linear Regression - R²: {r2_lr:.4f}, MSE: {mse_lr:.2f}, Avg Error: {AverageError_lr}%")
    print(f"Random Forest     - R²: {r2_rf:.4f}, MSE: {mse_rf:.2f}, Avg Error: {AverageError_rf}%")
except:
    print("\nLinear Regression model (Lm) not available for comparison.")

# ============ 10️⃣ ACTUAL VS PREDICTED PLOTS =======
# Linear Regression
plt.figure(figsize=(8,6))
plt.scatter(y_test, Lm_Predict, color='orange', alpha=0.6, label='Linear Regression')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label='Perfect Prediction')
plt.xlabel("Actual LeadTime")
plt.ylabel("Predicted LeadTime")
plt.title("Actual vs Predicted LeadTime (Linear Regression)")
plt.legend()
plt.show()

# Random Forest
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred_rf, color='blue', alpha=0.6, label='Random Forest')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label='Perfect Prediction')
plt.xlabel("Actual LeadTime")
plt.ylabel("Predicted LeadTime")
plt.title("Actual vs Predicted LeadTime (Random Forest)")
plt.legend()
plt.show()

# Combined Plot
plt.figure(figsize=(8,6))
plt.scatter(y_test, Lm_Predict, alpha=0.6, label='Linear Regression', color='orange')
plt.scatter(y_test, y_pred_rf, alpha=0.6, label='Random Forest', color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label='Perfect Prediction')
plt.xlabel("Actual LeadTime")
plt.ylabel("Predicted LeadTime")
plt.title("Model Comparison: Actual vs Predicted LeadTime")
plt.legend()
plt.show()

# ============ 11️⃣ RESIDUALS (Random Forest) ==========
residuals = y_test - y_pred_rf
sns.histplot(residuals, bins=20, kde=True, color='green')
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.title("Error Distribution (Random Forest)")
plt.show()

# ============ 12️⃣ FEATURE IMPORTANCE (Random Forest) ==========
importances = rf.feature_importances_
features = X_train.columns
plt.barh(features, importances)
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Random Forest Feature Importance")
plt.show()
