# -----------------------------
# Fraud Detection using Random Forest
# -----------------------------
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE  # For balancing dataset

# -----------------------------
# Step 1: Load datasets
# -----------------------------
train_file = "fraudTrain.csv"
test_file = "fraudTest.csv"

df_train = pd.read_csv(train_file)
df_test = pd.read_csv(test_file)

print("✅ Datasets loaded successfully!")
print("Training set shape:", df_train.shape)
print("Test set shape:", df_test.shape)

# -----------------------------
# Step 2: Use a smaller subset (optional for faster training)
# -----------------------------
df_train_small = df_train.sample(50000, random_state=42)
features = ['unix_time', 'amt']

X = df_train_small[features]
y = df_train_small['is_fraud']

# -----------------------------
# Step 3: Handle imbalanced dataset using SMOTE
# -----------------------------
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)

# -----------------------------
# Step 4: Scale features
# -----------------------------
scaler = StandardScaler()
X_res_scaled = scaler.fit_transform(X_res)
X_test_scaled = scaler.transform(df_test[features])
y_test = df_test['is_fraud']

# -----------------------------
# Step 5: Train Random Forest
# -----------------------------
model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')
model.fit(X_res_scaled, y_res)

print("\n🎯 Fraud Detection Model Ready!")

# -----------------------------
# Step 6: Evaluate on test set
# -----------------------------
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

print("\n📊 Model Evaluation on Test Set:")
print(classification_report(y_test, y_pred))
print("ROC-AUC Score:", roc_auc_score(y_test, y_prob))

# -----------------------------
# Step 7: Interactive prediction
# -----------------------------
print("\n💡 Enter transactions to predict (type 'exit' to quit):")
while True:
    user_input = input("\nEnter unix_time, amt (e.g., 1325376018, 123.45):\n")
    if user_input.lower() == "exit":
        break
    try:
        unix_time_val, amt_val = [float(x.strip()) for x in user_input.split(",")]
        input_df = pd.DataFrame([[unix_time_val, amt_val]], columns=features)
        input_scaled = scaler.transform(input_df)
        prob = model.predict_proba(input_scaled)[0, 1]
        label = "Fraudulent" if prob >= 0.5 else "Legitimate"
        print(f"✅ Predicted: {label} (Fraud Probability: {prob:.4f})")
    except Exception as e:
        print("❌ Error:", e)
        print("Format: unix_time, amt  e.g., 1325376018, 123.45")

# -----------------------------
# Step 8: Show top 20 suspicious transactions in test set
# -----------------------------
df_test_copy = df_test.copy()
df_test_copy['p_fraud'] = y_prob
print("\n🔝 Top 20 suspicious transactions in test set:")
print(df_test_copy.sort_values('p_fraud', ascending=False).head(20)[['unix_time', 'amt', 'p_fraud', 'is_fraud']])
