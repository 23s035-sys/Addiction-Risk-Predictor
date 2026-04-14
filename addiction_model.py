import pandas as pd
import joblib

# ML tools
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.utils import resample

# Algorithms
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier


# -----------------------------
# 1 LOAD DATASET
# -----------------------------
data = pd.read_excel("addiction_dataset.xlsx")

print("\nDataset Preview")
print(data.head())


# -----------------------------
# 2 FEATURES AND TARGET
# -----------------------------
X = data.drop("Addiction Risk Level", axis=1)
y = data["Addiction Risk Level"]


# -----------------------------
# 3 ONE HOT ENCODING
# -----------------------------
X = pd.get_dummies(X)


# -----------------------------
# 4 BALANCE DATASET
# -----------------------------
data_bal = pd.concat([X, y], axis=1)

low = data_bal[data_bal["Addiction Risk Level"] == "Low Risk"]
mod = data_bal[data_bal["Addiction Risk Level"] == "Moderate Risk"]
high = data_bal[data_bal["Addiction Risk Level"] == "High Risk"]

max_size = max(len(low), len(mod), len(high))

low_up = resample(low, replace=True, n_samples=max_size, random_state=42)
mod_up = resample(mod, replace=True, n_samples=max_size, random_state=42)
high_up = resample(high, replace=True, n_samples=max_size, random_state=42)

data_balanced = pd.concat([low_up, mod_up, high_up])

# shuffle dataset (important but does not change output)
data_balanced = data_balanced.sample(frac=1, random_state=42)

X = data_balanced.drop("Addiction Risk Level", axis=1)
y = data_balanced["Addiction Risk Level"]


# -----------------------------
# 5 TRAIN TEST SPLIT
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining size:", X_train.shape)
print("Testing size:", X_test.shape)


# -----------------------------
# 6 FEATURE SCALING
# -----------------------------
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# -------------------------------------------------
# FUNCTION TO PRINT ALL METRICS
# -------------------------------------------------
def evaluate_model(name, y_test, predictions, model, X_train_data):

    print("\n==============================")
    print(name)
    print("==============================")

    print("Accuracy :", accuracy_score(y_test, predictions))
    print("Precision:", precision_score(y_test, predictions, average='weighted'))
    print("Recall   :", recall_score(y_test, predictions, average='weighted'))
    print("F1 Score :", f1_score(y_test, predictions, average='weighted'))

    print("\nConfusion Matrix")
    print(confusion_matrix(y_test, predictions))

    cv = cross_val_score(model, X_train_data, y_train, cv=5)
    print("\nCross Validation:", cv.mean())


# -----------------------------
# 7 LOGISTIC REGRESSION
# -----------------------------
log_model = LogisticRegression(max_iter=5000)

log_model.fit(X_train_scaled, y_train)
log_pred = log_model.predict(X_test_scaled)

evaluate_model("Logistic Regression", y_test, log_pred, log_model, X_train_scaled)


# -----------------------------
# 8 DECISION TREE
# -----------------------------
dt_model = DecisionTreeClassifier(max_depth=10, random_state=42)

dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)

evaluate_model("Decision Tree", y_test, dt_pred, dt_model, X_train)


# -----------------------------
# 9 RANDOM FOREST
# -----------------------------
rf_model = RandomForestClassifier(
    n_estimators=400,
    max_depth=12,
    random_state=42
)

rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

evaluate_model("Random Forest", y_test, rf_pred, rf_model, X_train)


# -----------------------------
# 10 ARTIFICIAL NEURAL NETWORK
# -----------------------------
ann_model = MLPClassifier(
    hidden_layer_sizes=(60, 60),
    max_iter=5000,
    random_state=42
)

ann_model.fit(X_train_scaled, y_train)
ann_pred = ann_model.predict(X_test_scaled)

evaluate_model("ANN", y_test, ann_pred, ann_model, X_train_scaled)


# -----------------------------
# SAVE BEST MODEL
# -----------------------------
joblib.dump(log_model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(X.columns, "columns.pkl")

print("Model saved successfully")