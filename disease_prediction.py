import os
import warnings
import pandas as pd

warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# =====================================
# LOAD DATASET
# =====================================

print("=== Disease Prediction System ===\n")

current_dir = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(current_dir, "diabetes.csv")

df = pd.read_csv(dataset_path)

print("Dataset Loaded Successfully")
print("Shape:", df.shape)

# =====================================
# FEATURES & TARGET
# =====================================

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# =====================================
# TRAIN TEST SPLIT
# =====================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =====================================
# FEATURE SCALING
# =====================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =====================================
# MODELS
# =====================================

models = {

    "Logistic Regression":
        LogisticRegression(max_iter=1000),

    "Support Vector Machine":
        SVC(probability=True),

    "Random Forest":
        RandomForestClassifier(
            n_estimators=200,
            random_state=42
        )
}

# =====================================
# TRAIN & EVALUATE
# =====================================

for name, model in models.items():

    print("\n" + "=" * 50)
    print("Model:", name)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("Accuracy  :", round(accuracy, 4))
    print("Precision :", round(precision, 4))
    print("Recall    :", round(recall, 4))
    print("F1-Score  :", round(f1, 4))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

print("\nDisease Prediction Completed.")