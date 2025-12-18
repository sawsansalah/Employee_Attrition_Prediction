import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
import os

# Load data
df = pd.read_csv("data/employee_data.csv")

X = df.drop("left_company", axis=1)
y = df["left_company"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Employee attrition model accuracy: {accuracy}")

# Save model
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/attrition_model.pkl")
