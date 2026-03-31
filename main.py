import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Load dataset
df = pd.read_csv("data/student-mat.csv")

# Features & target
X = df[['studytime', 'absences', 'G1', 'G2']]
y = df['G3']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation
score = r2_score(y_test, y_pred)
print(f"Model Accuracy (R² Score): {score:.2f}")
