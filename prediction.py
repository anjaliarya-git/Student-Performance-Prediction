import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Dataset
data = {
    'Hours': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Result': [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
}

# Create DataFrame
df = pd.DataFrame(data)

print("\nStudent Study Dataset:\n")
print(df)

# Graph
plt.scatter(df['Hours'], df['Result'])
plt.xlabel("Study Hours")
plt.ylabel("Result (0 = Low, 1 = High)")
plt.title("Student Performance Prediction")
plt.show()

# Input and Output
X = df[['Hours']]
y = df['Result']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Model
model = LogisticRegression()
model.fit(X_train, y_train)

# User Input
hours = float(input("\nEnter study hours: "))

# Prediction
prediction = model.predict([[hours]])

if prediction[0] == 1:
    print("Prediction: Student may get HIGH marks")
else:
    print("Prediction: Student may get LOW marks")

# Accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", round(accuracy * 100, 2), "%")