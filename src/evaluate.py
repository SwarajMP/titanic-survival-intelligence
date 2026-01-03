import joblib
from preprocessing import load_and_clean_data
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

model = joblib.load("models/titanic_model.pkl")
df = load_and_clean_data("data/train.csv")

X = df.drop("Survived", axis=1)
y = df["Survived"]

preds = model.predict(X)

print("Accuracy:", accuracy_score(y, preds))
print("Confusion Matrix:\n", confusion_matrix(y, preds))
print("Classification Report:\n", classification_report(y, preds))
