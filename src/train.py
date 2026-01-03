import joblib
from preprocessing import load_and_clean_data
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier

# Load data
df = load_and_clean_data("data/train.csv")

X = df.drop("Survived", axis=1)
y = df["Survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
rf = RandomForestClassifier(random_state=42)

params = {
    'n_estimators': [100,200],
    'max_depth': [5,10,None]
}

grid = GridSearchCV(rf, params, cv=5, scoring='accuracy')
grid.fit(X_train, y_train)

best_model = grid.best_estimator_

joblib.dump(best_model, "models/titanic_model.pkl")

print("Model trained and saved successfully")
