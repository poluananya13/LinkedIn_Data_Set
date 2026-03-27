import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)

df = pd.read_csv("data/final_data.csv")

print(df.head())
print(df.shape)
print(df.columns)
print(df.info())
print(df.isnull().sum())
print(df['Class'].value_counts())

plt.figure()
df['Class'].value_counts().plot(kind='bar')
plt.savefig(os.path.join(output_dir, "class_distribution.png"))
plt.close()

X = df.select_dtypes(include=['int64', 'float64'])
y = df['Class']

X = X.fillna(0)

print(X.columns)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

svm_linear = SVC(kernel='linear')
svm_poly = SVC(kernel='poly', degree=3)
svm_rbf = SVC(kernel='rbf')

svm_linear.fit(X_train, y_train)
svm_poly.fit(X_train, y_train)
svm_rbf.fit(X_train, y_train)

def evaluate_model(model, name):
    y_pred = model.predict(X_test)
    print(name)
    print(accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)
    plt.figure()
    sns.heatmap(cm, annot=True, fmt='d')
    plt.savefig(os.path.join(output_dir, f"confusion_matrix_{name}.png"))
    plt.close()

evaluate_model(svm_linear, "linear")
evaluate_model(svm_poly, "poly")
evaluate_model(svm_rbf, "rbf")

accuracies = []
models = {
    "Linear": svm_linear,
    "Polynomial": svm_poly,
    "RBF": svm_rbf
}

for name, model in models.items():
    y_pred = model.predict(X_test)
    accuracies.append(accuracy_score(y_test, y_pred))

plt.figure()
plt.bar(models.keys(), accuracies)
plt.savefig(os.path.join(output_dir, "accuracy_comparison.png"))
plt.close()

def cross_validate_model(model, name):
    scores = cross_val_score(model, X, y, cv=5)
    print(name)
    print(scores)
    print(scores.mean())

cross_validate_model(SVC(kernel='linear'), "Linear")
cross_validate_model(SVC(kernel='poly', degree=3), "Polynomial")
cross_validate_model(SVC(kernel='rbf'), "RBF")

feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Weight': svm_linear.coef_[0]
})

feature_importance = feature_importance.sort_values(by='Weight', ascending=False)

print(feature_importance.head(10))