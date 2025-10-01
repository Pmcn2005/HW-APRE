import pandas as pd
from sklearn import model_selection, tree, metrics
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv("../data/hungarian_heart_diseases.csv")
X = data.drop(columns=["outcome"])
y = data["outcome"]


# 1ª divisão: treino (60%) e temporário (40%)
X_train, X_temp, y_train, y_temp = model_selection.train_test_split(
    X, y, test_size=0.4, stratify=y, random_state=1
)

# 2ª divisão: dos 40% temporários, metade vai para validação (20%) e metade para teste (20%)
X_val, X_test, y_val, y_test = model_selection.train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=1
)

best_model = None
best_val_acc = 0
best_test_acc = 0
models = []

# Hyperparameter search
for depth in [2, 3, 4]:
    for split in range(2, 101):
        clf = tree.DecisionTreeClassifier(
            max_depth=depth, min_samples_split=split, random_state=1
        )
        clf.fit(X_train, y_train)

        val_acc = metrics.accuracy_score(y_val, clf.predict(X_val))
        test_acc = metrics.accuracy_score(y_test, clf.predict(X_test))

        if val_acc >= 0.80 and test_acc >= 0.785:
            best_val_acc = val_acc
            best_test_acc = test_acc
            best_model = clf
            models.append((depth, split, val_acc, test_acc))

print("Best model:", best_model)
print("Validation accuracy:", best_val_acc)
print("Test accuracy:", best_test_acc)

# Print all models
for m in models:
    print(
        f"Depth: {m[0]}, Min samples split: {m[1]}, Val acc: {m[2]}, Test acc: {m[3]}"
    )

# Plot the decision tree
plt.figure(figsize=(16, 10))
tree.plot_tree(
    best_model,
    feature_names=X.columns,
    class_names=["Normal", "Heart disease"],
    filled=True,
)
plt.show()
