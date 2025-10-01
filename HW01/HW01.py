import pandas as pd
from sklearn import model_selection, tree, metrics
import matplotlib.pyplot as plt

# Importar dados
data = pd.read_csv("../data/hungarian_heart_diseases.csv")

X = data.drop(columns=["outcome"])
y = data["outcome"]

X_train, X_test, y_train, y_test = model_selection.train_test_split(
    X, y, train_size=0.8, random_state=1, stratify=y
)

clf = tree.DecisionTreeClassifier(random_state=1)

min_samples_leaf_values = [1, 3, 5, 10, 25, 50, 100]

train_accuracies, test_accuracies = [], []

for m in min_samples_leaf_values:
    clf.set_params(min_samples_leaf=m)
    clf.fit(X_train, y_train)

    train_acc = metrics.accuracy_score(y_train, clf.predict(X_train))
    test_acc = metrics.accuracy_score(y_test, clf.predict(X_test))

    train_accuracies.append(train_acc)
    test_accuracies.append(test_acc)

# Plot dos resultados
# plt.figure(figsize=(8, 5))
plt.plot(
    min_samples_leaf_values, train_accuracies, marker="o", label="Training Accuracy"
)
plt.plot(min_samples_leaf_values, test_accuracies, marker="o", label="Testing Accuracy")
plt.xlabel("min_samples_leaf")
plt.ylabel("Accuracy")
plt.title("Decision Tree Accuracy vs. min_samples_leaf")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.show()

# The plot shows that when min_samples_leaf = 1, the decision tree achieves almost perfect accuracy on the training set,
# but its testing accuracy is noticeably lower. This indicates overfitting, as the model memorizes the training data but
# fails to generalize well to unseen samples.

# As the value of min_samples_leaf increases, the training accuracy decreases gradually, since the tree becomes more
# constrained and less capable of fitting every detail in the data. However, the testing accuracy initially increases or
# stabilizes, showing an improvement in generalization capacity, as the tree avoids overfitting and captures only the most
# relevant patterns.

# For very large values such as min_samples_leaf = 50 or 100, both training and testing accuracies drop significantly.
# In this case, the model becomes too simple, failing to capture important relationships in the data, which corresponds to
# underfitting.

# Overall, the best generalization capacity is achieved at the value of min_samples_leaf that maximizes the testing
# accuracy, as this reflects the model’s ability to perform well on unseen data. Nonetheless, the comparison between
# training and testing accuracies across settings is crucial to understand the trade-off between overfitting (too complex)
# and underfitting (too simple).


# From the plot we can see that when min_samples_leaf = 1 the tree gets almost perfect accuracy on the training set,
# but the testing accuracy is lower. This means the model is overfitting, because it memorizes the training data and does
# not generalize so well.

# When we increase min_samples_leaf, the training accuracy goes down since the tree is more limited, but the testing
# accuracy becomes a bit better and more stable. This shows that the model is generalizing better, because it is not just
# fitting the noise in the training set.

# For very large values like 50 or 100, both training and testing accuracies drop. Here the tree is too simple and cannot
# capture enough information, which is underfitting.

# In general, the best value is the one with the highest testing accuracy (where min_samples_leaf = 10, 25), because this
# shows the model can generalize.
# Still, looking at the balance between training and testing also helps to see if the model is overfitting or underfitting.
