import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score

# 1. Prepare Data
data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

# 2. Build and Train Model
# We limit max_depth to 3 to keep the tree simple and clear
model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X_train, y_train)

# 3. Test the Model
predictions = model.predict(X_test)
print(f"Model Accuracy: {accuracy_score(y_test, predictions) * 100:.2f}%")

# 4. Show the Logic (Visual Tree)
plt.figure(figsize=(12, 8))
plot_tree(model, 
          feature_names=data.feature_names, 
          class_names=list(data.target_names), 
          filled=True)

plt.title("Decision Tree Flowchart")
plt.show()
