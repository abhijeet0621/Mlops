# Import required packages
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import os
import joblib

# Step 1: Load dataset
print("Reading dataset...")
iris_df = pd.read_csv('data/iris.csv')
print("Dataset loaded successfully!")
print(iris_df.head())

# Step 2: Prepare training and testing data
print("Preparing train and test splits...")
train_data, test_data = train_test_split(
    iris_df,
    test_size=0.4,
    stratify=iris_df['species'],
    random_state=42
)

X_train = train_data[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y_train = train_data['species']
X_test = test_data[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y_test = test_data['species']

# Step 3: Build and train decision tree classifier
print("Fitting Decision Tree model...")
tree_clf = DecisionTreeClassifier(max_depth=3, random_state=1)
tree_clf.fit(X_train, y_train)
print("Model training completed!")

# Step 4: Test model performance
print("Testing model performance...")
y_pred = tree_clf.predict(X_test)
accuracy = metrics.accuracy_score(y_test, y_pred)
print(f"Decision Tree Model Accuracy: {accuracy:.3f}")

# Step 5: Save trained model
print("Exporting model...")
os.makedirs("model", exist_ok=True)
joblib.dump(tree_clf, "model/iris_decision_tree.joblib")
print("Model saved as model/iris_decision_tree.joblib successfully!")
