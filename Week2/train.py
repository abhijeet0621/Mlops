# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import os
import joblib

# Load the data
print("Loading data...")
data = pd.read_csv('data/iris.csv')
print("Data loaded successfully!")
# Print the first 5 rows of the data
print(data.head())

# Split the data into training and testing sets
print("Splitting data into training and testing sets...")
train, test = train_test_split(data, test_size = 0.4, stratify = data['species'], random_state = 42)
X_train = train[['sepal_length','sepal_width','petal_length','petal_width']]
y_train = train.species
X_test = test[['sepal_length','sepal_width','petal_length','petal_width']]
y_test = test.species

# Train the model
print("Training the model...")
mod_dt = DecisionTreeClassifier(max_depth = 3, random_state = 1)
mod_dt.fit(X_train,y_train)
print("Model trained successfully!")

# Evaluate the model on test set
print("Evaluating the model on test set...")
prediction = mod_dt.predict(X_test)
print('The accuracy of the Decision Tree is',"{:.3f}".format(metrics.accuracy_score(prediction,y_test)))

# Save the model
print("Saving the model...")
os.makedirs("model", exist_ok=True)
joblib.dump(mod_dt, "model/model.joblib")
print("Model saved to model/model.joblib successfully!")