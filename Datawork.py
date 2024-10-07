import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV

# Load the Titanic dataset
df = pd.read_csv("Titanic-Dataset.csv")
print(df.head())

# Drop Cabin column which has many missing values
df = df.drop("Cabin", axis=1)

# Fill missing values in Age and Embarked columns
df["Age"].fillna(df["Age"].median(), inplace=True)
df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)

print(df.isnull().sum())

# Define numeric columns for scaling
num_vals = ['Age', 'SibSp(Number of Siblings or Spouse Abroad)', 
            'Parch(Number of Parents or Children abroad)', 
            'Fare(Money a passenger paid for this trip)']

# Normalize the numeric features using StandardScaler
scalar = StandardScaler()
df[num_vals] = scalar.fit_transform(df[num_vals])

# Display data types and correlation
print(df.dtypes)
corr_matrix = df.select_dtypes(include=['int64', 'float64']).corr()
print(corr_matrix)
print(corr_matrix['Survived'])

# Define features and target
X = df.select_dtypes(include=['int64', 'float64'])
y = df['Survived']

# Recursive Feature Elimination
estimator = RandomForestClassifier()
selector = RFE(estimator, n_features_to_select=5, step=1)
selector = selector.fit(X, y)

# Print selected features
print("Support: ", selector.support_)
print("Ranking: ", selector.ranking_)
print("Selected features: ", X.columns[selector.support_])

# Checking the skewness of the data
print(df.select_dtypes(include=['int64', 'float64']).skew())

# Log transformation of certain features
df['SibSp(Number of Siblings or Spouse Abroad)'] = np.log1p(df['SibSp(Number of Siblings or Spouse Abroad)'])
df['Parch(Number of Parents or Children abroad)'] = np.log1p(df['Parch(Number of Parents or Children abroad)'])
df['Fare(Money a passenger paid for this trip)'] = np.log1p(df['Fare(Money a passenger paid for this trip)'])

# Square root transformation
df['SibSp(Number of Siblings or Spouse Abroad)'] = np.sqrt(df['SibSp(Number of Siblings or Spouse Abroad)'])
df['Parch(Number of Parents or Children abroad)'] = np.sqrt(df['Parch(Number of Parents or Children abroad)'])
df['Fare(Money a passenger paid for this trip)'] = np.sqrt(df['Fare(Money a passenger paid for this trip)'])

# Re-fit StandardScaler
df[num_vals] = scalar.fit_transform(df[num_vals])

print(df.select_dtypes(include=['int64', 'float64']).skew())

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training data shape:", X_train.shape, y_train.shape)
print("Testing data shape:", X_test.shape, y_test.shape)

# Train a Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = model.predict(X_test)

# Evaluate the model's performance
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Hyperparameter tuning using GridSearchCV
param_grid = {
    'max_iter': [1000, 2000, 5000],
    'C': [0.1, 1, 10],
    'penalty': ['l2'],
    'solver': ['liblinear', 'lbfgs']
}

grid_search = GridSearchCV(LogisticRegression(), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

print("Best Parameters:", grid_search.best_params_)
print("Best Score:", grid_search.best_score_)

best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Make predictions on the testing data
y_pred = best_model.predict(X_test)

# Print the predictions
print(y_pred)

# Evaluate the model's performance on the testing data
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Create a new data point (e.g., a new passenger)
new_passenger = pd.DataFrame({'PassengerId': [0],'Survived':[0],'Passenger class (Pclass)': [0], 'Age': [10], 'SibSp(Number of Siblings or Spouse Abroad)': [2], 'Parch(Number of Parents or Children abroad)': [3], 'Fare(Money a passenger paid for this trip)': [1000]})

new_passenger[num_vals] = scalar.transform(new_passenger[num_vals])
# Make a prediction on the new data point
prediction = best_model.predict(new_passenger)
print("Prediction:", prediction)