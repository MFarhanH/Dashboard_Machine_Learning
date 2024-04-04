import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, classification_report, confusion_matrix, f1_score
from sklearn.neural_network import MLPClassifier
import pickle
# Load the Heart Disease dataset
dataset = 'https://storage.googleapis.com/dqlab-dataset/heart_disease.csv'
heart_data = pd.read_csv(
    'https://storage.googleapis.com/dqlab-dataset/heart_disease.csv')

# kategori
heart_data['cp'] = heart_data['cp'].replace(
    {'typical angina': 0, 'atypical angina': 1, 'non-anginal pain': 2, 'asymtomatic': 3})
heart_data['slope'] = heart_data['slope'].replace(
    {'downsloping': 0, 'flat': 1, 'upsloping': 2})
heart_data['exang'] = heart_data['exang'].replace({'No': 0, 'Yes': 1})
heart_data['ca'] = heart_data['ca'].replace(
    {'Number of major vessels: 0': 0, 'Number of major vessels: 1': 1, 'Number of major vessels: 2': 2, 'Number of major vessels: 3': 3})
heart_data['thal'] = heart_data['thal'].replace(
    {'normal': 1, 'fixed defect': 2, 'reversable defect': 3})
heart_data['sex'] = heart_data['sex'].replace({'Male': 1, 'Female': 0})
heart_data['target'] = heart_data['target'].replace(
    {'No disease': 0, 'Disease': 1})

heart_data = heart_data[['cp', 'thalach', 'slope', 'oldpeak',
                         'exang', 'ca', 'thal', 'sex', 'age', 'target']]

# Define x and y
X = heart_data.drop(['target'], axis=1)
y = heart_data['target']

# Standardize the features
scaler = StandardScaler()
heart_data_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=100)

# Build Random Forest Model
clf = RandomForestClassifier()

# train the classifier
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# printing the test accuracy
print("The test accuracy score of Random Forest Classifier is ",
      accuracy_score(y_test, y_pred))

print(classification_report(y_test, y_pred))

# Hyper parameter tunning Randforest
clf = RandomForestClassifier()

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'criterion': ['gini', 'entropy']
}

gs1 = GridSearchCV(
    estimator=clf,
    param_grid=param_grid,
    cv=5,
    n_jobs=-1,
    scoring='roc_auc'
)

fit_clf_rf = gs1.fit(X_train, y_train)

print("Best Hyperparameters: ", fit_clf_rf.best_params_)
print("Best Score: ", fit_clf_rf.best_score_)


# Saving Best Model with Pickle
pkl_name = "Best Model for heart disease.pkl"

with open(pkl_name, 'wb') as file:
    pickle.dump(fit_clf_rf, file)
