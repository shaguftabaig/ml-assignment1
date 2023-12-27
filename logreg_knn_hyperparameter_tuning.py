import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

#Loading the data set
file_path = '/Users/shaguftaafreenbaig/Documents/ML/assignment1/diabetes.csv'
data = pd.read_csv(file_path)

#Remove DiabetesPedigreeFunnction column
data = data.drop('DiabetesPedigreeFunction' , axis= 1)

#Define features (X) annd target variables (y)
X = data.drop('Outcome', axis=1)
y = data['Outcome']

#Splittinng dataset innto traininng and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state= 42)

# Scaling the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#Logistic Regression
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train_scaled, y_train)
y_pred_log_reg = log_reg.predict(X_test_scaled)
acc_log_reg = accuracy_score(y_test, y_pred_log_reg)
cv_scores_log_reg = cross_val_score(log_reg, X_train_scaled, y_train, cv=5)

#K-NN
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
acc_knn = accuracy_score(y_test , y_pred_knn)
cv_scores_knn = cross_val_score(knn, X_train, y_train, cv=5)

print(f"Logistic Regression Accuracy: {acc_log_reg * 100:.2f}%")
print("CV Average Score for Logistic Regression:", cv_scores_log_reg.mean())
print(f"KNN Accuracy: {acc_knn * 100:.2f}%")
print("CV Average Score for KNN:", cv_scores_knn.mean())

#Hyperparameter grid for Logistic Regression
param_grid_log_reg = {'C': [0.01, 0.1, 1, 10, 100], 'max_iter': [100, 200, 500, 1000]}

#Hyperparameter grid for k-NN
param_grid_knn = {'n_neighbors': [1, 3, 5, 7, 10, 15, 20]}

# Logistic Regression Grid Search
grid_search_log_reg = GridSearchCV(LogisticRegression(), param_grid_log_reg , cv=5)
grid_search_log_reg.fit(X_train_scaled, y_train)

#KNN Gris Search 
grid_search_knn = GridSearchCV(KNeighborsClassifier(), param_grid_knn, cv=5)
grid_search_knn.fit(X_train, y_train)

best_params_log_reg = grid_search_log_reg.best_params_
best_score_log_reg = grid_search_log_reg.best_score_

best_params_knn = grid_search_knn.best_params_
best_score_knn = grid_search_knn.best_score_

print("Best Params for Logistic Regression:", best_params_log_reg)
print("Best Cross-Validation Score for Logistic Regression:", best_score_log_reg)
print("Best Params for KNN:", best_params_knn)
print("Best Cross-Validation Score for KNN:", best_score_knn)