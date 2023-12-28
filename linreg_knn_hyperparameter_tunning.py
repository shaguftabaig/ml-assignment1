import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV , cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

#Load the file
file_path = 'diabetes.csv'
data = pd.read_csv(file_path)

#Remove Outcome column
data = data.drop('Outcome' , axis= 1)

#Define features X and target variable y
X = data.drop('DiabetesPedigreeFunction', axis=1)
y = data['DiabetesPedigreeFunction']

# Split the data into training and testing sets for regression
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply Linear Regression
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
y_pred_lin_reg = lin_reg.predict(X_test)
mse_lin_reg = mean_squared_error(y_test, y_pred_lin_reg)

# Apply K-Nearest Neighbors Regression
knn_reg = KNeighborsRegressor()
knn_reg.fit(X_train, y_train)
y_pred_knn_reg = knn_reg.predict(X_test)
mse_knn_reg = mean_squared_error(y_test, y_pred_knn_reg)


print(f"Linear Regression: {mse_lin_reg * 100:.2f}%")
print(f"KNN : {mse_knn_reg * 100:.2f}%")

# Define hyperparameter grid for KNN Regression
param_grid_knn_reg = {
    'n_neighbors': [3, 5, 7, 9, 11],
    'weights': ['uniform', 'distance'],
    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
}

# Hyperparameter tuning for KNN Regression using GridSearchCV
grid_search_knn_reg = GridSearchCV(KNeighborsRegressor(), param_grid_knn_reg, cv=5, scoring='neg_mean_squared_error')
grid_search_knn_reg.fit(X_train, y_train)

# Best parameters and best score (MSE) for KNN Regression
best_params_knn_reg = grid_search_knn_reg.best_params_
best_score_knn_reg = -grid_search_knn_reg.best_score_  # Negating because GridSearchCV returns negative MSE

print("Best score for knn", best_score_knn_reg)
print("Best Params for KNN:", best_params_knn_reg)
