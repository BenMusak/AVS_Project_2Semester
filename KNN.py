import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

def knn_model(x_train, x_test, y_train, y_test, model_name):

    x_train = np.array(x_train)
    x_test = np.array(x_test)
    y_train = np.array(y_train).ravel()
    y_test = np.array(y_test).ravel()

    print(f'Shape: {x_train.shape}')
    print(f'Observation: \n{x_train[0]}')
    print(f'Labels: {y_train}')

    # Scale the data to be between -1 and 1
    scaler = StandardScaler()
    scaler.fit(x_train)

    # Save the scaler
    joblib.dump(scaler, 'scaler.pkl')

    # Transform the training and testing data
    x_train_scaled = scaler.transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    # Create a KNN model
    grid_params = {
        'n_neighbors': [3, 5, 7, 9, 11, 15],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    }

    print("Training and Testing Model...")
    model = GridSearchCV(KNeighborsClassifier(), grid_params, cv=5, n_jobs=-1)
    model.fit(x_train_scaled, y_train)

    # Show the different parameters tested and the best one
    print(f'Best Parameters: {model.best_params_}')
    

    print(f'Model Score for {model_name}: {model.score(x_test_scaled, y_test)}')

    y_predict = model.predict(x_test_scaled)
    print(f'Confusion Matrix For {model_name} with Test Data: \n{confusion_matrix(y_predict, y_test)}')
    print(f'Accuracy Score for {model_name} with Test Data: {accuracy_score(y_predict, y_test)}')

    print("Best Model Score: ", model.best_score_)

    return model