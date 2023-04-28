from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

def knn_model(x_train, x_test, y_train, y_test):

    print(f'Shape: {x_train.shape}')
    print(f'Observation: \n{x_train[0]}')
    print(f'Labels: {y_train}')

    # Scale the data to be between -1 and 1
    scaler = StandardScaler()
    scaler.fit(x_train)

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

    print(f'Model Score: {model.score(x_test_scaled, y_test)}')

    y_predict = model.predict(x_test_scaled)
    print(f'Confusion Matrix For Test Data: \n{confusion_matrix(y_predict, y_test)}')
    print(f'Accuracy Score for Test Data: {accuracy_score(y_predict, y_test)}')

    print("Accuracy: ", model.best_score_)

    return model