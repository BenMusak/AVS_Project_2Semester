from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

def svm_model(x_train, x_test, y_train, y_test):
    print(f'Shape: {x_train.shape}')
    print(f'Observation: \n{x_train[0]}')
    print(f'Labels: {y_train}')

    # Scale the data to be between -1 and 1
    scaler = StandardScaler()
    scaler.fit(x_train)

    # Transform the training and testing data
    x_train_scaled = scaler.transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    # Create a SVM model
    grid_params = [{'kernel': ['rbf'], 'gamma': [1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2],
                    'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

    print("Training and Testing Model...")
    model = GridSearchCV(SVC(), grid_params, cv=5, n_jobs=-1)
    model.fit(x_train_scaled, y_train)

    print(f'Model Score: {model.score(x_test_scaled, y_test)}')

    y_predict = model.predict(x_test_scaled)
    print(f'Confusion Matrix For Test Data: \n{confusion_matrix(y_predict, y_test)}')
    print(f'Accuracy Score SVM for Test Data: {accuracy_score(y_predict, y_test)}')

    print("Accuracy SVM: ", model.best_score_)

    return model