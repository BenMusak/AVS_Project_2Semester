import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

def svm_model(x_train, x_test, y_train, y_test, model_name):
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

    print("Training and Testing SVM Model...")
    model = GridSearchCV(SVC(), grid_params, cv=5, n_jobs=-1)
    model.fit(x_train_scaled, y_train)

    # Show the grid of parameters tested and the accuracy for each one using pandas dataframe
    results = pd.DataFrame(model.cv_results_)
    print(results)

    # Show the different parameters tested and the best one
    print(f'Best Parameters: {model.best_params_}')
    print(f'Model Score for {model_name}: {model.score(x_test_scaled, y_test)}')
    print("Best Model Score: ", model.best_score_)

    y_predict = model.predict(x_test_scaled)
    print(f'Confusion Matrix best scored model for {model_name} with Test Data: \n{confusion_matrix(y_predict, y_test)}')
    
    # plot the confusion matrix
    plot_confusion_matrix(model, x_test_scaled, y_test, cmap=plt.cm.Blues)

    # Save the confusion matrix as an image
    plt.savefig(f'{model_name}_confusion_matrix.png')

    print(f'Accuracy Score best scored model for {model_name} with Test Data: {accuracy_score(y_predict, y_test)}')

    # Plot the precision-recall curve
    #precision, recall, thresholds = precision_recall_curve(y_test, y_predict)
    #plt.plot(recall, precision, marker='.')
    #plt.xlabel('Recall')
    #plt.ylabel('Precision')
    #plt.show()

    return model