import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

def knn_model(x_train, x_test, y_train, y_test, label_names, model_name):

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
    joblib.dump(scaler, 'models\scalars\scaler_{}.pkl'.format(model_name))

    # Transform the training and testing data
    x_train_scaled = scaler.transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    # Create a KNN model
    grid_params = {
        'n_neighbors': [3, 5, 7, 9, 11, 15],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    }

    print("Training and Testing KNN Model...")
    model = GridSearchCV(KNeighborsClassifier(), grid_params, cv=5, n_jobs=-1)
    model.fit(x_train_scaled, y_train)

    # Show the grid of parameters tested and the accuracy for each one using pandas dataframe
    results = pd.DataFrame(model.cv_results_)
    print(results)

    # Save pandas dataframe as csv file
    results.to_csv(f'dataframes/csv_models/model_scores/{model_name}_results.csv')

    # Show the different parameters tested and the best one
    print(f'Best Parameters: {model.best_params_}')
    print(f'Model Score for {model_name}: {model.score(x_test_scaled, y_test)}')
    print("Best Model Score: ", model.best_score_)

    y_predict = model.predict(x_test_scaled)
    print(f'Confusion Matrix best scored model for {model_name} with Test Data: \n{confusion_matrix(y_predict, y_test)}')
    
    # plot the confusion matrix
    plot_confusion_matrix(model, x_test_scaled, y_test, cmap=plt.cm.Blues, display_labels=label_names)

    # Save the confusion matrix as an image
    plt.savefig(f'dataframes/plots/{model_name}_confusion_matrix.png')

    print(f'Accuracy Score best scored model for {model_name} with Test Data: {accuracy_score(y_predict, y_test)}')

    # For each class, print the precision and recall
    report = classification_report(y_predict, y_test, output_dict=True)
    print(f'Classification Report for {model_name} best model: \n{report}')

    # Save the classification_report to a CSV file
    df = pd.DataFrame(report).transpose()
    df.to_csv(f'dataframes/csv_models/classification_reports/{model_name}_report.csv')

    return model