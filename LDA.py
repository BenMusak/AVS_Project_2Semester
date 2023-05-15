import numpy as np
from scipy.stats import multivariate_normal as norm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDASK
from scipy.stats import multivariate_normal


class LDA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.linear_discriminants = None
        
    def fit(self, data, labels):
        # First we get the features and labels. The function unique returns the unique values in the array, so that we have the different labels without repetitions.
        n_features = data.shape[1]
        class_labels = np.unique(labels)
        
        # Get the mean of the dataset
        mean_overall = np.mean(data, axis=0)
        # Initialize S_W and S_B with the size of the number of features
        SW = np.zeros((n_features, n_features))
        SB = np.zeros((n_features, n_features))
        
        # Fishers LDA for multiple classes
        for c in class_labels:
            d = labels == c
            X_c = []
            for i in range(len(d)):
                if d[i] == True:
                    X_c.append(data[i])
                else:
                    pass
            
            # Turn it into an array from a list
            X_c = np.array(X_c)
            
            # Get the mean of the current class
            mean_c = np.mean(X_c, axis=0)
            
            # Within-class covariance matrix, (n, n_c) * (n_c, n) = (n, n)
            SW += (X_c - mean_c).T.dot((X_c - mean_c))
            
            # Between-class covariance matrix
            n_c = X_c.shape[0]
            mean_diff = (mean_c - mean_overall).reshape(n_features, 1)
            SB += n_c * (mean_diff).dot(mean_diff.T)

        # Finding SW^-1 * SB. Since the determinant of SW and SB is 0 we can't use inv, but we can use pinv.
        W = np.linalg.pinv(SW).dot(SB)

        # Get eigenvalues and eigenvectors of A
        eigenvalues, eigenvectors = np.linalg.eig(W)

        # sort eigenvalues high to low
        eigenvectors = eigenvectors.T
        idxs = np.argsort(abs(eigenvalues))[::-1]
        eigenvalues = eigenvalues[idxs]
        eigenvectors = eigenvectors[idxs]

        # store first n eigenvectors in linear_discriminants since it is the biggest
        self.linear_discriminants = eigenvectors[0 : self.n_components]
    
    def transform(self, X):
        # project data onto linear discriminants
        return np.dot(X, self.linear_discriminants.T)


def plot_data(train_, trn_targets, title, labels):
    
    color = ["red", "green", "blue", "yellow", "black", "orange", "purple", "pink", "brown", "gray"]
    for i in range(len(train_)):
        if trn_targets[i] == 0:
            plt.scatter(train_[i][0], train_[i][1], color=color[0], edgecolors="none", label=labels[0], alpha=0.8)
        elif trn_targets[i] == 1:
            plt.scatter(train_[i][0], train_[i][1], color=color[1], edgecolors="none", label=labels[1], alpha=0.8)
        elif trn_targets[i] == 2:
            plt.scatter(train_[i][0], train_[i][1], color=color[2], edgecolors="none", label=labels[2], alpha=0.8)
        elif trn_targets[i] == 3:
            plt.scatter(train_[i][0], train_[i][1], color=color[3], edgecolors="none", label=labels[3], alpha=0.8)
        elif trn_targets[i] == 4:
            plt.scatter(train_[i][0], train_[i][1], color=color[4], edgecolors="none", label=labels[4], alpha=0.8)
        elif trn_targets[i] == 5:
            plt.scatter(train_[i][0], train_[i][1], color=color[5], edgecolors="none", label=labels[5], alpha=0.8)
        elif trn_targets[i] == 6:
            plt.scatter(train_[i][0], train_[i][1], color=color[6], edgecolors="none", label=labels[6], alpha=0.8)
        elif trn_targets[i] == 7:
            plt.scatter(train_[i][0], train_[i][1], color=color[7], edgecolors="none", label=labels[7], alpha=0.8)
        elif trn_targets[i] == 8:
            plt.scatter(train_[i][0], train_[i][1], color=color[8], edgecolors="none", label=labels[8], alpha=0.8)
        elif trn_targets[i] == 9:
            plt.scatter(train_[i][0], train_[i][1], color=color[9], edgecolors="none", label=labels[9], alpha=0.8)

    plt.suptitle(title)
    plt.xlabel("Linear Discriminant 1")
    plt.ylabel("Linear Discriminant 2")

    # Create a custom legend with the same colors as the corresponding class
    handles = [plt.Rectangle((0,0), 1, 1, color=color[i], alpha=0.8) for i in range(len(labels))]
    plt.legend(handles, labels)
    #plt.show()

    # Save the plot
    plt.savefig("dataframes/plots/" + title + ".png")


def plot_data_3D(train_, trn_targets, title, labels):
    
    train_ = np.absolute(train_)

    # Change backend for matplotlib to plot in python
    #matplotlib.use('TkAgg')

    fig = plt.figure()
    ax = plt.axes(projection='3d')

    color = ["red", "green", "blue", "yellow", "black", "orange", "purple", "pink", "brown", "gray"]
    for i in range(len(train_)):
        if trn_targets[i] == 0:
            #ax.scatter(train_[i][0], train_[i][1], train_[i][2], color=color[0], edgecolors="none", label=labels[0], alpha=0.8)
            ax.scatter3D(train_[i][0], train_[i][1], train_[i][2], color=color[0], edgecolors="none", label=labels[0], alpha=0.5)
        elif trn_targets[i] == 1:
            ax.scatter3D(train_[i][0], train_[i][1], train_[i][2], color=color[1], edgecolors="none", label=labels[1], alpha=0.5)
        elif trn_targets[i] == 2:
            ax.scatter3D(train_[i][0], train_[i][1], train_[i][2], color=color[2], edgecolors="none", label=labels[2], alpha=0.5)
        elif trn_targets[i] == 3:
            ax.scatter3D(train_[i][0], train_[i][1], train_[i][2], color=color[3], edgecolors="none", label=labels[3], alpha=0.5)
        elif trn_targets[i] == 4:
            ax.scatter3D(train_[i][0], train_[i][1], train_[i][2], color=color[4], edgecolors="none", label=labels[4], alpha=0.5)
        elif trn_targets[i] == 5:
            ax.scatter3D(train_[i][0], train_[i][1], train_[i][2], color=color[5], edgecolors="none", label=labels[5], alpha=0.5)

    fig.suptitle(title)
    ax.set_xlabel("Linear Discriminant 1")
    ax.set_xlabel("Linear Discriminant 2")
    ax.set_xlabel("Linear Discriminant 3")

    # Create a custom legend with the same colors as the corresponding class
    handles = [plt.Rectangle((0,0), 1, 1, color=color[i], alpha=0.8) for i in range(len(labels))]
    fig.legend(handles, labels)
    #plt.show()

    # Save the plot
    plt.savefig("dataframes/plots/" + title + ".png")


def regularize_covariance(covariance_matrix, shrinkage_factor=0.1):
    p = covariance_matrix.shape[0]
    shrunk_covariance_matrix = ((1 - shrinkage_factor) * covariance_matrix 
                                + shrinkage_factor * np.trace(covariance_matrix) / p * np.eye(p))
    return shrunk_covariance_matrix


def LDA_Fishers(trainset, trn_targets, test_x, test_y, n_components, label, label_names=["default", "default", "default", "default", "default"]):

    #trainset, trn_targets, test_x, test_y, label_names = load_dataset("LDA_Fishers_train.npz")

    lda = LDASK(n_components=n_components, solver="eigen", shrinkage="auto")
    #lda.fit(np.absolute(trainset), trn_targets)
    print("Fitting LDA...")
    lda.fit(trainset, trn_targets)

    train_ = lda.transform(trainset)
    train_tgt = trn_targets
    test_ = lda.transform(test_x)
    test_tgt = test_y

    print(label_names)

    print("Plotting...")
    if n_components == 2:
        plot_data(train_, train_tgt, "LDA_Fisher's_{}".format(label), label_names)
    elif n_components == 3:
        plot_data_3D(train_, train_tgt, "LDA_Fisher's_{}".format(label), label_names)
        
    # Predict the labels of test data
    y_pred = lda.predict(test_x)

    # Calculate the accuracy of the model
    accuracy = np.sum(test_y == y_pred) / len(test_y) * 100
    print("Accuracy of LDA: {:.2f}%".format(accuracy))

    print("Saving dataset...")
    # save the data
    np.savez("models/fishers_dataset/{}_LDA_Fishers_train".format(label), train_=train_, train_tgt=train_tgt, test_=test_, test_tgt=test_y, label_names=label_names)
    
    return lda, train_, train_tgt, test_, test_tgt


def load_dataset(dataset_path):
    # Load the numpy dataset
    dataset = np.load(dataset_path)
    # Split the dataset into features and targets
    print(dataset.files)
    features = np.absolute(dataset['train_'])
    targets = np.absolute(dataset['train_tgt'])
    label_names = dataset['label_names']
    test_x = np.absolute(dataset['test_'])
    test_y = np.absolute(dataset['test_tgt'])

    return features, targets, test_x, test_y, label_names
