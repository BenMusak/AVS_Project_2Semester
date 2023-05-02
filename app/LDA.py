import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


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
    

def plot_data(train_, trn_targets, title, labels, new_data):
    
    fig, ax = plt.subplots()

    color = ["red", "green", "blue", "yellow", "black", "orange"]
    for i in range(len(train_)):
        if trn_targets[i] == 0:
            ax.scatter(train_[i][0], train_[i][1], train_[i][2], color=color[0], edgecolors="none", label=labels[0], alpha=0.8)
        elif trn_targets[i] == 1:
            ax.scatter(train_[i][0], train_[i][1], train_[i][2], color=color[1], edgecolors="none", label=labels[1], alpha=0.8)
        elif trn_targets[i] == 2:
            ax.scatter(train_[i][0], train_[i][1], train_[i][2], color=color[2], edgecolors="none", label=labels[2], alpha=0.8)
        elif trn_targets[i] == 3:
            ax.scatter(train_[i][0], train_[i][1], train_[i][2], color=color[3], edgecolors="none", label=labels[3], alpha=0.8)
        elif trn_targets[i] == 4:
            ax.scatter(train_[i][0], train_[i][1], train_[i][2], color=color[4], edgecolors="none", label=labels[4], alpha=0.8)
        elif trn_targets[i] == 5:
            ax.scatter(train_[i][0], train_[i][1], train_[i][2], color=color[5], edgecolors="none", label=labels[5], alpha=0.8)

    # Plot the new data
    ax.scatter(new_data[0][0], new_data[0][1], color="purple", edgecolors="none", label="New Data", alpha=0.8, marker="x", s=100)

    print(new_data[0][0], new_data[0][1])

    fig.suptitle(title)
    plt.xlabel("Linear Discriminant 1")
    plt.ylabel("Linear Discriminant 2")

    # Create a custom legend with the same colors as the corresponding class
    handles = [plt.Rectangle((0,0), 1, 1, color=color[i], alpha=0.8) for i in range(len(labels))]
    fig.legend(handles, labels.values())

    return fig


def load_dataset(dataset_path):
    # Load the numpy dataset
    dataset = np.load(dataset_path)
    # Split the dataset into features and targets
    print(dataset.files)
    features = dataset['train_']
    targets = dataset['train_tgt']
    return features, targets