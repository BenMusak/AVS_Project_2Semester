from sklearn.neighbors import KNeighborsClassifier

class KNN:
    def __init__(self, k_num=3, weight='distance', metric='euclidean'):
        self.k_num = k_num #
        self.weight = weight 
        self.metric = metric
        self.model = KNeighborsClassifier(n_neighbors=self.k_num, weights=self.weight, metric=self.metric)
    
    # training on whole dataset to get model
    def train(self, X_train, y_train):
        print('Training data ...')
        self.model.fit(X_train, y_train)

    #def test(self):
    
    # make a prediction on data
    def predict(self, x_test): 
        print('Test data ...')
        y_pred = self.model.predict(x_test)
        return y_pred