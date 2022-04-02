import numpy as np

class LDA:
    def __init__(self):
        """
        This method imports into the object: 
        0) labels: set containing the different class labels. 
        1) mu: list of arrays in which each element is the array containing the mean of each feature in each class. 
        2) sigma: estimation of the Variance-Covariance matrix for the given dataset.
        3) transform_matrix: Cholesky factorization of sigma matrix (the inverse of this matrix is to be used to sphere the data). 
        4) prior: list of elements in which each element is the prior knowledge related to the class distribution.
        5) mu_transform: list of arrays in which each array is the array corresponding to the spherification of mu
                         through the inverse of the transform matrix. 
        """
        self.labels = None
        self.mu = None
        self.sigma = None
        self.transform_matrix = None
        self.prior = None
        self.mu_transform = None
      
    def fit(self, X, y):
        """
        This method performs all the operations needed to obtain the attributes presented in __init__() method. 
        """
        self.labels= set(y)
        indexes = [np.where(y==i)[0] for i in self.labels]
        X_s = [X[i] for i in indexes]
        self.prior = [i.shape[0]/y.shape[0] for i in indexes]
        self.mu = [I.sum(axis=0)/I.shape[0] for I in X_s]
        varcov_s = []
        for i in range(len(self.labels)):
            varcov_matrix = np.zeros((X_s[i].shape[1],X_s[i].shape[1]))
            for x in X_s[i]:
                varcov_matrix = varcov_matrix+((x-self.mu[i]).reshape(-1,1))@((x-self.mu[i]).reshape(-1,1).T)
           
            varcov_s.append(varcov_matrix/X_s[i].shape[0])
        
        varcov_s= [varcov_s[i]*self.prior[i] for i in range(len(self.labels))]
        self.sigma= np.array(varcov_s).sum(axis=0)
        self.transform_matrix= np.linalg.cholesky(self.sigma)
        self.mu_transform=[np.array(np.linalg.inv(self.transform_matrix))@mu for mu in self.mu]
        
    def predict(self, X):
        """
        This method performs a: 
        1) transformation of the given X in the space the prediction for the given dataset X. 
        2) prediction for X (which means that to each point is associated one class label)
        """
        X_transform = (np.array(np.linalg.inv(self.transform_matrix))@(X.T)).T
        y_pred = []
        for x in X_transform:
            y_pred.append(np.argmin([np.linalg.norm(x-self.mu_transform[i])**2+
                                     2*np.log(self.prior[i]) for i in range(len(self.labels)) ]))
        
        return np.array(y_pred)
    
    def transform(self, X):
        """
        This method performs the transformation onto the affine subspace spanned by [mu'_i - mu'_0], i = 1,...,c-1. 
        """
        W = np.zeros((self.mu_transform[0].shape[0],len(self.mu)-1))
        mu1 = self.mu_transform[0]
        for mu,i in zip(self.mu_transform[1:],range(1,len(self.mu_transform))):
            W[:,i-1] = mu-mu1
        (U,Stmp,V) = np.linalg.svd(W,full_matrices = True)
        S = np.zeros((self.mu_transform[0].shape[0],len(self.labels)-1))

        for i in range(len(self.labels)-1):
            S[i,i] = Stmp[i]

        Proj_Rot = S @ np.linalg.inv(S.T @ S) @ S.T @ U.T
        
        Proj_Rot = Proj_Rot[0:self.mu_transform[0].shape[0]-len(self.labels)-1-1,:]
        X_hat = (np.array(np.linalg.inv(self.transform_matrix)) @ (X.T)).T
        
        X_transform = (Proj_Rot@X_hat.T).T
        return X_transform