import numpy as np
import scipy
class my_GMM():
    
    def __init__(self, k, num_iter):
        '''
        Parameters:
        k: integer
            number of components
        
        Attributes:
        
        alpha_: np.array
            proportion of components
        mu_: np.array
            array containing means
        Sigma_: np.array
            array cointaining covariance matrix
        cond_prob_: (n, K) np.array
            conditional probabilities for all data points 
        labels_: (n, ) np.array
            labels for data points
        '''
        self.k = k
        self.num_iter = num_iter
        self.alpha_ = None
        self.mu_ = None
        self.sigma_ = None
        self.cond_prob = None
        self.labels = None
        self.P= None
        
    def fit(self, X):
        """ Find the parameters
        that better fit the data
        
        Parameters:
        -----------
        X: (n, p) np.array
            Data matrix
        
        Returns:
        -----
        self
        """
        np.random.seed(seed=3)
        self.alpha_ = np.ones(self.k)/self.k
        self.mu_ = 10*(np.random.rand(self.k,np.shape(X)[1])-0.5)
        self.sigma_ = np.random.rand(self.k,np.shape(X)[1],np.shape(X)[1])
        for i in range (self.sigma_.shape[0]):
            self.sigma_[i]=np.diag(np.diag(self.sigma_[i]))
  #      print(self.sigma_)
 #       for i in range(self.k):
  #          self.sigma_[i]=np.diag(100*np.random.rand(np.shape(X)[1]))
        N= np.shape(X)[0]
        K=self.k   
        for t in range(self.num_iter):
            P= np.zeros((N, K))
            for i in range(N):
                S=0
                for l in range(K):
                    S+= self.alpha_[l]* scipy.stats.multivariate_normal.pdf(X[i],self.mu_[l],self.sigma_[l],allow_singular=True)
                for j in range(K):
                    P[i][j]=(self.alpha_[j]* scipy.stats.multivariate_normal.pdf(X[i],self.mu_[j],self.sigma_[j],allow_singular=True))/S 
            self.alpha_= np.sum(P, axis = 0)/ N
            for j in range(K):
                self.mu_[j,:]= np.sum(X*np.expand_dims(P[:,j],axis=1), axis=0)/(N* self.alpha_[j])#
            for j in range(K):
                A= np.zeros((np.shape(X)[1],np.shape(X)[1]), dtype= "Float64")
                for i in range(N):
                    A= A+ P[i,j]*np.dot(np.expand_dims((X[i]-self.mu_[j]).T, axis=1),np.expand_dims(X[i]-self.mu_[j], axis=0))
                self.sigma_[j]= A/(N* self.alpha_[j])
                for i in range (self.sigma_.shape[0]):
                    self.sigma_[i]=np.diag(np.diag(self.sigma_[i]))
        self.P= P
        return( self.alpha_, self.mu_, self.sigma_ , np.shape(self.P))
    
    def predict(self, X):
        """ Predict labels for X
        
        Parameters:
        -----------
        X: (n, p) np.array
            New data matrix
        
        Returns:
        -----
        label assigment        
        """
        N= np.shape(X)[0]
        K=self.k 
        self.labels= np.zeros((N,1))
        for i in range(N):
            self.labels[i]=np.random.choice(np.arange(1, K+1), p=self.P[i])
        return(self.labels)
