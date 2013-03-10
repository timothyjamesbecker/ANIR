from PIL import Image
from numpy import *

def pca(X):
    #Principle Component Analysis
    #input:X,matrix with training data stored as flattened arrays in rows
    #return:projection matrix (with PC1, PC2,..PCN)variance and mean
    num_data,dim = X.shape
    mean_X = X.mean(axis=0)
    X = X - mean_X

    if dim>num_data:
        #PCA compact trick used
        M = dot(X,X.T) #covariance matrix
        e,EV = linalg.eigh(M) #eigenvalues and eigenvectors
        tmp = dot(X.T,EV).T #this is the trick
        V = tmp[::-1]     #reverse to get the last eigenvectors
        S = sqrt(e)[::-1] #reverse since eigenvalues are in increasing order
        for i in range(V.shape[1]):
            V[:,i] /= S
    else:
        #PCA - SVD used
        U,S,V = linalg.svd(X)
        V = V[:num_data]
    #return the projection matrix, the variance and the mean
    return V,S,mean_X
                
