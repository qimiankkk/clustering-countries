import numpy as np
from sklearn.cluster import KMeans
from scipy.stats import multivariate_normal

def run_kmeans(X, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)
    return labels

def custom_gmm_uniform_prior(X, n_clusters, max_iters=100, tol=1e-4):
    """
    Custom EM algorithm for Gaussian Mixture Model with strict uniform prior (pi_k = 1/K).
    """
    N, D = X.shape
    
    # Initialization using KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)
    
    means = np.zeros((n_clusters, D))
    covs = np.zeros((n_clusters, D, D))
    
    for k in range(n_clusters):
        X_k = X[labels == k]
        if len(X_k) > 1:
            means[k] = np.mean(X_k, axis=0)
            covs[k] = np.cov(X_k.T) + np.eye(D) * 1e-6
        else:
            means[k] = np.random.randn(D)
            covs[k] = np.eye(D)
            
    prior = 1.0 / n_clusters
    log_likelihoods = []
    
    for iteration in range(max_iters):
        # E-Step
        resp = np.zeros((N, n_clusters))
        
        for k in range(n_clusters):
            try:
                rv = multivariate_normal(mean=means[k], cov=covs[k], allow_singular=True)
                resp[:, k] = rv.pdf(X)
            except np.linalg.LinAlgError:
                covs[k] = np.eye(D) * 1e-4
                rv = multivariate_normal(mean=means[k], cov=covs[k], allow_singular=True)
                resp[:, k] = rv.pdf(X)
                
        # Handle zero probabilities
        sum_resp = np.sum(resp, axis=1, keepdims=True)
        sum_resp[sum_resp == 0] = 1e-10
        gamma = resp / sum_resp
        
        # Log Likelihood
        weighted_pdfs = resp * prior
        log_likelihood = np.sum(np.log(np.sum(weighted_pdfs, axis=1) + 1e-10))
        log_likelihoods.append(log_likelihood)
        
        if iteration > 0 and abs(log_likelihood - log_likelihoods[-1]) < tol:
            break
            
        # M-Step
        N_k = np.sum(gamma, axis=0)
        
        for k in range(n_clusters):
            if N_k[k] > 1e-4:
                means[k] = np.sum(gamma[:, k:k+1] * X, axis=0) / N_k[k]
                diff = X - means[k]
                cov_k = np.dot((gamma[:, k:k+1] * diff).T, diff) / N_k[k]
                covs[k] = cov_k + np.eye(D) * 1e-6
                
    final_labels = np.argmax(gamma, axis=1)
    return final_labels
