import numpy as np

def multivariate_normal_pdf(x, mean, cov):
    cov = np.clip(cov, 1e-12, 1e+8)
    d = x.shape[0]

    det_cov = np.linalg.det(cov)
    inv_cov = np.linalg.inv(cov)
    
    denom = np.sqrt((2 * np.pi)**d * det_cov)

    diff = x - mean
    exponent = -0.5 * diff.T @ inv_cov @ diff
    
    return np.exp(exponent) / denom

def initialize_parameters(X, K, random_state=42):
    np.random.seed(random_state)
    N, d = X.shape

    pis = np.ones(K) / K
    
    indices = np.random.choice(N, K, replace=False)

    mus = X[indices]

    covs = np.array([np.cov(X.T) for _ in range(K)])
    
    return pis, mus, covs

def e_step(X, pis, mus, covs):
    N, d = X.shape
    K = pis.shape[0]
    
    gamma = np.zeros((N, K))
    for i in range(N):
        for k in range(K):
            gamma[i, k] = pis[k] * multivariate_normal_pdf(X[i], mus[k], covs[k])
            
    gamma_sum = np.sum(gamma, axis=1, keepdims=True) 
    gamma = gamma / gamma_sum
    
    return gamma

def m_step(X, gamma):
    N, d = X.shape
    K = gamma.shape[1]

    N_k = np.sum(gamma, axis=0) 
    
    pis = N_k / N
    
    mus = np.zeros((K, d))
    for k in range(K):
        mus[k] = np.sum(gamma[:, k].reshape(-1, 1) * X, axis=0) / N_k[k]
        
    covs = np.zeros((K, d, d))
    for k in range(K):
        diff = X - mus[k]  
        cov_k = np.zeros((d, d))
        for i in range(N):
            diff_i = diff[i].reshape(-1, 1)
            cov_k += gamma[i, k] * (diff_i @ diff_i.T)
        covs[k] = cov_k / N_k[k]
    
    return pis, mus, covs

def log_likelihood(X, pis, mus, covs):
    N, d = X.shape
    K = pis.shape[0]
    
    ll = 0.0
    for i in range(N):
        tmp = 0.0
        for k in range(K):
            tmp += pis[k] * multivariate_normal_pdf(X[i], mus[k], covs[k])
        ll += np.log(tmp)
    return ll

def em_algorithm(X, K=3, max_iter=100, tol=1e-6, random_state=42):
    pis, mus, covs = initialize_parameters(X, K, random_state=random_state)
    ll_old = -np.inf
    log_likelihoods = []
    
    for iteration in range(max_iter):
        gamma = e_step(X, pis, mus, covs)

        pis, mus, covs = m_step(X, gamma)

        ll_new = log_likelihood(X, pis, mus, covs)
        log_likelihoods.append(ll_new)
        
        if np.abs(ll_new - ll_old) < tol:
            print(f"Converged on iteration {iteration}")
            break
        ll_old = ll_new
    
    return pis, mus, covs, log_likelihoods


def compute_cluster_distances(X, cluster_assignments, cluster_centers):
    K = cluster_centers.shape[0]
    
    intra_cluster_distances = []
    for k in range(K):
        cluster_points = X[cluster_assignments == k]
        
        if len(cluster_points) > 1:  
            distances = np.linalg.norm(
                cluster_points[:, np.newaxis] - cluster_points, axis=2
            )
            intra_cluster_distances.append(np.sum(distances) / (len(cluster_points) * (len(cluster_points) - 1)))
        elif len(cluster_points) == 1:
            intra_cluster_distances.append(0) 

    mean_intra_cluster_distance = np.mean(intra_cluster_distances)
    
    inter_cluster_distances = []
    for i in range(K):
        for j in range(i + 1, K): 
            dist = np.linalg.norm(cluster_centers[i] - cluster_centers[j])
            inter_cluster_distances.append(dist)
    
    mean_inter_cluster_distance = np.mean(inter_cluster_distances)
    
    return mean_intra_cluster_distance, mean_inter_cluster_distance