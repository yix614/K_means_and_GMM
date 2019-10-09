import numpy as np
from kmeans import KMeans

class GMM():
    '''
        Fits a Gausian Mixture model to the data.

        attrs:
            n_cluster : Number of mixtures (Int)
            e : error tolerance (Float) 
            max_iter : maximum number of updates (Int)
            init : initialization of means and variance
                Can be 'random' or 'kmeans' 
            means : means of Gaussian mixtures (n_cluster X D numpy array)
            variances : variance of Gaussian mixtures (n_cluster X D X D numpy array) 
            pi_k : mixture probabilities of different component ((n_cluster,) size numpy array)
    '''

    def __init__(self, n_cluster, init='k_means', max_iter=100, e=0.0001):
        self.n_cluster = n_cluster
        self.e = e
        self.max_iter = max_iter
        self.init = init
        self.means = None
        self.variances = None
        self.pi_k = None

    def fit(self, x):
        '''
            Fits a GMM to x.

            x: is a NXD size numpy array
            updates:
                self.means
                self.variances
                self.pi_k
        '''
        assert len(x.shape) == 2, 'x can only be 2 dimensional'

        np.random.seed(42)
        N, D = x.shape

        if (self.init == 'k_means'):
            # TODO
            # - comment/remove the exception
            # - initialize means using k-means clustering
            # - compute variance and pi_k (see P4.pdf)

            # DONOT MODIFY CODE ABOVE THIS LINE
            # raise Exception(
            #     'Implement initialization of variances, means, pi_k using k-means')
            kmeans = KMeans(self.n_cluster, self.max_iter, self.e)
            centroids, membership, _ = kmeans.fit(x)
            self.means = centroids
            self.variances = np.zeros((self.n_cluster,D,D))
            self.pi_k = np.zeros(self.n_cluster)
            r = np.eye(len(centroids))[membership]
            for k in range(len(centroids)):
                n_k = np.sum(r[:,k])
                r_k = r[:,k].reshape((N,1))
                self.variances[k] = np.matmul(np.multiply(r_k, x-centroids[k]).T, x-centroids[k])/n_k
                self.pi_k[k] = n_k/N
            # DONOT MODIFY CODE BELOW THIS LINE

        elif (self.init == 'random'):
            # TODO
            # - comment/remove the exception
            # - initialize means randomly
            # - initialize variance to be identity and pi_k to be uniform

            # DONOT MODIFY CODE ABOVE THIS LINE
            # raise Exception(
            #     'Implement initialization of variances, means, pi_k randomly')
            self.means = np.random.rand(self.n_cluster,D)
            self.variances = np.zeros((self.n_cluster,D,D))
            self.pi_k = np.zeros(self.n_cluster)
            for k in range(self.n_cluster):
                self.variances[k] = np.eye(D)
                self.pi_k[k] = 1.0/self.n_cluster
            # DONOT MODIFY CODE BELOW THIS LINE

        else:
            raise Exception('Invalid initialization provided')

        # TODO
        # - comment/remove the exception
        # - Use EM to learn the means, variances, and pi_k and assign them to self
        # - Update until convergence or until you have made self.max_iter updates.
        # - Return the number of E/M-Steps executed (Int) 
        # Hint: Try to separate E & M step for clarity
        # DONOT MODIFY CODE ABOVE THIS LINE
        # raise Exception('Implement fit function (filename: gmm.py)')
        iter = 0
        l = -np.inf
        r = np.empty((N, self.n_cluster))
        while iter < self.max_iter:
            # E
            for k in range(self.n_cluster):
                mu_k = self.means[k]
                temp = self.variances[k]
                while (np.linalg.matrix_rank(temp) < D):
                    temp = temp + 0.001 * np.eye(D)
                variance_k = temp
                r[:, k] = (np.exp(-0.5 * np.sum(np.multiply(np.dot(x-mu_k, np.linalg.inv(variance_k)), x-mu_k), axis=1)) * self.pi_k[k]
                    / np.sqrt((2 * np.pi) ** D * np.linalg.det(variance_k)))        
            l_new = np.sum(np.log(np.sum(r, axis=1)))
            r = (r.T / np.sum(r, axis=1)).T
            n_k = np.sum(r, axis=0)

            # M
            for k in range(self.n_cluster):
                self.means[k] = (np.sum(r[:, k] * x.T, axis=1)).T / n_k[k]
                self.variances[k] = np.dot(np.multiply((x-self.means[k]).T, r[:, k]), x-self.means[k]) / n_k[k]
            self.pi_k = n_k/N
            
            if (np.abs(l - l_new) <= self.e):
                break
            l = l_new
            iter += 1
        return iter

        # DONOT MODIFY CODE BELOW THIS LINE

		
    def sample(self, N):
        '''
        sample from the GMM model

        N is a positive integer
        return : NXD array of samples

        '''
        assert type(N) == int and N > 0, 'N should be a positive integer'
        np.random.seed(42)
        if (self.means is None):
            raise Exception('Train GMM before sampling')

        # TODO
        # - comment/remove the exception
        # - generate samples from the GMM
        # - return the samples

        # DONOT MODIFY CODE ABOVE THIS LINE
        # raise Exception('Implement sample function in gmm.py')
        samples = []
        self.pi_k /= np.sum(self.pi_k)
        random_k = np.random.choice(self.n_cluster, size=N, p=self.pi_k)
        for k in random_k:
            samples.append(np.random.multivariate_normal(self.means[k], self.variances[k]))
        samples = np.asarray(samples)

        # DONOT MODIFY CODE BELOW THIS LINE
        return samples       

    def compute_log_likelihood(self, x, means=None, variances=None, pi_k=None):
        '''
            Return log-likelihood for the data

            x is a NXD matrix
            return : a float number which is the log-likelihood of data
        '''
        assert len(x.shape) == 2,  'x can only be 2 dimensional'
        if means is None:
            means = self.means
        if variances is None:
            variances = self.variances
        if pi_k is None:
            pi_k = self.pi_k    
        # TODO
        # - comment/remove the exception
        # - calculate log-likelihood using means, variances and pi_k attr in self
        # - return the log-likelihood (Float)
        # Note: you can call this function in fit function (if required)
        # DONOT MODIFY CODE ABOVE THIS LINE
        # raise Exception('Implement compute_log_likelihood function in gmm.py')
        N, D = x.shape
        r = np.empty((N, self.n_cluster))
        for k in range(self.n_cluster):
            mu_k = means[k]
            temp = variances[k]
            while (np.linalg.matrix_rank(temp) < D):
                temp = temp + 0.001 * np.eye(D)
            variance_k = temp
            r[:, k] = (np.exp(-0.5 * np.sum(np.multiply(np.dot(x-mu_k, np.linalg.inv(variance_k)), x-mu_k), axis=1)) * pi_k[k]
                    / np.sqrt((2 * np.pi) ** D * np.linalg.det(variance_k)))
        l = np.sum(np.log(np.sum(r, axis=1)))
        log_likelihood = float(l)
        # DONOT MODIFY CODE BELOW THIS LINE
        return log_likelihood



        # DONOT MODIFY CODE BELOW THIS LINE
        #return log_likelihood

    class Gaussian_pdf():
        def __init__(self,mean,variance):
            self.mean = mean
            self.variance = variance
            self.c = None
            self.inv = None
            '''
                Input: 
                    Means: A 1 X D numpy array of the Gaussian mean
                    Variance: A D X D numpy array of the Gaussian covariance matrix
                Output: 
                    None: 
            '''
            # TODO
            # - comment/remove the exception
            # - Set self.inv equal to the inverse the variance matrix (after ensuring it is full rank - see P4.pdf)
            # - Set self.c equal to ((2pi)^D) * det(variance) (after ensuring the variance matrix is full rank)
            # Note you can call this class in compute_log_likelihood and fit
            # DONOT MODIFY CODE ABOVE THIS LINE
            # raise Exception('Impliment Guassian_pdf __init__')
            self.inv = np.linalg.inv(self.variance)
            self.c = (2*np.pi ** D) * np.linalg.det(variance)
            # DONOT MODIFY CODE BELOW THIS LINE

        def getLikelihood(self,x):
            '''
                Input: 
                    x: a 1 X D numpy array representing a sample
                Output: 
                    p: a numpy float, the likelihood sample x was generated by this Gaussian
                Hint: 
                    p = e^(-0.5(x-mean)*(inv(variance))*(x-mean)'/sqrt(c))
                    where ' is transpose and * is matrix multiplication
            '''
            #TODO
            # - Comment/remove the exception
            # - Calculate the likelihood of sample x generated by this Gaussian
            # Note: use the described implementation of a Gaussian to ensure compatibility with the solutions
            # DONOT MODIFY CODE ABOVE THIS LINE
            # raise Exception('Impliment Guassian_pdf getLikelihood')
            x_m = x - self.mean
            p = np.exp(-0.5*np.dot(np.dot(x_m, self.inv), x_m.T))/np.sqrt(self.c)
            # DONOT MODIFY CODE BELOW THIS LINE
            return p
