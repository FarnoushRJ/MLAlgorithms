import numpy as np
from KMeansClustering.KMeans import KMeans


class GMM:
    def __init__(self,
                 n_components: int,
                 max_iters: int = 100,
                 init_kmeans: bool = False,
                 tol: float = 1e-5):
        """
        :param n_components: Number of Gaussian components
        :param max_iters: Maximum number of iterations
        :param init_kmeans: Use K-Means initialization or not
        :param tol: Tolerance
        """

        self.n_components = n_components
        self.max_iters = max_iters
        self.init_kmeans = init_kmeans
        self.tol = tol

    @staticmethod
    def _pdf(X: np.ndarray,
             mu: np.ndarray,
             cov: np.ndarray):
        """
        :param X: Array_like of shape [numbers, dimensions]
        :param mu: Mean
        :param cov: Covariance
        :return: Normal PDF
        """
        (num, dim) = X.shape
        norm_pdf = np.zeros(num)
        mu = mu.reshape(1, dim)

        det = np.linalg.det(cov)
        cov_inverse = np.linalg.inv(cov)

        for i, x in enumerate(X):
            x = x.reshape(1, dim)
            coeff = 1 / (((2 * np.pi) ** (dim / 2)) * det ** (1 / 2))
            norm_pdf[i] = coeff * np.exp((np.dot(np.dot((x - mu), cov_inverse), (x - mu).T) / (-2)).flatten()[0])

        return norm_pdf

    def _expectation(self, X):
        """ Expectation Step """
        (num, dim) = X.shape
        likelihood = np.zeros(num)

        for i in range(self.n_components):
            self._sigma[i] = self._sigma[i] + (self.tol * np.eye(dim))  # regularization
            norm_pdf = self._pdf(X, self._mu[i], self._sigma[i])
            self._weights[:, i] = self._pi[i] * norm_pdf
            likelihood = likelihood + self._weights[:, i]

        self._weights = self._weights / np.sum(self._weights, axis=1).reshape(num, 1)
        return likelihood

    def _maximization(self, X):
        """ Maximization Step """
        (num, dim) = X.shape
        num_k = np.sum(self._weights, axis=0)
        self._pi = num_k / num
        self._mu = np.dot(self._weights.T, X) / num_k .reshape(self.n_components, 1)

        for i in range(self.n_components):
            diff = X - self._mu[i]
            self._sigma[i] = np.dot((self._weights[:, i] * diff.T), diff) / num_k[i]
        return self

    def fit(self, X):
        """
        :param X: Array_like of shape [numbers, dimensions]
        :return: self
        """
        (num, dim) = X.shape

        if self.init_kmeans:
            kmeans = KMeans(n_components=self.n_components, max_iter=self.max_iters)
            kmeans.fit(X)
            self._mu = kmeans.cluster_centers
        else:
            idx = np.random.choice(range(num), self.n_components, replace=False)
            self._mu = X[idx, :]

        self._pi = np.full([self.n_components], float(1) / float(self.n_components))
        self._weights = np.empty([num, self.n_components], dtype=float)
        self._sigma = np.full([num, dim, dim], np.eye(dim))

        llv_prev = 0.0  # previous log-likelihood
        llv_cur = 0.0  # current log-likelihood

        for iteration in range(self.max_iters):
            likelihood = self._expectation(X)
            self._maximization(X)
            llv_cur = np.sum(np.log(likelihood))

            # Termination
            if (llv_cur - llv_prev) == 0:
                break

            llv_prev = llv_cur
        return self

    @property
    def sigma(self):
        return self._sigma

    @property
    def pi(self):
        return self._pi

    @property
    def mu(self):
        return self._mu

    @property
    def weights(self):
        return self._weights
