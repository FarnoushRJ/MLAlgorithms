import numpy as np
from scipy.spatial import distance


class LLE:
    def __init__(self,
                 n_components: int,
                 tol: np.float = 1e-2,
                 neighbors_algorithm: str = "KNN",
                 n_neighbors: int = None,
                 epsilon: np.float = None):

        """
        Locally Linear Embedding
        :param n_components: Embedding dimension
        :param tol: Tolerance
        :param neighbors_algorithm: KNN or EPS_BALL
        :param n_neighbors: Number of neighbors
        :param epsilon: Epsilon value of the Epsilon Ball algorithm
        """

        self.n_components = n_components
        self.neighbors_algorithm = neighbors_algorithm
        self.n_neighbors = n_neighbors
        self.epsilon = epsilon
        self.tol = tol
        self.embedding = None

    def _knn(self, distances):
        """
        K-Nearest Neighbors
        """
        indices = np.argsort(distances.flatten())[1: self.n_neighbors+1]
        return indices

    def _epsilon_ball(self, distances):
        """
        Epsilon Ball
        """
        indices = np.where(distances.flatten()[1:] < self.epsilon)[0]
        if not list(indices):
            raise ValueError
        return indices

    def fit_transform(self, X):
        """
        :param X: Array_like of shape [numbers, dimensions]
        :return:  Array_like of shape [numbers, n_components]
        """

        (num, dim) = X.shape
        weight = np.zeros((num, num))

        for idx, x in enumerate(X):
            x = x.reshape(1, dim)

            # calculate the neighbors
            distances = distance.cdist(x, X)
            calc_neighbors = self._knn if self.neighbors_algorithm == 'KNN' else self._epsilon_ball
            indices = calc_neighbors(distances)
            neighbors = X[indices]
            self.n_neighbors = neighbors.shape[0] if self.neighbors_algorithm == 'EPS_BALL' else self.n_neighbors

            # compute the local covariance matrix
            cov = np.dot((x - neighbors), (x - neighbors).T)

            # if cov is not full rank, it should be regularized
            if self.n_neighbors > dim:
                cov = cov + (self.tol * np.eye(self.n_neighbors))

            # calculate the weight matrix
            ones = np.ones(self.n_neighbors).reshape(self.n_neighbors, 1)
            weight[idx, indices] = np.linalg.solve(cov, ones).flatten()
            weight[idx, indices] = weight[idx, indices] / np.sum(weight[idx, indices])

        # Create sparse matrix M
        identity = np.eye(num)
        M = np.dot((identity - weight).T, (identity - weight))

        # Find bottom d+1 eigen vectors of M
        u, s, v = np.linalg.svd(M)
        idx = np.argsort(s)[1: self.n_components + 1]
        self.embedding = u[:, idx]

        return self.embedding
