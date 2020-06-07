import numpy as np
from scipy.spatial.distance import cdist


class KMeans:
    def __init__(self,
                 n_components: int,
                 max_iter: int = 100):
        """
        :param n_components: Number of cluster
        :param max_iter: Maximum number of iterations
        """
        self.n_components = n_components
        self.max_iter = max_iter

    def _assignment(self, X):
        (num, dim) = X.shape
        for i, x in enumerate(X):
            x = x.reshape(1, dim)
            distances = cdist(x, self._cluster_centers, 'euclidean')
            self.labels[i] = np.argmin(distances.reshape(self.n_components, 1))
        return self

    def _update(self, X):
        criterion = 0
        (num, dim) = X.shape
        for i in range(self.n_components):
            indices = np.where(self._labels == i)[0]
            points = X[indices, :]

            if list(indices) is not []:
                self._cluster_centers[i, :] = np.mean(points, axis=0)
            else:
                index = np.random.choice(range(num), size=1)
                self._cluster_centers[i, :] = X[index, :]

            criterion = criterion + np.sum(cdist(points,
                                                 self._cluster_centers[i].reshape(1, dim),
                                                 'euclidean'))
        return criterion

    def fit(self, X):
        (num, dim) = X.shape
        self._labels = np.empty(num, dtype=int)
        prev_labels = np.empty(num, dtype=int)
        self._criterion = 0.0

        # initialize cluster centers
        indices = np.random.choice(range(num), size=self.n_components, replace=False)
        self._cluster_centers = X[indices, :]

        # iterative assignment and update
        for iteration in range(self.max_iter):
            self._assignment(X)
            self._criterion = self._update(X)

            # termination
            if np.array_equal(prev_labels, self._labels):
                break
            prev_labels = np.copy(self._labels)

        return self

    @property
    def labels(self):
        return self._labels

    @property
    def cluster_centers(self):
        return self._cluster_centers

    @property
    def criterion(self):
        return self.criterion




