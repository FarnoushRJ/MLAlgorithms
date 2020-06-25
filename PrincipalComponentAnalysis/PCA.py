import numpy as np


class PCA:
    def __init__(self,
                 n_components):
        """
        :param n_components: Embedding dimension
        """
        self.n_components = n_components

    def fit(self, X):
        """
        :param X: Array_like of shape [numbers, dimensions]
        :return: self
        """
        # calculate covariance matrix
        (num, dim) = X.shape
        self.mean = np.mean(X, axis=0).reshape(1, dim)
        X = X - self.mean
        covariance = np.cov(X.T)

        # singular value decomposition
        (unitary_arrays, singular_values, _) = np.linalg.svd(covariance)
        self.components = unitary_arrays[:, 0: self.n_components]

        return self

    def transform(self, X):
        """
        :param X: Array_like of shape [numbers, dimensions]
        :return: Array_like of shape [numbers, n_components]
        """
        X = X - self.mean
        new_X = np.dot(X, self.components)

        return new_X

    def inverse_transform(self, X):
        """
        :param X: Array_like of shape [numbers, n_components]
        :return: Array_like of shape [numbers, dimensions]
        """
        original_X = np.dot(X, self.components.T) + self.mean

        return original_X
