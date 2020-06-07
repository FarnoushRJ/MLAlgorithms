# **********************************************************************************
# Many parts of the visualization code are taken from
# "https://jakevdp.github.io/PythonDataScienceHandbook/05.12-gaussian-mixtures.html"
# **********************************************************************************

import matplotlib.pylab as plt
from scipy.spatial.distance import cdist
plt.style.use('ggplot')


def plot_kmeans(kmeans, X, xy_labels):
    labels = kmeans.labels
    centers = kmeans.cluster_centers

    fig = plt.figure(figsize=(10, 5), dpi=150)
    ax = fig.add_subplot(111)

    if xy_labels is not None:
        ax.set_xlabel(xy_labels[0])
        ax.set_ylabel(xy_labels[1])

    ax.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis', zorder=2, marker='^')
    ax.scatter(centers[:, 0], centers[:, 1], s=100, zorder=4, marker='*', c='red')
    ax.axis('equal')

    radius = [cdist(X[labels == i], [center]).max() for i, center in enumerate(centers)]
    for c, r in zip(centers, radius):
        ax.add_patch(plt.Circle(c, r, fc='#CCCCCC', lw=3, alpha=0.5, zorder=1))
    plt.show()
