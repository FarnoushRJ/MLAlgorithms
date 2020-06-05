# **********************************************************************************
# Many parts of the visualization code are taken from
# "https://jakevdp.github.io/PythonDataScienceHandbook/05.12-gaussian-mixtures.html"
# **********************************************************************************

import matplotlib.pylab as plt
import numpy as np
from matplotlib.patches import Ellipse
plt.style.use('ggplot')


# ---------------------
def draw_ellipse(mu, sigma, ax=None, **kwargs):
    """
    Draw an ellipse
    """
    # convert covariance to principal axes
    if sigma.shape == (2, 2):
        u, s, v = np.linalg.svd(sigma)
        angle = np.degrees(np.arctan2(u[1, 0], u[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt(sigma)

    # draw the ellipse
    for i in range(1, mu.shape[0]+1):
        width *= i
        height *= i
        ax.add_patch(Ellipse(mu, width, height, angle, **kwargs))


# ---------------------
def plot_gmm(gmm, X, colors, label=True, xy_labels=None):
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111)

    if xy_labels is not None:
        print('Hi')
        ax.set_xlabel(xy_labels[0])
        ax.set_ylabel(xy_labels[1])

    if label:
        ax.scatter(X[:, 0], X[:, 1], c=colors, s=40, zorder=2, marker='^')
        ax.scatter(gmm.mu[:, 0], gmm.mu[:, 1], s=100, zorder=4, marker='*', c='black')
    else:
        ax.scatter(X[:, 0], X[:, 1], s=50, zorder=2, marker='^', c='black')
    ax.axis('equal')

    for pos, cov in zip(gmm.mu, gmm.sigma):
        draw_ellipse(pos, cov, ax, color='orange', alpha=0.2)
    plt.show()
