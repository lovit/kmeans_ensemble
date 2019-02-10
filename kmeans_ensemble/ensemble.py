from collections import defaultdict
from sklearn.cluster import KMeans
from scipy.sparse import csr_matrix
import numpy as np
from .hierarchical import single_linkage


class KMeansEnsemble:
    """
    Arguments
    ---------
    n_clusters : int
        Number of clusters
    n_ensembles : int
        Iterations of base k-means
    n_ensemble_units : int
        Number of clusters in each base k-means
    max_iter : int
        Iterations of base k-means
    n_base_history : int
        Number of stored base k-means label results
    verbose : True
        If True, verbose mode on

    Attributes
    ----------
    labels : numpy.ndarray
        Cluster labels. size with (n_rows,)
    affinity : scipy.sparse.csr_matrix
        Pairwise affinity matrix size with (n_rows, n_rows)
    base_history; : list of numpy.ndarray
        Labels of base k-means
    """

    def __init__(self, n_clusters=10, n_ensembles=100, n_ensemble_units=100,
        max_iter=10, n_base_history=10, verbose=True):

        self.n_clusters = n_clusters
        self.n_ensembles = n_ensembles
        self.n_ensemble_units = n_ensemble_units
        self.max_iter = max_iter
        self.n_base_history = n_base_history
        self.verbose = verbose

        self.labels = None
        self.affinity = None
        self.base_history = None

    def fit_predict(self, X):
        self._ensemble(X)
        self._agglomerative(self.affinity)
        return self.labels

    def _ensemble(self, X):
        """
        Repetation of base k-means
        """

        if self.n_base_history > 0:
            self.base_history = []

        n_rows = X.shape[0]

        affinity = defaultdict(int)

        for i_iter in range(1, self.n_ensembles + 1):

            base_kmeans = KMeans(
                n_clusters = self.n_ensemble_units,
                n_init = 1,
                max_iter = self.max_iter
            )
            y = base_kmeans.fit_predict(X)

            if (self.n_base_history > 0) and (i_iter <= self.n_base_history):
                self.base_history.append(y)

            for label in np.unique(y):
                indices = np.where(y == label)[0]
                for i in indices:
                    for j in indices:
                        if i == j:
                            continue
                        key = (i, j)
                        affinity[key] += 1

            if self.verbose and i_iter % 10 == 0:
                print('\rIteration {} / {} ...'.format(i_iter, self.n_ensembles), end='')
        if self.verbose:
            print('\rIteration {0} / {0} was done'.format(self.n_ensembles))

        self.affinity = affinity_as_csr(affinity, n_rows)

    def _agglomerative(self, affinity):
        self.history, self.labels = single_linkage(
            affinity, self.n_clusters, self.verbose)

def affinity_as_csr(dok, n_rows):
    n = len(dok)
    rows = np.zeros(n)
    cols = np.zeros(n)
    data = np.zeros(n)
    for ptr, ((i, j), c) in enumerate(dok.items()):
        rows[ptr] = i
        cols[ptr] = j
        data[ptr] = c
    csr = csr_matrix((data, (rows, cols)), shape=(n_rows, n_rows))
    return csr