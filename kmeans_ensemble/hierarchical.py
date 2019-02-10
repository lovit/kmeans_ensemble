import numpy as np
from collections import namedtuple


Merge = namedtuple('Merge', 'parent child0 child1 similarity')


def single_linkage(affinity, n_clusters=2, verbose=True):
    """
    Arguments
    ---------
    affinity : scipy.sparse.matrix
        Affinity matrix
    n_clusters : int
        Number of clusters
    verbose : Boolean
        If True, verbose mode on

    Returns
    -------
    history : list of Merge
        Merge history. Merge is namedtuple
    labels : numpy.ndarray
        shape = (n_rows,)
        unique value = [0, 1, ..., n_clustsers-1]

    Usage
    -----
        >>> history, labels = single_linkage(
        >>>     affinity, n_clusters=2, verbose=True)
    """
    most_similars = []

    n = affinity.shape[0]
    rows, cols = affinity.nonzero()
    data = affinity.data

    for i, j, d in zip(rows, cols, data):
        if i < j:
            most_similars.append((i, j, d))
    sorted_affinity = sorted(most_similars, key=lambda x:x[2], reverse=True)

    idx_to_c = [i for i in range(n)]
    c_to_idxs = {i:{i} for i in range(n)}
    new_c = n

    history = []
    n_iters = 0

    while len(c_to_idxs) > n_clusters and sorted_affinity:

        # Find a new link
        i, j, sim = sorted_affinity.pop(0)
        ci = idx_to_c[i]
        cj = idx_to_c[j]

        # merge two clusters
        union = c_to_idxs[ci]
        union.update(c_to_idxs[cj])
        for u in union:
            idx_to_c[u] = new_c
        c_to_idxs[new_c] = union
        del c_to_idxs[ci]
        del c_to_idxs[cj]

        # log merging history
        history.append(Merge(new_c, ci, cj, sim))

        # Remove already merged links
        sorted_affinity = [pair for pair in sorted_affinity
            if not ((pair[0] in union) and (pair[1] in union))]

        # Increase new cluster idx
        new_c += 1

        n_iters += 1
        if n_iters > n:
            raise ValueError('Number of repeatation is larger than {}'.format(n))

        if verbose and n_iters % 10 == 0:
            print('\rAgglomerative clustering iters = {}'.format(n_iters), end='')

    if verbose:
        print('\rAgglomerative clustering iters = {} was done'.format(n_iters))

    labels = np.asarray(idx_to_c)
    unique = np.unique(labels)
    indices = [np.where(l == labels)[0] for l in unique]
    for l, idxs in enumerate(indices):
        labels[idxs] = l

    return history, labels
