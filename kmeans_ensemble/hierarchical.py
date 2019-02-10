from collections import namedtuple
Merge = namedtuple('Merge', 'parent child0 child1 similarity')


def single_linkage(affinity, n_clusters=2):
    most_similars = []

    n = affinity.shape[0]
    rows, cols = affinity.nonzero()
    data = affinity.data

    for i, j, d in zip(rows, cols, data):
        if i < j:
            most_similars.append((i, j, d))
    most_similars = sorted(most_similars, key=lambda x:x[2], reverse=True)

    idx_to_c = [i for i in range(n)]
    c_to_idxs = {i:{i} for i in range(n)}
    new_c = n

    history = []
    labeles = np.zeros(n)
    n_iters = 0

    while len(c_to_idxs) > 1 and most_similars:

        # Find a new link
        i, j, sim = most_similars.pop(0)
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
        most_similars = [pair for pair in most_similars
            if not (i in union) and not (j in union)]

        # Increase new cluster idx
        new_c += 1

        n_iter += 1
        if n_iter > n:
            raise ValueError('Number of repeatation is larger than {}'.format(n))

    for c, idxs in labels.items():
        for idx in idxs:
            labels[idx] = c

    return history, labeles