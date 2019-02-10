import numpy as np
from bokeh.plotting import figure
from bokeh.palettes import magma, viridis


def draw_scatterplot(X, y, title=None,
    width=800, height=800, size=5):

    """
    Arguments
    ---------
    X : numpy.ndarray
        shape = (n_rows, 2)
    y : numpy.ndarray
        shape = (n_rows,)
        Data labels
    title : str or None
        Figure title. Default is 'Untitled'
    width : int
        Figure width. default is 800
    height : int
        Figure height. default is 800
    size : int
        Point size

    Usage
    -----
        >>> from bokeh.plotting import output_notebook, show
        >>>
        >>> p = draw_scatterplot(X, y)
        >>> # output_notebook()
        >>> show(p)
    """

    if title is None:
        title = 'Untitled'

    p = figure(title=title, width=width, height=height)
    n_uniques = np.unique(y).shape[0]
    colors = viridis(n_uniques)

    for label in range(n_uniques):
        indices = np.where(y == label)[0]
        x_col = X[indices,0].tolist()
        y_col = X[indices,1].tolist()

        p.scatter(x=x_col, y=y_col, fill_color=colors[label],
            line_color='white', marker='circle', size=size)
    return p