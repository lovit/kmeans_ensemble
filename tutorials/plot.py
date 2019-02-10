import numpy as np
from bokeh.plotting import figure
from bokeh.palettes import magma


def draw_scatterplot(X, y, size=5):
    """
    Usage
    -----
        >>> from bokeh.plotting import output_notebook, show
        >>>
        >>> p = draw_scatterplot(X, y)
        >>> # output_notebook()
        >>> show(p)
    """

    p = figure()
    n_uniques = np.unique(y).shape[0]
    colors = magma(n_uniques + 1)

    for label in range(n_uniques):
        indices = np.where(y == label)[0]
        x_col = X[indices,0].tolist()
        y_col = X[indices,1].tolist()

        p.scatter(x=x_col, y=y_col, fill_color=colors[label],
            line_color='white', marker='circle', size=size)
    return p