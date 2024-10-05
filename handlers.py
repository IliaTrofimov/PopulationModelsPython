import matplotlib.pyplot as plt
import matplotlib.animation as animation
import traceback

import numpy as np

from utils import logger


class MultiPointsClickHandler:
    def __init__(self, fig: plt.Figure, axes: plt.Axes, max_points: int = 10, **kwargs):
        if max_points <= 0:
            raise ValueError("max_points must be greater then 0")

        self.__fig = fig
        self.__axes = axes
        self.__max_points = max_points
        self.__points_x = []
        self.__points_y = []
        self.kwargs = kwargs
        self.__plot = None
        self.__cid = fig.canvas.mpl_connect('button_press_event', self)

    def __call__(self, event):
        if event.inaxes != self.__axes: return

        if len(self.__points_x) == self.__max_points:
            self.__points_x.pop(0)
            self.__points_y.pop(0)

        x, y = event.xdata, event.ydata
        self.__points_x.append(x)
        self.__points_y.append(y)
        self.__init_plot()
        logger.debug(f'on_add_point: x={x:.3f} y={y:.3f} points_count={len(self.__points_x)}')
        self.__plot.set_data(self.__points_x, self.__points_y)
        self.__fig.canvas.draw()

    def __init_plot(self):
        if self.__plot is not None: return
        if self.kwargs is None or self.kwargs == {}:
            logger.debug('on_add_point: CONFIG using default scatter arguments')
            self.__plot, = self.__axes.plot(self.__points_x, self.__points_y, lw=0, ms=5, marker='.')
        else:
            self.__plot, = self.__axes.plot(self.__points_x, self.__points_y, **self.kwargs)

    def __str__(self):
        return f"MultiPointsClickHandler(mpl_cid={self.__cid})"


class OnSetPointHandler:
    def __init__(self, fig: plt.Figure, axes: plt.Axes, action=None, **kwargs):
        self._fig = fig
        self._axes = axes
        self._plot = None
        self.kwargs = kwargs
        self.action = action
        self._cid = fig.canvas.mpl_connect('button_press_event', self)

    def __call__(self, event):
        if event.inaxes != self._axes: return

        x, y = event.xdata, event.ydata
        self.__init_plot(x, y)
        self.__call_action(x, y)
        self._fig.canvas.draw()

    def __init_plot(self, x, y):
        if self._plot is None:
            self._plot, = self._axes.plot(x, y, **self.kwargs)
        else:
            self._plot.set_data([x], [y])

    def __call_action(self, x, y):
        if self.action is not None:
            try:
                self.action(x, y, self._axes, self._fig)
                logger.debug(f'on_set_point: x={x:.3f} y={y:.3f} ACTION')
            except Exception:
                logger.error(f'on_set_point: x={x:.3f} y={y:.3f} ACTION failed')
                logger.error(f'on_set_point: exception trace: {traceback.format_exc()}')

        else:
            logger.debug(f'on_set_point: x={x:.3f} y={y:.3f}')

    def __str__(self):
        return f"OnSetPointHandler(mpl_cid={self._cid})"


class HeatmapUpdateHandler:
    locked = False

    def __init__(self, fig: plt.Figure, axes: plt.Axes, heatmap_fxy, xlim: tuple[float], ylim: tuple[float],
                 steps: int = 20, **kwargs):
        self._fig = fig
        self._axes = axes
        self._heatmap = None
        self._quiver = None
        self._cbar = None
        self.kwargs = kwargs
        self.heatmap_fxy = heatmap_fxy
        self.xlim = xlim
        self.ylim = ylim
        self.steps = steps

    def __call__(self, event):
        if HeatmapUpdateHandler.locked:
            logger.warn(f'on_heatmap_update: thread lock')
            return

        HeatmapUpdateHandler.locked = True
        logger.debug(f'on_heatmap_update: calculating...')

        X, Y = np.mgrid[self.xlim[0]:self.xlim[1]:complex(0, self.steps),
                        self.ylim[0]:self.ylim[1]:complex(0, self.steps)]
        U, V = self.heatmap_fxy(0, [X, Y])
        Z = np.hypot(U, V)

        logger.debug(f'on_heatmap_update: drawing...')

        if self._heatmap is None:
            self._heatmap = self._axes.contourf(X, Y, Z, cmap='plasma', levels=20)
            self._quiver = self._axes.quiver(X, Y, U, V)
            self._cbar = self._fig.colorbar(self._heatmap, ax=self._axes)
        else:
            self._cbar.remove()
            self._heatmap.remove()
            self._axes.cla()
            self._heatmap = self._axes.contourf(X, Y, Z, cmap='plasma', levels=20)
            self._quiver = self._axes.quiver(X, Y, U, V)
            self._cbar = self._fig.colorbar(self._heatmap, ax=self._axes)

        self._fig.show()
        logger.debug(f'on_heatmap_update: done')
        HeatmapUpdateHandler.locked = False

    def init(self):
        self.__call__(None)
