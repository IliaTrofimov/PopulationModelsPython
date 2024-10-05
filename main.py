import random

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.widgets as widgets

import numpy as np
from matplotlib.gridspec import GridSpec
from scipy.integrate import solve_ivp

import models
import utils
from utils import logger
from handlers import *


model = models.BazykinModelA()
sliders: list[tuple[plt.Axes, widgets.Slider]] = []


def draw_trace(x0, y0, axes: plt.Axes, fig: plt.Figure):
    logger.debug(f'draw_trace: {x0:.3f} {y0:.3f}')
    trace = model.solve(x0, y0, 10, dt=0.1)
    axes.plot(trace[0], trace[1], label=f"$x_0={x0:.3f}$, $y_0={y0:.3f}$")
    fig.show()


def clear_axes(*axes: plt.Axes):
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.tick_params(left=False, bottom=False, labelbottom=False)
        ax.spines[:].set_visible(False)


def init_sliders(heatmap_updater: HeatmapUpdateHandler):
    for ax, slider in sliders:
        ax.clear()
    sliders.clear()

    def on_slider_update(n, v):
        last_v = model.get_param(n)
        model.set_param(n, v)
        if abs(last_v - v) > 0.05:
            heatmap_updater.init()

    for i, (p_name, (p_value, p_min, p_max)) in enumerate(model.enum_parameters().items()):
        ax = plt.axes((0.03, 0.95 - 0.1*(i+1), 0.20, 0.08))
        slider = widgets.Slider(ax=ax, label=p_name, valmin=p_min, valmax=p_max, valinit=p_value)
        sliders.append((ax, slider))
        slider.on_changed(lambda v: on_slider_update(p_name, v))


if __name__ == "__main__":
    utils.parse_args()
    logger.info('Starting execution...')

    fig, ax_main = plt.subplots()
    ax_main.grid()
    ax_main.set_xlabel('Prey, $x$')
    ax_main.set_ylabel('Predators, $y$')

    fig.subplots_adjust(left=0.32, right=0.99, top=0.98, bottom=0.2)
    fig.canvas.manager.set_window_title('Bazykin population model')

    ax_corner = plt.axes((0.03, 0.03, 0.20, 0.10))
    ax_bottom = plt.axes((0.7, 0.05, 0.25, 0.075))

    clear_axes(ax_corner, ax_bottom)

    heatmap_update = HeatmapUpdateHandler(fig, ax_main, model,
                                          [0, 5], [0, 5])
    heatmap_update.init()

    clicker = OnSetPointHandler(fig, ax_main, action=draw_trace)

    btn_update = widgets.Button(ax_corner, 'Update', )
    btn_update.on_clicked(heatmap_update)

    init_sliders(heatmap_update)

    plt.show()


