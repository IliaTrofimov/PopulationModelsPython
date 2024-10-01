import random

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.widgets as widgets

import numpy as np
from scipy.integrate import solve_ivp

import utils
from utils import logger
from handlers import *


alpha = 1.0
epsilon = 1.0
gamma = 1.0
mu = 1.0
dt = 0.01


def bazykin_model_a(t, *y):
    a = random.random()
    return (y[0] - y[0]*y[1]/(1 + a*y[0]) - epsilon*y[0]**2,
            -gamma*y[1] + y[0]*y[1]/(1 + a*y[0]) - mu*y[1]**2)


def bazykin_solution_a(x0: float, y0: float, t: float):
    return solve_ivp(bazykin_model_a, (0, t), (x0, y0), first_step=dt).y


def draw_trace(x0, y0, axes: plt.Axes, fig: plt.Figure):
    logger.debug(f'draw_trace: {x0:.3f} {y0:.3f}')
    trace = bazykin_solution_a(x0, y0, 5.0)
    axes.plot(trace[:, 0], trace[:, 1], label=f"$x_0={x0:.3f}$, $y_0={y0:.3f}$")
    fig.show()


if __name__ == "__main__":
    utils.parse_args()
    logger.info('Starting execution...')

    fig, ax_main = plt.subplots()
    ax_next = fig.add_axes([0.81, 0.05, 0.1, 0.075])
    btn_update = widgets.Button(ax_next, 'Update')

    heatmap_update = HeatmapUpdateHandler(fig, ax_main, bazykin_model_a,
                                          [0, 5], [0, 5])
    heatmap_update.init()

    clicker = OnSetPointHandler(fig, ax_main, action=draw_trace)
    btn_update.on_clicked(heatmap_update)

    plt.show()


