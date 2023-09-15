"""Symplectic methods."""
import numpy as np


def step_velocity_verlet(t_prev, x_prev, v_prev, f_prev, f_fun, h):
    v_half = v_prev + 0.5*h*f_prev
    x_next = x_prev + h*v_half
    f_next = f_fun(x_next)
    v_next = v_half + 0.5*h*f_next
    return x_next, v_next, f_next


def step_position_verlet(t_prev, x_prev, v_prev, f_fun, h):
    x_half = x_prev + 0.5*h*v_prev
    v_next = v_prev + h*f_fun(t_prev + 0.5*h, x_half)
    x_next = x_half + 0.5*h*v_next
    return x_next, v_next


def step_position_verlet_ct(t_prev, x_prev, v_prev, f_fun, h):
    x_half = x_prev + 0.5*h*v_prev
    v_next = v_prev + h*f_fun(t_prev, x_half)
    x_next = x_half + 0.5*h*v_next
    return x_next, v_next


def solve_velocity_verlet(ti, tf, x0, v0, f_fun, h):
    t = np.arange(ti, tf, h)
    x = np.zeros(len(t))
    v = np.zeros(len(t))
    f = np.zeros(len(t))
    ax = np.zeros(len(t))
    av = np.zeros(len(t))

    x[0] = x0
    v[0] = v0
    f[0] = f_fun(ti, x0)
    ax[0] = 1.
    av[0] = 0.5/np.sqrt(ti)

    for i in range(len(t) - 1):
        x[i + 1], v[i + 1], f[i + 1] = step_velocity_verlet(t[i],
                                                            x[i],
                                                            v[i],
                                                            f[i],
                                                            f_fun,
                                                            h)

    return {
        't': t,
        'x': x,
        'v': v
    }


def solve_position_verlet(ti, tf, x0, v0, f_fun, h):
    t = np.arange(ti, tf, h)
    x = np.zeros(len(t))
    v = np.zeros(len(t))

    x[0] = x0
    v[0] = v0

    for i in range(len(t) - 1):
        x[i + 1], v[i + 1], = step_position_verlet(t[i],
                                                   x[i],
                                                   v[i],
                                                   f_fun,
                                                   h)

    return {
        't': t,
        'x': x,
        'v': v
    }


def solve_position_verlet_ct(ti, tf, x0, v0, f_fun, h):
    t = np.arange(ti, tf, h)
    x = np.zeros(len(t))
    v = np.zeros(len(t))

    x[0] = x0
    v[0] = v0

    for i in range(len(t) - 1):
        x[i + 1], v[i + 1], = step_position_verlet_ct(t[i],
                                                      x[i],
                                                      v[i],
                                                      f_fun,
                                                      h)

    return {
        't': t,
        'x': x,
        'v': v
    }
