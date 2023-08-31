# -*- coding:utf-8 -*-
"""
Author:Alarak
Date:2023/07/29
"""
import tensorflow as tf

xmin, xmax, ymin, ymax = -2.5, 2.5, -2.5, 2.5
g = tf.constant(0,dtype=tf.float32)
g_gamma = tf.constant(0,dtype=tf.float32)


def equation(x,y,U):
    with tf.GradientTape(watch_accessed_variables=False, persistent=True) as g:
        g.watch([x, y])
        with tf.GradientTape(watch_accessed_variables=False, persistent=True) as gg:
            gg.watch([x, y])
            u = U(x, y)
        u_x = gg.gradient(u, x)
        u_y = gg.gradient(u, y)
        au_x = a(x, y) * u_x
        au_y = a(x, y) * u_y
    au_x_x = g.gradient(au_x, x)
    au_y_y = g.gradient(au_y, y)
    return -au_x_x - au_y_y - f(x,y)


def true_sol(x, y):  # input: x,y:1-d tensorflow, dtype=tf.float32
    return 100*tf.sin(x)*tf.sin(y)


def a(x, y):
    return tf.ones_like(x)


def a_p(x, y):
    return 2 + tf.sin(x) * tf.sin(y)


def a_n(x, y):
    return 1e3 * a_p(x, y)


def gamma(x, y):
    return x ** 4 + 2 * y ** 4 - 2


def f(x, y):
    with tf.GradientTape(watch_accessed_variables=False, persistent=True) as g:
        g.watch([x, y])
        with tf.GradientTape(watch_accessed_variables=False, persistent=True) as gg:
            gg.watch([x, y])
            u = true_sol(x, y)
        u_x = gg.gradient(u, x)
        u_y = gg.gradient(u, y)
        au_x = a(x, y) * u_x
        au_y = a(x, y) * u_y
    au_x_x = g.gradient(au_x, x)
    au_y_y = g.gradient(au_y, y)
    return -au_x_x - au_y_y


def g1(x, y):
    with tf.GradientTape(watch_accessed_variables=False) as g:
        g.watch(x)
        u = true_sol(x, y)
    u_x = g.gradient(u, x)
    return -u_x + (tf.cos(y) + 2) * u


def g2(x, y):
    return true_sol(x, y)


def g3(x, y):
    with tf.GradientTape(watch_accessed_variables=False) as g:
        g.watch(y)
        u = true_sol(x, y)
    u_y = g.gradient(u, y)
    return -u_y + (tf.sin(x) + 2) * u


def g4(x, y):
    return true_sol(x, y)


####Extra Calls####
def true_sol_x(x, y):  # input: x,y:1-d tensorflow, dtype=tf.float32
    with tf.GradientTape(watch_accessed_variables=False) as g:
        g.watch(x)
        true_s = true_sol(x, y)
    true_s_x = g.gradient(true_s, x)
    return true_s_x


def true_sol_y(x, y):  # input: x,y:1-d tensorflow, dtype=tf.float32
    with tf.GradientTape(watch_accessed_variables=False) as g:
        g.watch(y)
        true_s = true_sol(x, y)
    true_s_y = g.gradient(true_s, y)
    return true_s_y


def true_sol_xx(x, y):  # input: x,y:1-d tensorflow, dtype=tf.float32
    with tf.GradientTape(watch_accessed_variables=False) as g:
        g.watch(x)
        true_s_x = true_sol_x(x, y)
    true_s_xx = g.gradient(true_s_x, x)
    return true_s_xx


def true_sol_yy(x, y):  # input: x,y:1-d tensorflow, dtype=tf.float32
    with tf.GradientTape(watch_accessed_variables=False) as g:
        g.watch(y)
        true_s_y = true_sol_y(x, y)
    true_s_yy = g.gradient(true_s_y, y)
    return true_s_yy