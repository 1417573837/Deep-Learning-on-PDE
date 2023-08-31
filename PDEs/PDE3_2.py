# -*- coding:utf-8 -*-
"""
Author:Alarak
Date:2023/07/29
"""
from math import pi
import tensorflow as tf
xmin, xmax, ymin, ymax = -2, 2, -2, 2


def theta(x,y):
    infinitesimal=1e-6
    x+=infinitesimal
    y+=infinitesimal
    return tf.atan(y/x)


def gamma(x, y):
    return x ** 2 + y**2 - (pi/3+0.4*tf.sin(8*theta(x,y))**2)


def gamma_r(theta):
    return pi/3+0.4*tf.sin(8*theta)


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
    return tf.where(gamma(x, y) > 0,true_sol_p(x,y),true_sol_n(x,y))


def true_sol_p(x,y):
    return tf.cos(x)


def true_sol_n(x,y):
    return 1e3*tf.sin(3*pi*y)+1500


def a(x, y):
    return tf.where(gamma(x, y) > 0, a_p(x, y), a_n(x, y))


def a_p(x, y):
    return 1*tf.ones_like(x)


def a_n(x, y):
    return 1e-3*tf.ones_like(x)


def f(x, y):
    return tf.where(gamma(x, y) > 0, f_p(x, y), f_n(x, y))


def f_p(x,y):
    return tf.cos(x)


def f_n(x,y):
    return (3*pi)**2*tf.sin(3*pi*y)


def g1(x, y):
    return true_sol(x,y)


def g2(x, y):
    return true_sol(x,y)


def g3(x, y):
    return true_sol(x,y)


def g4(x, y):
    return true_sol(x,y)


def g(x,y):
    return true_sol_p(x,y)-true_sol_n(x,y)


def g_gamma(x,y,normal_vector_x,normal_vector_y):
    with tf.GradientTape(watch_accessed_variables=False, persistent=True) as g:
        g.watch([x, y])
        u_p = true_sol_p(x, y)
        u_n = true_sol_n(x, y)
    u_p_x = g.gradient(u_p, x)
    u_p_y = tf.zeros_like(y)
    u_n_x = tf.zeros_like(x)
    u_n_y = g.gradient(u_n, y)
    grad_p = tf.concat((u_p_x, u_p_y), axis=1)
    grad_n = tf.concat((u_n_x, u_n_y), axis=1)
    gamma_normal_vector=tf.concat((normal_vector_x,normal_vector_y), axis=1)
    return tf.reduce_mean(tf.square(
        a_p(x, y) * grad_p @ tf.transpose(gamma_normal_vector)
        - a_n(x, y) * grad_n @ tf.transpose(gamma_normal_vector)))
    return


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

