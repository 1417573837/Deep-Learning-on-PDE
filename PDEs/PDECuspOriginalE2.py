# -*- coding:utf-8 -*-
"""
Author:Alarak
Date:2023/07/29
"""
import tensorflow as tf

xmin, xmax, ymin, ymax = -1, 1, -1, 1
g = tf.constant(0, dtype=tf.float32)


def equation(x, y, U):
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
    return au_x_x + au_y_y -u - f(x, y)


def true_sol(x, y):  # input: x,y:1-d tensorflow, dtype=tf.float32
    return tf.where(gamma(x, y) > 0,
                    true_sol_p(x,y),
                    true_sol_n(x,y)
                    )


def true_sol_p(x, y):
    return -2*tf.math.log(gamma(x, y) + 1)


def true_sol_n(x, y):
    return 1 - tf.exp(0.1 * gamma(x, y))


def a(x, y):
    return tf.where(gamma(x, y) > 0, a_p(x, y), a_n(x, y))


def a_p(x, y):
    return tf.constant(1, dtype=tf.float32)


def a_n(x, y):
    return tf.constant(10, dtype=tf.float32)


def gamma(x, y):
    return x ** 2 / 0.25 + y ** 2 / 0.25 - 1


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
    return au_x_x + au_y_y -u


def g1(x, y):
    return true_sol(x, y)


def g2(x, y):
    return true_sol(x, y)


def g3(x, y):
    return true_sol(x, y)


def g4(x, y):
    return true_sol(x, y)


def g_gamma(gamma_boundary_points, gamma_normal_vector):
    return tf.constant(-4,dtype=tf.float32)
    # x = gamma_boundary_points[:, 0:1]
    # y = gamma_boundary_points[:, 1:2]
    # with tf.GradientTape(watch_accessed_variables=False, persistent=True) as g:
    #     g.watch([x,y])
    #     u_p = true_sol_p(x,y)
    #     u_n = true_sol_n(x,y)
    # u_p_x = g.gradient(u_p, x)
    # u_p_y = g.gradient(u_p, y)
    # u_n_x = g.gradient(u_n, x)
    # u_n_y = g.gradient(u_n, y)
    # grad_p = tf.concat((u_p_x, u_p_y), axis=1)
    # grad_n = tf.concat((u_n_x, u_n_y), axis=1)
    # return a_p(x, y) * grad_p @ tf.transpose(gamma_normal_vector) - a_n(x, y) * grad_n @ tf.transpose(gamma_normal_vector)


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
