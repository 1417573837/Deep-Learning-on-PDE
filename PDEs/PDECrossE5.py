# -*- coding:utf-8 -*-
"""
Author:Alarak
Date:2023/07/29
"""
from math import pi
import tensorflow as tf
xmin, xmax, ymin, ymax = 0, 1, 0, 1
ksi,zeta= pi/4,pi/10


# def gamma(x, y, p):
#     if p == 1:
#         return 1e4
#     if p == 2:
#         return 1e-6
#     if p == 3:
#         return 1e5
#     if p == 4:
#         return 1e-5



def equation(x,y,U, p):
    with tf.GradientTape(watch_accessed_variables=False, persistent=True) as g:
        g.watch([x, y])
        with tf.GradientTape(watch_accessed_variables=False, persistent=True) as gg:
            gg.watch([x, y])
            u = U(x, y, p)
        u_x = gg.gradient(u, x,unconnected_gradients=tf.UnconnectedGradients.ZERO)
        u_y = gg.gradient(u, y,unconnected_gradients=tf.UnconnectedGradients.ZERO)
        au_x = a(x, y, p) * u_x
        au_y = a(x, y, p) * u_y
    au_x_x = g.gradient(au_x, x,unconnected_gradients=tf.UnconnectedGradients.ZERO)
    au_y_y = g.gradient(au_y, y,unconnected_gradients=tf.UnconnectedGradients.ZERO)
    return -au_x_x - au_y_y - f(x,y,p)


def true_sol(x, y,p):  # input: x,y:1-d tensorflow, dtype=tf.float32
    if p==1:
        return tf.sin(16*y)
    if p==2:
        return tf.cos(16 * x)
    if p==3:
        return tf.sin(16 * y)
    if p==4:
        return tf.cos(16 * x)


def a(x, y,p):
    if p==1:
        return 1e4
    if p==2:
        return 1e-6
    if p==3:
        return 1e5
    if p==4:
        return 1e-5


def f(x, y, p):
    with tf.GradientTape(watch_accessed_variables=False, persistent=True) as g:
        g.watch([x, y])
        with tf.GradientTape(watch_accessed_variables=False, persistent=True) as gg:
            gg.watch([x, y])
            u = true_sol(x, y, p)
        u_x = gg.gradient(u, x,unconnected_gradients=tf.UnconnectedGradients.ZERO)
        u_y = gg.gradient(u, y,unconnected_gradients=tf.UnconnectedGradients.ZERO)
        au_x = a(x, y, p) * u_x
        au_y = a(x, y, p) * u_y
    au_x_x = g.gradient(au_x, x,unconnected_gradients=tf.UnconnectedGradients.ZERO)
    au_y_y = g.gradient(au_y, y,unconnected_gradients=tf.UnconnectedGradients.ZERO)
    return -au_x_x - au_y_y


def phi(x,y,p):
    if p==1:
        return true_sol(x,y,2)-true_sol(x,y,1)
    if p==2:
        return true_sol(x,y,3)-true_sol(x,y,4)
    if p==3:
        return true_sol(x,y,2)-true_sol(x,y,3)
    if p==4:
        return true_sol(x,y,1)-true_sol(x,y,4)


def psi(x,y,p):
    if p==1:
        return a(x,y,2)*true_sol_x(x,y,2)-a(x,y,1)*true_sol_x(x,y,1)
    if p==2:
        return a(x,y,3)*true_sol_x(x,y,3)-a(x,y,4)*true_sol_x(x,y,4)
    if p==3:
        return a(x,y,2)*true_sol_y(x,y,2)-a(x,y,3)*true_sol_y(x,y,3)
    if p==4:
        return a(x,y,1)*true_sol_y(x,y,1)-a(x,y,4)*true_sol_y(x,y,4)


####Extra Calls####
def true_sol_x(x, y, p):  # input: x,y:1-d tensorflow, dtype=tf.float32
    with tf.GradientTape(watch_accessed_variables=False) as g:
        g.watch(x)
        true_s = true_sol(x, y,p)
    true_s_x = g.gradient(true_s, x,unconnected_gradients=tf.UnconnectedGradients.ZERO)
    return true_s_x


def true_sol_y(x, y,p):  # input: x,y:1-d tensorflow, dtype=tf.float32
    with tf.GradientTape(watch_accessed_variables=False) as g:
        g.watch(y)
        true_s = true_sol(x, y,p)
    true_s_y = g.gradient(true_s, y,unconnected_gradients=tf.UnconnectedGradients.ZERO)
    return true_s_y

# 
# def true_sol_xx(x, y):  # input: x,y:1-d tensorflow, dtype=tf.float32
#     with tf.GradientTape(watch_accessed_variables=False) as g:
#         g.watch(x)
#         true_s_x = true_sol_x(x, y)
#     true_s_xx = g.gradient(true_s_x, x)
#     return true_s_xx
# 
# 
# def true_sol_yy(x, y):  # input: x,y:1-d tensorflow, dtype=tf.float32
#     with tf.GradientTape(watch_accessed_variables=False) as g:
#         g.watch(y)
#         true_s_y = true_sol_y(x, y)
#     true_s_yy = g.gradient(true_s_y, y)
#     return true_s_yy

def true_sol_plotting(x,y):
    return tf.where(x>ksi,
                    tf.where(y>zeta, true_sol(x,y,2),true_sol(x,y,3)),
                    tf.where(y>zeta,true_sol(x,y,1),true_sol(x,y,4)))

