# -*- coding:utf-8 -*-
"""
Author:Alarak
Date:2023/07/31

gpustat --force-color -cpu -i 1
^^^This line of command is used in terminal to monitor GPU state

This program solves a PDE using customized PINN.
For more preliminaries, see file 'readme'.
"""
import matplotlib.pyplot as plt
import tensorflow as tf
from math import pi
import os
import sys
import time

"""
packages above is imported in other files.
"""
import plotting  # customized plotting
import DataGenerateCross  # data generate
import condtional_print  # coloful printing in console
from PDEs.PDECrossE5Simplified import *  # PDE problem
from VectorDenseLayerV2 import *  # customized layer
from SAIS import *  # failure-informed sampling method

# make folder for results
PDEname = 'CrossE5Simplified VDLV2 SAIS Vx.x'
if not os.path.exists(PDEname):
    os.makedirs(PDEname)

# Model Settings
"""
Vector Dense Layer Model 
Allowing to put every activations that might work
"""
# define your activations
act1 = lambda x, y: tf.sin(x + y)
act2 = lambda x, y: tf.sin(x * y)
act3 = lambda x, y: tf.sin(x) * tf.sin(y)
# assemble the model
Model = tf.keras.Sequential([tf.keras.layers.Dense(30, input_dim=2),
                             VectorDenseLayer(30, activations=[act1, act2, act3]),
                             tf.keras.layers.Dense(30),
                             VectorDenseLayer(30, activations=[act1, act2, act3]),
                             tf.keras.layers.Dense(30),
                             VectorDenseLayer(30, activations=[act1, act2, act3]),
                             tf.keras.layers.Dense(30),
                             VectorDenseLayer(30, activations=[act1, act2, act3]),
                             tf.keras.layers.Dense(30),
                             VectorDenseLayer(30, activations=[act1, act2, act3]),
                             tf.keras.layers.Dense(4)
                             ])

"""
traditional neural network
allowing a customized activation"""
# my_activation=lambda x:tf.sin(x)
# my_act = tf.keras.activations.get(my_activation)
# Model=tf.keras.Sequential([tf.keras.layers.Dense(30,activation=my_act),
#                            tf.keras.layers.Dense(30,activation=my_act),
#                            tf.keras.layers.Dense(30,activation=my_act),
#                            tf.keras.layers.Dense(30,activation=my_act),
#                            tf.keras.layers.Dense(30,activation=my_act),
#                            tf.keras.layers.Dense(4,activation=None)])

"""
reload previous model, or continue training
"""
# Model = tf.keras.models.load_model("./CrossE5Simplified VDLV2 SAIS v2.1/CrossE5Simplified VDLV2 SAIS v2.1 1400.h5",
#                                    custom_objects={"<lambda>":my_activation})
# Model = tf.keras.models.load_model("./CrossE5Simplified VDLV2 SAIS v2.1/CrossE5Simplified VDLV2 SAIS v2.1 11900.h5",custom_objects={"VectorDenseLayer":VectorDenseLayer})
"""
set optimizer and learning rate schedule, but empirically Adam don't need adjust on learning rate. Just don't set a too big learning rate at start"""
# lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
#     initial_learning_rate=0.01,
#     decay_steps=1,
#     decay_rate=0.01)
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

# U(Model) Setup
scale = 1  # the scale of the solution. if known, may accelerate training


def U(x, y, p):
    """
    :param x:1-d tensorflow, dtype=tf.float32
    :param y:same as above
    :param p:serial number of the subdomain, int
    """
    format_input = tf.concat((x, y), axis=1)
    if p == 1:
        return scale * Model(format_input)[:, 0:1]
    if p == 2:
        return scale * Model(format_input)[:, 1:2]
    if p == 3:
        return scale * Model(format_input)[:, 2:3]
    if p == 4:
        return scale * Model(format_input)[:, 3:4]


"""
special method for plotting, only for this PDE

For future code optimize: remove this part
"""


def U_plotting(x, y):
    return tf.where(x > ksi,
                    tf.where(y > zeta, U(x, y, 2), U(x, y, 3)),
                    tf.where(y > zeta, U(x, y, 1), U(x, y, 4)))


def U_x_plotting(x, y):
    return tf.where(x > ksi,
                    tf.where(y > zeta, U_x(x, y, 2), U_x(x, y, 3)),
                    tf.where(y > zeta, U_x(x, y, 1), U_x(x, y, 4)))


def U_y_plotting(x, y):
    return tf.where(x > ksi,
                    tf.where(y > zeta, U_y(x, y, 2), U_y(x, y, 3)),
                    tf.where(y > zeta, U_y(x, y, 1), U_y(x, y, 4)))


"""
methods for gradients of U
"""


def U_x(x, y, p):
    with tf.GradientTape(watch_accessed_variables=False) as g:
        g.watch(x)
        u = U(x, y, p)
    u_x = g.gradient(u, x)
    return u_x


def U_y(x, y, p):
    with tf.GradientTape(watch_accessed_variables=False) as g:
        g.watch(y)
        u = U(x, y, p)
    u_y = g.gradient(u, y)
    return u_y


def U_xx(x, y, p):
    with tf.GradientTape(watch_accessed_variables=False) as g:
        g.watch(x)
        u_x = U_x(x, y, p)
    u_xx = g.gradient(u_x, x)
    return u_xx


def U_yy(x, y, p):
    with tf.GradientTape(watch_accessed_variables=False) as g:
        g.watch(y)
        u_y = U_y(x, y, p)
    u_yy = g.gradient(u_y, y)
    return u_yy


"""
Definition of losses

For future code optimize: move to PDE definition file
"""


def loss_omega_boundary(omega_boundary_points_p, p):
    x = omega_boundary_points_p[:, 0:1]
    y = omega_boundary_points_p[:, 1:2]
    return tf.reduce_mean(tf.square(
        U(x, y, p) - true_sol(x, y, p)))


def loss_jump(gamma_points_p, p):
    x = gamma_points_p[:, 0:1]
    y = gamma_points_p[:, 1:2]
    if p == 1:
        jump_predicted = U(x, y, 2) - U(x, y, 1)
    if p == 2:
        jump_predicted = U(x, y, 3) - U(x, y, 4)
    if p == 3:
        jump_predicted = U(x, y, 2) - U(x, y, 3)
    if p == 4:
        jump_predicted = U(x, y, 1) - U(x, y, 4)
    return tf.reduce_mean(tf.square(
        jump_predicted - phi(x, y, p)))


def loss_neumann(gamma_points_p, p):
    x = gamma_points_p[:, 0:1]
    y = gamma_points_p[:, 1:2]
    if p == 1:
        neumann_predicted = a(x, y, 2) * U_x(x, y, 2) - a(x, y, 1) * U_x(x, y, 1)
    if p == 2:
        neumann_predicted = a(x, y, 3) * U_x(x, y, 3) - a(x, y, 4) * U_x(x, y, 4)
    if p == 3:
        neumann_predicted = a(x, y, 2) * U_y(x, y, 2) - a(x, y, 3) * U_y(x, y, 3)
    if p == 4:
        neumann_predicted = a(x, y, 1) * U_y(x, y, 1) - a(x, y, 4) * U_y(x, y, 4)
    return tf.reduce_mean(tf.square(
        neumann_predicted - psi(x, y, p)))


def loss_domain(omega_points, p):
    x = omega_points[:, 0:1]
    y = omega_points[:, 1:2]
    return tf.reduce_mean(tf.square(
        equation(x, y, U, p)))


# Training Settings
def train_step():
    gradients = None
    for i in range(1):  # for gradient accumulate, actual batch_size=range*Num
        with tf.GradientTape(watch_accessed_variables=False, persistent=True) as g:
            # training=True is only needed if there are layers with different
            # behavior during training versus inference (e.g. Dropout).
            g.watch(Model.trainable_variables)
            loss_ob = tf.reduce_sum(
                [loss_omega_boundary(omega_boundary_points[p], p) for p in range(1, 5)])  # omega_boundary
            loss_d = tf.reduce_sum([loss_domain(omega_points[p], p) for p in range(1, 5)])
            loss_j = tf.reduce_sum([loss_jump(gamma_points[p], p) for p in range(1, 5)])
            loss_n = tf.reduce_sum([loss_neumann(gamma_points[p], p) for p in range(1, 5)])
            train_loss = weight['omega boundary'] * loss_ob + weight['domain'] * loss_d + weight['jump'] * loss_j + \
                         weight['neumann'] * loss_n
        if gradients is not None:
            gradients += g.gradient(train_loss, Model.trainable_variables)
        else:
            gradients = g.gradient(train_loss, Model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, Model.trainable_variables))
    loss_j, loss_n, loss_d = [tf.squeeze(_).numpy() for _ in [loss_j, loss_n, loss_d]]
    return {'omega boundary': loss_ob, 'domain': loss_d, 'jump': loss_j,
            'neumann': loss_n,
            'train': train_loss}


"""
test settings

For future code optimize: use package functions
"""


def test_step():
    # infinite norm
    test_loss_infinite = tf.reduce_max([tf.reduce_max(tf.abs(
        true_sol(test_points[p][:, 0:1], test_points[p][:, 1:2], p) - U(test_points[p][:, 0:1], test_points[p][:, 1:2],
                                                                        p))) for p in range(1, 5)]) \
                         / tf.reduce_max([tf.reduce_max(tf.abs(
        true_sol(test_points[p][:, 0:1], test_points[p][:, 1:2], p))) for p in range(1, 5)])
    # 2nd norm
    test_loss_2nd = tf.reduce_sum([tf.reduce_sum(tf.square(
        true_sol(test_points[p][:, 0:1], test_points[p][:, 1:2], p) - U(test_points[p][:, 0:1], test_points[p][:, 1:2],
                                                                        p))) for p in range(1, 5)]) \
                    / tf.reduce_sum([tf.reduce_sum(tf.square(
        true_sol(test_points[p][:, 0:1], test_points[p][:, 1:2], p))) for p in range(1, 5)])
    # infinite norm of gradient
    test_loss_grad_infinite = tf.reduce_max([
        tf.reduce_max([tf.reduce_max(tf.abs(
            true_sol_x(test_points[p][:, 0:1], test_points[p][:, 1:2], p) - U_x(test_points[p][:, 0:1],
                                                                                test_points[p][:, 1:2], p))) for p in
            range(1, 5)]),
        tf.reduce_max([tf.reduce_max(tf.abs(
            true_sol_y(test_points[p][:, 0:1], test_points[p][:, 1:2], p) - U_y(test_points[p][:, 0:1],
                                                                                test_points[p][:, 1:2], p))) for p in
            range(1, 5)])
    ]) \
                              / tf.reduce_max([
        tf.reduce_max([tf.reduce_max(tf.abs(
            true_sol_x(test_points[p][:, 0:1], test_points[p][:, 1:2], p))) for p in range(1, 5)]),
        tf.reduce_max([tf.reduce_max(tf.abs(
            true_sol_y(test_points[p][:, 0:1], test_points[p][:, 1:2], p))) for p in range(1, 5)])
    ])

    # 2nd norm of gradient
    test_loss_grad_2nd = tf.reduce_sum([
        tf.reduce_sum([tf.reduce_sum(tf.square(
            true_sol_x(test_points[p][:, 0:1], test_points[p][:, 1:2], p) - U_x(test_points[p][:, 0:1],
                                                                                test_points[p][:, 1:2], p))) for p in
            range(1, 5)]),
        tf.reduce_sum([tf.reduce_sum(tf.square(
            true_sol_y(test_points[p][:, 0:1], test_points[p][:, 1:2], p) - U_y(test_points[p][:, 0:1],
                                                                                test_points[p][:, 1:2], p))) for p in
            range(1, 5)])
    ]) \
                         / tf.reduce_sum([
        tf.reduce_sum([tf.reduce_sum(tf.square(
            true_sol_x(test_points[p][:, 0:1], test_points[p][:, 1:2], p))) for p in range(1, 5)]),
        tf.reduce_sum([tf.reduce_sum(tf.square(
            true_sol_y(test_points[p][:, 0:1], test_points[p][:, 1:2], p))) for p in range(1, 5)])
    ])
    return {"relatively_infinite_loss": test_loss_infinite,
            "relatively_2nd_loss": test_loss_2nd,
            "relatively_gradient_infinite_loss": test_loss_grad_infinite,
            "relatively_gradient_2nd_loss": test_loss_grad_2nd}


"""
training parameters settings
EPOCHS: max iteration number, but usually view output results during training and KeyboardInterrupt
weight: reweighting is useless, but I kept this choice
NumOmega, NumBoundary: Number of points in training
epsilon_r, epsilon_p: threshold in SAIS
Warning: 
In SAIS mode, NumOmega isn't actual number of collocation points, since SAIS will endlessly add points. 
So I use a threshold, which is 5000 here, if the number of points is too big, then randomly abandon half of it.
"""
EPOCHS = 100000
weight = {'omega boundary': 1, 'domain': 1, 'jump': 1, 'neumann': 1, 'train': 1}
NumOmega = 1024
NumBoundary = 1024
epsilon_r = 10
epsilon_p = 0.1
losses = {}

for epoch in range(EPOCHS):
    # initial training points
    if epoch == 0:
        gamma_points, omega_points, omega_boundary_points, test_points = DataGenerateCross.generate_points(
            xmin, xmax, ymin, ymax, ksi, zeta, NumGamma=NumBoundary, NumOmegaBoundary=NumBoundary, NumOmega=NumOmega,
            ShowPoints=False, SaveFig=False)
    """
    perform SAIS
    
    Future code optimize: move to SAIS file
    """
    if epoch % 10 == 0:
        print("resampling...")
        # drop omega points if too much
        num_current_collection_points = sum([tf.shape(omega_points[p])[0] for p in range(1, 5)])
        print("Current number of collection points: ", num_current_collection_points)
        if num_current_collection_points > 5000:
            for p in range(1, 5):
                omega_points[p] = tf.random.shuffle(omega_points[p])
                omega_points[p] = tf.slice(omega_points[p], [0, 0], [int(tf.shape(omega_points[p])[0] / 2), 2])


        # using SAIS to sample omega_points
        def g(points):
            points = tf.cast(points, dtype=tf.float32)
            x = points[:, 0:1]
            y = points[:, 1:2]
            result = tf.where(x > ksi,
                              tf.where(y > zeta, loss_domain(points, 2), loss_domain(points, 3)),
                              tf.where(y > zeta, loss_domain(points, 1), loss_domain(points, 4))) \
                     - epsilon_r
            result = result.numpy()[:, 0]
            return result


        P_F, adaptive_omega_points = SAIS(g)
        if P_F < epsilon_p:
            print("P_F small enough")
        adaptive_omega_points = tf.cast(adaptive_omega_points, dtype=tf.float32)
        for p in range(1, 5):
            x = adaptive_omega_points[:, 0]
            y = adaptive_omega_points[:, 1]
            if p == 1:
                condition = tf.logical_and(x < ksi, y > zeta)
            if p == 2:
                condition = tf.logical_and(x > ksi, y > zeta)
            if p == 3:
                condition = tf.logical_and(x > ksi, y < zeta)
            if p == 4:
                condition = tf.logical_and(x < ksi, y < zeta)
            omega_points[p] = tf.concat((tf.boolean_mask(adaptive_omega_points, condition), omega_points[p]), axis=0)

            if epoch % 10 == 0:
                plt.scatter(omega_points[p][:, 0], omega_points[p][:, 1], s=1)
        if epoch % 10 == 0:
            plt.savefig(os.path.join(PDEname, 'points_for_solving_PDE ' + str(epoch) + ".jpg"))
            print("points_for_solving_PDE saved")
            plt.clf()

    # picture result output
    if epoch % 100 == 0:
        if true_sol is not None:
            test_loss = test_step()
            print('Epoch', epoch, end=' ')
            for name, loss in test_loss.items():
                if name not in losses.keys():
                    losses[name] = []
                condtional_print.conditional_print(losses[name], test_loss[name], name)
                losses[name].append(test_loss[name])
            print('')

        print("saving model...")
        Model.save(os.path.join(PDEname, PDEname + " " + str(epoch) + ".h5"))
        print("model saved")

        print("plotting pictures...")
        plotting.plot_2D_result(PDEname, U_plotting, true_sol_plotting, xmin, xmax, ymin, ymax, epoch,
                                figName='U', max_batch_size=NumOmega)
        # plotting.plot_3D_result(PDEname, U_plotting, true_sol_plotting, xmin, xmax, ymin, ymax, epoch,figName='U')
        # plotting.plot_2D_result(PDEname, U_x_plotting, true_sol_x_plotting, xmin, xmax, ymin, ymax, epoch, figName='Ux')
        # # # plotting.plot_3D_result(PDEname, U_x, true_sol_x, xmin, xmax, ymin, ymax, epoch, figName='Ux')
        # plotting.plot_2D_result(PDEname, U_y_plotting, true_sol_y_plotting, xmin, xmax, ymin, ymax, epoch, figName='Uy')
        # # plotting.plot_3D_result(PDEname, U_xx, true_sol_xx, xmin, xmax, ymin, ymax, epoch, figName='Uxx')

    # calculate train loss
    train_loss = train_step()
    if epoch % 1 == 0:
        print('Epoch', epoch, end=' ')
        for name, loss in train_loss.items():
            if name not in losses.keys():
                losses[name] = []
            condtional_print.conditional_print(losses[name], train_loss[name], name)
            losses[name].append(train_loss[name])
        print('')
