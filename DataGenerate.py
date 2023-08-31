# -*- coding:utf-8 -*-
"""
Author:Alarak
Date:2023/07/20
Basic version of DataGenerate
"""
import numpy as np
import tensorflow as tf
import math
import matplotlib.pyplot as plt
pi = math.pi
def generate_points(xmin,xmax,ymin,ymax,gamma=None,gamma_r=None,InnerPointsExcludeGamma=False,NumGammaBoundary=256, NumBoundary=256, NumOmega=1024,ShowPoints=False,
    SaveFig=False):
    #Generate gamma_boundary_points & gamma_normal_vector
    if gamma is not None:
        gamma_boundary_points = []
        for theta in np.linspace(0, 2 * pi, NumGammaBoundary, endpoint=False):
            for r in np.linspace(0, 2, 1000):
                x = r * np.cos(theta)
                y = r * np.sin(theta)
                if gamma(x, y) > 0:
                    gamma_boundary_points.append([x, y])
                    break

        gamma_boundary_points = np.array(gamma_boundary_points)
        gamma_normal_vector = []
        for i in range(gamma_boundary_points.shape[0]):
            k = i + 1
            if i == gamma_boundary_points.shape[0] - 1:
                k = 0
            dx, dy = (gamma_boundary_points[k, :] - gamma_boundary_points[i, :])
            n = [dy, -dx] / np.sqrt(dx ** 2 + dy ** 2)
            gamma_normal_vector.append(n)
        gamma_normal_vector = np.array(gamma_normal_vector)
        gamma_boundary_points, gamma_normal_vector=[tf.cast(_, tf.float32) for _ in [gamma_boundary_points, gamma_normal_vector]]

    elif gamma_r is not None:
        gamma_boundary_points = []
        for theta in np.linspace(0, 2 * pi, NumGammaBoundary, endpoint=False):
            r=gamma_r(theta)
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            gamma_boundary_points.append([x, y])

        gamma_boundary_points = np.array(gamma_boundary_points)
        gamma_normal_vector = []
        for i in range(gamma_boundary_points.shape[0]):
            k = i + 1
            if i == gamma_boundary_points.shape[0] - 1:
                k = 0
            dx, dy = (gamma_boundary_points[k, :] - gamma_boundary_points[i, :])
            n = [dy, -dx] / np.sqrt(dx ** 2 + dy ** 2)
            gamma_normal_vector.append(n)
        gamma_normal_vector = np.array(gamma_normal_vector)
        gamma_boundary_points, gamma_normal_vector = [tf.cast(_, tf.float32) for _ in
                                                      [gamma_boundary_points, gamma_normal_vector]]
    else:
        gamma_boundary_points=None
        gamma_normal_vector=None
    
    
    omega_points = np.vstack(
        (np.random.uniform(xmin, xmax, NumOmega), np.random.uniform(ymin, ymax, NumOmega))).T
    if InnerPointsExcludeGamma==True:
        infinitesimal=1e-3
        omega_points= omega_points[np.where(abs(gamma(omega_points[:,0],omega_points[:,1]))>infinitesimal)]
    test_points = np.vstack(
        (np.random.uniform(xmin, xmax, NumOmega), np.random.uniform(ymin, ymax, NumOmega))).T
    gamma_1 = np.vstack((xmin * np.ones(NumBoundary), np.random.uniform(ymin, ymax, NumBoundary))).T
    gamma_2 = np.vstack((xmax * np.ones(NumBoundary), np.random.uniform(ymin, ymax, NumBoundary))).T
    gamma_3 = np.vstack((np.random.uniform(xmin, xmax, NumBoundary), ymin * np.ones(NumBoundary))).T
    gamma_4 = np.vstack((np.random.uniform(xmin, xmax, NumBoundary), ymax * np.ones(NumBoundary))).T
    if gamma is not None:
        inner_points = omega_points[(np.where(gamma(omega_points[:, 0], omega_points[:, 1]) < 0))]
        outer_points = omega_points[(np.where(gamma(omega_points[:, 0], omega_points[:, 1]) > 0))]
        inner_points,outer_points = [tf.cast(_, tf.float32) for _ in
            [inner_points,outer_points]]
    else:
        inner_points=None
        outer_points=None
    omega_points, test_points, gamma_1, gamma_2, gamma_3, gamma_4 = [tf.cast(_, tf.float32) for _ in
                                                                                         [omega_points, test_points,
                                                                                          gamma_1, gamma_2, gamma_3,
                                                                                          gamma_4]]


    if ShowPoints==True:
        plt.scatter(omega_points[:, 0], omega_points[:, 1], s=1)
        plt.scatter(gamma_1[:, 0], gamma_1[:, 1], s=1)
        plt.scatter(gamma_2[:, 0], gamma_2[:, 1], s=1)
        plt.scatter(gamma_3[:, 0], gamma_3[:, 1], s=1)
        plt.scatter(gamma_4[:, 0], gamma_4[:, 1], s=1)
        if gamma is not None:
            plt.scatter(gamma_boundary_points[:, 0], gamma_boundary_points[:, 1], s=1)
        plt.show()
    if SaveFig==True:
        plt.savefig('points_for_solving_PDE.jpg')


    return omega_points, test_points, gamma_1, gamma_2, gamma_3, gamma_4, gamma_boundary_points, gamma_normal_vector,inner_points,outer_points



