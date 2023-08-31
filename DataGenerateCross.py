# -*- coding:utf-8 -*-
"""
Author:Alarak
Date:2023/07/20

Generate training points for Cross problems
"""
import numpy as np
import tensorflow as tf
import math
import matplotlib.pyplot as plt
pi = math.pi
def generate_points(xmin,xmax,ymin,ymax,ksi,zeta,NumGamma=256,NumOmegaBoundary=256, NumOmega=1024,ShowPoints=False,
    SaveFig=False):
    #Generate gamma_points
    gamma_points={}
    num=int(NumGamma/4)
    gamma_points[1] = np.vstack((ksi * np.ones(num), np.random.uniform(zeta, ymax, num))).T
    gamma_points[2] = np.vstack((ksi * np.ones(num), np.random.uniform(ymin, zeta, num))).T
    gamma_points[3] = np.vstack((np.random.uniform(ksi, xmax, num), zeta * np.ones(num))).T
    gamma_points[4] = np.vstack((np.random.uniform(xmin, ksi, num), zeta * np.ones(num))).T
    
    omega_points={}
    num = int(NumOmega/ 4)
    omega_points[1] = np.vstack(
        (np.random.uniform(xmin, ksi, num), np.random.uniform(zeta, ymax, num))).T
    omega_points[2] = np.vstack(
        (np.random.uniform(ksi, xmax, num), np.random.uniform(zeta, ymax, num))).T
    omega_points[3] = np.vstack(
        (np.random.uniform(ksi, xmax, num), np.random.uniform(ymin, zeta, num))).T
    omega_points[4] = np.vstack(
        (np.random.uniform(xmin, ksi, num), np.random.uniform(ymin, zeta, num))).T
    
    omega_boundary_points={}
    num = int(NumOmegaBoundary/8)
    omega_boundary_points[1] = np.vstack((np.vstack(
        (xmin* np.ones(num), np.random.uniform(zeta, ymax, num))).T,
                                np.vstack(
        (np.random.uniform(xmin, ksi, num), ymax* np.ones(num))).T))
    
    omega_boundary_points[2] = np.vstack((np.vstack(
        (xmax * np.ones(num), np.random.uniform(zeta, ymax, num))).T,
                                np.vstack(
                                    (np.random.uniform(ksi, xmax, num), ymax * np.ones(num))).T))
    
    omega_boundary_points[3] = np.vstack((np.vstack(
        (xmax * np.ones(num), np.random.uniform(ymin, zeta, num))).T,
                                np.vstack(
                                    (np.random.uniform(ksi, xmax, num), ymin * np.ones(num))).T))
    
    omega_boundary_points[4] = np.vstack((np.vstack(
        (xmin * np.ones(num), np.random.uniform(ymin, zeta, num))).T,
                                np.vstack(
                                    (np.random.uniform(xmin, ksi, num), ymin * np.ones(num))).T))

    test_points = {}
    num = int(NumOmega / 4)
    test_points[1] = np.vstack(
        (np.random.uniform(xmin, ksi, num), np.random.uniform(zeta, ymax, num))).T
    test_points[2] = np.vstack(
        (np.random.uniform(ksi, xmax, num), np.random.uniform(zeta, ymax, num))).T
    test_points[3] = np.vstack(
        (np.random.uniform(ksi, xmax, num), np.random.uniform(ymin, zeta, num))).T
    test_points[4] = np.vstack(
        (np.random.uniform(xmin, ksi, num), np.random.uniform(ymin, zeta, num))).T
    
    
    for points_set in [gamma_points,omega_points,omega_boundary_points,test_points]:
        for key,subpoints_set in points_set.items():
            points_set[key]=tf.cast(subpoints_set, tf.float32)
            if ShowPoints == True:
                plt.scatter(subpoints_set[:, 0], subpoints_set[:, 1], s=1)

    if ShowPoints == True:
        plt.show()
    if SaveFig==True:
        plt.savefig('points_for_solving_PDE.jpg')


    return gamma_points,omega_points,omega_boundary_points,test_points


