# -*- coding:utf-8 -*-
"""
Author:Alarak
Date:2023/07/20

plot 2D, 3D results
1: predicted solution
2: true solution
3: error
"""
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.ticker import FuncFormatter
import os


def plot_result(PDEname,U,true_sol,xmin,xmax,ymin,ymax,epoch,figName='',plot_2D=True,plot_3D=True,
                resolution=256,max_batch_size=128):# Need to be power of 2
    # Plot the solution over a square grid with 100 points per wavelength in each direction
    Nx = resolution
    Ny = Nx
    num_batches=int(resolution**2/max_batch_size)

    # Grid points

    plot_grid = np.mgrid[xmin: xmax: Nx * 1j, ymin: ymax: Ny * 1j]
    points = np.vstack(
        (plot_grid[0].ravel(), plot_grid[1].ravel(), np.zeros(plot_grid[0].size))
    )

    points_2d = points[:2, :]
    points_2d_x = tf.constant(points_2d.T[:, 0:1], dtype=tf.float32)
    points_2d_y = tf.constant(points_2d.T[:, 1:2], dtype=tf.float32)
    # Split points_2d_x and points_2d_y into 10 smaller tensors each
    points_2d_x_split = tf.split(points_2d_x, num_or_size_splits=num_batches, axis=0)
    points_2d_y_split = tf.split(points_2d_y, num_or_size_splits=num_batches, axis=0)

    # Loop over the smaller tensors and call U on each pair
    u_pred_split = []
    for x, y in zip(points_2d_x_split, points_2d_y_split):
        # print(x,y,U(x,y))
        u_pred_split.append(U(x, y))

    # Concatenate the results to get the final u_pred
    u_pred = tf.concat(u_pred_split, axis=0).numpy()
    u_pred = u_pred.reshape((Nx, Ny))


    if plot_2D:
        plt.rc("font", family="serif", size=22)

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=(36, 12))




        if true_sol is not None:
            u_exact = true_sol(points_2d_x, points_2d_y).numpy()
            u_exact = u_exact.reshape((Nx, Ny))
            u_diff = u_exact - u_pred
            matrix = np.fliplr(u_exact).T

            pcm2 = ax2.imshow(
                matrix,
                extent=[xmin, xmax, ymin, ymax],
                cmap=plt.cm.get_cmap("seismic"),
                interpolation="spline16",
                label="Exact",
            )
            fig.colorbar(pcm2, ax=ax2)
            ax2.set_title("Exact")


            """
            plotting error
            made some adjust to plot 0 (no error) in white (color bar: red-white-blue)"""
            matrix = np.fliplr(u_diff).T
            matrix_normalized = np.where(matrix > 0, matrix/matrix.max(), -matrix/matrix.min())

            pcm3 = ax3.imshow(
                matrix_normalized,
                extent=[xmin, xmax, ymin, ymax],
                cmap=plt.cm.get_cmap("bwr"),
                interpolation="spline16",
                label="Error",
            )

            cbar = fig.colorbar(pcm3, ax=ax3)
            cbar.set_ticks([-1,-0.5,0,0.5,1])
            cbar.set_ticklabels([matrix.min(), matrix.min()/2, 0, matrix.max()/2, matrix.max()])
            ax3.set_title("Error")

        matrix = np.fliplr(u_pred).T
        pcm1 = ax1.imshow(
            matrix,
            extent=[xmin, xmax, ymin, ymax],
            cmap=plt.cm.get_cmap("seismic"),
            interpolation="spline16",
            label="PINN",
        )

        fig.colorbar(pcm1, ax=ax1)
        ax1.set_title("PINNs")

        plt.savefig(os.path.join(PDEname, figName + "2D " + str(epoch) + ".jpg"))
        print(figName + "2D Successfully printed")

        plt.close()



    if plot_3D:
        Fig = plt.figure(figsize=(12, 12))
        ax = Fig.add_subplot(1, 1, 1, projection='3d')
        matrix = np.fliplr(u_pred).T
        X, Y = np.meshgrid(np.linspace(xmin, xmax, Nx), np.linspace(ymin, ymax, Ny))
        ax.plot_surface(X, Y, matrix, cmap=plt.cm.get_cmap("seismic"))
        plt.savefig(os.path.join(PDEname, figName + "3DPred" + str(epoch) + ".jpg"))
        print(figName + "3D Successfully printed")

        plt.close()

        if true_sol is not None:
            u_exact = true_sol(points_2d_x, points_2d_y).numpy()
            u_exact = u_exact.reshape((Nx, Ny))
            u_diff = u_exact - u_pred
            Fig = plt.figure(figsize=(12, 12))
            ax = Fig.add_subplot(1, 1, 1, projection='3d')
            matrix = np.fliplr(u_exact).T
            X, Y = np.meshgrid(np.linspace(xmin, xmax, Nx), np.linspace(ymin, ymax, Ny))
            ax.plot_surface(X, Y, matrix, cmap=plt.cm.get_cmap("seismic"))
            plt.savefig(os.path.join(PDEname, figName + "3DTrue" + str(epoch) + ".jpg"))
            print(figName + "3D Successfully printed")

            plt.close()

            Fig = plt.figure(figsize=(12, 12))
            ax = Fig.add_subplot(1, 1, 1, projection='3d')
            matrix = np.fliplr(u_diff).T
            X, Y = np.meshgrid(np.linspace(xmin, xmax, Nx), np.linspace(ymin, ymax, Ny))
            ax.plot_surface(X, Y, matrix, cmap=plt.cm.get_cmap("seismic"))
            plt.savefig(os.path.join(PDEname, figName + "3DDiff" + str(epoch) + ".jpg"))
            print(figName + "3D Successfully printed")

            plt.close()


def plot_2D_result(PDEname,U,true_sol,xmin,xmax,ymin,ymax,epoch,figName='',
                resolution=256,max_batch_size=128):
    plot_result(PDEname,U,true_sol,xmin,xmax,ymin,ymax,epoch,figName=figName,plot_2D=True,plot_3D=False,
                resolution=resolution,max_batch_size=max_batch_size)


def plot_3D_result(PDEname,U,true_sol,xmin,xmax,ymin,ymax,epoch,figName='',
                resolution=256,max_batch_size=128):
    plot_result(PDEname,U,true_sol,xmin,xmax,ymin,ymax,epoch,figName=figName,plot_2D=False,plot_3D=True,
                resolution=resolution,max_batch_size=max_batch_size)