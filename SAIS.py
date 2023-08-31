# -*- coding:utf-8 -*-
"""
Author:Alarak
Date:2023/08/25

Self-adaptive Importance Sampling
"""
import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
import os


def SAIS(g,omega=multivariate_normal(mean=[0.5,0.5],cov=[[1,0],[0,1]]),N1=300, N2=1000, p0=0.1, M=10):
    """
    :param N1: Number of samples
    :param N2: Number of samples
    :param p0:
    :param g: LSF
    :param omega: prior distribution, an scipy gauss distribution object
    :param M: maximum updated number
    :return P_SIAS_F: estimator of the failure probability
    :return D_adaptive:
    """
    # Number of samples N1 and N2, the parameter p0, the LSF g, the prior distribution omega and the maximum updated number M.
    h = omega # set h to omega
    for k in range(M):
        S0 = h.rvs(N1) # generate N1 samples from h, shape=(N1,2)
        mask = (S0[:, 0] >= 0) & (S0[:, 0] <= 1) & (S0[:, 1] >= 0) & (S0[:, 1] <= 1)
        S0 = S0[mask]
        g_S0 = g(S0)
        S0 = S0[np.argsort(-g_S0,axis=0)]  # sort the samples according to g in descending order, candidate_dataset

        N_eta = np.count_nonzero(g_S0 > 0)  # count the number of elements that are greater than 0
        N_p = int(p0 * N1) # calculate N_p
        if N_eta < N_p: # if condition
          mu = np.mean(S0[:N_p], axis=0) # calculate mu using Eq.(10)
          sigma = np.cov(S0[:N_p], rowvar=False) # calculate sigma using Eq.(10)
          h = multivariate_normal(mu, sigma) # set h to NT(mu, sigma)
        else: # else condition
          break # break the loop
    mu_opt = np.sum(S0[:N_eta]*np.expand_dims(omega.pdf(S0[:N_eta]),axis=1), axis=0)/ np.sum(omega.pdf(S0[:N_eta]), axis=0)# calculate mu_opt using Eq.(11)
    sigma_opt = np.cov(S0[:N_eta], rowvar=False) # calculate sigma_opt using Eq.(11)
    h_opt = multivariate_normal(mu_opt,sigma_opt)
    S = h_opt.rvs(N2) # generate N2 samples from NT(mu_opt, sigma_opt)
    mask = (S[:, 0] >= 0) & (S[:, 0] <= 1) & (S[:, 1] >= 0) & (S[:, 1] <= 1)
    S = S[mask]
    g_S = g(S)  # apply g to the whole array
    omega_S = omega.pdf(S)  # apply omega.pdf to the whole array
    h_opt_S = h_opt.pdf(S)  # apply h_opt.pdf to the whole array
    P_SIAS_F = np.mean(omega_S[g_S > 0] / h_opt_S[g_S > 0])  # calculate P_SIAS_F using Eq.(12)

    # n = np.count_nonzero(g_S > 0)  # 计算 g_S 中大于零的元素的个数
    # if n > np.shape(S)[0] / 2:  # 如果大于零的元素超过 S 的一半
    #     idx = np.argsort(g_S)[::-1]  # 按 g_S 从大到小排序，得到索引数组
    #     D_adaptive = S[idx[:n // 2]]  # 取 S 中前一半的元素，按 g_S 的顺序
    # else:
    #     D_adaptive = S[g_S > 0]  # 否则，取 S 中 g_S 大于零的元素
    D_adaptive = S[g_S > 0]
    return P_SIAS_F, D_adaptive # return the results
