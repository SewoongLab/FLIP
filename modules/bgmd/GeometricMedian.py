"""
Implementation GeometricMedian lightly modified by Rishi Jha.
Original credit goes to Anish Acharya

# Copyright (c) Anish Acharya.
# Licensed under the MIT License
"""

import numpy as np
import time
from typing import List
from scipy.spatial.distance import cdist, euclidean


class GeometricMedian():
    def __init__(self, geo_med_alg='vardi', eps=1e-5, max_iter=500):
        self.geo_med_alg = geo_med_alg
        self.eps = eps
        self.max_iter = max_iter

    def get_gm(self, X: np.ndarray):

        if self.geo_med_alg == 'vardi':
            gm = self.vardi(X=X, eps=self.eps, max_iter=self.max_iter)
        elif self.geo_med_alg == 'wzfld':
            gm = self.weiszfeld(X=X, eps=self.eps, max_iter=self.max_iter)
        elif self.geo_med_alg == 'cvx_opt':
            gm = self.cvx_opt(X=X, eps=self.eps, max_iter=self.max_iter)
        else:
            raise NotImplementedError

        return gm

    def aggregate(self, G: np.ndarray, ix: List[int] = None) -> np.ndarray:
        # if ix given only aggregate along the indexes ignoring the rest of the ix
        if ix is not None:
            g_agg = np.zeros_like(G[0, :])
            G = G[:, ix]
            low_rank_gm = self.get_gm(X=G)
            g_agg[ix] = low_rank_gm
            return g_agg
        else:
            return self.get_gm(X=G)

    # ------------------------------------ #
    # Different GM Algorithms implemented  #
    # ------------------------------------ #

    def vardi(self, X, eps, max_iter) -> np.ndarray:
        # Copyright (c) Orson Peters
        # Licensed under zlib License
        # Reference: https://stackoverflow.com/questions/30299267/geometric-median-of-multidimensional-points
        """
        Implementation of "The multivariate L1-median and associated data depth;
        Yehuda Vardi and Cun-Hui Zhang; PNAS'2000"
        """
        # initial guess
        t0 = time.time()
        mu = np.mean(X, 0)
        mu = np.nan_to_num(mu, copy=False, nan=0, posinf=0, neginf=0)
        num_iter = 1
        while num_iter < max_iter:
            # noinspection PyTypeChecker
            D = cdist(X, [mu]).astype(mu.dtype)
            non_zeros = (D != 0)[:, 0]
            D_inv = 1 / D[non_zeros]
            W = np.divide(D_inv, sum(D_inv))
            T = np.sum(W * X[non_zeros], 0)
            num_zeros = len(X) - np.sum(non_zeros)

            if num_zeros == 0:
                mu1 = T
            elif num_zeros == len(X):
                self.agg_time = time.time() - t0
                self.num_iter = num_iter
                return mu
            else:
                r = np.linalg.norm((T - mu) * sum(D_inv))
                r_inv = 0 if r == 0 else num_zeros / r
                mu1 = max(0, 1 - r_inv) * T + min(1, r_inv) * mu

            mu1 = np.nan_to_num(mu1, copy=False, nan=0, posinf=0, neginf=0)

            if euclidean(mu, mu1) < eps:
                self.agg_time = time.time() - t0
                self.num_iter = num_iter
                return mu

            mu = mu1
            num_iter += 1

        self.agg_time = time.time() - t0
        self.num_iter = num_iter
        print('Ran out of Max iter for GM - returning hem zero')
        return np.zeros_like(mu)

    def weiszfeld(self, X, eps, max_iter):
        # inspired by: https://github.com/mrwojo
        """
        Implements: On the point for which the sum of the distances to n given points is minimum
        E Weiszfeld, F Plastria: Annals of Operations Research
        """
        # initial Guess : centroid / empirical mean
        t0 = time.time()
        mu = np.mean(X, 0)
        num_iter = 0
        while num_iter < max_iter:
            # noinspection PyTypeChecker
            distances = cdist(X, [mu]).astype(mu.dtype)
            distances = np.where(distances == 0, 1, distances)
            mu1 = (X / distances).sum(axis=0) / (1. / distances).sum(axis=0)
            guess_movement = np.sqrt(((mu - mu1) ** 2).sum())

            mu = mu1
            if guess_movement <= eps:
                self.agg_time = time.time() - t0
                return mu
            num_iter += 1

        self.agg_time = time.time() - t0
        print('Ran out of Max iter for GM - returning sub optimal answer')
        return mu

    def cvx_opt(self, X, eps=1e-5, max_iter=1000):
        raise NotImplementedError