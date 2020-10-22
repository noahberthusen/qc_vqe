import os
import sys
import numpy as np
import pdb
from numpy import linalg as LA
import copy
from numpy import random


def prepare_Q_matrix(h1,eri,n,ne):

    #define the initial slater determinat
    #psi = np.random.rand(n,ne)
    '''
    Declare initial state
    '''
    h_init = copy.deepcopy(h1)
    E,V = LA.eigh(h_init)
    energy = 2*sum(E[:int(n/2)])
    psi = V[:,:int(n/2)]

    psi_d = np.transpose(np.conjugate(psi))
    rho = np.matmul(psi,psi_d)
    e_prev = E[0]
    eps = 1e-12
    error = 100000
    while error > eps:
        HU = np.zeros((n,n))
        for i in range(n):
            HU[i,i] += 0.5*eri[i][i][i][i]*(rho[i,i])

        HK = np.zeros((n,n))
        for i in range(n):
            for j in range(n):
                HK[i,j] = h1[i,j]
        H = HK+HU
        E,V = LA.eigh(H)
        rho_prev = copy.deepcopy(rho)
        psi = V[:,:int(n/2)]
        psi_d = np.transpose(np.conjugate(psi))
        rho = np.matmul(psi,psi_d)
        error = sum(sum(abs(rho-rho_prev)))
        e_prev = E[0]
        energy = 2*sum(E[:int(n/2)])
        print(repr(psi_d))
    print("Converged Energy = "+str(energy))
    return psi_d

def prepare_general_Q_matrix(h1,eri,n,n2):
    A = np.random.rand(n,n)
    At = np.transpose(A)
    E,V = LA.eigh(A+At)
    psi = V[:,:int(n/2)]
    psi_d = np.transpose(np.conjugate(psi))
    print("Q-matrix")
    print(psi_d)
    rho = np.matmul(psi,psi_d)
    HU = np.zeros((n,n))
    for i in range(n):
        HU[i,i] += 0.5*eri[i][i][i][i]*(rho[i,i])
    HK = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            #HK[i,j] = h1[i,j]*rho[i,j]
            HK[i,j] = h1[i,j]
    H = HK+HU
    '''
    Compute energy expectation <psi|H|psi>
    '''
    print("Energy")
    print(2*np.trace(np.matmul(rho,H)) )
    return psi_d


if __name__ == "__main__":
    # 1D Hubbard model at half filling
    n = 2
    ne = n
    h1 = np.zeros((n, n))
    for i in range(n-1):
        h1[i, i+1] = h1[i+1, i] = -1.0
    # h1 = np.array([ [ 0., -1.,  0., -0.99],
    #                 [-1.,  0., -0.99,  0.],
    #                 [ 0., -0.99,  0., -1.],
    #                 [-0.99,  0., -1.,  0.]])
    print(h1)
    eri = np.zeros((n, n, n, n))
    for i in range(n):
        if (False):
            eri[i, i, i, i] = 2.0
        else:
            eri[i, i, i, i] = 1.0
    prepare_Q_matrix(h1,eri,n,ne)
