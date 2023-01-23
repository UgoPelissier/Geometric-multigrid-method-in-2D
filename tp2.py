#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 16:27:31 2020

@author: christophe
"""

from math import pi, sin, cos
import pylab as plt
import numpy as np
import scipy as sp
import scipy.sparse as spa
import scipy.linalg as la
import time
#from matplotlib import rc, rcParams
#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
#rc('text', usetex=True)
#rcParams.update({'font.size': 16})

# Some paramters
_eps =1e-12
_maxiter=500

def _basic_check(A, b, x0):
    """ Common check for clarity """
    n, m = A.shape
    if(n != m):
        raise ValueError("Only square matrix allowed")
    if(b.size != n):
        raise ValueError("Bad rhs size")
    if (x0 is None):
        x0 = np.zeros(n)
    if(x0.size != n):
        raise ValueError("Bad initial value size")
    return x0


def laplace(n):
    """ Construct the 1D laplace operator """
    k = [-np.ones(n-1),2*np.ones(n),-np.ones(n-1)]
    offset = [-1,0,1]
    A = spa.diags(k,offset).toarray()
    return A


def JOR(A, b, x0=None, omega=0.5, eps=_eps, maxiter=_maxiter):
    """
    Methode itérative stationnaire de sur-relaxation (Jacobi over relaxation)
    Convergence garantie si A est à diagonale dominante stricte
    A = D - E - F avec D diagonale, E (F) tri inf. (sup.) stricte
    Le préconditionneur est diagonal M = (1./omega) * D

    Output:
        - x is the solution at convergence or after maxiter iteration
        - residual_history is the norm of all residuals
    """
    x = _basic_check(A, b, x0)
    r = 1e3*np.ones(x.shape)
    residual_history = np.array([])
    
    M = (1/omega) * np.diag(A.diagonal())
    M_inv = np.linalg.inv(M)
    
    i=0
    while ( (i<maxiter) and (np.linalg.norm(r)>eps) ) :
        r = b-np.dot(A,x)
        residual_history = np.append(residual_history, np.linalg.norm(r))
        z = np.dot(M_inv,r)
        x += z
        i += 1
    r = b-np.dot(A,x)
        
    return x, r, residual_history


def SOR(A, b, x0=None, omega=1.5, eps=_eps, maxiter=_maxiter):
    """
    Methode itérative stationnaire de sur-relaxation successive
    (Successive Over Relaxation)

    A = D - E - F avec D diagonale, E (F) tri inf. (sup.) stricte
    Le préconditionneur est tri. inf. M = (1./omega) * D - E

    * Divergence garantie pour omega <= 0. ou omega >= 2.0
    * Convergence garantie si A est symétrique définie positive pour
    0 < omega  < 2.
    * Convergence garantie si A est à diagonale dominante stricte pour
    0 < omega  <= 1.

    Output:
        - x is the solution at convergence or after maxiter iteration
        - residual_history is the norm of all residuals

    """
    if (omega > 2.) or (omega < 0.):
        raise ArithmeticError("SOR will diverge")

    x = _basic_check(A, b, x0)
    r = 1e3*np.ones(x.shape)
    residual_history = list()
    
    D = (1/omega) * np.diag(A.diagonal())
    E = - np.tril(A)-np.diag(np.diag(A))
    M = D-E
    M_inv = np.linalg.inv(M)
    
    i=0
    while ( (i<maxiter) and (np.linalg.norm(r)>eps) ) :
        r = b-np.dot(A,x)
        residual_history.append(np.linalg.norm(r))
        z = np.dot(M_inv,r)
        x += z
        i += 1
        
    return x, r, residual_history


def injection(fine):
    """ 
    Classical injection, that only keep the value of the coarse nodes 
    The modification of coarse is done inplace
    """
    return fine[1::2]

def interpolation(coarse, fine):
    """ 
    Classical linear interpolation (the modification of fine is done inplace)
    """
    fine[0] = coarse[0]/2
    fine[len(fine)-1] = coarse[len(coarse)-1]/2
    
    i = 1
    j = 0
    while ( (i<len(fine)-1) and (j<len(coarse-1)) ):
        if(i%2==0):
            fine[i] = (coarse[j] + coarse[j+1]) / 2
            j += 1
        else:
            fine[i] = coarse[j]
        i += 1
    return fine

def plot(x, y, custom, label=""):
    """ 
    A custom plot function, usage: 
        plot(x, y,'-x', label="u")
    """
    plt.plot(x, y, label=label);     
    plt.xlabel(r"x")    
    plt.ylabel(r"u(x)")
    plt.legend()
    
def smoother(nsegment):

    if(nsegment%2): raise ValueError("nsegment must be even")
    
    # Beware that : nsegment
    # n = number of nodes 
    # n_inc = number of unknowns 
    n = nsegment + 1    
    h = 1.0 / nsegment
    
    k = nsegment//4
    
    # Full points
    xh = np.linspace(0.,1., n) 
     
    # Inner points
    xih = xh[1:-1]
    
    # construction of Laplace operator 
    Ah = laplace(nsegment-1)

    it = [10, 20, 30]
    
    for i in it:
        # Initial guess    
        u0 = 0.5 * (np.sin(k * xih * pi))
        u = np.zeros(u0.shape)
        
        # Solve
        u, residual_history = JOR(Ah, u, u0, omega=0.5, eps=_eps, maxiter=i)
        x = np.concatenate((np.array([0]), xih, np.array([1])), axis=0)
        u = np.concatenate((np.array([0]), u, np.array([0])), axis=0)
        plot(x, u,'-x', label=str(i))
    plt.show()

def tgcyc(nsegment, b, engine=JOR, n1=5, n2=5):
    """ 
    Two grid cycle:
        - nsegment: the number of segment so that h = 1.0/nsegment
        - engine: the stationary iterative method used for smoothing 
    
    Warning: make the good distinction between the number of segments, the 
    number of nodes and the number of unknowns
    """
    start = time.time()
    
    if(nsegment%2): raise ValueError("nsegment must be even")
    
    # Beware that : nsegment
    # n = number of nodes 
    # n_inc = number of unknowns 
    n = nsegment + 1    
    h = 1.0 / nsegment
    H = 2. * h
    
    n_inc_h = n-2
    n_inc_H = (n - 2) // 2
    
    # Full points
    xh = np.linspace(0.,1., n) 
    xH = np.linspace(0.,1., n//2 + 1)
     
    # Inner points
    xih = xh[1:-1]
    xiH = xH[1:-1]
    
    # Construction of Laplace operator 
    Ah = (1./(h*h)) * laplace(n_inc_h)
    AH = (1./(H*H)) * laplace(n_inc_H)
    
    # RHS
    if(b is None):
        b = np.zeros(xih.shape)
    
    # Initial guess    
    u0 = 0.5 * (np.sin(16. * xih * pi) + np.sin(40. * xih * pi))    
    uh = np.zeros(u0.shape)
    
    plt.plot(xih, u0,'-', label="Initial guess")
    # plt.show()
        
    # Pre-smoothing Relaxation
    uh, dh, _ = engine(Ah, b, u0, omega=0.5, eps=_eps, maxiter=n1)
    
    plt.plot(xih, u0,'-', label="Pre smoothing")
    # plt.show()

    # Restriction with injection
    dH = np.zeros(len(dh))
    dH = injection(dh)
        
    # Solve on the coarse grid
    vH = np.zeros(dH.shape)
    vH = np.dot(np.linalg.inv(AH),dH)
            
    # Prolongation
    vh = np.zeros(uh.shape)
    vh = interpolation(vH,vh)
        
    # Update solution
    uh += vh
    
    plt.plot(xih, uh,'-', label="Coarse grid solve")
    # plt.show()
        
    # Post-smoothing Relaxation
    uh, dh, _ = engine(Ah, b, x0=uh, omega=0.5, eps=_eps, maxiter=n2)
    
    plt.plot(xih, uh,'-', label="Post smoothing")
    plt.legend()
    plt.show()
    
    # Initial guess    
    u0 = 0.5 * (np.sin(16. * xih * pi) + np.sin(40. * xih * pi))    
    uh = np.zeros(u0.shape)
        
    # Pre-smoothing Relaxation
    uh, dh, _ = engine(Ah, b, u0, omega=0.5, eps=_eps, maxiter=n1+3)
        
    # Post-smoothing Relaxation
    uh, dh, _ = JOR(Ah, b, x0=uh, omega=0.5, eps=_eps, maxiter=n2)
    
    # plot(xih, uh,'-x', label="3 iterations of smoothing")
    
    end = time.time()
    print('\nTwo grid cycle took {:.2f}s to compute.'.format(end - start))
    
    return uh

def mgcyc(lmax, l, gamma, nsegment, u0, b, engine=JOR, n1=5, n2=5):
    """ 
    Multi grid cycle:
        - nsegment: the number of segment so that h = 1.0/nsegment
        - engine: the stationary iterative method used for smoothing 
    
    Warning: make the good distinction between the number of segments, the 
    number of nodes and the number of unknowns
    """
    start = time.time()
    
    if(nsegment%2): raise ValueError("nsegment must be even")
    
    # Beware that : nsegment
    # n = number of nodes 
    # n_inc = number of unknowns 
    n = nsegment + 1    
    h = 1.0 / nsegment
    H = 2. * h
    
    n_inc_h = nsegment-1
    n_inc_H = (nsegment - 1) // 2
    
    # Full points
    xh = np.linspace(0.,1., n) 
    xH = np.linspace(0.,1., n//2 + 1)
     
    # Inner points
    xih = xh[1:-1]
    xiH = xH[1:-1]
    
    # construction of Laplace operator 
    Ah = (1./(h*h)) * laplace(n_inc_h)
    AH = (1./(H*H)) * laplace(n_inc_H)
    
    # RHS
    if(b is None):
        b = np.zeros(xih.shape)
        
    # Init
    uh = np.zeros(xih.shape)
    
    for i in range(gamma):
        if(i==0):
            # Initial guess
            if(u0 is None):
                u0 = 0.5 * (np.sin(16. * xih * pi) + np.sin(40. * xih * pi))
                # plot(xih, u0,'-x', label="Initial guess")
        else:
            u0 = uh
        
        # Pre-smoothing Relaxation
        uh, dh, _ = engine(Ah, b, u0, omega=0.5, eps=_eps, maxiter=n1)
    
        # Restriction with injection
        dH = np.zeros(len(dh))
        dH = injection(dh)
            
        # Solve
        vH = np.zeros(dH.shape)
        if(l==1):
            vH = np.dot(np.linalg.inv(AH),dH)
        else:
            for j in range(gamma):
                vH = mgcyc(lmax, l-1, gamma, nsegment=n_inc_H+1, u0=vH, b=dH, engine=JOR, n1=n1, n2=n2)
                
        # Prolongation
        vh = np.zeros(uh.shape)
        vh = interpolation(vH,vh)
            
        # Update solution
        uh += vh
            
        # Post-smoothing Relaxation
        uh, dh, _ = JOR(Ah, b, x0=uh, omega=0.5, eps=_eps, maxiter=n2)
    
        if(l==lmax):
            label = "$\gamma = $" + str(i)
            plot(xih, uh,'-x', label=label)
            title = str(l) + " levels multigrid method"
            plt.title(title)
            
        plt.show()
    
    if(l==lmax):
        end = time.time()
        print('\nMultigrid cycle took {:.2f}s to compute.'.format(end - start))
    
    return uh
    
tgcyc(nsegment=64, b=None)
mgcyc(lmax=2, l=2, gamma=2, nsegment=64, u0=None, b=None)