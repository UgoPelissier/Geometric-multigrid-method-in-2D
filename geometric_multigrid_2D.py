# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 22:44:53 2023

@author: ugo.pelissier
"""

from math import pi, sin, cos
import pylab as plt
import numpy as np
import scipy as sp
import scipy.sparse as spa
import scipy.linalg as la
import time

# Some paramters
_eps =1e-12
_maxiter=500

def discrete_grid():
    N=10
    h=1/N
    x=np.arange(0,1.0001,h)
    y=np.arange(0,1.0001,h)
    X, Y = np.meshgrid(x, y)
    fig = plt.figure()
    plt.plot(x[1],y[1],'ro',label='Unknown');
    plt.plot(X,Y,'ro');
    plt.plot(np.ones(N+1),y,'go',label='Boundary Condition');
    plt.plot(np.zeros(N+1),y,'go');
    plt.plot(x,np.zeros(N+1),'go');
    plt.plot(x, np.ones(N+1),'go');
    plt.xlim((-0.1,1.1))
    plt.ylim((-0.1,1.1))
    plt.xlabel('x')
    plt.ylabel('y')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.title(r'Discrete Grid $\Omega_h,$ h= %s'%(h),fontsize=24,y=1.08)
    plt.show()

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

def laplace(n,sigma,h):
    """ Construct the 2D laplace operator """
    A=np.zeros((n*n,n*n))
    
    # DIAGONAL            
    for i in range (0,n):
        for j in range (0,n):           
            A[i+n*j,i+n*j]=4+sigma*h*h
    
    # LOWER DIAGONAL        
    for i in range (1,n):
        for j in range (0,n):           
            A[i+n*j,i+n*j-1]=-1   
    # UPPPER DIAGONAL        
    for i in range (0,n-1):
        for j in range (0,n):           
            A[i+n*j,i+n*j+1]=-1   
    
    # LOWER IDENTITY MATRIX
    for i in range (0,n):
        for j in range (1,n):           
            A[i+n*j,i+n*(j-1)]=-1        
            
            
    # UPPER IDENTITY MATRIX
    for i in range (0,n):
        for j in range (0,n-1):           
            A[i+n*j,i+n*(j+1)]=-1
            
    return A

def plot_laplace(n,sigma,h):
    A = laplace(n,sigma,h)
    Ainv=np.linalg.inv(A)   
    fig = plt.figure(figsize=(12,4));
    plt.subplot(121)
    plt.imshow(A,interpolation='none');
    clb=plt.colorbar();
    clb.set_label('Matrix elements values');
    plt.title('Matrix A ',fontsize=24)
    plt.subplot(122)
    plt.imshow(Ainv,interpolation='none');
    clb=plt.colorbar();
    clb.set_label('Matrix elements values');
    plt.title(r'Matrix $A^{-1}$ ',fontsize=24)
    
    fig.tight_layout()
    plt.show()

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

def injection(fine,nh,nH):
    """ 
    Classical injection, that only keep the value of the coarse nodes 
    """
    coarse = np.zeros(nH*nH)
    k = 0
    for i in range(0,nH):
        for j in range(0,nH):
            coarse[k] = fine[(2*i+1)*nh + (2*j+1)]
            k += 1
    return coarse

def interpolation(coarse, fine):
    """ 
    Classical linear interpolation
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

def mgcyc(l, gamma, nsegment, u0, b, engine=JOR, n1=5, n2=5):
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
    
    yh = np.linspace(0.,1., n) 
    yH = np.linspace(0.,1., n//2 + 1)
     
    # Inner points
    xih = xh[1:-1]
    xiH = xH[1:-1]
    
    yih = xh[1:-1]
    yiH = xH[1:-1]
    
    # construction of Laplace operator 
    Ah = (1./(h*h)) * laplace(n_inc_h,sigma=0,h=h)
    AH = (1./(H*H)) * laplace(n_inc_H,sigma=0,h=H)
    
    # RHS
    if(b is None):
        b = np.zeros((xih.shape[0],yih.shape[0]))
        b = b.reshape(n_inc_h*n_inc_h)
        
    # Init
    uh = np.zeros((xih.shape[0],yih.shape[0]))
    uh = uh.reshape(n_inc_h*n_inc_h)
    
    if(u0 is None):
        u0 = np.zeros((xih.shape[0],yih.shape[0]))
        for i in range(len(xih)):
            for j in range(len(yih)):
                u0[i,j] = 0.5 * (np.sin(16. * xih[i] * pi) + np.sin(40. * yih[j] * pi))
    u0 = u0.reshape(n_inc_h*n_inc_h)
                
    # Pre-smoothing Relaxation
    uh, dh, _ = engine(Ah, b, u0, omega=0.5, eps=_eps, maxiter=n1)

    # Restriction with injection
    dH = injection(dh,n_inc_h,n_inc_H)
        
    # Solve
    vH = np.zeros(dH.shape)
    if(l==1):
        vH = np.dot(np.linalg.inv(AH),dH)
    else:
        for j in range(gamma):
            vH = mgcyc(l-1, gamma, nsegment=n_inc_H+1, u0=vH, b=dH, engine=JOR, n1=n1, n2=n2)
            
    # Prolongation
    vh = np.zeros(uh.shape)
    # vh = interpolation(vH,vh)
        
    # # Update solution
    # uh += vh
        
    # # Post-smoothing Relaxation
    # uh, dh, _ = JOR(Ah, b, x0=uh, omega=0.5, eps=_eps, maxiter=n2)

    # label = "$(\gamma, l) = ($" + str(gamma) + "," + str(l) + ")"
    # plot(xih, uh,'-x', label=label)
    # # title = str(l) + " levels multigrid method"
    # # plt.title(title)
        
    # plt.show()
    
    return uh

def time_mgcyc(l, gamma, nsegment, u0, b):
    start = time.time()
    
    mgcyc(l, gamma, nsegment, u0, b)
    
    end = time.time()
    print('\nMultigrid cycle took {:.2f}s to compute.'.format(end - start))
    
discrete_grid()    

plot_laplace(n=10,sigma=0,h=0)
    
time_mgcyc(1, 1, nsegment=6, u0=None, b=None)