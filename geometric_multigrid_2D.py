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
from mpl_toolkits import mplot3d

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
    plt.title('Discrete Grid $\Omega_h$')
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

def f(xih,yih):
    b = np.zeros((xih.shape[0],yih.shape[0]))
    b[len(xih)//4,len(xih)//4] = 10000
    b[len(xih)//4,3*len(xih)//4] = 10000
    b[3*len(xih)//4,len(xih)//4] = 10000
    b[3*len(xih)//4,3*len(xih)//4] = 10000
    return b

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

def injection(fine,nh,nH,option):
    """ 
    3 options of injection: classical, half-weighting and full-weighting
    """
    fine = fine.reshape((nh,nh))
    coarse = np.zeros((nH,nH))
    
    if (option is None):
        k = 0
        l=0
        for i in range(1,nh,2):
            for j in range(1,nh,2):
                coarse[k,l] = fine[i,j]
                l+=1
            k+=1
            l=0
    elif (option == "half-weighting"):      
        M = np.array([
            [0,1,0],
            [1,4,1],
            [0,1,0]])/8
        k = 0
        l=0
        for i in range(1,nh,2):
            for j in range(1,nh,2):
                coarse[k,l] = np.sum(M*fine[i-1:i+2,j-1:j+2])
                l+=1
            k+=1
            l=0
    elif (option == "full-weighting"):
        M = np.array([
            [1,2,1],
            [2,4,2],
            [1,2,1]])/16
        k = 0
        l=0
        for i in range(1,nh,2):
            for j in range(1,nh,2):
                coarse[k,l] = np.sum(M*fine[i-1:i+2,j-1:j+2])
                l+=1
            k+=1
            l=0
    else:
        print("Injection option not allowed. Try one of [None, half-weighting, full-weighting]")
    return coarse.reshape(nH*nH)

def interpolation(coarse,n_inc_H,fine,n_inc_h):
    """ 
    Classical linear interpolation
    """
    
    coarse_boundary = np.zeros((n_inc_H+2)**2)
    for i in range(1,n_inc_H+1):
        coarse_boundary[i*(n_inc_H+2)+1:i*(n_inc_H+2)+1+n_inc_H] = coarse[(i-1)*n_inc_H:i*n_inc_H]
    coarse_boundary = coarse_boundary.reshape((n_inc_H+2),(n_inc_H+2))
    
    fine_boundary = np.zeros(((n_inc_h+2),(n_inc_h+2)))
    
    k=0
    l=0
    for i in range(0,(n_inc_h),2):
        for j in range(0,(n_inc_h),2):
            
            # Known values of the fine grid
            if(i==0 or j==0):
                if(i==0 and j==0):
                    fine_boundary[i,j] = coarse_boundary[k,l]
                    fine_boundary[i,j+2] = coarse_boundary[k,l+1]
                    fine_boundary[i+2,j] = coarse_boundary[k+1,l]
                    fine_boundary[i+2,j+2] = coarse_boundary[k+1,l+1]
                elif(i==0 and j!=0):
                    fine_boundary[i,j+2] = coarse_boundary[k,l+1]
                    fine_boundary[i+2,j+2] = coarse_boundary[k+1,l+1]
                elif(i!=0 and j==0):
                    fine_boundary[i+2,j] = coarse_boundary[k+1,l]
                    fine_boundary[i+2,j+2] = coarse_boundary[k+1,l+1]
            else:
                fine_boundary[i+2,j+2] = coarse_boundary[k+1,l+1]
            l+=1
            
            # Interpolation
            fine_boundary[i,j+1] = (fine_boundary[i,j]+fine_boundary[i,j+2])/2
            fine_boundary[i+1,j] = (fine_boundary[i,j]+fine_boundary[i+2,j])/2
            fine_boundary[i+1,j+2] = (fine_boundary[i,j+2]+fine_boundary[i+2,j+2])/2
            fine_boundary[i+2,j+1] = (fine_boundary[i+2,j]+fine_boundary[i+2,j+2])/2
            fine_boundary[i+1,j+1] = (fine_boundary[i,j]+fine_boundary[i+2,j]+fine_boundary[i,j+2]+fine_boundary[i+2,j+2])/4
            
        k+=1
        l=0
    
    fine = fine_boundary[1:n_inc_h+1,1:n_inc_h+1].reshape(n_inc_h*n_inc_h)
    
    return fine

def plot(x,y,z):
    if not (z.shape == (x.shape[0],y.shape[0])):
        z = z.reshape(x.shape[0],y.shape[0])
    xs, ys = np.meshgrid(x, y)
    fig = plt.figure()
    ax = plt.axes(projection='3d')    
    ax.plot_surface(xs, ys, z, cmap='viridis')
    plt.show()

def mgcyc(l, gamma, nsegment, u0, b, f, engine=JOR, n1=20, n2=20):
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
    # b = f(xih,yih)
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
                u0[i,j] = 0.5 * (np.sin(5. * xih[i] * pi) * np.sin(5. * yih[j] * pi))
    u0 = u0.reshape(n_inc_h*n_inc_h)
    plot(xih,yih,u0)
                
    # Pre-smoothing Relaxation
    uh, dh, _ = engine(Ah, b, u0, omega=0.5, eps=_eps, maxiter=n1)

    # Restriction with injection
    dH = injection(dh,n_inc_h,n_inc_H,option=None)
        
    # Solve
    vH = np.zeros(dH.shape)
    if(l==1):
        vH = np.dot(np.linalg.inv(AH),dH)
    else:
        for j in range(gamma):
            vH = mgcyc(l-1, gamma, nsegment=n_inc_H+1, u0=vH, b=dH, f=None, engine=JOR, n1=n1, n2=n2)
            
    # Prolongation
    vh = np.zeros(uh.shape)
    vh = interpolation(vH,n_inc_H,vh,n_inc_h)
        
    # Update solution
    uh += vh
        
    # Post-smoothing Relaxation
    uh, dh, _ = engine(Ah, b, x0=uh, omega=0.5, eps=_eps, maxiter=n2)

    label = "$(\gamma, l) = ($" + str(gamma) + "," + str(l) + ")"
    
    return xih, yih, uh

def time_mgcyc(l, gamma, nsegment, u0, b, f):
    start = time.time()
    
    xih, yih, uh = mgcyc(l, gamma, nsegment, u0, b, f)
    plot(xih,yih,uh)
    
    end = time.time()
    print('\nMultigrid cycle took {:.2f}s to compute.'.format(end - start))
    
# discrete_grid()    

# plot_laplace(n=10,sigma=0,h=0)
    
time_mgcyc(l=1, gamma=1, nsegment=64, u0=None, b=None, f=f)