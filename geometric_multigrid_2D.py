# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 22:44:53 2023

@author: meryame.boudhar & ugo.pelissier
"""

#-----------------------------------------------------------------------------#
# IMPORT DEPENDENCIES
#-----------------------------------------------------------------------------#
from math import pi, sin, cos
import pylab as plt
import numpy as np
import scipy as sp
import scipy.sparse as spa
import scipy.linalg as la
from scipy import sparse
import pyamg
import time
from mpl_toolkits import mplot3d

#-----------------------------------------------------------------------------#
# MULTIGRID CYCLE
#-----------------------------------------------------------------------------#
def discrete_grid():
    fig, ax = plt.subplots()
    
    N=10
    h=1/N
    
    x=np.arange(0,1.0001,h)
    y=np.arange(0,1.0001,h)
    colors = ['k']*len(x)
    
    ax.set_xlim((0,1))
    ax.set_ylim((0,1))
    ax.set_xticks(x)
    ax.set_yticks(y)
    
    x0,x1 = ax.get_xlim()
    y0,y1 = ax.get_ylim()
    ax.set_aspect(abs(x1-x0)/abs(y1-y0))
    
    ax.grid(visible=True, which = "both", color='k')
    for i in range(len(x)):
        ax.scatter(np.array([x[i]]*len(x)), y, c=colors, marker=".")
        
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(r"$\bf{\Omega_h}$")
    fig.savefig('discrete_grid.png', dpi=600)
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

def laplace(n,sigma,h,epsilon):
    """ Construct the anisotropic 2D laplace operator """
    A=np.zeros((n*n,n*n))
    
    # DIAGONAL            
    for i in range (0,n):
        for j in range (0,n):           
            A[i+n*j,i+n*j]=2*(1+epsilon)+sigma*h*h
    
    # LOWER DIAGONAL        
    for i in range (1,n):
        for j in range (0,n):           
            A[i+n*j,i+n*j-1]=-epsilon   
    # UPPPER DIAGONAL        
    for i in range (0,n-1):
        for j in range (0,n):           
            A[i+n*j,i+n*j+1]=-epsilon  
    
    # LOWER IDENTITY MATRIX
    for i in range (0,n):
        for j in range (1,n):           
            A[i+n*j,i+n*(j-1)]=-1        
            
            
    # UPPER IDENTITY MATRIX
    for i in range (0,n):
        for j in range (0,n-1):           
            A[i+n*j,i+n*(j+1)]=-1
            
    return A

def plot_laplace(n,sigma,h,epsilon):
    """ Plot the laplace matrix coefficients for clarity """
    A = laplace(n,sigma,h,epsilon)
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
    fig.savefig('laplace_matrix.png', format='png', dpi=900)
    plt.show()
    
def initial_guess(xih,yih,n_inc_h):
    u0 = np.zeros((xih.shape[0],yih.shape[0]))
    for i in range(len(xih)):
        for j in range(len(yih)):
            u0[i,j] = 0.5 * (np.sin(5. * xih[i] * pi) * np.sin(5. * yih[j] * pi))
    u0 = u0.reshape(n_inc_h*n_inc_h)
    return u0

def JOR(A, b, x0, omega, eps, maxiter):
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

def SOR(A, b, x0, omega, eps, maxiter):
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

def restriction(fine,nh,nH,option):
    """ 
    3 options of restriction: injection, half-weighting and full-weighting
    """
    fine = fine.reshape((nh,nh))
    coarse = np.zeros((nH,nH))
    
    if (option == "injection"):
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
        print("restriction option not allowed. Try one of [injection, half-weighting, full-weighting]")
    return coarse.reshape(nH*nH)

def interpolation(coarse,n_inc_H,fine,n_inc_h):
    """ Classical multi-linear interpolation """
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

def mgcyc(strategy, l, gamma, nsegment, u0, b, f, sigma, epsilon, engine, n1, n2, eps, omega):
    """ 
    Multi grid cycle:
        - nsegment: the number of segment so that h = 1.0/nsegment
        - engine: the stationary iterative method used for smoothing
    """
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
    
    # Construction of Laplace operator
    Ah = (1./(h*h)) * laplace(n_inc_h,sigma,h,epsilon)
    AH = (1./(H*H)) * laplace(n_inc_H,sigma,H,epsilon)
    
    # RHS
    if(b is None):
        b = np.zeros((xih.shape[0],yih.shape[0]))
        b = f(xih,yih)
        b = b.reshape(n_inc_h*n_inc_h)
        
    # Init
    uh = np.zeros((xih.shape[0],yih.shape[0]))
    uh = uh.reshape(n_inc_h*n_inc_h)
    
    if(u0 is None):
        u0 = initial_guess(xih,yih,n_inc_h)
        # plot(xih,yih,u0,title="Initial")
                
    # Pre-smoothing Relaxation
    uh, dh, _ = engine(Ah, b, u0, omega=omega, eps=eps, maxiter=n1)

    # Restriction
    dH = restriction(dh,n_inc_h,n_inc_H,option=strategy)
        
    # Solve
    vH = np.zeros(dH.shape)
    if(l==1):
        vH = np.dot(np.linalg.inv(AH),dH)
    else:
        for j in range(gamma):
            _, _, vH, _ = mgcyc(strategy, l=l-1, gamma=gamma, nsegment=n_inc_H+1, u0=vH, b=dH, f=f, sigma=sigma, epsilon=epsilon, engine=engine, n1=n1, n2=n2, eps=eps, omega=omega)
            
    # Prolongation
    vh = np.zeros(uh.shape)
    vh = interpolation(vH,n_inc_H,vh,n_inc_h)
        
    # Update solution
    uh += vh
        
    # Post-smoothing Relaxation
    uh, dh, _ = engine(Ah, b, x0=uh, omega=omega, eps=eps, maxiter=n2)

    label = "$(\gamma, l) = ($" + str(gamma) + "," + str(l) + ")"
    
    return xih, yih, uh, dh
    
def v_cycle_res(strategy, l, nsegment, u0, b, f, sigma, epsilon, engine, n1, n2, eps, omega):
    """ 
    Run multiple iterations of V-cycle and return an array containing the norm of the residual at each iteration
    To perform sensitivity study for a parameter, it can be passed as a list (epsilon in this case).
    """
    gamma = 1
    res = []
    
    if (type(epsilon)==int):
        xih, yih, uh, dh = mgcyc(strategy, l, gamma, nsegment, u0, b, f, sigma, epsilon, engine, n1, n2, eps, omega)
        res.append(np.linalg.norm(dh))
        for i in range(20):
            xih, yih, uh, dh = mgcyc(strategy, l, gamma, nsegment, uh, b, f, sigma, epsilon, engine, n1, n2, eps, omega)
            res.append(np.linalg.norm(dh))
            
    elif(type(epsilon)==list):
        for elt in epsilon:
            temp_res = []
            xih, yih, uh, dh = mgcyc(strategy, l, gamma, nsegment, u0, b, f, sigma, elt, engine, n1, n2, eps, omega)
            temp_res.append(np.linalg.norm(dh))
            for i in range(20):
                xih, yih, uh, dh = mgcyc(strategy, l, gamma, nsegment, uh, b, f, sigma, elt, engine, n1, n2, eps, omega)
                temp_res.append(np.linalg.norm(dh))
            res.append(temp_res)
            
    print("\nV-cycle residual done.")
            
    return np.array(res)

def w_cycle_res(strategy, l, nsegment, u0, b, f, sigma, epsilon, engine, n1, n2, eps, omega):
    """ 
    Run multiple iterations of W-cycle and return an array containing the norm of the residual at each iteration
    To perform sensitivity study for a parameter, it can be passed as a list (epsilon in this case).
    """
    gamma = 2
    res = []
    
    if (type(epsilon)==int):
        xih, yih, uh, dh = mgcyc(strategy, l, gamma, nsegment, u0, b, f, sigma, epsilon, engine, n1, n2, eps, omega)
        res.append(np.linalg.norm(dh))
        for i in range(20):
            xih, yih, uh, dh = mgcyc(strategy, l, gamma, nsegment, uh, b, f, sigma, epsilon, engine, n1, n2, eps, omega)
            res.append(np.linalg.norm(dh))
            
    elif(type(epsilon)==list):
        for elt in epsilon:
            temp_res = []
            xih, yih, uh, dh = mgcyc(strategy, l, gamma, nsegment, u0, b, f, sigma, elt, engine, n1, n2, eps, omega)
            temp_res.append(np.linalg.norm(dh))
            for i in range(20):
                xih, yih, uh, dh = mgcyc(strategy, l, gamma, nsegment, uh, b, f, sigma, elt, engine, n1, n2, eps, omega)
                temp_res.append(np.linalg.norm(dh))
            res.append(temp_res)
            
    print("\nW-cycle residual done.")
            
    return np.array(res)

#-----------------------------------------------------------------------------#
# POST-PROCESS
#-----------------------------------------------------------------------------#
def plot(x,y,z,title):
    """ 3D plot of field z(x,y) """
    if not (z.shape == (x.shape[0],y.shape[0])):
        z = z.reshape(x.shape[0],y.shape[0])
    xs, ys = np.meshgrid(x, y)
    
    fig = plt.figure()
    ax = plt.axes(projection='3d')    
    ax.plot_surface(xs, ys, z, cmap='viridis')
    
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_zlabel('$u(x,y)$')
    
    plt.title(r"$\bf{" + title + "}$")
    # fig.savefig(title+'.png', format='png', dpi=900)
    plt.show()
    
def plot_initial_final(x,y,n_segment,uh,title):
    """ 3D plot of the initial guess compared to the final solution """
    n_inc_h = nsegment-1
    u0 = initial_guess(x,y,n_inc_h)
    u0 = u0.reshape(x.shape[0],y.shape[0])
    
    if not (uh.shape == (x.shape[0],y.shape[0])):
        uh = uh.reshape(x.shape[0],y.shape[0])
        
    xs, ys = np.meshgrid(x, y)
    
    fig = plt.figure()
    ax = plt.axes(projection='3d')    
    surf_u0 = ax.plot_surface(xs, ys, u0, alpha=0.35, label="Initial guess")
    surf_uh = ax.plot_surface(xs, ys, uh, alpha=1, label="Final solution")
    
    surf_u0._facecolors2d=surf_u0._facecolor3d
    surf_u0._edgecolors2d=surf_u0._edgecolor3d
    
    surf_uh._facecolors2d=surf_uh._facecolor3d
    surf_uh._edgecolors2d=surf_uh._edgecolor3d
    
    ax.legend()
    
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_zlabel('$u(x,y)$')
    
    # plt.title(r"$\bf{" + title + "}$")
    fig.savefig('initial_final_comp.png', format='png', dpi=900)
    plt.show()

def post_process_mgcyc(strategy, l, gamma, nsegment, u0, b, f, sigma, epsilon, engine, n1, n2, eps, omega):
    """
    Plot the final solution and the comparison of this latter with the initial guess
    """
    start = time.time()
    xih, yih, uh, dh = mgcyc(strategy, l, gamma, nsegment, u0, b, f, sigma, epsilon, engine, n1, n2, eps, omega)
    end = time.time()
    
    # plot(xih,yih,uh,title="Solution")
    plot_initial_final(xih,yih,nsegment,uh,"Solution")
    
    print('\nResidual - ||r|| = {:.2f}'.format(np.linalg.norm(dh)))
    
    print('\nMultigrid cycle took {:.2f}s to compute.'.format(end - start))

def plot_res_cycle(res,epsilon):
    """
    Plot the norm of the residual as a function of the iteration for various values of one paramater
    """
    m,n = res.shape
    it = range(1,n)
    col = ["blue", "red", "black"]
    for i in range(m):        
        plt.plot(it, res[i][0:n-1], 'o', color=col[i])
        plt.plot(it, res[i][0:n-1], '--', color=col[i], label="$\epsilon$ = {}".format(epsilon[i]))
        
    plt.xticks(it)    
    plt.xlabel("Iterations")
    
    plt.yscale("log")
    plt.ylabel("||r||")
    
    plt.legend()
    plt.title("W-cycle convergence")
    plt.savefig('w_cycle_res.png', format='png', dpi=900)
    plt.show()
    
def compare_plot_res_cycle(res_v,res_w,epsilon):
    """
    Plot the norm of the residual as a function of the iteration for various values of one paramater for both V-cycle and W-cycle
    """
    m,n = res_v.shape
    index = [0,2]
    it = range(1,n)
    col_v = ["turquoise", "lightseagreen"]
    j = 0
    for i in index:        
        plt.plot(it, res_v[i][0:n-1], 'o', color=col_v[j])
        plt.plot(it, res_v[i][0:n-1], '--', color=col_v[j], label="V-cycle - l={}".format(epsilon[i]))
        j += 1
        
    col_w = ["chocolate", "saddlebrown"]
    j = 0
    for i in index:        
        plt.plot(it, res_w[i][0:n-1], 'o', color=col_w[j])
        plt.plot(it, res_w[i][0:n-1], '--', color=col_w[j], label="W-cycle - l= {}".format(epsilon[i]))
        j += 1
        
    plt.xticks(it)    
    plt.xlabel("Iterations")
    
    plt.yscale("log")
    plt.ylabel("||r||")
    
    plt.legend()
    plt.title("Cycles convergence")
    plt.savefig('cycle_res_comp.png', format='png', dpi=900)
    plt.show()

#-----------------------------------------------------------------------------#
# PARAMETERS
#-----------------------------------------------------------------------------#
# Smoothing solver
eps =1e-12
omega_JOR = 0.5
omega_SOR = 1.5

# Smoothing iterations
engine=JOR
omega = omega_JOR
n1 = 5
n2 = 5

# PDE
sigma = 0
epsilon = 1

def f(xih,yih):
    """ Construct the RHS """
    b = np.zeros((xih.shape[0],yih.shape[0]))
    return b

def source(xih,yih):
    """ Construct the RHS """
    b = np.zeros((xih.shape[0],yih.shape[0]))
    value = 1e5
    b[len(xih)//4,len(xih)//4] = value
    b[len(xih)//4,3*len(xih)//4] = value
    b[3*len(xih)//4,len(xih)//4] = value
    b[3*len(xih)//4,3*len(xih)//4] = value
    return b

# Grid Cycle
l = 4
gamma = 2
nsegment = 64
u0 = None
b = None
strategy = "injection"

#-----------------------------------------------------------------------------#
# COMPUTATION
#-----------------------------------------------------------------------------#
# plot_laplace(n=10,sigma=sigma,h=0.1,epsilon=epsilon)   
    
post_process_mgcyc(strategy, l, gamma, nsegment, u0, b, f, sigma, epsilon, engine, n1, n2, eps, omega)

# res_v = v_cycle_res(strategy, l, nsegment, u0, b, f, sigma, epsilon, engine, n1, n2, eps, omega)
# res_w = w_cycle_res(strategy, l, nsegment, u0, b, f, sigma, epsilon, engine, n1, n2, eps, omega)

#-----------------------------------------------------------------------------#
# Coarsening Strategies
#-----------------------------------------------------------------------------#

n = 100
a = laplace(n=n ,sigma=1,h=0.1,epsilon=0.01)
A =spa.csr_matrix(a)
ml = pyamg.ruge_stuben_solver(A) # Ruge Stuben 
#ml = pyamg.smoothed_aggregation_solver(A) # Smooth Aggregation
#ml = pyamg.aggregation.rootnode_solver(A)

print(ml)
b = np.random.rand(A.shape[0])                      
x = ml.solve(b, tol=1e-10)                          
print("residual: ", np.linalg.norm(b-A*x)) 


#-----------------------------------------------------------------------------#
# Compatible Relaxation
#-----------------------------------------------------------------------------#

n = 20
b = laplace(n=n ,sigma=1,h=0.1,epsilon=0.01)
B =spa.csr_matrix(b)

xx = np.linspace(0,1,n)
x,y = np.meshgrid(xx,xx)
V = np.concatenate([[x.ravel()],[y.ravel()]],axis=0).T

splitting = pyamg.classical.cr.CR(B)

C = np.where(splitting == 0)[0]
F = np.where(splitting == 1)[0]

fig, ax = plt.subplots()
ax.scatter(V[C, 0], V[C, 1], marker='s', s=18,
           color=[232.0/255, 74.0/255, 39.0/255], label='C-pts')
ax.scatter(V[F, 0], V[F, 1], marker='s', s=18,
           color=[19.0/255, 41.0/255, 75.0/255], label='F-pts')
plt.legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
           borderaxespad=0, ncol=2)

ax.axis('square')
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')


plt.show()