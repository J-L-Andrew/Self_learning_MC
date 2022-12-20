"""
* Implementation of the periodic divide and concur (PDC) developped by 
* Yoav kallus, adapated from 'Method for dense packing discovery'

* The present code is based on the original C version written by Yoav Kallus
* who kindly provides his code on demand.

* u [lattice
    centroid
    rotmat ]
"""

import numpy as np
from particle.superellipsoid import *
from LLL import LLL_reduction
from utils import Transform
from exclusion import *


# Follow the notation of Kallus
dim = 3 # d in paper
nB = 4 # p in paper
nA = 10 # number of rows for matrix A (nA, dim+nB), may change

# a pair of particles
replica = [SuperEllipsoid(2., 1., 1., 1.) for i in range(2)]

# here nP=dim+1
nP = dim + 1 # number of vertices (dim of single particle)

# Xij for all i != j, we use 2*NUM_REPLICA because 
NUM_REPLICA = 6

class Lambda(object):
    """ still do not understand """
    def __init__(self):
        pass
    
    def func0(self, x): return 0.5 * (1. + np.sqrt(1.+x**2))
    def func1(self, x): return 0.5 * (1. + np.sqrt(1.-x**2))
    def func2(self, x): return 0.5 * (1. - np.sqrt(1.-x**2))
    
    def dfunc0(self, x): return 0.5 * x * (1. + np.sqrt(1.+x**2))
    def dfunc1(self, x): return -0.5 * x * (1. + np.sqrt(1.-x**2))
    def dfunc2(self, x): return 0.5 * x * (1. - np.sqrt(1.-x**2))

#=========================================================================#
#  Divide projections                                                     #
#=========================================================================#

def proj_nonoverlap(pair: np.array):
    """ the divide projection acts independently on each replica pair """
    # rotation matrix & translation vector (centroid)
    R1, r1 = pair[0:dim][:], pair[dim][:] # (dim, dim) & (1, dim)
    R2, r2 = pair[nP:nP+dim][:], pair[nP+dim][:]
    
    pair_new = pair.copy()
    
    replica[0].rot_mat, replica[0].centroid = R1, r1
    replica[1].rot_mat, replica[1].centroid = R2, r2

    # check centroid-centroid distance
    dist = np.linalg.norm(r1 - r2)
    outscribed_d = (replica[0].outscribed_d + replica[1].outscribed_d)/2.
    if (dist < outscribed_d):
        delta = overlap_measure(replica[0], replica[1])
        if (delta > 0.): 
            resolve_overlap(replica[0], replica[1])
            
            pair_new[0:dim][:], pair_new[dim][:] = replica[0].rot_mat, replica[0].centroid
            pair_new[np:2*dim+1][:], pair_new[2*dim+1][:] = replica[1].rot_mat, replica[1].centroid
    
    return pair_new

def divide(x_u: np.array, x_r: np.array): 
    out = input.copy() # (nA, dim)
    for i in range(0, nA, 2*np):
        pair = input[i:i+2*np][:]
        pair_new = proj_nonoverlap(pair)
        out[i:i+2*np][:] = pair_new
    
    return out


def divide(input: np.array): 
    out = input.copy() # (nA, dim)
    for i in range(0, nA, 2*np):
        pair = input[i:i+2*np][:]
        pair_new = proj_nonoverlap(pair)
        out[i:i+2*np][:] = pair_new
    
    return out


#=========================================================================#
#  Concur projections                                                     #
#=========================================================================#

def proj_rigid(single: np.array):
    """ project basis points onto rigid bodies """
    
    # single: means the parameter of a single particle
    R = single[0:dim][:]
    
    U, Sigma, V = np.linalg.svd(R, full_matrices=True)
    Sigma = np.ones(dim)
    
    temp = np.matmul(Sigma, V)
    Rnew = np.matmul(U, temp)
    
    single_new = np.concatenate(Rnew, single[dim])
    
    return single_new


def zbrent(l1: np.double, l2: np.double, singval: np.array, branch: int):
    """ Brent's method: a root-finding algorithm """
    # hyperparameters
    brent_itmax = 25
    brent_prec = 1.e-5
    brent_acc = 1.e-8
    
    f_goal = 1.e-12
    refin_max = 70
    refin_prec = 1.e-7
    refin_acc = 1.e-11
    
    a = l1
    b = c = l2
    
    # fa, fb: bisection method
    if (branch == -1):
        fa = np.prod(singval)
        for i in range(dim): fa *= Lambda().func0(a/singval[i])
        fa = fa/V1 - 1.
        
        fb = np.prod(singval)
        for i in range(dim): fb *= Lambda().func0(b/singval[i])
        fb = fb/V1 - 1.
    else:
        fa = np.prod(singval)
        for i in range(dim-1): fa *= Lambda().func1(a/singval[i])
        if (branch == 0): fa *= Lambda().func1(a/singval[dim-1]) 
        else: fa *= Lambda().func2(a/singval[dim-1])
        fa = fa/V1 - 1.
        
        fb = np.prod(singval)
        for i in range(dim-1): fb *= Lambda().func1(b/singval[i])
        if (branch == 0): fb *= Lambda().func1(b/singval[dim-1]) 
        else: fb *= Lambda().func2(b/singval[dim-1])
        fb = fb/V1 - 1.
    
    precb = np.fabs(a)/2. + np.fabs(b)/2.
    if ((fa > 0. and fb > 0.) or  (fa < 0. and fb < 0.)):
        print("root isn't bracketed in brent")
        
    d = e = np.fabs(b-a)
    fc = fb
    for iter in range(brent_itmax):
        if ((fb > 0. and fc > 0.) or (fb < 0. and fc < 0.)):
            c = a
            fc = fa
            e = d = b-a
            
        if (np.fabs(fc) < np.fabs(fb)):
            a = b
            b = c
            c = a
            
        tol1 = 2.*brent_prec*np.fabs(precb) + 0.5*brent_acc
        xm = 0.5*(c-b)
        if (np.fabs(xm) <= tol1 or fb == 0.): break
        if (np.fabs(e) >= tol1 and np.fabs(fa) > np.fabs(fb)):
            s = fb/fa
            if (a == c):
                p = 2.*xm*s
                q = 1.-s
            else:
                q = fa/fc
                r = fb/fc
                p = s*(2.*xm*q*(q-r)-(b-a)*(r-1.))
                q=(q-1.)*(r-1.)*(s-1.)
                
            if (p > 0.): q = -q
            p = np.fabs(p)
            min1 = 3.*xm*q-np.fabs(tol1*q)
            min2 = np.fabs(e*q)
                
            if (2.*p < np.min(min1, min2)):
                e = d
                d = p/q
            else:
                d = xm
                e = d
            
        else:
            d = xm
            e = d
                
        a = b
        fa = fb
        if (np.fabs(d) > tol1): b += d
        else: 
            temp = np.fabs(tol1) if xm>0. else -np.fabs(tol1)
            b += temp
            
        if (branch == -1):
            fb = np.prod(singval)
            for i in range(dim): fb *= Lambda().func0(b/singval[i])
            fb = fb/V1 - 1.
        else:
            fb = np.prod(singval)
            for i in range(dim-1): fb *= Lambda().func1(b/singval[i])
            if (branch == 0): fb *= Lambda().func1(b/singval[dim-1])
            else: fb *= Lambda().func2(b/singval[dim-1])
            fb = fb/V1 - 1.
          
    if (iter == brent_itmax): print("too many iterations in root finding!")
    # switch over to Newton's method
          
    if (a > b): 
        c = a
        a = b
        b = c
        fc = fa
        fa = fb
        fb = fc
    if (fb > fa): dir = 1.
    else: dir = -1.
          
    xx = b
    ff = fb
          
    if (branch == -1):
        dff = 0.
        for i in range(dim):
            dff += Lambda().dfunc0(xx/singval[i])/(singval[i]*Lambda().func0(xx/singval[i]))
        dff *= ff
    else:
        dff = 0.
        for i in range(dim-1):
            dff += Lambda().dfunc1(xx/singval[i])/(singval[i]*Lambda().func1(xx/singval[i]))
        if (branch == 0): dff += Lambda().dfunc1(xx/singval[dim-1])/(singval[dim-1]*Lambda().func1(xx/singval[dim-1]))
        else: dff += Lambda().dfunc2(xx/singval[dim-1])/(singval[dim-1]*Lambda().func2(xx/singval[dim-1]))
        dff *= ff
          
    if (ff < f_goal): return xx
          
    for iter in range(refin_max):
        dxx = ff/dff
        xx -= dxx
        tol1 = 2.*refin_prec*np.fabs(precb) + 0.5*refin_acc
              
        if (np.fabs(dxx) < tol1): return xx
        if (xx < a or xx > b):
            xx = (a+b)/2.
            if (branch == -1):
                ff = np.prod(singval)
                for i in range(dim): ff *= Lambda().func0(xx/singval[i])
                ff = ff/V1 - 1.
                      
                dff = 0.
                for i in range(dim): dff += Lambda().dfunc0(xx/singval[i])/(singval[i]*Lambda().func0(xx/singval[i]))
                dff *= ff
            else:
                ff = np.prod(singval)
                for i in range(dim-1): ff *= Lambda().func1(xx/singval[i])
                if (branch == 0): ff *= Lambda().func1(xx/singval[dim-1])
                else: ff *= Lambda().func2(xx/singval[dim-1])
                ff = ff/V1 - 1.
                      
                dff = 0.
                for i in range(dim-1): dff += Lambda().dfunc1(xx/singval[i])/(singval[i]*Lambda().func1(xx/singval[i]))
                if (branch == 0): dff += Lambda().dfunc1(xx/singval[dim-1])/(singval[dim-1]*Lambda().func1(xx/singval[dim-1]))
                else: dff += Lambda().dfunc2(xx/singval[dim-1])/(singval[dim-1]*Lambda().func2(xx/singval[dim-1]))
                dff *= ff
                  
            if (dir*ff > 0.): b = xx
            else: a = xx
              
        else:
            if (branch == -1):
                ff = np.prod(singval)
                for i in range(dim): ff *= singval[i]*Lambda().func0(xx/singval[i])
                ff = ff/V1 - 1.
                      
                dff = 0.
                for i in range(dim): dff+=Lambda().dfunc0(xx/singval[i])/(singval[i]*Lambda().func0(xx/singval[i]))
                dff *= ff
            else:
                ff = np.prod(singval)
                for i in range(dim-1): ff *= singval[i]*Lambda().func1(xx/singval[i])
                if (branch == 0): ff *= singval[dim-1]*Lambda().func1(xx/singval[dim-1])
                else: ff *= singval[dim-1]*Lambda().func2(xx/singval[dim-1])
                ff = ff/V1 - 1.
                      
                dff = 0.
                for i in range(dim-1): dff += Lambda().dfunc1(xx/singval[i])/(singval[i]*Lambda().func1(xx/singval[i]))
                if (branch == 0): dff += Lambda().dfunc1(xx/singval[dim-1])/(singval[dim-1]*Lambda().func1(xx/singval[dim-1]))
                else: dff += Lambda().dfunc2(xx/singval[dim-1])/(singval[dim-1]*Lambda().func2(xx/singval[dim-1]))
                dff *= ff
                  
            if (dir*ff > 0.): b = xx
            else: a = xx
              
        if (np.fabs(a-b) < tol1): return xx
          
    print("too many iterations in newton refinement")
    return xx
   

def concur(input: np.array):
      
      # u = M_bar = atwainv . ( Atran . W*input ) input: X
      
      for i in range(nA):
          out[i] = W[i/(2*nP)]*input[i]
      
      AtranWin = np.matmul(Ad.T, out) # (dim+nB, dim)
      u = np.matmul(atwainv, AtranWin) # (dim+nB, dim)
      
      # L = Qinv*M0
      L = np.matmul(Qinv[0:dim][0:dim], u[0:dim][0:dim])
      # U=plu (P), V=plv (R) (Kallus)
      plu, singval, plv = np.linalg.svd(L,full_matrices=False)
      
      # we need to make sure that sigma 从大到小排序
      detL = np.prod(singval)
      
      if (np.fabs(detL) > V1): 
          for i in range(dim): detL *= Lambda().func1(singval[dim-1]/singval[i])
          
          if (np.fabs(detL) > V1):
              # need to use branch 2 for i=dim-1
              bracket = 0.
              
              while True:
                  bracket = singval[dim-1] - ((singval[dim-1] - bracket)/2.)
                  detL = np.prod(singval)
                  for i in range(dim-1): detL *= Lambda().func1(bracket/singval[i])
                  detL *= Lambda().func2(bracket/singval[dim-1])
                  if (np.fabs(detL) < V1): break
              
              mu = zbrent(0., bracket, singval, 1)
              for i in range(dim-1): singval[i] *= Lambda().func1(mu/singval[i])
              singval[dim-1] *= Lambda().func2(mu/singval[dim-1])
          else:
              bracket = 0
              while True:
                  bracket = singval[dim-1] - ((singval[dim-1] - bracket)/2.)
                  detL = np.prod(singval)
                  for i in range(dim): detL *= Lambda().func1(bracket/singval[i])
                  if (np.fabs(detL) > V1): break
              
              mu = zbrent(0., bracket, singval, 0)
              for i in range(dim): singval[i] *= Lambda().func1(mu/singval[i])
      else:
          bracket = singval[dim-1]/1024.
          
          while True:
              bracket *= 2.
              detL = np.prod(singval)
              for i in range(dim): detL *= Lambda().func0(bracket/singval[i])
              if (np.fabs(detL) < V1): break
              
          mu = zbrent(0., bracket, singval, -1)
          for i in range(dim): singval[i] *= Lambda().func0(mu/singval[i])
      
      # let U0 = Q.P.SINGVAL.R
      singval = np.diag(singval)
      temp = np.matmul(singval, plv)
      AtranWin = np.matmul(plu, temp)
      
      # dU = Q.AtranWin - U
      plu = np.matmul(Q, temp) - u[0:dim][0:dim]
      
      # then let U1 += (-atwa11inv . atwa10) . (U0 - U0init)
      u[dim:][:] -= np.matmul(mw11iw1, plu)
      
      # U += dU
      u += plu
      
      for i in range(0, nB, nP): 
          u[dim+i, dim+i+nP][:] = proj_rigid(u[dim+i, dim+i+nP][:])
      
      out = np.matmul(Ad, u)
      
      return out


def Ltrd():
    """ Lattice reduction """
    # H = G = (G0 0, G1 1)
    Hd = np.zero(dim+nP, dim+nP)
  
    # u: (dim+nB, dim)
    LRrnew[0:nB][0:dim] = u[dim:][:] # u1
    Hinvd[0:dim][0:dim] = u[0:dim][:] # u0
    
    # u1=LRrnew*u0, then unew = 
    LRrnew[0:nP][0:dim] = np.matmul(LRrnew[0:nP][0:dim], np.linalg.inv(Hinvd[0:dim][0:dim]))
    
    Lattice = u[0:dim][:]
    unew[0:dim][:], H = LLL_reduction(dim, Lattice) # (dim, dim)
    
    Hd[0:dim][0:dim] = np.double(H) # G0
    
    # all primitive particles' centroids: -0.5<=Lambda<0.5
    for i in range(nP): Hd[dim+i][:] = -np.floor(0.5+LRrnew[i][:])
    
    Hd[dim:][dim:] = np.diag(np.ones(nP))
    
    Hinvd = np.diag(np.ones(dim+nP)) 
    Hinvd[0:dim][0:dim] = np.linalg.inv(Hd[0:dim][0:dim])
    
    # unew = H.u
    unew = np.matmul(Hd, u)
    
    # Anew = A.Hinv
    Anew = np.matmul(Ad, Hinvd) # (nA, dim+nP)
    # A = Anew
    Anew, A = A, Anew
    
    # LRr_new = H.LRr(old)
    LRrnew = np.matmul(Hd, LRr)
    LRr = LRrnew.copy()
    Al = Ad.copy()
    
    sortAold(atmp,btmp,xtmp,mida,olda,newa,Ad,Al,nA)


#=========================================================================#
#  Difference map                                                         #
#=========================================================================#  

def dm_step(x: np.array):
    err = 0.
    
    """ 
    f_D(X) = (1-1/beta)*pi_D(X) + 1/beta*X = X
    f_C(X) = (1+1/beta)*pi_C(X) - 1/beta*X = 2*pi_C(X) - X
    """
    x1 = divide(x) # pi_D(X)
    
    fc = 2.*x1 - x # f_C(X)
    
    x2 = concur(fc) # pi_
    # err <- ||XC - XD||
    err = np.dot(x1-x2, x1-x2)
    
    # iterate X = X + beta*(X_D-X_C)
    x -= (x1-x2)
    
    return x, err/nA


def weight_func(pair: np.array, alpha: np.double):
    """
    w(Xc): A function that assigns replica weights based on their 
    configuration in the concur estimate.
    """
    # rotation matrix and translation vector (centroid)
    R1, r1 = pair[0:dim][:], pair[dim][:]
    R2, r2 = pair[dim+1:2*dim+1][:], pair[2*dim+1][:]
    
    replica[0].rot_mat, replica[0].centroid = R1, r1
    replica[1].rot_mat, replica[1].centroid = R2, r2
  
    dist = np.linalg.norm(r1 - r2)
    outscribed_d = (replica[0].outscribed_d + replica[1].outscribed_d)/2.
    is_overlap = False
    if (dist < outscribed_d):
        delta = overlap_measure(replica[0], replica[1])
        if (delta > 0.): 
            is_overlap = True
            delta_square = delta**2
            
            # Todo: ret??
    
    inscribed_d = (replica[0].inscribed_d + replica[1].inscribed_d)/2.
    if (is_overlap): return np.exp(alpha*delta_square)
    else: 
        y = (1. + dist**2 - inscribed_d**2)**(-2)
        return y


def update_weights(W: np.array, x: np.array, tau: np.double):
    """ perform the weight adjustments according to Eq. (47) """
    for i in range(0, nA, 2*np):
        pair = x[i:i+2*np][:] # slice first
        W[i/(2*np)] =  (tau*W[i/(2*np)] + weight_func(pair)) / (tau+1.)
        
        # Todo: ret?
    return W


def RotOpt(u: np.array):
    """ 
    Algorithm CLOSEPOINT adpated from "Closest Point Search in Lattices". step2: QR decomposition
    
    Parameters
    ----------
    u: (dim+nB, dim)
    
    Return
    ----------
    g, h
    """
    lattice = u[0:dim][0:dim]
    
    h = np.zeros(dim)
    for i in range(dim): h[i] = i

    uu = np.zeros(dim, dim)
    # in the order u[h[0]], u[h[1]], ...
    for i in range(dim):
        uu[i] = lattice[h[i]]

    # Gram schimidt, gs = Q (orthonormal matrix)
    gs = np.zeros(dim, dim) # (dim, dim)
    for i in range(dim):
        gs[i] = uu[i]
        # gs[k] -= (gs[k].gs[l<k]) gs[l]
        for j in range(i): gs[i] -= np.dot(gs[i], gs[j])*gs[j]
        
        # normalized
        gs[i] /= np.linalg.norm(gs[i])
    
    # g[0:dim][:] = G3 = G2 * Q^T (lower-triangular matrix)
    # let x = x * Q^T
    g = np.zeros(dim+nB, dim)
    g[0:dim][:] = np.matmul(uu, gs.T)
    g[dim:dim+nB][:] = np.matmul(u[dim:dim+nB][:], gs.T)
    
    return g, h


def ListClosest(rho0: np.double):
    """
    Using the generating matrix obtained in the concur projection, find 
    all replicas to represent.
    
    Parameters
    ----------
    rho0: pper bound on ||x^ - x||
    
    Return
    ----------
    g, h
    """
    # initilization
    max_breadth = 100
    xx = np.zeros(dim)
    pair_idx = np.zeros(max_breadth*(dim+1), dim)
    idx = np.zeros(dim)
    toadd = np.zeros(dim)
    
    # perm: coordinate permutations
    g, perm = RotOpt(u)
    
    # where x^ denote the closest lattice point to x
    npairs = 0
    pair_level = []
    pair_x = []
    pair_rho = []
    pair_b1 = pair_b2 = []
    for j in range(nB-nP, -1, -nP):
        for i in range(j, -1, -nP):
            # from vnn to vn-1 n-1
            pair_level.append(dim)
            pair_x.append(-(g[j+dim] - g[i+dim])) # centroid
            pair_rho.append(rho0)
            # starting index in nB
            pair_b1.append(j)
            pair_b2.append(i)
            npairs += 1
    
    """
    Our criterion for which replicas to represent is based
    on the difference map's current concur estimate: we include
    a replica pair for each pair of particles whose centroids in the
    concur estimate are closer than some cutoff distance """
    nAnew = 0
    while (npairs > 0):
        npairs -= 1
        level = pair_level[npairs]
        xx[0:level] = pair_x[npairs][0:level]

        rho = pair_rho[npairs]
        p1, p2 = pair_b1[npairs], pair_b2[npairs]
        idx[0:dim-level] = pair_idx[npairs][0:dim-level]
        
        if (level > 0):
            # start from 0
            k = level - 1
            xperp = xx[k]
            vperp = g[k][k] # vnn
            
            # xperp (u_n^*||v_perp||)
            # The indices of these layers are u_n:
            indice_min = np.ceil((xperp - rho)/vperp)
            indice_max = np.floor((xperp + rho)/vperp)
            
            for indice in range(indice_min, indice_max+1):
                pair_level[npairs] = level - 1
                
                for i in range(level-1):
                    pair_x[npairs][i] = xx[i] - indice*g[k][i]
                
                pair_rho[npairs] = np.sqrt(rho**2 - (indice*vperp-vperp)**2)
                pair_b1[npairs], pair_b2[npairs] = p1, p2
                pair_idx[npairs][0] = indice
                pair_idx[npairs][1:dim-level+1] = idx[0:dim-level]
                
                npairs += 1
        else:
            for i in range(dim): toadd[perm[i]] = -idx[i]
            
            for i in range(dim):
                if (toadd[i] != 0): break
            
            if (p1 != p2 or i != dim):
                for l in range(nP):
                    Anew[nAnew] = -0.6*toadd
                    Anew[nAnew][dim:dim+nB] = np.zeros(nB)
                    Anew[nAnew][2*dim+p1] = 1.
                    nAnew += 1
                
                for l in range(nP):
                    Anew[nAnew] = 0.4*toadd
                    Anew[nAnew][dim:dim+nB] = np.zeros(nB)
                    Anew[nAnew][2*dim+p2] = 1.
                    nAnew += 1
    return nAnew
                

def sortAold(atmp: np.array, btmp: np.array, xtmp: np.array, Atosort: np.array, Altosort: np.array, nAtosort: int):
    rra = np.array(dim+nB)
    rra1 = np.array(dim+nB)
    rra2 = np.array(dim+nB)
    
    n = nAtosort/(2*nP)
    if (n < 2): return
    
    l = n-1
    ir = n-1
    
    while True:
        if (l > 0):
            l -= 1
            for k in range(dim+nB):
                rra[k] = 0
                for m in range(2*nP):
                    if (rra[k] == 0): rra[k] += np.floor(2*(Altosort[2*nP*l+m][k])+0.5)
            
            atmp[0:2*nP][0:dim+nB] = Atosort[2*nP*l:2*nP*(l+1)][0:dim+nB]
            btmp[0:2*nP][0:dim+nB] = Altosort[2*nP*l:2*nP*(l+1)][0:dim+nB]
            xtmp[0:2*nP][0:dim] = x[2*nP*l:2*nP*(l+1)][0:dim]
        
            Wtemp = W[l]
        else:
            for k in range(dim+nB):
                rra[k] = 0
                for m in range(2*nP):
                    if (rra[k] == 0): rra[k] += np.floor(2*(Altosort[2*nP*ir+m][k])+0.5)
            
            atmp[0:2*nP][0:dim+nB] = Atosort[2*nP*ir:2*nP*(ir+1)][0:dim+nB]
            btmp[0:2*nP][0:dim+nB] = Altosort[2*nP*ir:2*nP*(ir+1)][0:dim+nB]
            xtmp[0:2*nP][0:dim] = x[2*nP*l:2*nP*(ir+1)][0:dim]
        
            Wtemp = W[ir]
            # put Atosort[0] into Atosort[ir]
            Atosort[2*nP*ir:2*nP*(ir+1)][0:dim+nB] = Atosort[0:2*nP][0:dim+nB]
            Altosort[2*nP*ir:2*nP*(ir+1)][0:dim+nB] = Altosort[0:2*nP][0:dim+nB]
            W[ir] = W[0]
            
            ir -= 1
            if (ir == 0):
                Atosort[0:2*nP][0:dim+nB] = atmp[0:2*nP][0:dim+nB]
                Altosort[0:2*nP][0:dim+nB] = btmp[0:2*nP][0:dim+nB]
                x[0:2*nP][0:dim] = xtmp[0:2*nP][0:dim]
                W[0] = Wtemp
                break
        
        i = l
        j = l+1
        while (j <= ir):
            for k in range(dim+nB):
                rra1[k] = rra2[k] = 0
                for m in range(2*nP):
                    if (rra1[k] == 0): rra1[k] += np.floor(2*(Altosort[2*nP*j+m][k])+0.5)
                    if (rra2[k] == 0): rra2[k] += np.floor(2*(Altosort[2*nP*(j+1)+m][k])+0.5)
            
            comp = 0
            for k in range(dim+nB-1, -1, -1):
                if (rra1[k] < rra2[k]): 
                    comp = -1
                    break
                if (rra1[k] > rra2[k]):
                    comp = 1
                    break
            
            if (j < ir and comp == -1):
                j += 1
                rra1[0:dim+nB] = rra2[0:dim+nB]
            
            comp = 0
            for k in range(dim+nB-1, -1, -1):
                if (rra[k] < rra1[k]): 
                    comp = -1
                    break
                if (rra[k] > rra1[k]):
                    comp = 1
                    break
            
            if (comp == -1):
                Atosort[2*nP*i:2*nP*(i+1)][0:dim+nB] = Atosort[2*nP*j:2*nP*(j+1)][0:dim+nB]
                Altosort[2*nP*i:2*nP*(i+1)][0:dim+nB] = Altosort[2*nP*j:2*nP*(j+1)][0:dim+nB]
                x[2*nP*i:2*nP*(i+1)][0:dim] = x[2*nP*j:2*nP*(j+1)][0:dim]
                W[i] = W[j]
                i = j
                j <<= 1
            else: j = ir+1
            
        Atosort[2*nP*i:2*nP*(i+1)][0:dim+nB] = atmp[0:2*nP][0:dim+nB]
        Altosort[2*nP*i:2*nP*(i+1)][0:dim+nB] = btmp[0:2*nP][0:dim+nB]
        x[2*nP*i:2*nP*(i+1)][0:dim] = xtmp[0:2*nP][0:dim]
        W[i] = Wtemp      
                 

def update_A():

    
    olda = np.empty(dim+nB)
    newa = np.empty(dim+nB)
    
    outscribed_d = replica[0].outscribed_d
    nAnew = ListClosest(outscribed_d)
    
    Alnew = Anew.copy() # (nAnew, dim+nB)
    xt = np.matmul(Anew, u) # (nAnew, dim)
    x2 = xt.copy()
    
    j = i = 0
    if (nA > 0):
        for k in range(dim+nB):
            olda[k] = 0
            for m in range(2*nP):
                if (olda[k] == 0): olda[k] += np.floor(2*(Al[2*j*nP+m][k])+0.5)
            
            newa[k] = 0
            for m in range(2*nP):
                if (newa[k] == 0): newa[k] += np.floor(2*(Al[2*j*nP+m][k])+0.5)
    
    while (True):
        if (j >= nA/(2*nP) or i >= nAnew/(2*nP)): break
        comp = 0
        for k in range(dim+nB-1, -1, -1):
            if (olda[k] < newa[k]):
                comp = -1
                break
            if (olda[k] > newa[k]):
                comp = 1
                break
        
        if (comp == 0):
            xt[2*nP*i:2*nP*(i+1)][0:dim] = x[2*nP*j:2*nP*(j+1)][0:dim]
            Wnew[i] = W[j]
            i += 1
            if (i >= nAnew/(2*np)): break
            
            for k in range(dim+nB):
                newa[k] = 0
                for m in range(2*nP):
                    if (newa[k] == 0): newa[k] += np.floor(2*(Alnew[2*nP*i+m][k])+0.5)
            
            j += 1
            if (j >= nA/(2*nP)): break
            
            for k in range(dim+nB):
                olda[k] = 0
                for m in range(2*nP):
                    if (olda[k] == 0): olda[k] += np.floor(2*(Al[2*nP*j+m][k])+0.5)
        
        elif (comp == -1): # newa > olda
            j += 1
            if (j >= nA/(2*nP)): break
            for k in range(dim+nB):
                olda[k] = 0
                for m in range(2*nP):
                    if (olda[k] == 0): olda[k] += np.floor(2*(Al[2*nP*j+m][k])+0.5)
        
        else:
            Wnew[i] = weight_func()
            if (Wnew[i] > 1.): Wnew[i] = 1.
            i += 1
            if (i >= nAnew/2*nP): break
            for k in range(dim+nB):
                newa[k] = 0
                for m in range(2*nP):
                    if (newa[k] == 0): newa[k] += np.floor(2*(Alnew[2*nP*i+m][k])+0.5)
                    
    if (i < nAnew/(2*nP)):
        for t in range(i, nAnew/(2*nP)):
            Wnew[t] = weight_func()
            if (Wnew[t] > 1.): Wnew[t] = 1.
      
    
    # replace x with xt
    x, xt = xt, x
    # replace W with Wnew
    W, Wnew = Wnew, W
    # replace A with Anew, nA with nAnew
    Ad, Anew = Anew, Ad
    Al, Alnew = Alnew, Al
    nA = nAnew

    # set x2 = A . u
    x2 = np.matmul(Ad, u) # (nA, dim)
  
  
def calc_atwa(Anew: np.array):
    """ Used in Lattice constraint """
    # W is a diagonal matrix whose diagonal elements wi are the metric weights of different replicas
    
    # atwa = A^T . (W*A)
    # Anew = W*A
    for i in range(nA):
        Anew[i][:] = W[i/(2*nP)]*Ad[i][:] # (nA, dim+nB)
    
    # atwa = A^T . Anew
    atwa = np.matmul(Anew.T, temp) # (dim+nB, dim+nB)
    
    # temp = (W'11)^-1 * W'10 | (nB, nB)*(nB, dim)
    wtemp = np.linalg.pinv(atwa[dim:][dim:])
    # w' = atwa
    # i means inverse
    temp = np.matmul(wtemp, atwa[dim:][0:dim]) # (nB, dim)
    
    # atwa2 = W'' = W'00 - W'01*(W'11)^-1 * W'10
    atmp = atwa
    atmp[0:dim][0:dim] = atwa[0:dim][0:dim] - np.matmul(atwa[0:dim][dim:], temp) # (dim, dim)
    
    eigs, featurevector = np.linalg.eig(atmp[0:dim+nB][0:dim])
    
    # let Q = W^-1/2, then V1 (V_target) = V0*det(Q)
    V1 = V0
    for i in range(dim): V1*=np.sqrt(eigs[i])
    
    # atwa = A.L.AT, eig_work=atmp=A
    # Qinv = W^1/2, so Qinv = A.(sqrt(L).AT) = A. (A.sqrt(L))T
    # (A.sqrt(L)) = Aij sqrt(Lj)
    for i in range(dim):
        for j in range(dim):
            temp[i][j] = np.sqrt(eigs[i])*atmp[i][j]
    
    Qinv = np.matmul(atmp.T, temp)
    
    for i in range(dim):
        for j in range(dim):
            temp[i][j] /=eigs[i]
    Q = np.matmul(atmp.T, temp)


if __name__ == '__main__':
    # specify your problem
    dim = 3
    nP = 5 # number of particles
  
  
    # global declaration
    max_nA = 400000
    
    # 
    u = np.empty([dim+nP, dim])
    r = np.empty([dim*nP, dim])
    
    x = np.empty([max_nA, dim])
    x1 = np.empty([max_nA, dim]) # divide projection
    x2 = np.empty([max_nA, dim]) # concur projection
    xt = np.empty([max_nA, dim])
    
    Ad = np.empty([max_nA, dim+nP])
    Anew = np.empty([max_nA, dim+nP])
    Al = np.empty([max_nA, dim+nB])
    Alnew = np.empty([max_nA, dim+nB])
    atwa = np.empty([dim+nB, dim+nB])
    atwainv = np.empty([dim+nB, dim+nB])
    Q = np.empty([dim, dim])
    Qinv = np.empty([dim, dim])
    mw11iw10 = np.empty([nB, dim])
    
    W = np.empty(max_nA)
    Wnew = np.empty(max_nA)
    
    maxstep = 500000
    
    np.random.seed(1)
    
    
    # start from a random initial configuration
    V0 = replica[0].volume*(nB/nP)/efficiency
    
    u[0:dim][:] = 0.02*(-0.5 + np.random.random((dim, dim)))
    for i in range(dim): u[i][i] += 1. # np.cbrt(V0)
    
    u[dim+nP-1:dim+nB:nP][:] = -0.5 + np.random.random((nB/nP, dim))
    for i in range(nB/nP): u[dim+i*nP:dim+(i+1)*nP][:] = Transform().mat_random()
    
    LRr = np.diag(np.ones(dim+nB))

    
    Ltrd()
    update_A()
    update_weights(W, x, )
    calc_atwa()
    
    for i in range(maxstep):
        err = dm_step(x)
        if ((i%50) == 49): Ltrd()
        update_A()
        update_weights()
        calc_atwa
        
        if (err < 8.e-11):
            update_A()
            err = update_weights()
            
            if (err < 8.e-11): break
            
    
            
        
        
    
    
