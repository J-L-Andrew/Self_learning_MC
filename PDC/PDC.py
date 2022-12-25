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
from tenpy.linalg.svd_robust import svd
from numpy.linalg import norm
import sys
sys.path.append(r'/mnt/Edisk/andrew/Self_learning_MC')

from LLL import LLL_reduction
from global_var import *
from particle.superellipsoid import *
from exclusion import *
from utils import Transform
from cell import Cell

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
def proj_nonoverlap(pair_c: np.array, pair_r: np.array):
    """ 
    The divide projection acts independently on each replica pair 
    
    parameters: translation vector (centroid) & rotation matrix
    """
    pnew_c, pnew_r = pair_c.copy(), pair_r.copy()
    
    dist = norm(pair_c[0] - pair_c[1])
    if (dist > outscribed_d): return pnew_c, pnew_r
    
    replica[0].rot_mat, replica[0].centroid = pair_r[0:dim].copy(), pair_c[0].copy()
    replica[1].rot_mat, replica[1].centroid = pair_r[dim:2*dim].copy(), pair_c[1].copy()

    delta = overlap_measure(replica[0], replica[1])

    if (delta < 0.):
        return pnew_c, pnew_r
    else:
        resolve_overlap(replica[0], replica[1])
        pnew_r[0:dim], pnew_c[0] =  replica[0].rot_mat.copy(), replica[0].centroid.copy()
        pnew_r[dim:2*dim], pnew_c[1] =  replica[1].rot_mat.copy(), replica[1].centroid.copy()
    
    return pnew_r, pnew_c

def proj_nonoverlap(pair: np.array):
    """ 
    The divide projection acts independently on each replica pair 
    
    parameters: translation vector (centroid) & rotation matrix
    """
    pair_new = pair.copy()
    
    dist = norm(pair[0] - pair[nV])
    if (dist > outscribed_d): return pair_new
    
    replica[0].centroid, replica[0].rot_mat = pair[0].copy(), pair[1:nV].copy()
    replica[1].centroid, replica[1].rot_mat = pair[nV].copy(), pair[nV+1:2*nV].copy()

    delta = overlap_measure(replica[0], replica[1])

    if (delta < 0.): return pair_new
    else:
        resolve_overlap(replica[0], replica[1])
        pair_new[0], pair_new[1:nV] =  replica[0].centroid.copy(), replica[0].rot_mat.copy()
        pair_new[nV], pair_new[nV+1:2*nV] =  replica[1].centroid.copy(),  replica[1].rot_mat.copy()
    
    return pair_new

def divide(input: np.array): 
    out = input[0:pdc.nA,:].copy() # (nA, dim)
    for i in range(0, pdc.nA, 2*np):
        pair = input[i:i+2*np][:]
        pair_new = proj_nonoverlap(pair)
        out[i:i+2*np][:] = pair_new
    
    return out


#=========================================================================#
#  Concur projections                                                     #
#=========================================================================#
def proj_rigid(single: np.array):
    """ project a general matrix into the subset of orthogonal matrices """
    single_new = single.copy()
    
    rot = single[1:dim+1,:].copy()
    
    # setting all the singular values to unity
    V, singval, U = svd(rot)
    temp = np.matmul(U.T, V.T)
    
    single_new[1:dim+1,:] = temp.copy()
    
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
        fa = 1.
        for i in range(dim): fa *= singval[i]*Lambda().func0(a/singval[i])
        fa = fa/pdc.V1 - 1.
        
        fb = 1.
        for i in range(dim): fb *= singval[i]*Lambda().func0(b/singval[i])
        fb = fb/pdc.V1 - 1.
    else:
        fa = 1.
        for i in range(dim-1): fa *= singval[i]*Lambda().func1(a/singval[i])
        if (branch == 0): fa *= singval[dim-1]*Lambda().func1(a/singval[dim-1]) 
        else: fa *= singval[dim-1]*Lambda().func2(a/singval[dim-1])
        fa = fa/pdc.V1 - 1.
        
        fb = 1.
        for i in range(dim-1): fb *= singval[i]*Lambda().func1(b/singval[i])
        if (branch == 0): fb *= singval[dim-1]*Lambda().func1(b/singval[dim-1]) 
        else: fb *= singval[dim-1]*Lambda().func2(b/singval[dim-1])
        fb = fb/pdc.V1 - 1.
    
    precb = np.fabs(a)/2. + np.fabs(b)/2.
    if ((fa > 0. and fb > 0.) or  (fa < 0. and fb < 0.)):
        print("root isn't bracketed in brent")
        return 0.
        
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
            fa = fb
            fb = fc
            fc = fa
            
        tol1 = 2.*brent_prec*np.fabs(precb) + 0.5*brent_acc
        xm = 0.5 * (c-b)
        if (np.fabs(xm) <= tol1 or fb == 0.): break
        if (np.fabs(e) >= tol1 and np.fabs(fa) > np.fabs(fb)):
            s = fb/fa
            if (a == c):
                p = 2.*xm*s
                q = 1. - s
            else:
                q = fa/fc
                r = fb/fc
                p = s*(2.*xm*q*(q-r)-(b-a)*(r-1.))
                q = (q-1.)*(r-1.)*(s-1.)
                
            if (p > 0.): q = -q
            p = np.fabs(p)
            min1 = 3.*xm*q - np.fabs(tol1*q)
            min2 = np.fabs(e*q)
                
            if (2.*p < min(min1, min2)):
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
            temp = np.fabs(tol1) if xm > 0. else -np.fabs(tol1)
            b += temp
            
        if (branch == -1):
            fb = 1.
            for i in range(dim): fb *= singval[i]*Lambda().func0(b/singval[i])
            fb = fb/pdc.V1 - 1.
        else:
            fb = 1.
            for i in range(dim-1): fb *= singval[i]*Lambda().func1(b/singval[i])
            if (branch == 0): fb *= singval[dim-1]*Lambda().func1(b/singval[dim-1])
            else: fb *= singval[dim-1]*Lambda().func2(b/singval[dim-1])
            fb = fb/pdc.V1 - 1.
          
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
                ff = 1.
                for i in range(dim): ff *= singval[i]*Lambda().func0(xx/singval[i])
                ff = ff/pdc.V1 - 1.
                      
                dff = 0.
                for i in range(dim): dff += Lambda().dfunc0(xx/singval[i])/(singval[i]*Lambda().func0(xx/singval[i]))
                dff *= ff
            else:
                ff = 1.
                for i in range(dim-1): ff *= singval[i]*Lambda().func1(xx/singval[i])
                if (branch == 0): ff *= singval[dim-1]*Lambda().func1(xx/singval[dim-1])
                else: ff *= singval[dim-1]*Lambda().func2(xx/singval[dim-1])
                ff = ff/pdc.V1 - 1.
                      
                dff = 0.
                for i in range(dim-1): dff += Lambda().dfunc1(xx/singval[i])/(singval[i]*Lambda().func1(xx/singval[i]))
                if (branch == 0): dff += Lambda().dfunc1(xx/singval[dim-1])/(singval[dim-1]*Lambda().func1(xx/singval[dim-1]))
                else: dff += Lambda().dfunc2(xx/singval[dim-1])/(singval[dim-1]*Lambda().func2(xx/singval[dim-1]))
                dff *= ff
                  
            if (dir*ff > 0.): b = xx
            else: a = xx
              
        else:
            if (branch == -1):
                ff = 1.
                for i in range(dim): ff *= singval[i]*Lambda().func0(xx/singval[i])
                ff = ff/pdc.V1 - 1.
                      
                dff = 0.
                for i in range(dim): dff+=Lambda().dfunc0(xx/singval[i])/(singval[i]*Lambda().func0(xx/singval[i]))
                dff *= ff
            else:
                ff = 1.
                for i in range(dim-1): ff *= singval[i]*Lambda().func1(xx/singval[i])
                if (branch == 0): ff *= singval[dim-1]*Lambda().func1(xx/singval[dim-1])
                else: ff *= singval[dim-1]*Lambda().func2(xx/singval[dim-1])
                ff = ff/pdc.V1 - 1.
                      
                dff = 0.
                for i in range(dim-1): dff += Lambda().dfunc1(xx/singval[i])/(singval[i]*Lambda().func1(xx/singval[i]))
                if (branch == 0): dff += Lambda().dfunc1(xx/singval[dim-1])/(singval[dim-1]*Lambda().func1(xx/singval[dim-1]))
                else: dff += Lambda().dfunc2(xx/singval[dim-1])/(singval[dim-1]*Lambda().func2(xx/singval[dim-1]))
                dff *= ff
                  
            if (dir*ff > 0.): b = xx
            else: a = xx
              
        if (np.fabs(a-b) < tol1): return xx
          
    print("too many iterations in newton refinement")
    return 

def concur(input: np.array):
    out = input[0:pdc.nA,:].copy() # (nA, dim)
    # u = M_bar = atwainv . (Atran . W*input)
    for i in range(pdc.nA):
        out[i] = pdc.W[int(i/(2*nV))]*input[i]

    AtranWin = np.matmul(pdc.Ad[0:pdc.nA,:].T, out) # (dim+nB, dim)
    
    pdc.u = np.matmul(pdc.atwainv, AtranWin) # (dim+nB, dim)
      
    # L = Qinv*M0
    L = np.matmul(pdc.u[0:dim,:].T, pdc.Qinv.T)
    # L = np.matmul(pdc.Qinv, pdc.u[0:dim,:]) # (dim, dim)
    # U=plu (P), V=plv (R) (Kallus)
    ### do not why, but plu and plv is changed
    plv, singval, plu = svd(L)
    
    detL = np.prod(singval)
    if (np.fabs(detL) > pdc.V1): 
        detL = 1.
        for i in range(dim): detL *= singval[i]*Lambda().func1(singval[dim-1]/singval[i])
          
        if (np.fabs(detL) > pdc.V1):
            # need to use branch 2 for i=dim-1
            bracket = 0.
              
            while (True):
                bracket = singval[dim-1] - (singval[dim-1] - bracket)/2.
                detL = 1.
                for i in range(dim-1): detL *= singval[i]*Lambda().func1(bracket/singval[i])
                detL *= singval[dim-1]*Lambda().func2(bracket/singval[dim-1])
                if (np.fabs(detL) > pdc.V1): break
              
            mu = zbrent(0., bracket, singval, 1)
            for i in range(dim-1): singval[i] *= Lambda().func1(mu/singval[i])
            singval[dim-1] *= Lambda().func2(mu/singval[dim-1])
        else:
            bracket = 0
            while (True):
                bracket = singval[dim-1] - (singval[dim-1] - bracket)/2.
                detL = 1.
                for i in range(dim): detL *= singval[i]*Lambda().func1(bracket/singval[i])
                if (np.fabs(detL) < pdc.V1): break
              
            mu = zbrent(0., bracket, singval, 0)
            for i in range(dim): singval[i] *= Lambda().func1(mu/singval[i])
    else:
        bracket = singval[dim-1] / 1024.
        while True:
            bracket *= 2.
            detL = 1.
            for i in range(dim): detL *= singval[i]*Lambda().func0(bracket/singval[i])
            if (np.fabs(detL) > pdc.V1): break
           
        mu = zbrent(0., bracket, singval, -1)
        for i in range(dim): singval[i] *= Lambda().func0(mu/singval[i])
      
    # let U0 = Q.P.SINGVAL.R
    # R = SINGVAL.R = SINGVAL_i R_ij
    for i in range(dim):
        for j in range(dim):
            plv[i][j] *= singval[j]

    AtranWin = np.matmul(plu.T, plv.T) # (dim, dim)
    # dU = Q.AtranWin - U
    plu = np.matmul(pdc.Q, AtranWin) - pdc.u[0:dim,:]
    
    # then let U1 += (-atwa11inv . atwa10) . (U0 - U0init)
    pdc.u[dim:,:] += np.matmul(pdc.mw11iw10, plu)
    
    # U += dU
    pdc.u[0:dim,:] += plu
      
    # Rigidity constraint, project basis points onto rigid bodies:
    for i in range(0, nB, nV): 
        pdc.u[dim+i:dim+i+nV,:] = proj_rigid(pdc.u[dim+i:dim+i+nV,:])
    
    # out = A.u
    out = np.matmul(pdc.Ad[0:pdc.nA,:], pdc.u)
      
    return out



#=========================================================================#
#  Difference map                                                         #
#=========================================================================#  
def initialize(pd_target: np.double):
    """ start from random initial configurations """
    pdc.nA = 0
    pdc.V0 = nP*replica[0].volume / pd_target
    
    pdc.LRr= np.diag(np.ones(dim+nB))
    
    np.random.seed()
    pdc.u[0:dim,:] = 0.02*(-0.5 + np.random.random((dim, dim)))
    for i in range(dim): pdc.u[i,i] += np.cbrt(pdc.V0)
    pdc.u[dim:,:] = -0.5 + np.random.random((nB, dim))
    
    # For xyz file
    packing.particle_type = 'sphere'
    packing.num_particles = nB
    
    # add particles
    packing.dim = 3
    packing.particles = [Sphere(1.) for i in range(packing.num_particles)]
    for i, particle in enumerate(packing.particles):
        particle.name = 'sphere %d' % i
        particle.color = np.array([0.51,0.792,0.992])
    
    # add cell
    packing.cell = Cell(packing.dim)
    packing.cell.color = np.array([0.25,0.25,0.25])
  
def dm_step(x: np.array):
    """ simple difference map solver """
    # f_D(X) = (1-1/beta)*pi_D(X) + 1/beta*X = X
    # f_C(X) = (1+1/beta)*pi_C(X) - 1/beta*X = 2*pi_C(X) - X
    pdc.x1[0:pdc.nA,:] = divide(pdc.x) # pi_D(X)
    
    pdc.xt[0:pdc.nA,:] = 2.*pdc.x1[0:pdc.nA,:] - pdc.x[0:pdc.nA,:] # f_C(X)

    pdc.x2[0:pdc.nA,:] = concur(pdc.xt) # pi_
  
    # err <- ||XC - XD||
    delta = pdc.x1[0:pdc.nA] - pdc.x2[0:pdc.nA]
    
    err = np.sum(delta*delta)
    
    if (err > pdc.nA*maxstep):
        pdc.x1[0:pdc.nA] *= np.sqrt(maxstep*pdc.nA/err)
        err = pdc.nA*maxstep
    
    # iterate X = X + beta*(X_D-X_C)
    pdc.x[0:pdc.nA] -= delta
    
    return err/pdc.nA


def weight_func(pair: np.array, alpha: np.double):
    """
    A function that assigns replica weights based on their 
    configuration in the concur estimate.
    
    return: w(Xc) and overlap measure (dist here)
    """
    # rotation matrix and translation vector (centroid)
    r1, R1 = pair[0].copy(), pair[1:nV].copy()
    r2, R2 = pair[nV].copy(), pair[nV+1:2*nV].copy()
    
    replica[0].centroid, replica[0].rot_mat = r1, R1
    replica[1].centroid, replica[1].rot_mat = r2, R2
  
    dist = np.linalg.norm(r1 - r2)
    is_overlap = False
    if (dist < outscribed_d):
        delta = overlap_measure(replica[0], replica[1])
        if (delta > 0.): 
            is_overlap = True
            delta_square = delta**2
            
            # Todo: ret??
    
    if (is_overlap): return np.exp(alpha*delta_square)
    else: 
        y = (1. + dist**2 - inscribed_d**2)**(-2)
        return y


def update_weights():
    """ perform the weight adjustments according to Eq. (47) """
    ret = 0
    for i in range(0, pdc.nA, 2*nV):
        w, s = weight_func(pdc.x2[i:i+2*nV])
        pdc.W[int(i/(2*nV))] = (tau*pdc.W[int(i/(2*nV))] + w) / (tau+1.)
        ret += s
        
    return ret


#=========================================================================#
#  Formal configuration-space maintenance                                 #
#=========================================================================# 
def Ltrd():
    """ Lattice reduction """
    LRrnew = np.empty([dim+nB, dim+nB])
    Hinvd = np.empty([dim+nB, dim+nB])
    unew = np.empty([dim+nB, dim])
    
    # u' = H.u LLL-reduced, H = G = (G0 0, G1 1)
    LRrnew[0:nP,0:dim] = pdc.u[dim::nV,:].copy() # u1 (centroid only)
    Hinvd[0:dim,0:dim] = pdc.u[0:dim,:].copy() # u0
    
    # u1 = LRrnew*u0, then LRrnew = u1*u0^-1
    LRrnew[0:nP,0:dim] = np.matmul(LRrnew[0:nP,0:dim], np.linalg.inv(Hinvd[0:dim,0:dim]))
    
    unew[0:dim,:], H = LLL_reduction(pdc.u[0:dim,:], dim) # (dim, dim)
    
    Hd = np.zeros([dim+nB, dim+nB])
    Hd[0:dim,0:dim] = np.double(H) # G0
    # all primitive particles' centroids: -0.5<=Lambda<0.5
    for i in range(dim, dim+nB, nV):
        for j in range(dim):
            Hd[i,j] = -np.floor(0.5 + LRrnew[(i-dim)/nV,j])
            Hd[i+1:i+nV,j] = np.zeros(nV-1, dtype=int)
    Hd[dim:,dim:] = np.diag(np.ones(nB))
    
    # Hinv = H^-1
    Hinvd = np.diag(np.ones(dim+nB)) 
    Hinvd[0:dim,0:dim] = np.linalg.inv(Hd[0:dim,0:dim])
    
    # unew = H.u
    unew = np.matmul(Hd, pdc.u)
    pdc.u = unew.copy()
    
    # Anew = A.Hinv
    pdc.Anew[0:pdc.nA,:] = np.matmul(pdc.Ad[0:pdc.nA,:], Hinvd) # (nA, dim+nB)
    # A = Anew
    pdc.Anew, pdc.Ad = pdc.Ad.copy(), pdc.Anew.copy()
    
    # LRr_new = H.LRr(old)
    LRrnew = np.matmul(Hd, pdc.LRr) # (dim+nB, dim+nB)
    pdc.LRr = LRrnew.copy()
    
    pdc.Al[0:pdc.nA,:] = pdc.Ad[0:pdc.nA,:].copy()
    
    if (pdc.nA >= 2*nV): pdc.Ad, pdc.Al = sortAold(pdc.Ad, pdc.Al, pdc.nA)

def RotOpt():
    """ 
    Algorithm CLOSEPOINT adpated from "Closest Point Search in Lattices".
    
    Return
    ----------
    g(lower-triangular matrix), h(int)
    """
    h = np.arange(0, dim, dtype=int)

    uu = np.zeros([dim, dim])
    # in the order u[h[0]], u[h[1]], ...
    for i in range(dim):
        uu[i] = pdc.u[h[i]].copy()

    # Gram schimidt, gs = Q (orthonormal matrix)
    # then G2 = G3 * Q, and G3 becomes a lower-triangular matrix
    gs = np.empty([dim, dim])
    for i in range(dim):
        gs[i] = uu[i].copy()
        # gs[k] -= (gs[k].gs[l<k]) gs[l]
        for j in range(i): gs[i] -= np.dot(gs[i], gs[j])*gs[j]
        
        gs[i] /= norm(gs[i]) # normalized
    
    # G3 = G2 * Q^T
    g = np.empty([dim+nB, dim])
    g[0:dim,:] = np.matmul(uu, gs.T)
    g[dim:,:] = np.matmul(pdc.u[dim:,:], gs.T)
    
    return g, h

def ListClosest(rho0: np.double):
    """
    Using the generating matrix obtained in the concur projection, find 
    all replicas to represent (recreate Anew).
    
    Parameters
    ----------
    rho0: pper bound on ||x^ - x||
    """
    # initilization
    max_breadth = 100
    xx = np.empty(dim)
    idx = np.empty(dim, dtype=int)
    toadd = np.empty(dim, dtype=int)
    
    pair_idx = np.empty([max_breadth*(dim+1), dim], dtype=int)
    pair_x = np.empty([max_breadth*(dim+1), dim])
    pair_level = np.empty(max_breadth*(dim+1), dtype=int)
    pair_rho = np.empty(max_breadth*(dim+1))
    pair_p1 = np.empty(max_breadth*(dim+1), dtype=int)
    pair_p2 = np.empty(max_breadth*(dim+1), dtype=int)
    
    # perm: coordinate permutations
    g, perm = RotOpt()
    
    # where x^ denote the closest lattice point to x
    npairs = 0
    for j in range(nB-nV, -1, -nV):
        for i in range(j, -1, -nV):
            # from vnn to vn-1 n-1
            pair_level[npairs] = dim
            
            pair_x[npairs] = -(g[dim+j] - g[dim+i]) # centroid
            pair_rho[npairs] = rho0

            pair_p1[npairs], pair_p2[npairs] = j, i
            npairs += 1
    
    # Our criterion for which replicas to represent is based
    # on the difference map's current concur estimate: we include
    # a replica pair for each pair of particles whose centroids in the
    # concur estimate are closer than some cutoff distance
    nAnew = 0
    while (npairs > 0):
        npairs -= 1
        level = pair_level[npairs]
        xx[0:level] = pair_x[npairs,0:level].copy()

        rho = pair_rho[npairs]
        p1, p2 = pair_p1[npairs], pair_p2[npairs]
        idx[0:dim-level] = pair_idx[npairs,0:dim-level].copy()
        
        if (level > 0):
            # start from 0
            k = level - 1
            xperp = xx[k]
            vperp = g[k,k] # vnn
            
            # xperp (u_n^*||v_perp||)
            # The indices of these layers are u_n:
            indice_min = int(np.ceil((xperp - rho)/vperp))
            indice_max = int(np.floor((xperp + rho)/vperp))
            
            for indice in range(indice_min, indice_max+1):
                pair_level[npairs] = level - 1
                pair_x[npairs,0:level-1] = xx[0:level-1] - indice*g[k,0:level-1]
                
                # yn := |un-un^| * ||v_perp||
                pair_rho[npairs] = np.sqrt(rho**2 - (indice*vperp-xperp)**2)
                pair_p1[npairs], pair_p2[npairs] = p1, p2
                pair_idx[npairs,0] = indice
                pair_idx[npairs,1:dim-level+1] = idx[0:dim-level].copy()
                
                npairs += 1
        else:
            for i in range(dim): toadd[perm[i]] = -idx[i]
            
            count = 0
            for i in range(dim):
                if (toadd[i] != 0): break
                count += 1
            
            if (p1 != p2 or count != dim):
                for n in range(nV):
                    if (n == 0): pdc.Anew[nAnew,0:dim] = -0.6*toadd
                    else: pdc.Anew[nAnew,0:dim] = np.zeros(dim)
                    pdc.Anew[nAnew,dim:] = np.zeros(nB)
                    pdc.Anew[nAnew,dim+p1] = 1.
                    nAnew += 1
                
                for n in range(nV):
                    if (n == 0): pdc.Anew[nAnew,0:dim] = 0.4*toadd
                    else: pdc.Anew[nAnew,0:dim] = np.zeros(dim)
                    pdc.Anew[nAnew,dim:] = np.zeros(nB)
                    pdc.Anew[nAnew,dim+p2+n] = 1.
                    nAnew += 1
            
            if (nAnew > max_nA - 2*nV): print("memory overflow")

    return nAnew             

def update_A():
    olda = np.empty(dim+nB, dtype=int)
    newa = np.empty(dim+nB, dtype=int)
    
    nAnew = int(ListClosest(4.))
    
    pdc.Alnew[0:nAnew,:] = pdc.Anew[0:nAnew,:].copy() # (nAnew, dim+nP)
    pdc.xt[0:nAnew,:] = np.matmul(pdc.Anew[0:nAnew,:], pdc.u) # (nAnew, dim)
    pdc.x2[0:nAnew,:] = pdc.xt[0:nAnew,:].copy()
    
    j = i = int(0)
    # olda = A[j].LRr
    if (pdc.nA > 0):
        for k in range(dim+nB):
            olda[k] = 0
            for m in range(2*nV):
                if (olda[k] == 0): olda[k] += int(np.floor(2*(pdc.Al[2*nV*j+m,k])+0.5))
            
            newa[k] = 0
            for m in range(2*nV):
                if (newa[k] == 0): newa[k] += int(np.floor(2*(pdc.Alnew[2*nV*j+m,k])+0.5))
    
    while (True):
        if (j >= int(pdc.nA/(2*nV)) or i >= int(nAnew/(2*nV))): break
        # compare olda and newa
        comp = 0
        for k in range(dim+nB-1, -1, -1):
            if (olda[k] < newa[k]):
                comp = -1
                break
            if (olda[k] > newa[k]):
                comp = 1
                break
        
        if (comp == 0):
            pdc.xt[2*nV*i:2*nV*(i+1),:] = pdc.x[2*nV*j:2*nV*(j+1),:].copy()
            pdc.Wnew[i] = pdc.W[j]
            i += 1
            if (i >= int(nAnew/(2*nV))): break
            for k in range(dim+nB):
                newa[k] = 0
                for m in range(2*nV):
                    if (newa[k] == 0): newa[k] += int(np.floor(2*(pdc.Alnew[2*nV*i+m,k])+0.5))
            
            j += 1
            if (j >= int(pdc.nA/(2*nV))): break
            for k in range(dim+nB):
                olda[k] = 0
                for m in range(2*nV):
                    if (olda[k] == 0): olda[k] += int(np.floor(2*(pdc.Al[2*nV*j+m,k])+0.5))
        
        elif (comp == -1): # newa > olda
            j += 1
            if (j >= int(pdc.nA/(2*nV))): break
            for k in range(dim+nB):
                olda[k] = 0
                for m in range(2*nV):
                    if (olda[k] == 0): olda[k] += int(np.floor(2*(pdc.Al[2*nV*j+m,k])+0.5))
        
        else: # newa < olda
            pdc.Wnew[i], s = weight_func(pdc.xt[2*nV*i:2*nV*(i+1)])
            if (pdc.Wnew[i] > 1.): pdc.Wnew[i] = 1.
            i += 1
            if (i >= int(nAnew/(2*nV))): break
            for k in range(dim+nB):
                newa[k] = 0
                for m in range(2*nV):
                    if (newa[k] == 0): newa[k] += int(np.floor(2*(pdc.Alnew[2*nV*i+m,k])+0.5))
    
    # broken without populating Anew entirely               
    if (i < int(nAnew/(2*nV))):
        for t in range(i, int(nAnew/(2*nV))):
            pdc.Wnew[t], s = weight_func(pdc.xt[2*nV*t:2*nV*(t+1)])
            if (pdc.Wnew[t] > 1.): pdc.Wnew[t] = 1.
      
    # replace x with xt
    pdc.x, pdc.xt = pdc.xt.copy(), pdc.x.copy()
    # replace W with Wnew
    pdc.W, pdc.Wnew = pdc.Wnew.copy(), pdc.W.copy()
    # replace A with Anew, nA with nAnew
    pdc.Ad, pdc.Anew = pdc.Anew.copy(), pdc.Ad.copy()
    pdc.Al, pdc.Alnew = pdc.Alnew.copy(), pdc.Al.copy()
    pdc.nA = nAnew

    # set x2 = A . u
    pdc.x2[0:pdc.nA,:] = np.matmul(pdc.Ad[0:pdc.nA,:], pdc.u) # (nA, dim)     
     
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

def sortAold(atmp: np.array, btmp: np.array, xtmp: np.array, Atosort: np.array, Altosort: np.array, nAtosort: int):
    rra = np.empty(dim+nB)
    rra1 = np.empty(dim+nB)
    rra2 = np.empty(dim+nB)
    
    n = int(nAtosort/2)
    
    l = n - 1
    ir = n - 1
    
    while (True):
        if (l > 0):
            l -= 1
            for k in range(dim+nB):
                rra[k] = 0
                for m in range(2):
                    if (rra[k] == 0): rra[k] += np.floor(2*(Altosort[2*l+m,k])+0.5)
            
            atmp = Atosort[2*l:2*(l+1),:].copy()
            btmp = Altosort[2*l:2*(l+1),:].copy()
            xtmp = pdc.x[2*l:2*(l+1),:].copy()
            Wtemp = pdc.W[l]
        else:
            for k in range(dim+nB):
                rra[k] = 0
                for m in range(2):
                    if (rra[k] == 0): rra[k] += np.floor(2*(Altosort[2*ir+m,k])+0.5)
            
            atmp = Atosort[2*ir:2*(ir+1),:].copy()
            btmp = Altosort[2*ir:2*(ir+1),:].copy()
            xtmp = pdc.x[2*ir:2*(ir+1),:].copy()
            Wtemp = pdc.W[ir]
          
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




if __name__ == '__main__':

    pdc = pdc()
    pdc.allocate()
    
    initialize(pd_target=0.75)
    
    Ltrd()
    update_A()
    
    err = update_weights()
    calc_atwa()
    
    # 500000
    for i in range(500000):
        err = dm_step()
        
        print("err and mult:", err, err*pdc.nA)
        
        if ((i%50) == 49): Ltrd()
        update_A()
        err = update_weights()
        calc_atwa()
        
        print(err, pdc.nA)
        
        if (err < 8.e-11):
            update_A()
            err = update_weights()
            
            if (err < 8.e-11): break

    print("iteration count: ", i+1)
    
    print("Solution\n", pdc.u)