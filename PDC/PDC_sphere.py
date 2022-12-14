"""
* Implementation of the periodic divide and concur (PDC) developped by 
* Yoav kallus, adapated from 'Method for dense packing discovery'

* The present code is based on the original C version written by Yoav Kallus
* who kindly provides his code on demand.
"""

import numpy as np
from tenpy.linalg.svd_robust import svd
from numpy.linalg import norm
import sys
sys.path.append(r'/mnt/Edisk/andrew/Self_learning_MC')

from LLL import LLL_reduction
from global_var import *
from particle.sphere import *
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
def proj_nonoverlap(pair: np.array):
    """ the divide projection acts independently on each replica pair """
    pair_new = pair.copy()
    
    # check centroid-centroid distance
    dist = norm(pair[0] - pair[1])
    if (dist < outscribed_d):
        pair_new[0] += (2.-dist)/2./dist*(pair[0] - pair[1])
        pair_new[1] -= (2.-dist)/2./dist*(pair[0] - pair[1])
    
    return pair_new

def divide(input: np.array): 
    out = input[0:pdc.nA,:].copy() # (nA, dim)
    for i in range(0, pdc.nA, 2):
        out[i:i+2,:] = proj_nonoverlap(input[i:i+2,:])
    
    return out


#=========================================================================#
#  Concur projections                                                     #
#=========================================================================#
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
    return xx
  
def concur(input: np.array):
      out = input[0:pdc.nA,:].copy() # (nA, dim)
      # u = M_bar = atwainv . (Atran . W*input)
      for i in range(pdc.nA):
          out[i] = pdc.W[int(i/2)]*input[i]
      
      AtranWin = np.matmul(pdc.Ad[0:pdc.nA,:].T, out) # (dim+nB, dim)
      pdc.u = np.matmul(pdc.atwainv, AtranWin) # (dim+nB, dim)
      
      # L = Qinv*M0
      L = np.matmul(pdc.u[0:dim,:].T, pdc.Qinv.T)
      # U=plu (P), V=plv (R) (Kallus)
      plv, singval, plu = svd(L)
      
      # we need to make sure that sigma ??????????????????
      detL = np.prod(singval)
      if (np.fabs(detL) > pdc.V1): 
          detL = 1.
          for i in range(dim): detL *= singval[i]*Lambda().func1(singval[dim-1]/singval[i])
          
          if (np.fabs(detL) > pdc.V1):
              # need to use branch 2 for i=dim-1
              bracket = 0.
              
              while True:
                  bracket = singval[dim-1] - ((singval[dim-1] - bracket)/2.)
                  detL = 1.
                  for i in range(dim-1): detL *= singval[i]*Lambda().func1(bracket/singval[i])
                  detL *= singval[dim-1]*Lambda().func2(bracket/singval[dim-1])
                  if (np.fabs(detL) > pdc.V1): break
              
              mu = zbrent(0., bracket, singval, 1)
              for i in range(dim-1): singval[i] *= Lambda().func1(mu/singval[i])
              singval[dim-1] *= Lambda().func2(mu/singval[dim-1])
          else:
              bracket = 0
              while True:
                  bracket = singval[dim-1] - ((singval[dim-1] - bracket)/2.)
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
    packing.num_particles = nP
    
    # add particles
    packing.dim = 3
    packing.particles = [Sphere(1.) for i in range(packing.num_particles)]
    for i, particle in enumerate(packing.particles):
        particle.name = 'sphere %d' % i
        particle.color = np.array([0.51,0.792,0.992])
    
    # add cell
    packing.cell = Cell(packing.dim)
    packing.cell.color = np.array([0.25,0.25,0.25])
    
def dm_step():
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
    
    return: w(Xc) and overlap measure (4-dist^2 here)
    """
    dist = norm(pair[0] - pair[1])
    ret = outscribed_d**2 - dist**2
    
    if (ret < 0.): 
        y = (dist**2 - 3)**(-2-dim/2)
    else: 
        r2 = alpha*ret
        if (r2 > 5.): r2 = 5.
        y = np.exp(r2)
    
    return y, max(ret, 0.)

def update_weights():
    """ perform the weight adjustments according to Eq. (47) """
    ret = 0
    for i in range(0, pdc.nA, 2):
        w, s = weight_func(pdc.x2[i:i+2], 20)
        pdc.W[int(i/2)] = (tau*pdc.W[int(i/2)] + w) / (tau+1.)
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
    LRrnew[0:nB,0:dim] = pdc.u[dim:,:].copy() # u1
    Hinvd[0:dim,0:dim] = pdc.u[0:dim,:].copy() # u0
    
    # u1 = LRrnew*u0, then LRrnew = u1*u0^-1
    LRrnew[0:nB,0:dim] = np.matmul(LRrnew[0:nB,0:dim], np.linalg.inv(Hinvd[0:dim,0:dim]))
    
    unew[0:dim,:], H = LLL_reduction(pdc.u[0:dim,:], dim) # (dim, dim)
    
    Hd = np.zeros([dim+nB, dim+nB])
    Hd[0:dim,0:dim] = np.double(H) # G0
    # all primitive particles' centroids: -0.5<=Lambda<0.5
    for i in range(dim, dim+nB):
        for j in range(dim):
            Hd[i,j] = -np.floor(0.5 + LRrnew[i-dim,j])
    Hd[dim:,dim:] = np.diag(np.ones(nB))
    
    # Hinv = H^-1
    Hinvd = np.diag(np.ones(dim+nB)) 
    Hinvd[0:dim,0:dim] = np.linalg.inv(Hd[0:dim,0:dim])
    
    # unew = H.u
    unew = np.matmul(Hd, pdc.u)
    pdc.u = unew.copy()
    
    # Anew = A.Hinv (A' = )
    pdc.Anew[0:pdc.nA,:] = np.matmul(pdc.Ad[0:pdc.nA,:], Hinvd) # (nA, dim+nP)
    # A = Anew
    pdc.Anew, pdc.Ad = pdc.Ad.copy(), pdc.Anew.copy()
    
    # LRr_new = H.LRr(old)
    LRrnew = np.matmul(Hd, pdc.LRr)
    pdc.LRr = LRrnew.copy()
    
    pdc.Al[0:pdc.nA,:] = pdc.Ad[0:pdc.nA,:].copy()
    
    if (pdc.nA > 2): pdc.Ad, pdc.Al = sortAold(pdc.Ad, pdc.Al, pdc.nA)

def RotOpt():
    """ 
    Algorithm CLOSEPOINT adpated from "Closest Point Search in Lattices". step2: QR decomposition
    
    Return
    ----------
    g, h(int)
    """
    h = np.arange(0, dim, dtype=int)

    uu = np.zeros([dim, dim])
    # in the order u[h[0]], u[h[1]], ...
    for i in range(dim):
        uu[i] = pdc.u[h[i]].copy()

    # Gram schimidt, gs = Q (orthonormal matrix)
    gs = np.empty([dim, dim]) # (dim, dim)
    for i in range(dim):
        gs[i] = uu[i].copy()
        # gs[k] -= (gs[k].gs[l<k]) gs[l]
        for j in range(i): gs[i] -= np.dot(gs[i], gs[j])*gs[j]
        
        # normalized
        gs[i] /= norm(gs[i])
    
    # g[0:dim][:] = G3 = G2 * Q^T (lower-triangular matrix)
    # let x = x * Q^T
    g = np.empty([dim+nB, dim])
    g[0:dim,:] = np.matmul(uu, gs.T)
    g[dim:,:] = np.matmul(pdc.u[dim:,:], gs.T)
    
    return g, h

def ListClosest(rho0: np.double):
    """
    Using the generating matrix obtained in the concur projection, find 
    all replicas to represent.
    
    recreate Anew
    
    Parameters
    ----------
    rho0: pper bound on ||x^ - x||
    """
    # initilization
    max_breadth = 100
    xx = np.zeros(dim)
    idx = np.zeros(dim, dtype=int)
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
    for j in range(nB-1, -1, -1):
        for i in range(j, -1, -1):
            # from vnn to vn-1 n-1
            pair_level[npairs] = dim
            pair_x[npairs] = -(g[dim+j] - g[dim+i]) # centroid
            pair_rho[npairs] = rho0

            # starting index in nB
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
                pdc.Anew[nAnew,0:dim] = -0.6*toadd
                pdc.Anew[nAnew,dim:] = np.zeros(nB)
                pdc.Anew[nAnew,dim+p1] = 1.
                nAnew += 1
                
                pdc.Anew[nAnew,0:dim] = 0.4*toadd
                pdc.Anew[nAnew,dim:] = np.zeros(nB)
                pdc.Anew[nAnew,dim+p2] = 1.
                nAnew += 1
        
        if (nAnew > max_nA - 2):
            print("memory overflow")

    return nAnew


def update_A():
    olda = np.empty(dim+nB, dtype=int)
    newa = np.empty(dim+nB, dtype=int)
    
    nAnew = int(ListClosest(2*outscribed_d))
    
    pdc.Alnew[0:nAnew,:] = pdc.Anew[0:nAnew,:].copy() # (nAnew, dim+nP)
    pdc.xt[0:nAnew,:] = np.matmul(pdc.Anew[0:nAnew,:], pdc.u) # (nAnew, dim)
    pdc.x2[0:nAnew,:] = pdc.xt[0:nAnew,:].copy()
    
    j = i = int(0)
    if (pdc.nA > 0):
        # olda = A[j].LRr
        for k in range(dim+nB):
            olda[k] = 0
            for m in range(2):
                if (olda[k] == 0): olda[k] += int(np.floor(2*(pdc.Al[2*j+m,k])+0.5))
            
            newa[k] = 0
            for m in range(2):
                if (newa[k] == 0): newa[k] += int(np.floor(2*(pdc.Alnew[2*j+m,k])+0.5))
    
    while (True):
        if (j >= int(pdc.nA/2) or i >= int(nAnew/2)): break
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
            pdc.xt[2*i:2*(i+1),:] = pdc.x[2*j:2*(j+1),:].copy()
            pdc.Wnew[i] = pdc.W[j]
            i += 1
            if (i >= int(nAnew/2)): break
            for k in range(dim+nB):
                newa[k] = 0
                for m in range(2):
                    if (newa[k] == 0): newa[k] += int(np.floor(2*(pdc.Alnew[2*i+m,k])+0.5))
            
            j += 1
            if (j >= int(pdc.nA/2)): break
            for k in range(dim+nB):
                olda[k] = 0
                for m in range(2):
                    if (olda[k] == 0): olda[k] += int(np.floor(2*(pdc.Al[2*j+m,k])+0.5))
        
        elif (comp == -1): # newa > olda
            j += 1
            if (j >= int(pdc.nA/2)): break
            for k in range(dim+nB):
                olda[k] = 0
                for m in range(2):
                    if (olda[k] == 0): olda[k] += int(np.floor(2*(pdc.Al[2*j+m,k])+0.5))
        
        else: # newa < olda
            pdc.Wnew[i], s = weight_func(pdc.xt[2*i:2*(i+1)], 20)
            if (pdc.Wnew[i] > 1.): pdc.Wnew[i] = 1.
            i += 1
            if (i >= int(nAnew/2)): break
            for k in range(dim+nB):
                newa[k] = 0
                for m in range(2):
                    if (newa[k] == 0): newa[k] += int(np.floor(2*(pdc.Alnew[2*i+m,k])+0.5))
    
    # broken without populating Anew entirely               
    if (i < int(nAnew/2)):
        for t in range(i, int(nAnew/2)):
            pdc.Wnew[t], s = weight_func(pdc.xt[2*t:2*(t+1)], 20)
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

def calc_atwa():
    """ Used in Lattice constraint """
    # W is a diagonal matrix whose diagonal elements wi are the metric weights of different replicas
    # atwa = A^T . (W*A), Anew = W*A
    for i in range(pdc.nA):
        pdc.Anew[i,:] = pdc.W[int(i/2)] * pdc.Ad[i,:] # (nA, dim+nB)
    
    # atwa = A^T . Anew
    pdc.atwa = np.matmul(pdc.Ad[0:pdc.nA,:].T, pdc.Anew[0:pdc.nA,:]) # (dim+nB, dim+nB)
    
    # atwainv = atwa^-1
    pdc.atwainv = np.linalg.pinv(pdc.atwa)
    
    # let w' = atwa, the wtemp = (W'11)^-1, i.e., (atwa11)^-1
    wtemp = np.linalg.pinv(pdc.atwa[dim:,dim:]) # (nP, nP)
    # mw11iw10 = -(W'11)^-1 * W'10 | (nB, nB)*(nB, dim)
    # = - (atwa11)^-1 . atwa10, and m means minus, i means inverse
    pdc.mw11iw10 = -np.matmul(wtemp, pdc.atwa[dim:,0:dim]) # (nP, dim)
    
    # atwa2 = W'' = W'00 - W'01*(W'11)^-1 * W'10 = W'00 - W'01*temp
    atmp = pdc.atwa.copy()
    print(atmp[0:dim,0:dim])
    atmp[0:dim,0:dim] += np.matmul(pdc.atwa[0:dim,dim:], pdc.mw11iw10) # (dim, dim)
    
    print(atmp[0:dim,0:dim])
    eigs, featurevector = np.linalg.eigh(atmp[0:dim,0:dim])
    featurevector = featurevector.T
    
    # let Q = W^-1/2, then V1 (V_target) = V0*det(Q)
    pdc.V1 = pdc.V0
    for i in range(dim): pdc.V1 *= np.sqrt(eigs[i])
    
    # atwa = A.L.AT, eig_work=atmp=A
    # Qinv = W^1/2, so Qinv = A.(sqrt(L).AT) = A. (A.sqrt(L))T
    # (A.sqrt(L)) = Aij sqrt(Lj)
    eig_work = np.empty([dim, dim])
    for i in range(dim):
        eig_work[i,:] = np.sqrt(eigs[i])*featurevector[i,:]
    
    pdc.Qinv = np.matmul(featurevector.T, eig_work)
    
    for i in range(dim): 
        for j in range(dim):
            eig_work[i,j] /= eigs[i]
    pdc.Q = np.matmul(featurevector.T, eig_work)

def sortAold(Atosort: np.array, Altosort: np.array, nAtosort: int):
    rra = np.empty(dim+nB)
    rra1 = np.empty(dim+nB)
    rra2 = np.empty(dim+nB)
    
    n = int(nAtosort/2)
    
    l = n-1
    ir = n-1
    
    while True:
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
            Atosort[2*ir:2*(ir+1),:] = Atosort[0:2,:].copy()
            Altosort[2*ir:2*(ir+1),:] = Altosort[0:2,:].copy()
            pdc.x[2*ir:2*(ir+1),:] = pdc.x[0:2,:].copy()
            pdc.W[ir] = pdc.W[0]
            
            ir -= 1
            if (ir == 0):
                Atosort[0:2,:] = atmp.copy()
                Altosort[0:2,:] = btmp.copy()
                pdc.x[0:2,0:dim] = xtmp.copy()
                pdc.W[0] = Wtemp
                break
        
        i = l
        j = l+1
        while (j <= ir):
            for k in range(dim+nB):
                rra1[k] = rra2[k] = 0
                for m in range(2):
                    if (rra1[k] == 0): rra1[k] += np.floor(2*(Altosort[2*j+m,k])+0.5)
                    if (rra2[k] == 0): rra2[k] += np.floor(2*(Altosort[2*(j+1)+m,k])+0.5)
            
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
                rra1 = rra2.copy()
            
            comp = 0
            for k in range(dim+nB-1, -1, -1):
                if (rra[k] < rra1[k]): 
                    comp = -1
                    break
                if (rra[k] > rra1[k]):
                    comp = 1
                    break
            
            if (comp == -1):
                Atosort[2*i:2*(i+1),:] = Atosort[2*j:2*(j+1),:].copy()
                Altosort[2*i:2*(i+1),:] = Altosort[2*j:2*(j+1),:].copy()
                pdc.x[2*i:2*(i+1),:] = pdc.x[2*j:2*(j+1),:].copy()
                pdc.W[i] = pdc.W[j]
                i = j
                j <<= 1
            else: j = ir + 1
            
        Atosort[2*i:2*(i+1),:] = atmp.copy()
        Altosort[2*i:2*(i+1),:] = btmp.copy()
        pdc.x[2*i:2*(i+1),:] = xtmp.copy()
        pdc.W[i] = Wtemp
    
    return Atosort, Altosort
                 

#=========================================================================#
#  Visualization                                                          #
#=========================================================================#
def plot(repeat: bool):
    """ XYZ file """
    for i, particle in enumerate(packing.particles):
        particle.centroid = pdc.u[dim+i]
    
    packing.cell.lattice = pdc.u[0:dim,:]

    f = f'config.xyz'
    packing.output_xyz(f, repeat)



  
if __name__ == '__main__':
  
    pdc = pdc()
    pdc.allocate()
    
    initialize(pd_target=0.74047)
    
    # pdc.u[0:dim,:] = np.array([[3.852803, -0.002112, 0.005662],
    #                           [0.005969, 3.854232, -0.006049],
    #                           [-0.003296, 0.005365, 3.841554]])
    
    Ltrd()
    update_A()
    
    # # plot()
    err = update_weights()
    calc_atwa()
    
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
    
    plot(repeat=True)
    print(packing.fraction)

            
    
            
        
        
    
    
