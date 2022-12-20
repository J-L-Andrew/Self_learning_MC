"""
* Implementation of the periodic divide and concur (PDC) developped by 
* Yoav kallus, adapated from 'Method for dense packing discovery'

* The present code is based on the original C version written by Yoav Kallus
* who kindly provides his code on demand.
"""
import numpy as np
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
    
    js = np.empty(2*nV, dtype=int)
    minjs = np.empty(2*nV, dtype=int)
    gs = np.empty([dim, dim])
    quad = np.empty([dim, dim])
    
    s1 = np.sum(pair[0:nV,:], axis=0) / nV
    s2 = np.sum(pair[nV:2*nV,:], axis=0) / nV
    dist = norm(s1 - s2)
    
    if (dist**2 > 4.): return pair_new
    
    # check if pair is already non-overlapping
    js[0:dim-1] = np.arange(0, dim-1)
    js[dim-1] = nV
        
    while (True):
        s1 = s2 = 0. # delta^2+(S) & delta^2-(S)
        
        for i in range(dim-1):
            gs[i] = pair[js[i+1]] - pair[js[0]]
        
        quad[dim-1,0] = gs[0,1]*gs[1,2] - gs[0,2]*gs[1,1]
        quad[dim-1,1] = gs[0,2]*gs[1,0] - gs[0,0]*gs[1,2]
        quad[dim-1,2] = gs[0,0]*gs[1,1] - gs[0,1]*gs[1,0]
        quad[dim-1] /= norm(quad[dim-1])
        
        for i in range(nV):
            s = 0.
            for j in range(dim):
                s += (pair[i,j] - pair[js[0],j])*quad[dim-1,j]
            if (s < 0.): s1 += s**2
            else: s2 += s**2
            
        for i in range(nV, 2*nV):
            s = 0.
            for j in range(dim):
                s += (pair[i,j] - pair[js[0],j])*quad[dim-1,j]
            if (s > 0.): s1 += s**2
            else: s2 += s**2
            
        if (s1 < PLANE_TOL or s2 < PLANE_TOL): return pair_new
            
        js[dim-1] += 1
        if (js[dim-1] == 2*nV):
            k = 1
            while (True):
                js[dim-1-k] += 1
                if (js[dim-1-k] == 2*nV-k): k += 1
                else: break
                
            if (js[0] >= nV): break
            for i in range(k-1, -1, -1):
                js[dim-1-i] = js[dim-2-i] + 1
            if (js[dim-1] < nV): js[dim-1] = nV
      
    # no co-1 plane saparates the two hulls
    nj = dim + 1
    js[0:2*dim] = np.arange(0, 2*dim)
    js[nj-1] = nV
    
    mind = bestatlevel = np.Infinity
    while (True):
        # distance to coplane is given by smallest eig of X^T.X matrix
        xavg = np.zeros(dim)
        for i in range(nj):
            for j in range(dim):
                xavg[j] += pair[js[i],j] / nj
        quad = np.zeros([dim, dim])
        
        for i in range(dim):
            for j in range(dim):
                for k in range(nj):
                    quad[i,j] += (pair[js[k],i] - xavg[i])*(pair[js[k],j] - xavg[j])
        
        eigs, featurevector = np.linalg.eig(quad)
        sorted_indices = np.argsort(eigs)
        eigs = eigs[sorted_indices].copy()
        featurevector = featurevector[sorted_indices].copy()
        
        # check if projecting to the coplane satisfies the constraint
        # first row of quad has lowest eigenvector
        if (bestatlevel > eigs[0]):
            bestatlevel = eigs[0]
            bestchekedout = 0
        
        sign = 0.
        index = 0
        for i in range(nV):
            count = 0
            for j in range(nj):
                if (i == js[j]): break
                count += 1
            if (count == nj):
                s = 0.
                for k in range(dim):
                    s += (pair[i,k] - xavg[k])*featurevector[0,k]
                if (sign == 0.):
                    if (s < -PLANE_TOL): sign = -1.
                    if (s > PLANE_TOL): sign = 1.
                    
                if (s*sign < -PLANE_TOL): break
            index += 1
        
        if (index == nV):
            count = 0
            for i in range(nV, 2*nV):
                for j in range(nj):
                    if (i == js[j]): break
                    count += 1
                if (count == nj):
                    s = 0.
                    for k in range(dim):
                        s += (pair[i,k] - xavg[k])*featurevector[0,k]
                        if (s*sign > PLANE_TOL): break
                index += 1
            
            if (index == 2*nV):
                bestchekedout = 1
                if (mind > eigs[0]):
                    mind = eigs[0]
                    minjs[0:nj] = js[0:nj].copy()
                    mineig = featurevector[0,:].copy()
                    minnj = nj
                    
        js[nj-1] += 1
        if (js[nj-1] == 2*nV):
            k = 1
            while (True):
                js[nj-1-k] += 1
                if (js[nj-1-k] == 2*nV-k): k += 1
                else: break
            for i in range(k-1, -1, -1):
                js[nj-1-i] = js[nj-2-i] + 1
            if (js[nj-1] < nV): js[nj-1] = nV
            if (js[0] >= nV):
                if (bestchekedout == 1): break
                nj += 1
                if (nj > 2*nV-1): break
                js[0:nj-1] = np.arange(0, nj-1)
                js[nj-1] = nV
    
    if (mind != np.Infinity):
        s = 0.
        for i in range(minnj):
            s += np.dot(pair[minjs[i]], mineig)
        s /= minnj
        
        for i in range(minnj):
            t = np.dot(pair[minjs[i]], mineig) - s
            pair_new[minjs[i]] -= t*mineig
    
    return pair_new

def divide(input: np.array): 
    out = input[0:pdc.nA,:].copy() # (nA, dim)
    for i in range(0, pdc.nA, 2*nV):
        out[i:i+2*nV,:] = proj_nonoverlap(input[i:i+2*nV,:])
    
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
        fa = fa/pdc.V1 - 1.
        
        fb = np.prod(singval)
        for i in range(dim): fb *= Lambda().func0(b/singval[i])
        fb = fb/pdc.V1 - 1.
    else:
        fa = np.prod(singval)
        for i in range(dim-1): fa *= Lambda().func1(a/singval[i])
        if (branch == 0): fa *= Lambda().func1(a/singval[dim-1]) 
        else: fa *= Lambda().func2(a/singval[dim-1])
        fa = fa/pdc.V1 - 1.
        
        fb = np.prod(singval)
        for i in range(dim-1): fb *= Lambda().func1(b/singval[i])
        if (branch == 0): fb *= Lambda().func1(b/singval[dim-1]) 
        else: fb *= Lambda().func2(b/singval[dim-1])
        fb = fb/pdc.V1 - 1.
    
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
            fb = fb/pdc.V1 - 1.
        else:
            fb = np.prod(singval)
            for i in range(dim-1): fb *= Lambda().func1(b/singval[i])
            if (branch == 0): fb *= Lambda().func1(b/singval[dim-1])
            else: fb *= Lambda().func2(b/singval[dim-1])
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
                ff = np.prod(singval)
                for i in range(dim): ff *= Lambda().func0(xx/singval[i])
                ff = ff/pdc.V1 - 1.
                      
                dff = 0.
                for i in range(dim): dff += Lambda().dfunc0(xx/singval[i])/(singval[i]*Lambda().func0(xx/singval[i]))
                dff *= ff
            else:
                ff = np.prod(singval)
                for i in range(dim-1): ff *= Lambda().func1(xx/singval[i])
                if (branch == 0): ff *= Lambda().func1(xx/singval[dim-1])
                else: ff *= Lambda().func2(xx/singval[dim-1])
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
                ff = np.prod(singval)
                for i in range(dim): ff *= singval[i]*Lambda().func0(xx/singval[i])
                ff = ff/pdc.V1 - 1.
                      
                dff = 0.
                for i in range(dim): dff+=Lambda().dfunc0(xx/singval[i])/(singval[i]*Lambda().func0(xx/singval[i]))
                dff *= ff
            else:
                ff = np.prod(singval)
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
          out[i] = pdc.W[int(i/(2*nV))]*input[i]
      
      AtranWin = np.matmul(pdc.Ad[0:pdc.nA,:].T, out) # (dim+nP, dim)
      pdc.u = np.matmul(pdc.atwainv, AtranWin) # (dim+nP, dim)
      
      # L = Qinv*M0
      L = np.matmul(pdc.u[0:dim,:].T, pdc.Qinv.T)
      # To do: what hell is fortran style?
      #L = np.matmul(pdc.Qinv, pdc.u[0:dim,:]) # (dim, dim)
      # U=plu (P), V=plv (R) (Kallus)
      plu, singval, plv = np.linalg.svd(L,full_matrices=False)
      
      # we need to make sure that sigma 从大到小排序
      detL = np.prod(singval)
      
      if (np.fabs(detL) > pdc.V1): 
          for i in range(dim): detL *= Lambda().func1(singval[dim-1]/singval[i])
          
          if (np.fabs(detL) > pdc.V1):
              # need to use branch 2 for i=dim-1
              bracket = 0.
              
              while True:
                  bracket = singval[dim-1] - ((singval[dim-1] - bracket)/2.)
                  detL = np.prod(singval)
                  for i in range(dim-1): detL *= Lambda().func1(bracket/singval[i])
                  detL *= Lambda().func2(bracket/singval[dim-1])
                  if (np.fabs(detL) < pdc.V1): break
              
              mu = zbrent(0., bracket, singval, 1)
              for i in range(dim-1): singval[i] *= Lambda().func1(mu/singval[i])
              singval[dim-1] *= Lambda().func2(mu/singval[dim-1])
          else:
              bracket = 0
              while True:
                  bracket = singval[dim-1] - ((singval[dim-1] - bracket)/2.)
                  detL = np.prod(singval)
                  for i in range(dim): detL *= Lambda().func1(bracket/singval[i])
                  if (np.fabs(detL) > pdc.V1): break
              
              mu = zbrent(0., bracket, singval, 0)
              for i in range(dim): singval[i] *= Lambda().func1(mu/singval[i])
      else:
          bracket = singval[dim-1]/1024.
          
          while True:
              bracket *= 2.
              detL = np.prod(singval)
              for i in range(dim): detL *= Lambda().func0(bracket/singval[i])
              if (np.fabs(detL) < pdc.V1): break
              
          mu = zbrent(0., bracket, singval, -1)
          for i in range(dim): singval[i] *= Lambda().func0(mu/singval[i])
      
      # let U0 = Q.P.SINGVAL.R
      for i in range(dim):
          for j in range(dim):
              plv[i] *= singval[j]

      AtranWin = np.matmul(plu.T, plv.T) # (dim, dim)
      
      # dU = Q.AtranWin - U
      plu = np.matmul(pdc.Q, AtranWin) - pdc.u[0:dim,:]
      
      # then let U1 += (-atwa11inv . atwa10) . (U0 - U0init)
      pdc.u[dim:,:] += np.matmul(pdc.mw11iw10, plu)
      
      # U += dU
      pdc.u[0:dim,:] += plu
      
      # Rigidity constraint
      # for i in range(nP): 
      #     u[dim+i, dim+i+nP][:] = proj_rigid(u[dim+i, dim+i+nP][:])
      
      out = np.matmul(pdc.Ad[0:pdc.nA,:], pdc.u)
      
      return out


#=========================================================================#
#  Difference map                                                         #
#=========================================================================#  
def initialize(pd_target: np.double):
    """ start from random initial configurations """
    pdc.nA = 0
    pdc.V0 = nB*replica[0].volume / pd_target
    
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
    
def dm_step():
    err = 0.
    
    # f_D(X) = (1-1/beta)*pi_D(X) + 1/beta*X = X
    # f_C(X) = (1+1/beta)*pi_C(X) - 1/beta*X = 2*pi_C(X) - X
    pdc.x1[0:pdc.nA,:] = divide(pdc.x) # pi_D(X)
    
    pdc.xt[0:pdc.nA,:] = 2.*pdc.x1[0:pdc.nA,:] - pdc.x[0:pdc.nA,:] # f_C(X)
    
    pdc.x2[0:pdc.nA,:] = concur(pdc.xt) # pi_
    # err <- ||XC - XD||
    delta = pdc.x1[0:pdc.nA] - pdc.x2[0:pdc.nA]
    err = np.dot(delta, delta)
    
    if (err > pdc.nA*maxstep):
        pdc.x1[0:pdc.nA] *= np.sqrt(maxstep*pdc.nA/err)
        err = pdc.nA*maxstep
    
    # iterate X = X + beta*(X_D-X_C)
    pdc.x[0:pdc.nA] += delta
    
    return err/pdc.nA

# def weight_func(pair: np.array):
#     """
#     A function that assigns replica weights based on their 
#     configuration in the concur estimate.
    
#     return: w(Xc) and overlap measure (dist here)
#     """
#     js = np.empty(10, dtype=int)
#     gs = np.empty([dim, dim])
#     quad = np.empty([dim, dim])
    
#     s1 = np.sum(pair[0:nV,:], axis=0) / nV
#     s2 = np.sum(pair[nV:2*nV,:], axis=0) / nV
#     dist = norm(s1 - s2)
    
#     if (dist < outscribed_d):
#         js[0:dim-1] = np.arange(0, dim-1)
#         js[dim-1] = nV
        
#         r2 = np.Infinity
        
#         while (True):
#             s1 = s2 = 0. # delta^2+(S) & delta^2-(S)
        
#             for i in range(dim-1):
#                 gs[i] = pair[js[i+1]] - pair[js[0]]
        
#             quad[dim-1,0] = gs[0,1]*gs[1,2] - gs[0,2]*gs[1,1]
#             quad[dim-1,1] = gs[0,2]*gs[1,0] - gs[0,0]*gs[1,2]
#             quad[dim-1,2] = gs[0,0]*gs[1,1] - gs[0,1]*gs[1,0]
        
#             quad[dim-1] /= norm(quad[dim-1])
        
#             for i in range(nV):
#                 s = 0.
#                 for j in range(dim):
#                     s += (pair[i,j] - pair[js[0],j])*quad[dim-1,j]
#                 if (s < 0.): s1 += s**2
#                 else: s2 += s**2
            
#             for i in range(nV, 2*nV):
#                 s = 0.
#                 for j in range(dim):
#                     s += (pair[i,j] - pair[js[0],j])*quad[dim-1,j]
#                 if (s > 0.): s1 += s**2
#                 else: s2 += s**2
            
#             s2 = min(s1, s2)
#             if (s2 < r2): r2 = s2
            
#             js[dim-1] += 1
#             if (js[dim-1] == 2*nV):
#                 k = 1
#                 while (True):
#                     js[dim-1-k] += 1
#                     if (js[dim-1-k] == 2*nV-k): k += 1
#                     else: break
                
#                 if (js[0] >= nV): break
#                 for i in range(k-1, -1, -1):
#                     js[dim-1-i] = js[dim-2-i] + 1
#                 if (js[dim-1] < nV): js[dim-1] = nV
      
#     else: r2 = 0.
#     ret = r2
    
#     if (r2 < PLANE_TOL): r2 = -(dist**2 - inscribed_d**2) # (ri^2 - 4rin^2)
#     else: r2 *= 10.
    
#     if (r2 > 5.): r2 = 5.
#     if (r2 > 0.): y = np.exp(r2)
#     else: y = (1 - r2)**(-2)
    
#     return y, ret

def weight_func(pair: np.array):
    """
    A function that assigns replica weights based on their 
    configuration in the concur estimate.
    
    return: w(Xc) and overlap measure (dist here)
    """
    js = np.empty(10, dtype=int)
    gs = np.empty([dim, dim])
    quad = np.empty([dim, dim])
    
    s1 = np.sum(pair[0:nV], axis=0) / nV
    s2 = np.sum(pair[nV:2*nV], axis=0) / nV
    dist = norm(s1 - s2)
    
    if (dist**2 < 4.):
        js[0:dim-1] = np.arange(0, dim-1)
        js[dim-1] = nV
        
        r2 = np.Infinity
        while (True):
            s1 = s2 = 0. # delta^2+(S) & delta^2-(S)
        
            for i in range(dim-1):
                gs[i] = pair[js[i+1]] - pair[js[0]]
        
            quad[dim-1,0] = gs[0,1]*gs[1,2] - gs[0,2]*gs[1,1]
            quad[dim-1,1] = gs[0,2]*gs[1,0] - gs[0,0]*gs[1,2]
            quad[dim-1,2] = gs[0,0]*gs[1,1] - gs[0,1]*gs[1,0]
            quad[dim-1] /= norm(quad[dim-1])
        
            for i in range(nV):
                s = 0.
                for j in range(dim):
                    s += (pair[i,j] - pair[js[0],j])*quad[dim-1,j]
                if (s < 0.): s1 += s**2
                else: s2 += s**2
            
            for i in range(nV, 2*nV):
                s = 0.
                for j in range(dim):
                    s += (pair[i,j] - pair[js[0],j])*quad[dim-1,j]
                if (s > 0.): s1 += s**2
                else: s2 += s**2
            
            s2 = min(s1, s2)
            if (s2 < r2): r2 = s2
            
            js[dim-1] += 1
            if (js[dim-1] == 2*nV):
                k = 1
                while (True):
                    js[dim-1-k] += 1
                    if (js[dim-1-k] == 2*nV-k): k += 1
                    else: break
                
                if (js[0] >= nV): break
                for i in range(k-1, -1, -1):
                    js[dim-1-i] = js[dim-2-i] + 1
                if (js[dim-1] < nV): js[dim-1] = nV
      
    else: r2 = 0.
    ret = r2
    
    if (r2 < PLANE_TOL): r2 = -(dist**2 - 4./3.) # (ri^2 - 4rin^2)
    else: r2 *= 10.
    
    if (r2 > 5.): r2 = 5.
    if (r2 > 0.): y = np.exp(r2)
    else: y = (1 - r2)**(-2)
    
    return y, ret

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
    LRrnew[0:nB,0:dim] = pdc.u[dim:,:].copy() # u1
    Hinvd[0:dim,0:dim] = pdc.u[0:dim,:].copy() # u0
    
    # u1 = LRrnew*u0, then LRrnew = u1*u0^-1
    LRrnew[0:nB,0:dim] = np.matmul(LRrnew[0:nB,0:dim], np.linalg.inv(Hinvd[0:dim,0:dim]))
    
    unew[0:dim,:], H = LLL_reduction(pdc.u[0:dim,:], dim) # (dim, dim)
    
    Hd = np.zeros([dim+nB, dim+nB])
    Hd[0:dim,0:dim] = np.double(H) # G0
    # all primitive particles' centroids: -0.5<=Lambda<0.5
    for i in range(dim, dim+nB, nV):
        for j in range(dim):
            s = np.sum(LRrnew[i-dim:i-dim+nV,j]) / nV
            Hd[i:i+nV,j] = -np.floor(0.5 + s)
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
    
    if (pdc.nA >= 2*nV): pdc.Ad, pdc.Al = sortAold(pdc.Ad, pdc.Al, pdc.nA)

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
        
        gs[i] /= norm(gs[i]) # normalized
    
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
            
            pair_x[npairs,:] = np.zeros(dim)
            for n in range(nV):
                pair_x[npairs] += -(g[dim+j+n] - g[dim+i+n])/nV # centroid
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
            # cout += 1
            for i in range(dim): toadd[perm[i]] = -idx[i]
            
            count = 0
            for i in range(dim):
                if (toadd[i] != 0): break
                count += 1
            
            if (p1 != p2 or count != dim):
                for n in range(nV):
                    pdc.Anew[nAnew,0:dim] = -0.6*toadd
                    pdc.Anew[nAnew,dim:] = np.zeros(nB)
                    pdc.Anew[nAnew,dim+p1+n] = 1.
                    nAnew += 1
                
                for n in range(nV):
                    pdc.Anew[nAnew,0:dim] = 0.4*toadd
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
    pdc.xt = np.matmul(pdc.Anew[0:nAnew,:], pdc.u) # (nAnew, dim)
    pdc.x2[0:nAnew,:] = pdc.xt[0:nAnew,:].copy()
    
    j = i = int(0)
    if (pdc.nA > 0):
        # olda = A[j].LRr
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

def calc_atwa():
    """ Used in Lattice constraint """
    # W is a diagonal matrix whose diagonal elements wi are the metric weights of different replicas
    # atwa = A^T . (W*A), Anew = W*A
    for i in range(pdc.nA):
        pdc.Anew[i,:] = pdc.W[int(i/(2*nV))] * pdc.Ad[i,:] # (nA, dim+nB)
    
    # atwa = A^T . Anew
    pdc.atwa = np.matmul(pdc.Ad[0:pdc.nA,:].T, pdc.Anew[0:pdc.nA,:]) # (dim+nB, dim+nB)
    
    # atwainv = atwa^-1
    pdc.atwainv = np.linalg.pinv(pdc.atwa)
    
    # let w' = atwa, the wtemp = (W'11)^-1, i.e., (atwa11)^-1
    wtemp = np.linalg.pinv(pdc.atwa[dim:,dim:]) # (nB, nB)
    # mw11iw10 = -(W'11)^-1 * W'10 | (nB, nB)*(nB, dim)
    # = - (atwa11)^-1 * atwa10, and m means minus, i means inverse
    pdc.mw11iw10 = -np.matmul(wtemp, pdc.atwa[dim:,0:dim]) # (nB, dim)
    
    # atwa2 = W'' = W'00 - W'01*(W'11)^-1 * W'10 = W'00 - W'01*temp
    atmp = pdc.atwa.copy()
    atmp[0:dim,0:dim] += np.matmul(pdc.atwa[0:dim,dim:], pdc.mw11iw10) # (dim, dim)
    
    eigs, featurevector = np.linalg.eig(atmp[0:dim,0:dim])
    sorted_indices = np.argsort(eigs)
    eigs = eigs[sorted_indices].copy()
    featurevector = featurevector[sorted_indices].copy()

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
    
    n = int(nAtosort/(2*nV))
    
    l = n-1
    ir = n-1
    
    while True:
        if (l > 0):
            l -= 1
            for k in range(dim+nB):
                rra[k] = 0
                for m in range(2*nV):
                    if (rra[k] == 0): rra[k] += np.floor(2*(Altosort[2*nV*l+m,k])+0.5)
            
            atmp = Atosort[2*nV*l:2*nV*(l+1),:].copy()
            btmp = Altosort[2*nV*l:2*nV*(l+1),:].copy()
            xtmp = pdc.x[2*nV*l:2*nV*(l+1),:].copy()
            Wtemp = pdc.W[l]
        else:
            for k in range(dim+nB):
                rra[k] = 0
                for m in range(2*nV):
                    if (rra[k] == 0): rra[k] += np.floor(2*(Altosort[2*nV*ir+m,k])+0.5)
            
            atmp = Atosort[2*nV*ir:2*nV*(ir+1),:].copy()
            btmp = Altosort[2*nV*ir:2*nV*(ir+1),:].copy()
            xtmp = pdc.x[2*nV*ir:2*nV*(ir+1),:].copy()
            Wtemp = pdc.W[ir]
            
            # put Atosort[0] into Atosort[ir]
            Atosort[2*nV*ir:2*nV*(ir+1),:] = Atosort[0:2*nV,:].copy()
            Altosort[2*nV*ir:2*nV*(ir+1),:] = Altosort[0:2*nV,:].copy()
            pdc.x[2*nV*ir:2*nV*(ir+1),:] = pdc.x[0:2*nV,:].copy()
            pdc.W[ir] = pdc.W[0]
            
            ir -= 1
            if (ir == 0):
                Atosort[0:2*nV,:] = atmp.copy()
                Altosort[0:2*nV,:] = btmp.copy()
                pdc.x[0:2*nV,0:dim] = xtmp.copy()
                pdc.W[0] = Wtemp
                break
        
        i = l
        j = l + 1
        while (j <= ir):
            for k in range(dim+nB):
                rra1[k] = rra2[k] = 0
                for m in range(2*nV):
                    if (rra1[k] == 0): rra1[k] += np.floor(2*(Altosort[2*nV*j+m,k])+0.5)
                    if (rra2[k] == 0): rra2[k] += np.floor(2*(Altosort[2*nV*(j+1)+m,k])+0.5)
            
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
                rra1 = rra2
            
            comp = 0
            for k in range(dim+nB-1, -1, -1):
                if (rra[k] < rra1[k]): 
                    comp = -1
                    break
                if (rra[k] > rra1[k]):
                    comp = 1
                    break
            
            if (comp == -1):
                Atosort[2*nV*i:2*nV*(i+1),:] = Atosort[2*nV*j:2*nV*(j+1),:].copy()
                Altosort[2*nV*i:2*nV*(i+1),:] = Altosort[2*nV*j:2*nV*(j+1),:].copy()
                pdc.x[2*nV*i:2*nV*(i+1),:] = pdc.x[2*nV*j:2*nV*(j+1),:].copy()
                pdc.W[i] = pdc.W[j]
                i = j
                j <<= 1
            else: j = ir + 1
            
        Atosort[2*nV*i:2*nV*(i+1),:] = atmp.copy()
        Altosort[2*nV*i:2*nV*(i+1),:] = btmp.copy()
        pdc.x[2*nV*i:2*nV*(i+1),:] = xtmp.copy()
        pdc.W[i] = Wtemp
    
    return Atosort, Altosort
                

#=========================================================================#
#  Visualization                                                          #
#=========================================================================#
def plot():
    """ XYZ file """
    for i, particle in enumerate(packing.particles):
        particle.centroid = pdc.u[dim+i]
    
    packing.cell.lattice = pdc.u[0:dim,:]

    f = f'config.xyz'
    packing.output_xyz(f, repeat=False)



  
if __name__ == '__main__':
  
    pdc = pdc()
    pdc.allocate()
    
    initialize(pd_target=0.75)
    
    # [2., 20., 30.],
    #                           [22., 1., 0.],
    #                           [0., 10., 1.],
    
    # [3.852803, -0.002112, 0.005662],
    #                           [0.005969, 3.854232, -0.006049],
    #                           [-0.003296, 0.005365, 3.841554],
    
    pdc.u[0:dim+nB,:] = np.array([[3.852803, -0.002112, 0.005662],
                              [0.005969, 3.854232, -0.006049],
                              [-0.003296, 0.005365, 3.841554],
                              [0.001079, -0.000452, 0.002577],
                              [-0.002704, 0.000268, 0.009045],
                              [0.008324, 0.002714, 0.004346],
                              [-0.007168, 0.002139, -0.009674],
                              [-0.005142, -0.007255, 0.006084],
                              [-0.006866, -0.001981, -0.007404],
                              [-0.007824, 0.009978, -0.005635],
                              [0.000259, 0.006782, 0.002253],
                              [-0.004079, 0.002751, 0.000486],
                              [-0.000128, 0.009456, -0.004150],
                              [0.005427, 0.000535, 0.005398],
                              [-0.001995, 0.007831, -0.004334],
                              [-0.002951, 0.006154, 0.008381],
                              [-0.008605, 0.008987, 0.000520],
                              [-0.008279, -0.006156, 0.003265],
                              [0.007805, -0.003022, -0.008717],
                              [-0.009600, -0.000846, -0.008738],
                              [-0.005234, 0.009413, 0.008044],
                              [0.007018, -0.004667, 0.000795],
                              [-0.002496, 0.005205, 0.000251],
                              [0.003354, 0.000632, -0.009214],
                              [-0.001247, 0.008637, 0.008616],
                              [0.004419, -0.004314, 0.004771],
                              [0.002800, -0.002919, 0.003757]])
    
    Ltrd()
    update_A()
    
    # # # plot()
    err = update_weights()
    
    calc_atwa()
    
    for i in range(500000):
        err = dm_step()
        
        if ((i%50) == 49): Ltrd()
        update_A()
        err = update_weights()
        calc_atwa()
        
        if (err < 8.e-11):
            update_A()
            err = update_weights()
            
            if (err < 8.e-11): break
            
    
            
        
        
    
    
