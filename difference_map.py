import numpy as np
from particle.superellipsoid import *

# Follow the notation of Kallus
dim = 3 # d in paper
nB = 4 # p in paper
nA = 10 # number of rows for matrix A (nA, dim+nB), may change

# a pair of particles
replica = [SuperEllipsoid(2., 1., 1., 1.) for i in range(2)]

# here nP=dim+1
nP = 6 # number of vertices (dim of single particle)

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



def Ltrd():
    """ Lattice reduction """
    # H = G  = (G0 0, G1 1)
    Hd = np.zero(dim+nB, dim+nB) # (dim+nB, dim+nB)
  
    # u: (dim+nB, dim)
    LRrnew[0:nB][0:dim] = u[dim:][:]
    Hinvd[0:dim][0:dim] = u[0:dim][:]
    
    u0 = u[dim:][:]
    u1 = u[0:dim][:]
    
    Lattice = u[0:dim][:]
    Lattice_new, H = LLL(dim, Lattice) # (dim, dim)
    
    unew[0:dim][:] = Lattice_new
    
    Hd[0:dim][0:dim] = np.double(H) # H: int
    
    # To do: how to compute G1
    
    
  
    
    Hd[dim:][dim:] = np.diag(np.ones(nB))
    Hinvd = np.diag(np.ones(dim+nB)) # (dim+nB, dim+nB)
    
    Hd[0:dim][0:dim] = np.double(H[0:dim][0:dim])
    
    
    # 1253
    unew = np.matmul(Hd, u) # (dim+nB, dim)
    u = unew.copy()
    
    # Anew = A.Hinv
    Anew = np.matmul(Ad, Hinvd) # (nA, dim+nB)
    # A = Anew
    Anew, A = A, Anew
    
    # LRr_new = H.LRr(old)
    LRrnew = np.matmul(Hd, LRr)
    
    LRr = LRrnew.copy()
    Al = Ad.copy()
    
    sortAold()
            
        
   



def dm_step():
    err = 0.
    
    """ 
    f_D(X) = (1-1/beta)*pi_D(X) + 1/beta*X = X
    f_C(X) = (1+1/beta)*pi_C(X) - 1/beta*X = 2*pi_C(X) - X
    """
    x1 = p1(x) # pi_D(X)
    
    fc = 2.*x1 - x # f_C(X)
    
    x2 = p2(fc) # pi_
    
    err = np.dot(x1-x2, x1-x2)
    
    # iterate X = X + beta*(X_D-X_C)
    x -= (x1-x2)
    
    
def p1(x: np.array):
    # x: (dim+nB, dim)
    
    x_new = x.copy()
    for i in range(0, nA, 2*np):
        pair = x[i:i+2*np][:]
        pair_new = proj_nonoverlap(pair)
        x_new[i:i+2*np][:] = pair_new
    
    return x_new
    

def proj_nonoverlap(pair: np.array):
    # 输入的in应该是一对颗粒
    # check centroid-centroid distance
    # rotation matrix and translation vector (centroid)
    R1, r1 = pair[0:dim][:], pair[dim][:]
    R2, r2 = pair[np:2*dim+1][:], pair[2*dim+1][:]
    
    pair_new = pair.copy()
  
    dist = np.linalg.norm(r1 - r2)
    if (dist < outscribed_D):
        replica[0].centroid = r1
        replica[0].rot_mat = R1
        replica[1].centroid = r2
        replica[1].rot_mat = R2
        
        delta = overlap_measure(replica[0], replica[1])
        if (delta > 0.): 
            resolve_overlap(replica[0], replica[1])
            
            pair_new[0:dim][:], pair_new[dim][:] = replica[0].rot_mat, replica[0].centroid
            pair_new[np:2*dim+1][:], pair_new[2*dim+1][:] = replica[1].rot_mat, replica[1].centroid
    
    return pair_new

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

def zbrent(l1: np.double, l2: np.double, singval: np.array, branch: np.int):
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
                  
def p2():
      # u = M_bar = atwainv . ( Atran . W*in ) in: X
      out = np.matmul(W, in)
      
      AtranWin = np.matmul(Ad.T, out)
      u = np.matmul(atwainv, AtranWin)
      
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

def divide():
    for i in range():
        proj_nonoverlap()

def sortAold():
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
            
            atemp[0:2*nP][0:dim+nB] = Atosort[2*nP*l:2*nP*(l+1)][0:dim+nB]
            btemp[0:2*nP][0:dim+nB] = Altosort[2*nP*l:2*nP*(l+1)][0:dim+nB]
            xtmp[0:2*nP][0:dim] = x[2*nP*l:2*nP*(l+1)][0:dim]
        
            Wtemp = W[l]
        else:
            for k in range(dim+nB):
                rra[k] = 0
                for m in range(2*nP):
                    if (rra[k] == 0): rra[k] += np.floor(2*(Altosort[2*nP*ir+m][k])+0.5)
            
            atemp[0:2*nP][0:dim+nB] = Atosort[2*nP*ir:2*nP*(ir+1)][0:dim+nB]
            btemp[0:2*nP][0:dim+nB] = Altosort[2*nP*ir:2*nP*(ir+1)][0:dim+nB]
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
    ListClosest()
    
    A_back = Anew.copy() # (nAnew, dim+nB)
    # x_temp?
    xt = np.matmul(Anew, u) # (nAnew, dim)
    
    for k in range(dim+nB): 
        pass
      
    
    # replace x with xt
    x, xt = xt, x
    # replace W with Wnew
    W, Wnew = Wnew, W
    # replace A with Anew, nA with nAnew
    Ad, Anew = Anew, Ad
    Al, Alnew = Alnew, Al
    nA = nAnew
    
    # set x2 = A . u
    x2 = np.matmul(Ad, u)
    
    


def weight_func(pair: np.array):
    """
    assigns replica weights based on 
    their configuration in the concur estimate
    """
    
    # rotation matrix and translation vector (centroid)
    R1, r1 = pair[0:dim][:], pair[dim][:]
    R2, r2 = pair[dim+1:2*dim+1][:], pair[2*dim+1][:]
  
    dist = np.linalg.norm(r1 - r2)
    is_overlap = False
    if (dist < outscribed_D):
        replica[0].centroid = r1
        replica[0].rot_mat = R1
        replica[1].centroid = r2
        replica[1].rot_mat = R2
        
        delta = overlap_measure(replica[0], replica[1])
        if (delta > 0.): 
            is_overlap = True
            delta_square = delta**2
            
            # Todo: ret??
    
    if (is_overlap): return np.exp(alpha*delta_square)
    else: 
        y = (1. + dist**2 - inscribed_D**2)**(-2)
        return y
      
    



def update_weights(tau: np.double):
    for i in range(0, nA, 2*np):
        pair = x[i:i+2*np][:] # slice first
        W[i/(2*np)] =  (tau*W[i/(2*np)] + weight_func(pair)) / (tau+1.)
        
        # Todo: ret?
 

def RotOpt():
    """ 
    Algorithm CLOSEPOINT adpated from "Closest Point Search in Lattices".
    
    step2: QR decomposition
    """
    
    # input: (dim+nB)*dim
    lattice = input[0:dim][:]
    
    h = np.zeros(dim)
    for i in range(dim): h[i] = i

    
    uu = np.zeros(dim, dim)
    for i in range(dim):
        uu[i][:] = lattice[h[i]][:]

    # Gram schimidt in the order u[h[0]], u[h[1]], ...
    # gs = Q (orthonormal matrix)
    gs = np.zeros(dim, dim) # (dim, dim)
    for i in range(dim):
        gs[i] = u[h[i]]
        # gs[k] -= (gs[k].gs[l<k]) gs[l]
        for j in range(i): gs[i] -= np.dot(u[h[i]], gs[j])*gs[j]
        
        # normalized
        gs[i] /= np.norm.linalg(gs[i])
    
    # g[0:dim][:] = G3 = G2 * Q^T (lower-triangular matrix)
    # let x = x * Q^T
    g = np.matmul(input, gs.T) # (dim+nB, dim)
    
    return h, g

def ListClosest():
    """ Gram schimit
    """
    
    # input: rho0: upper bound on ||x^ - x||
    # where x^ denote the closest lattice point to x
    npairs = 0
    for j in range(nB-nP, 0, -nP):
        for i in range(j, 0, -nP):
            # from vnn to vn-1 n-1
            pair_level[npairs] = dim
            pair_x[npairs] = g[j+dim] - g[i+dim] # centroid
            pair_rho[npairs] = roh0
            # starting index in nB
            pair_b1[npairs], pair_b2[npairs] = j, i
            npairs += 1
    
    """
    Our criterion for which replicas to represent is based
    on the difference map’s current concur estimate: we include
    a replica pair for each pair of particles whose centroids in the
    concur estimate are closer than some cutoff distance """
    nAnew = 0
    while (npairs > 0):
        npairs -= 1
        level = pair_level[npairs]
        
        xx[0:level] = pair_x[npairs][0:level]

        x = pair_x[npairs]
        rho = pair_rho[npairs]
        p1, p2 = pair_b1[npairs], pair_b2[npairs]
        
        
        
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
                pair_leve[npairs] = level - 1
                
                for i in range(level-1):
                    pair_x[npairs][i] = xx[i] - indice*g[k][i]
                
                pair_rho[npairs] = np.sqrt(rho**2 - (indice*vperp-vperp)**2)
                

    
    # 1476-1493: add a pair
    # p1, p2: particle id of a pair
    if (p1 != p2): pass


    
    
    
    # perp: perpendicular???
    
  
  
def calc_atwa():

    # W is a diagonal matrix whose diagonal elements wi are the metric weights of different replicas
    # temp = WA
    temp = np.matmul(W, Ad)
    
    atwa = np.matmul(Ad.T, temp) # (dim+nB, dim+nB)
    atwainv = np.linalg.pinv(atwa)
    
    # temp = (W'11)^-1 * W'10 | (nB, nB)*(nB, dim)
    wtemp = np.linalg.pinv(atwa[dim:][dim:])
    # w' = atwa
    # i means inverse
    temp = np.matmul(wtemp, atwa[dim:][0:dim]) # (nB, dim)
    
    # atwa2 = W'' = W'00 - W'01*(W'11)^-1 * W'10
    atwa2 = atwa[0:dim][0:dim] - np.matmul(atwa[0:dim][dim:], temp) # (dim, dim)
    
    eigenvalue, featurevector = np.linalg.eig(atwa2)
    
    # let Q = W^-1/2, then V1 (V_target) = V0*det(Q)
    V1 = V0
    for i in range(dim): V1*=np.sqrt(eigenvalue[i])




if __name__ == '__main__':
    LRr = np.diag(np.ones(dim+nB))
    
    Ltrd()
    
