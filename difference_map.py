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

def p2():
      # u = M_bar = atwainv . ( Atran . W*in ) in: X
      out = np.matmul(W, in)
      
      AtranWin = np.matmul(Ad.T, out)
      u = np.matmul(atwainv, AtranWin)
      
      # L = Qinv*M0
      L = np.matmul(Qinv[0:dim][0:dim], u[0:dim][0:dim])
      # U=plu (P), V=plv (R) (Kallus)
      plu, Sigma, plv = np.linalg.svd(L,full_matrices=False)
      
      
      # we need to make sure that sigma 从大到小排序
      detL = np.prod(Sigma)
      
      if (np.fabs(detL) > V1): pass
      else: pass
      
      # let U0 = Q.P.SINGVAL.R
      singval = np.diag(Sigma)
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



def update_A():
    ListClosest()
    
    A_back = Anew.copy()
    xt = np.matmul(Anew, u) # (nAnew, dim)
    
    
def Ltrd():
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
    
    
def  


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
    

def ListClosest():
    """ Gram schimit
    """
    
    # input: dim*dim (lattice)
    uu = np.zeros(dim, dim)
    for i in range(dim):
        uu[i][:] = lattice[h[i]][:]

    # Gram schimidt in the order u[h[0]], u[h[1]], ...
    
    gs = uu.copy()
    
    
    

    
    
    pass
  
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
    pass
    