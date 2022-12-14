"""
* Implementation of the LLL reduction (named after Lenstra, Lenstra, and Lovász)
* Adapated from 'Factoring Polynomials with Rational Coefficients'

* The present code is based on the original C version written by Yoav Kallus
* who kindly provides his code on demand.

Note that all matrix are in row-major.
"""
import numpy as np

LLL_tiny = 1e-10

def LLL_swap(k: int, kmax: int, b: np.array, mu: np.array, H: np.array, B: np.array, bstar: np.array):
    """
    Fig. 1. The reduction algorithm. Step(2)
    
    Parameters
    ----------
    b: basis
    mu: defined in Eq. (1.3)
    B: B_i = |b*_i|^2
    
    Return
    ----------
    b, mu, H, bstar, work
    """
    # (b_(k-1) b_k) := (b_k b_(k-1))
    b[k-1], b[k] = b[k], b[k-1]
    H[k-1], H[k] = H[k], H[k-1]

    # (mu_(k-1)j mu_kj) := (mu_kj mu_(k-1)j) for j= 0, ..., 
    if (k > 1):
        for j in range(k-1):
            mu[k-1,j], mu[k,j] = mu[k,j], mu[k-1,j]
    
    # mu := mu[k][k-1]
    mubar = mu[k,k-1]
    # B := B_k + mu*mu*B_(k-1)
    Bbar = B[k] + mubar**2 * B[k-1]
    
    if (np.fabs(Bbar) < LLL_tiny):
        B[k], B[k-1] = B[k-1], B[k]
        bstar[k], bstar[k-1] = bstar[k-1], bstar[k]
        mu[k+1:kmax+1,k], mu[k+1:kmax+1,k-1] = mu[k+1:kmax+1,k-1], mu[k+1:kmax+1,k]
    elif (np.fabs(B[k]) < LLL_tiny and mubar != 0.):
        # B_(k-1) = B
        B[k-1] = Bbar
        # bstar_(k-1) = mu * bstar_(k-1)
        bstar[k-1] *= mubar
        mu[k,k-1] = 1. / mubar
        mu[k+1:kmax+1,k-1] /= mubar
    elif (B[k] != 0.):
        # t = B_(k-1)/B
        t = B[k-1] / Bbar
        mu[k,k-1] = mubar * t
        
        bbar = bstar[k-1].copy()
        bstar[k-1] = bstar[k] + mubar * bbar
        bstar[k] = -mu[k,k-1]*bstar[k] + (B[k]/Bbar) * bbar
        
        # B_k := B_(k-1)*B_k/B
        B[k] *= t
        # B_(k-1) := B
        B[k-1] = Bbar
        for i in range(k+1, kmax+1):
            t = mu[i,k]
            # mu_ik := mu_ik-1 - mu*mu_ik
            mu[i,k] = mu[i,k-1] - mubar * t
            # mu_ik-1 := mu_ik + mu_kk-1*(...), where (...)=mu_ik
            mu[i,k-1]  = t + mu[k,k-1] * mu[i,k]
    
    return b, mu, H, bstar
            

def LLL_star(k: int, l: int, b: np.array, mu: np.array, H: np.array):
    """
    Fig. 1. The reduction algorithm. Step(*)
    
    Parameters
    ----------
    K & l: subindex of mu
    b: basis
    mu: defined in Eq. (1.3)
    H: 
    
    Return
    ----------
    b, mu, H
    """
    if (np.fabs(mu[k,l]) > 0.5): 
        # r := integer nearest to mu_kl
        r = int(np.floor(0.5 + mu[k,l]))
        # b_k := b_k - r*b_l
        b[k] -= r * b[l]
        H[k] -= r * H[l]
        
        # mu_kj := mu_kj - r*mu_lj for j = 0, ..., l-1
        mu[k,0:l] -= r*mu[l,0:l]
        # mu_kl := mu_kl - r
        mu[k,l] -= r
    
    return b, mu, H


def LLL_reduction(inbasis: np.array, dim: int):
    """
    Fig. 1. The reduction algorithm. Full
    
    Parameters
    ----------
    dim: dimension of input basis
    inbasis: input basis
    
    Return
    ----------
    basis, H (dim, dim)
    H indicate the lattice translate in local framework
    """
    basis = inbasis.copy() # (dim, dim)
    
    mu = np.zeros([dim, dim])
    H = np.diag(np.ones(dim, dtype=int))
    b_star = np.zeros([dim, dim])
    B = np.zeros(dim)
    subwork = np.empty(dim)

    k = 1
    kmax = 0
    
    """ Fig. 1. The reduction algorithm. The first part """
    # case1: i = 0
    b_star[0] = basis[0].copy()
    B[0] = np.dot(b_star[0], b_star[0])
    
    # case2: i > 0
    # if k=dim, terminate
    while (k < dim):
        if (k > kmax):
            kmax = k
            b_star[k] = basis[k].copy()
              
            for j in range(k):
                # mu_ij := (bi, b*_j)/B_j
                if (np.fabs(B[j]) < LLL_tiny): mu[k,j] = 0.
                else: mu[k,j] = np.dot(basis[k], b_star[j]) / B[j]
                # b*_i := b*_i - mu_ij*b*_j
                b_star[k,:] -= mu[k,j]*b_star[j,:]
              
            # B_i := (b*_i, b*_i) 
            B[k] = np.dot(b_star[k], b_star[k])
        
        while (1):
            # perform (*) for l = k- 1
            basis, mu, H = LLL_star(k, k-1, basis, mu, H)
            if (B[k] < (0.75 - mu[k,k-1]**2)*B[k-1]):
                # go to step(2)
                basis, mu, H, b_star = LLL_swap(k, kmax, basis, mu, H, B, b_star)
                k -= 1
                
                if (k < 1): k = 1
            else:
                # perform (*) for l = k- 2, k- 3 ..... 1
                for l in range(k-2, -1, -1):
                    basis, mu, H = LLL_star(k, l, basis, mu, H)
                k += 1
                break
    
    return basis, H