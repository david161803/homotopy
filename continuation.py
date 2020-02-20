"""
A collection of continuation functions for SH
"""
import numpy as np
from scipy import sparse
from scipy import linalg
from scipy import interpolate
import matplotlib.pyplot as plt
import weight
import mesh


H = mesh.H
dx = H
ch = mesh.chi
CH = sparse.diags(ch,0)
m = mesh.m

##Not the same q from mesh
q = mesh.xu.size
## L = mesh.q*(2pi)... the phase condition is implemented over the second to last period
Start = mesh.q - 2
Stop = Start + 1

##An Euler-Newton Method based on the More-Penrose Pseudoinverse.
#This version adapts the step size and reiterates a new trial soln is the norm of the orthogonal projection ever increases
def ENStep(f,J,ds,t0,y0):
    tol = 10**-6
    maxiter = 60
    #coordinitization by standard basis element in largest component of previous tangent (may preserve sparsity better)
    e = np.zeros(y0.size)
    e[np.argmax(t0)] = 1
    
    fprime = J(y0)
    A = sparse.vstack([fprime,e])
    A = A.tocsr()
    #get the tangent t
    tau = sparse.linalg.spsolve( A , np.append(fprime*e,0) )   ##The Solver here should depend on the structure of J (Probably a sparse arrow matrix with 5 diagonals and 2 cols & rows)
    #tau = sparse.linalg.qmr( A , np.append(fprime*e,0) )[0]   ##The Solver here should depend on the structure of J
    tau = e - tau  #tau in ker(fprime)
    t = ( np.sign( weight.dot(tau,t0,dx) )/np.sqrt( weight.dot(tau,tau,dx)) )*tau  #orient and normalize
    
    #Check if t is correct
    ker = J(y0)*t
    ker = np.abs(ker)   
    if np.sum(ker) > .0001:
        print('BAD TANGENT!?  ...maybe ignore this if dx is small' )
        print('norm ker is '+ str( weight.dot(ker,ker,dx) ) + ' this should be zero')
        print('max ker is '+ str(np.max(ker)) + ' this should be zero')
    yzero = y0
    y0 = y0 + ds*t #yhat the tangent predictor
    ##Corrector loop: Simulated More-Penrose Pseudoinverse Newton map using arbitrary linear solver
    iter = 0
    b = np.append(f(y0),0)
    A = sparse.vstack([fprime,t]).tocsr()
    normz2 = 10.0
    while iter < maxiter:
        iter = iter + 1
        ##The Solver here should depend on the structure of J 
        z = sparse.linalg.spsolve(A , b )
        #z = sparse.linalg.qmr(A , b )[0]
        z = z - weight.dot(t,z,dx)*t  #othogonal projection
        y1 = y0 - z
        normz1 = np.sqrt( weight.dot(z,z,dx) )   ##Should maybe be the norm for a weighted space
        print('The norm of z is ' + str(normz1) )
        if normz1 < tol:
            image = f(y1)
            res = weight.dot(image,image,dx) 
            print('Residual squared is ' + str(res) + ' using just pseudoinverse' + ' ds is ' + str(ds))
            break
        if normz1 > 5*normz2:
            print('Orthogonal projection is not decreasing fast enough; adapting pseudo-arclength')  
            ##Document k,mu,gamma values for when this occurs...mu and gamma are awkward
            ds = 0.61803*ds
          #  print('ds is ' + str(ds))
            y0 = yzero + ds*t
        normz2 = normz1
        y0 = y1
        b = np.append(f(y0),0)
        fprime = J(y0)
        A = sparse.vstack([fprime,t])
        A = A.tocsr()

    return t,y0,iter
##      Given a boundary layer, wavenumber and phase z=(w,k,phi), linear operator LmuBC and mesh xw; The function kphi gives k,phi continuation curve starting 
##     at k0,phi0 and continuing in the pos and neg k dir.
def Fw(w,k,phi,mu,gamma):
    """Defines finite difference approximation to the equation 
    -(\partial_xx + 1)^2 u + \mu u - u^3 = 0 
    with variable boundary conditions depending on gamma
    Also contains the phase condition so that the entire k and phi derivative columns in the Jacobian are computed numerically"""
    LmuBC = mesh.Lmew(k,gamma,mu)
    
    y = mesh.Y(phi)
    tck = mesh.getroll(k,mu)   
    U = interpolate.splev(y,tck,der=0)
    Up = interpolate.splev(y,tck,der=1)

    pc = np.zeros(w.size + 2)
    pc[Start*q:Stop*q] = Up[Start*q:Stop*q]
    pc[Start*q] = 0.5*pc[Start*q]
    pc[Stop*q-1] = 0.5*pc[Stop*q-1]
    
    ChiU = ch*U
    
    F = LmuBC*w - 3*ChiU**2*w - 3*ChiU*w**2 - w**3 + (LmuBC*CH - CH*LmuBC)*U + (CH - CH**3)*U**3
    
    return np.append( Fw, H*np.dot(pc,z) )

def Fz(z,mu,gamma):
    """ Discrtization of the equation 
    -(\partial_xx + 1)^2 v + \mu v - v^3 = 0 
    fow w where v = \chi*u_r + w
    with homogeneous variable boundary conditions depending on gamma """
    w, k, phi = z[:-2], z[-2], z[-1]
    #roll solution interpolant representation
    tck = mesh.getroll(k,mu)
    #linear operator
    Lmu = mesh.Lmew(k,gamma,mu)
    #Shifted coordinates
    y = mesh.Y(phi)
    #Interpolate
    #tck = interpolate.splrep(xubig,u,k=3)   
    U = interpolate.splev(y,tck,der=0)        
    Up = interpolate.splev(y,tck,der=1)

    pc = np.zeros(z.size)
    pc[Start*q:Stop*q] = Up[Start*q:Stop*q]
    pc[Start*q] = 0.5*pc[Start*q]
    pc[Stop*q-1] = 0.5*pc[Stop*q-1]
    
    #operator for w
    chiU = ch*U        
    Fw  = Lmu*w - (3*(chiU)**2)*w - 3*(chiU)*w**2 - w**3  + (Lmu*CH - CH*Lmu)*U + (CH-CH**3)*U**3
    return np.append( Fw, H*np.dot(pc,z) )
#the m+1 x m+2 Jacobian with respect to the variable z=(w,k,phi)
def Jz(z,mu,gamma):
    
    w0, k, phi = z[:-2], z[-2], z[-1]
    #roll solution interpolant representation
    tck = mesh.getroll(k,mu)
    y = mesh.Y(phi)
    ##define D_wF 
    #tck = interpolate.splrep(xubig,u,k=3)
    U = interpolate.splev(y,tck,der=0)
    Up = interpolate.splev(y,tck,der=1)
    chiU = ch*U
    dwFw = mesh.Lmew(k,gamma,mu) - sparse.diags(3*chiU**2 + 6*chiU*w0 + 3*w0**2,0)   
    
    dwPhi = np.zeros(z.size-2)
    dwPhi[Start*q:Stop*q] = Up[Start*q:Stop*q]
    ##2nd order dwdPHI
    dwPhi[Start*q] = 0.5*dwPhi[Start*q]
    dwPhi[Stop*q-1] = 0.5*dwPhi[Stop*q-1]
    ##
    dwPhi.shape = (1,dwPhi.size)
    dwPhi = sparse.bsr_matrix(dwPhi)
    
    A = sparse.bmat( [ [dwFw],[dwPhi]  ] )    
    #Numerically approximate k and phi derivatives with forward diference   
    t = 10**-4
    phihat = np.zeros(z.size)
    phihat[-1] = t
    khat = np.zeros(z.size)
    khat[-2] = t

    z_var_phi = z + phihat
    z_var_k = z + khat

    F0 = Fz(z,mu,gamma)
    dphiFw = (Fz(z_var_phi,mu,gamma) - F0)/t
    dphiFw.shape = (m+1,1)
    
    dkFw = (Fz(z_var_k,mu,gamma) - F0)/t 
    dkFw.shape = (m+1,1)

    return sparse.bmat( [ [A,dkFw,dphiFw]  ] )

def kphiSTEP(mu,gamma,ds,t0,z0):
    
    def F(z):
        return Fz(z,mu,gamma)
    def J(z):
        return Jz(z,mu,gamma)
    
    return ENStep(F,J,ds,t0,z0)

def kphiGETDATA(z,mu,gamma):
    '''
    Ansatz: v = w + chi*U(k0,phi0)
    '''
    def F(z):
        return Fz(z,mu,gamma)
    def J(z):
        return Jz(z,mu,gamma)
    
    z0 = z
    t0 = np.zeros(m+2)
    t0[-1] = -1
    
    k0=z[-2]
    
    data_k=[]
    data_phi=[]
    data_sol=[]
    
    ds = 10**-8
    t0,z0,iter = ENStep(F,J,ds,t0,z0)
    data_k.append(z0[-2])
    data_phi.append(z0[-1])
    data_sol.append(z0)

    dstol = .0005
    ds=.1
    Iter=0
    MaxIter = 200
#Currently written to continue in phi direction assuming k = k(phi)
    k=k0
    phi0 = z0[-1]
    print('phi is ' + str(phi0) + ' continueing in negative phi direction' )
    ######################kphi continue right/down
    phi = phi0
    while phi > -np.pi and ds>dstol:

        Iter = Iter + 1
        t0,z0,iter = kphiSTEP(mu,gamma,ds,t0,z0)
        k = z0[-2]
        phi = z0[-1]
        data_k.append(z0[-2])
        data_phi.append(z0[-1])
        data_sol.append(z0)
        ##ADAPTATION        
        if iter < 3 and ds<.051:
            ds = 2*ds
        elif iter > 5:
            ds = 0.5*ds
        if Iter == MaxIter:
            break
        #print('k is ' + str(k) + ' ds is ' + str(ds) + ' mu is ' + str(mu))
    data_sol.append(z0)
    #rearrange data to plot correctly
    data_k = data_k[::-1]
    data_phi = data_phi[::-1]
    data_sol = data_sol[::-1]
    #####################
    
    #####################kphi continue left/up
    z0=z
    t0 = np.zeros(m+2)
    t0[-1] = 1
    #short step
    ds = 10**-8
    t0,z0,iter = kphiSTEP(mu,gamma,ds,t0,z0)
    data_k.append(z0[-2])
    data_phi.append(z0[-1])
    data_sol.append(z0)
    #steploop
    ds=.1
    Iter=0
    
    
    phi0 = z0[-1]
    print('phi is ' + str(phi0) + ' continueing in positive phi direction' )
    phi = phi0
    while phi < np.pi and ds>dstol:  
    #While k < Eleft and k > Eright:
        Iter = Iter + 1
        t0,z0,iter = kphiSTEP(mu,gamma,ds,t0,z0)
        k = z0[-2]
        phi = z0[-1]
        data_k.append(z0[-2])
        data_phi.append(z0[-1])
        data_sol.append(z0)
        ##ADAPTATION
        if iter < 3 and ds<.051:
            ds = 2*ds
        elif iter > 5:
            ds = 0.5*ds
        if Iter == MaxIter:
            break		
       # print('k is ' + str(k) + ' ds is ' + str(ds) + ' mu is ' + str(mu))
    
    #####################################################
    return data_k, data_phi, data_sol