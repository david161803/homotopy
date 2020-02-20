import numpy as np
from scipy import sparse
from scipy import linalg
from scipy import interpolate

print 'how often is this preamble called'

###Mesh/Domain
#solution spatial scale in number of periods
q = 9
Q = 2*q
lw = Q*np.pi
#Create mesh for \Omega_w
xw, H = np.linspace(0,lw,(Q/32.0)*10000+2,endpoint=True,retstep=True)
print ('step size is ' + str(H) )
print ('domain is ' + str(lw) )
m = xw.size

#Shifted coordinates
def Y(phi):
    return (- xw + phi)%lu

N=xw[xw<2*np.pi].size
xu = xw[0:N]
lu = xu[-1]
print (' The small grid has been shortened by ' + str(2*np.pi-lu))
h=H
#make a larger xu domain for interpolation in getroll
NN = 7
#xubig = np.linspace(-NN*h,lu+NN*h,N +2*NN)
xubig = np.concatenate(( np.linspace(-NN*h,-h,NN) , xw[0:N+7]))

###Finite Difference Matrices
## "transparent BCs" u + u''=0, Neumann BCs u'=u'''=0 & Dirichlet BCs u=u''=0 are all treated as perturbations of stationary BCs u=u'=0 
##given by the constant diagonal matrices D2 and D4 
##WARNING! These stationary BCs are first order
##Right end point is now set to Neumann
bc = sparse.lil_matrix( (N,N) )
bc[0,1]=1
bc[N-1,N-2]=1
bc.tocsr()
Lh = (1/h/h)*( sparse.diags([np.ones(N-1),-2*np.ones(N),np.ones(N-1)],[-1,0,1]) + bc )

D4 = sparse.diags([np.ones(m-2),-4*np.ones(m-1),6*np.ones(m),-4*np.ones(m-1),np.ones(m-2)],[-2,-1,0,1,2])
D2 = sparse.diags([np.ones(m-1),-2*np.ones(m),np.ones(m-1)],[-1,0,1])
N4 = sparse.lil_matrix( (m,m) )
N2 = sparse.lil_matrix( (m,m) )
#Neaumann Right
N4[m-1,m-2] = -4
N4[m-1,m-3] = 1
N4[m-2,m-2] = 1
N2[m-1,m-2] = 1
#Stationary left
D4 = D4 + N4
D2 = D2 + N2

#The Linear operator Lmew is a sparse diferential operator whose BCs are a SECOND ORDER 
#homotopy of SH neumann in H^1 and transparent the natural H^2 BCs.  Created as perturbation from stationary BCs
def Lmew(k,gamma,mu):
    BC4 = sparse.lil_matrix( (m,m) )
    BC2 = sparse.lil_matrix( (m,m) )
    
    Htilde = H/k
    
    def A(gamma):
        return (gamma*(Htilde**2-2))/(Htilde/2 - gamma*(1+Htilde/2))
    def B(gamma):
        return (Htilde/2 + gamma*(1-Htilde/2))/(Htilde/2 - gamma*(1+Htilde/2))
    
    BC4[0,0] = -A(gamma)*(2+gamma*Htilde**2)
    BC4[0,1] = (2-gamma*Htilde**2)*(B(gamma)-1) - 4*B(gamma)
    BC4[0,2] = 1
    
    BC4[1,0] = A(gamma)
    BC4[1,1] = B(gamma)
    BC2[0,0] = A(gamma)
    BC2[0,1] = B(gamma)
    return (-k**4/H**4)*(D4+BC4) - (2*k**2/H**2)*(D2+BC2) + (mu-1)*sparse.eye(m)

tol = 10**-6
maxiter = 10
NN = 7
##Given k,mu outputs the abstract interpolant for the periodic roll solution
def getroll(k,mu):
        
    Diff_OP = -(k*k*Lh + sparse.eye(N))**2
    Lmu = Diff_OP + mu*sparse.eye(N)
    
    u0 = np.sqrt(4*mu/3)*np.cos(xu)
    
    def f(u,mu):
        return Lmu*u - u**3
    def J(u,mu):
        return Lmu - sparse.diags(3*u**2,0)
    #Newton loop
    iter = 0
    while iter < maxiter:
        u0 = u0 - sparse.linalg.spsolve(J(u0,mu),f(u0,mu))
        z = f(u0,mu)
        res = h**0.5*np.sqrt(np.dot(z,z))
        #print res
        iter = iter + 1
        if res < tol:
            break
        if iter == maxiter:
            print('Did not reach tolerance for roll solution after ' + str(iter) + ' iterations.  The residual is ' + str(res))
    u1, u2 = u0[-NN:], u0[0:NN]
    u = np.concatenate((u1,u0,u2))  
#        plt.plot(xubig, np.concatenate((u1,u0,u2)) )
#        plt.show()
    return interpolate.splrep(xubig,u,k=3)

#################
##CREATE CUTOFF##
#################
def A(n):
    A = np.ones((n,n))
    R = np.linspace(n,2*n-1,n)
    R0 = R
    for k in range(1,n):
        A[k,:] = R
        R0 = R0 - 1
        R = R*R0
    return A	
def b(n):
    b = np.zeros(n)
    b[0] = 1
    return b
#n co-efficients for g^* = "homogeneous cutoff" g_n(x) = x^{n+1}*g^*(x)
def c(n):
    return linalg.solve(A(n),b(n))

#piecewise polynomial function which is zero to the left of zero and one to the right of one which is in C^N(R)
def g(x,N):
    co = c(N+1)
    #print co
    g = 0
    for k in range(0,N+1):
        gamma = co[k]*x**k
        g = g + gamma
    one = x>1
    g[x<0] = 0
    g[x>1] = 0
    return g*x**(N+1) + one

#f is zero to the left of a and one to the right of b with N continuous derivatives at a & b. polynomial in between.
def f(x,a,b,N):
    return g((x-a)/(b-a),N)

#characteristic parameters
l=6
eps = 6
chi = f(xw,l,l+eps,4)

#def chi(x):
#    return f

print 'mesh done'
