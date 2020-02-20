import time
import numpy as np
from scipy import interpolate
from scipy import optimize
from scipy import sparse
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import continuation as cont
from scipy import optimize
import cutoff


start = time.time()

mu=0.5
mu0=0.5

MESHsize=[]

for jj in range(0,25):
    ###Mesh/Domain
    #solution spatial scale
    q = 8 + jj
    lw = q*np.pi
    #interior mesh points
    #N = 5000 + jj*5000
    #Create grid points for \Omega_w
    xw, H = np.linspace(0,lw,(q/16.0)*10000+2,endpoint=True,retstep=True)
    #xw, H = np.linspace(0,lw,N+2,endpoint=True,retstep=True)    
    print ('step size is ' + str(H) )
    print ('domain is ' + str(lw) )
    MESHsize.append(H)
    m = xw.size

    ch = cutoff.chi(xw)

    ##Initial parameters
    #initialize wavenumber
    k0 = 1 
    #initial shift
    phi0 = 0
    #Neumann BC on left
    gamma0 = 0

    ###Finite Difference Matrices
    ## "transparent BCs" u + u''=0, Neumann BCs u'=u'''=0 & Dirichlet BCs u=u''=0 are all treated as perturbations of stationary BCs u=u'=0 
    ##given by the constant diagonal matrices D2 and D4 
    ##WARNING! These stationary BCs are first order
    ##Right end point is now set to Neumann
    D4 = sparse.diags([np.ones(m-2),-4*np.ones(m-1),6*np.ones(m),-4*np.ones(m-1),np.ones(m-2)],[-2,-1,0,1,2])
    D2 = sparse.diags([np.ones(m-1),-2*np.ones(m),np.ones(m-1)],[-1,0,1])
    N4 = sparse.lil_matrix( (m,m) )
    N2 = sparse.lil_matrix( (m,m) )
#Neaumann Right
    N4[m-1,m-2] = -4
    N4[m-1,m-3] = 1
    N4[m-2,m-2] = 1
    N2[m-1,m-2] = 1
    
    #u'=0 at x_{N+1}
    #u''=0 at x_{N+2}
#    N4[m-1,m-1] = -1
#    N4[m-1,m-2] = -2
#    N4[m-2,m-2] = 1
#    N2[m-1,m-2] = 1

###u''=u'''=0 at at x_{N+1}
#    N4[m-1,m-3] = -1
#    N4[m-1,m-2] = 8
#    N4[m-1,m-1] = -10
#    N4[m-2,m-1] = 2
#    N4[m-2,m-2] = -1
#    N2[m-1,m-2] = -1    
#    N2[m-1,m-1] = 2    


    #Dirichlet Right u=u''=0 at x_{N+2}
    #N4[m-1,m-1] = 1
    D4 = D4 + N4
    D2 = D2 + N2
    
    #The Linear operator Lmew is a sparse diferential operator whose BCs are a SECOND ORDER homotopy of SH neumann in H^1 and transparent the natural H^2 BCs
    def Lmew(gamma,mu):
        BC4 = sparse.lil_matrix( (m,m) )
        BC2 = sparse.lil_matrix( (m,m) )
        
        def A(gamma):
            return (gamma*(H**2-2))/(H/2 - gamma*(1+H/2))
        def B(gamma):
            return (H/2 + gamma*(1-H/2))/(H/2 - gamma*(1+H/2))
        
        BC4[0,0] = -A(gamma)*(2+gamma*H**2)
        BC4[0,1] = (2-gamma*H**2)*(B(gamma)-1) - 4*B(gamma)
        BC4[0,2] = 1
        
        BC4[1,0] = A(gamma)
        BC4[1,1] = B(gamma)
        BC2[0,0] = A(gamma)
        BC2[0,1] = B(gamma)
        return (-1/H**4)*(D4+BC4) - (2/H**2)*(D2+BC2) + (mu-1)*sparse.eye(m)
##    HOMOTOPY OF SECOND ORDER BCS IS NOT A SCOND ORDER BC
##The Linear operator Lmew is a homotopy in gamma of the sparse diferential operators for SH neumann and transparent
##    def Lmew(gamma,mu):
##        #gamma=0, Neumann
##        #gamma=1, transparent
##        BC4 = gamma*BC4tra + (1-gamma)*BC4neu
##        BC2 = gamma*BC2tra + (1-gamma)*BC2neu
##        return (-1/H**4)*(D4+BC4) - (2/H**2)*(D2+BC2) + (mu-1)*sparse.eye(m)
#    #Fv is m equations with m unknowns for profile v.  mu,gamma fixed
    def Fv(v,mu,gamma):
        """Defines finite difference approximation to the equation 
        -(\partial_xx + 1)^2 u + \mu u - u^3 = 0 
        with variable boundary conditions depending on gamma"""
        Lmu = Lmew(gamma,mu)
        return Lmu*v - v**3
    #the m x m Jacobian of Fv with respect to the variable v
    def Jv(v,mu,gamma):
        Lmu = Lmew(gamma,mu)
        return Lmu - sparse.diags(3*v**2,0)
    ##Initial guess for Neumann v0 is vstar the asymptotic soln
    v0 = np.sqrt(4*mu0/3)*np.cos(xw) 
    ##Newton Solver for initial Profile v0
    iter = 0
    maxiter = 10
    tol = (1/H)*10**-7
#    print 'tol is'
#    print tol
    while iter < maxiter:
        v0 = v0 - sparse.linalg.spsolve(Jv(v0,mu,gamma0),Fv(v0,mu,gamma0))
        z = Fv(v0,mu,gamma0)
        res = np.sqrt(H*np.dot(z,z))
        print(res)
#        plt.plot(v0)
#        plt.show()
        iter = iter + 1
        if res < tol:
            break
        if iter == maxiter:
            print('Did not reach tolerance for initialized Neumann profile after ' + str(iter) + ' iterations.  The residual is ' + str(res))

    w0 = v0*(1-ch)
    z0 = np.append(w0,k0)
    z0 = np.append(z0,phi0)

    data_k=[]
    data_phi=[]
    data_sol=[]
    
    dgamma = 1./20
    jend = 21

    ###tangent continuation in k,phi
    for j in range(0,jend):
        gamma = gamma0 + j*dgamma
        LmuBC = Lmew(gamma,mu)
        print('gamma is ' + str(gamma))
        print('Total time is ' + str(time.time() - start))
        z0 = cont.kphiSolve(z0,LmuBC,mu,xw,H)
        K,PHI,SOL = cont.kphi(z0,LmuBC,mu,xw,H,gamma)
        K=np.array(K)
        PHI=np.array(PHI)
        SOL=np.array(SOL)
        data_k.append(K)
        data_phi.append(PHI)
        data_sol.append(SOL)
    print('Total time is ' + str(time.time() - start))
    print('jj is' + str(jj) )
    
    x=np.array(data_k)
    y=np.array(data_phi)
    z=np.array(data_sol)
    
    np.save('Ldatak'+str(jj),x)
    np.save('Ldataphi'+str(jj),y)
    np.save('Ldatasol'+str(jj),data_sol)

np.save('meshSIZE',np.array(MESHsize))

####Plotting #########
#make figure
fig=plt.figure(1)
#add space for slider bar
plt.subplots_adjust(bottom=0.25)
#initialize the curve parameterization
s0=0

##Find plot window
x=np.array(data_k)
y=np.array(data_phi)

L,R,T,B = np.zeros(jend), np.zeros(jend), np.zeros(jend), np.zeros(jend)
for j in range(0,jend):
    L[j],R[j],T[j],B[j] = np.min(x[j]),np.max(x[j]),np.max(y[j]),np.min(y[j])
left,r,t,b = np.min(L),np.max(R),np.max(T),np.min(B)

#plot initial guess
l, = plt.plot(data_k[s0],data_phi[s0])
plt.axis([left,r,b,t])
plt.xlabel('$k$')
plt.ylabel('$ \phi $')

axcolor = 'lightgoldenrodyellow'
#set size of slider
axs = plt.axes([0.25, 0.1, 0.65, .03], axisbg=axcolor)
#set slider values
s_slide = Slider(axs, '$\gamma$', 0, jend-1, valinit=s0)

def update(val):
    s = s_slide.val
    l.set_xdata(data_k[int(s)])
    l.set_ydata(data_phi[int(s)])
    plt.draw()
s_slide.on_changed(update)

plt.show()