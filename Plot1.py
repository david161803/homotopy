import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import pandas
from matplotlib.widgets import Slider
import matplotlib.colors as colors
import matplotlib.cm as cmx


S = np.load('datasol.npy')

##make list of good indices
list=[]
Shape=S[0].shape
s0=0
for n in range(0,51):
    s=S[n]
    if s.shape==Shape:
        list.append(n)
        s0 = s0+1
print list

s = S[0]
data_sol = s[:,:-2]
data_k = s[:,-2]
data_phi = s[:,-1]

data_sol.shape = (data_sol.shape[0],data_sol.shape[1],1)
data_k.shape = (data_k.shape[0],1)
data_phi.shape = (data_phi.shape[0],1)

for n in list:
    s = S[n] #continuation data for fixed gamma
#    data_sol.append(s[:,:-2])
#    data_k.append(s[:,-2])
#    data_phi.append(s[:,-1]) 
    d = s[:,:-2]
    K = s[:,-2]
    PHI = s[:,-1]
    d.shape = (d.shape[0],d.shape[1],1)
    K.shape = (K.shape[0],1)
    PHI.shape = (PHI.shape[0],1)

    data_sol = np.concatenate((data_sol,d),axis=2)
    data_k = np.concatenate((data_k,K),axis=1)
    data_phi = np.concatenate((data_phi,PHI),axis=1)
    print n

q = 9
Q = 2*q
lw = Q*np.pi
#Create mesh for \Omega_w
x, H = np.linspace(0,lw,(Q/32.0)*10000+2,endpoint=True,retstep=True)



##############
############

jend = data_sol.shape[0]

##Plot family of solutions
#dk marks the continuation
dk=int(jend/4)
k1=dk
k2=2*dk
k3=3*dk
k4=4*dk
#k5=5*dk

#k1 = np.ceil(0.5*(k1 + k2))
#k3 = np.floor(0.5*(k3 + k2))


#s is slider for gamma

#make figure
fig=plt.figure(1)
#add plot of 2d soln
fig.add_subplot(211)
#add space for slider bar
plt.subplots_adjust(bottom=0.25)
#initialize the curve parameterization
s0=s0

#m=np.max(data_sol)

#plot initial
l, = plt.plot(x,data_sol[0,:,s0],'b')
m, = plt.plot(x,data_sol[k1,:,s0],'m')
n, = plt.plot(x,data_sol[k2,:,s0],'r')
o, = plt.plot(x,data_sol[k3,:,s0],'y')
p, = plt.plot(x,data_sol[k4,:,s0],'k')

plt.xlabel('$x$',fontsize='26')
plt.ylabel('$ w $',fontsize='26')

#add plot to figure
ax = fig.add_subplot(212)
z, = ax.plot(data_k[:,s0],data_phi[:,s0])
plt.xlabel('$k$',fontsize='26')
plt.ylabel('$ \phi $',fontsize='26')
ax.axis([0.75,1.25,-np.pi,np.pi])
z1, = ax.plot(data_k[0,s0],data_phi[0,s0],'bo')
z2, = ax.plot(data_k[k1,s0],data_phi[k1,s0],'mo')
z3, = ax.plot(data_k[k2,s0],data_phi[k2,s0],'ro')
z4, = ax.plot(data_k[k3,s0],data_phi[k3,s0],'yo')
z5, = ax.plot(data_k[k4,s0],data_phi[k4,s0],'ko')

axcolor = 'lightgoldenrodyellow'
#set size of slider
axs = plt.axes([0.25, 0.1, 0.65, .03], axisbg=axcolor)
#set slider values
s_slide = Slider(axs, '$\gamma$', 1, s0, valinit=s0)


def update(val):
    s = s_slide.val
    l.set_ydata(data_sol[0,:,int(s)])
    m.set_ydata(data_sol[k1,:,int(s)])
    n.set_ydata(data_sol[k2,:,int(s)])
    o.set_ydata(data_sol[k3,:,int(s)])
    p.set_ydata(data_sol[k4,:,int(s)])
    
    z.set_xdata(data_k[:,int(s)])
    z.set_ydata(data_phi[:,int(s)])
    
    #redraw the scatter data
    z1.set_xdata(data_k[0,int(s)])
    z1.set_ydata(data_phi[0,int(s)])

    z2.set_xdata(data_k[k1,int(s)])
    z2.set_ydata(data_phi[k1,int(s)])
    
    z3.set_xdata(data_k[k2,int(s)])
    z3.set_ydata(data_phi[k2,int(s)])

    z4.set_xdata(data_k[k3,int(s)])
    z4.set_ydata(data_phi[k3,int(s)])

    z5.set_xdata(data_k[k4,int(s)])
    z5.set_ydata(data_phi[k4,int(s)])
#    m.set_xdata(MU[int(s)])
#    m.set_ydata(K[int(s)])
#    m.set_zdata(data_norm[int(s)])
    plt.draw()
s_slide.on_changed(update)

plt.show()

############
##############

#
#fig2=plt.figure(2)
#plt.subplot(121)
##5colors
##plt.plot(x,uinitial,'b-',x,data_sol[k1],'m-',x,data_sol[k2],'r-',x,data_sol[k3],'y-',x,u0,'g-')
##7colors
#plt.plot(x,data_sol[0],'c-',x,data_sol[k1],'b-',x,data_sol[k2],'m-',x,data_sol[k3],'r-',x,data_sol[k4],'y-',x,data_sol[k5],'g-',x,data_sol[-1],'k-')
##plt.legend(('u initial','u last'))
#plt.xlabel('$x$',fontsize='26')
#plt.ylabel('$ w $',fontsize='26')
#
#plt.subplot(122)
#plt.plot(data_k,data_phi)
##5 colors
##plt.plot([data_mu[0]] ,[data_norm[0]],'bo', [data_mu[k1] ],[data_norm[k1]],'mo',[data_mu[k2] ],[data_norm[k2]],'ro',[data_mu[k3] ],[data_norm[k3]],'yo',[data_mu[jend-2]] ,[data_norm[jend-2]],'go')
##7colors
#plt.plot([data_k[0]] ,[data_phi[0]],'co', [data_k[k1] ],[data_phi[k1]],'bo',[data_k[k2] ],[data_phi[k2]],'mo',[data_k[k3] ],[data_phi[k3]],'ro',[data_k[k4] ],[data_phi[k4]],'yo',[data_k[k5] ],[data_phi[k5]],'go',[data_k[jend-2]] ,[data_phi[jend-2]],'ko')
#plt.xlabel('$k$',fontsize='26')
#plt.ylabel('$ \phi $',fontsize='26')
##plt.plot(x,u0,x,sol.x)
#plt.show()
#

