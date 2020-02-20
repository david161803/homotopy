import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import pandas
from matplotlib.widgets import Slider
import matplotlib.colors as colors
import matplotlib.cm as cmx


##Choose gamma
jj = 48

##Choose spread
ds = 5

S = np.load('datasol.npy')
S = S[jj]

data_sol = []
data_k = []
data_phi = []
#for n in list:
for n in range(S.shape[0]):
    s = S[n] 

    w = s[:-2]
    k = s[-2]
    phi = s[-1]

    data_sol.append(w)
    data_k.append(k)
    data_phi.append(phi)

##MAKE SURE SPATIAL MESH IS THE SAME
q = 9
Q = 2*q
lw = Q*np.pi
#Create mesh for \Omega_w
x, H = np.linspace(0,lw,(Q/32.0)*10000+2,endpoint=True,retstep=True)

#s is slider for mu

#make figure
fig=plt.figure(1)
#add plot of 2d soln
fig.add_subplot(211)
#add space for slider bar
plt.subplots_adjust(bottom=0.25)
#initialize the curve parameterization
s0=0

#plot initial
l1, = plt.plot(x,data_sol[0+ 0*ds],'b')
l2, = plt.plot(x,data_sol[0+ 1*ds],'m')
l3, = plt.plot(x,data_sol[0+ 2*ds],'r')
l4, = plt.plot(x,data_sol[0+ 3*ds],'y')
l5, = plt.plot(x,data_sol[0+ 4*ds],'g')


plt.xlabel('$x$',fontsize='26')
plt.ylabel('$ w $',fontsize='26')

#add plot to figure
ax = fig.add_subplot(212)
ax.plot(data_k[:],data_phi[:],'k')

plt.xlabel('$k$',fontsize='26')
plt.ylabel('$ \phi $',fontsize='26')
#ax.axis([0.85,1.15,-1,1])
p1, = ax.plot(data_k[s0],data_phi[s0],'bo')
p2, = ax.plot(data_k[s0+ ds],data_phi[s0+ ds],'mo')
p3, = ax.plot(data_k[s0+ 2*ds],data_phi[s0+ 2*ds],'ro')
p4, = ax.plot(data_k[s0+ 3*ds],data_phi[s0+ 3*ds],'yo')
p5, = ax.plot(data_k[s0+ 4*ds],data_phi[s0+ 4*ds],'go')

axcolor = 'lightgoldenrodyellow'
#set size of slider
axs = plt.axes([0.25, 0.1, 0.65, .03], axisbg=axcolor)
#set slider values
s_slide = Slider(axs, 'not $\mu$', 0, n - 4*ds, valinit=s0)


def update(val):
    s = s_slide.val
    l1.set_ydata(data_sol[int(s)+ 0*ds])
    l2.set_ydata(data_sol[int(s)+ 1*ds])
    l3.set_ydata(data_sol[int(s)+ 2*ds])
    l4.set_ydata(data_sol[int(s)+ 3*ds])
    l5.set_ydata(data_sol[int(s)+ 4*ds])
    
    #redraw the scatter data
    p1.set_xdata(data_k[int(s)+ 0*ds])
    p1.set_ydata(data_phi[int(s)+ 0*ds])
    p2.set_xdata(data_k[int(s)+ 1*ds])
    p2.set_ydata(data_phi[int(s)+ 1*ds])
    p3.set_xdata(data_k[int(s)+ 2*ds])
    p3.set_ydata(data_phi[int(s)+ 2*ds])
    p4.set_xdata(data_k[int(s)+ 3*ds])
    p4.set_ydata(data_phi[int(s)+ 3*ds])
    p5.set_xdata(data_k[int(s)+ 4*ds])
    p5.set_ydata(data_phi[int(s)+ 4*ds])
    
    plt.draw()
s_slide.on_changed(update)

plt.show()

############
##############


