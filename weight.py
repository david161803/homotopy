import numpy as np

def dot(x,y,H):
    d = H*np.dot(x,y) + x[-1]*y[-1] + 10*x[-2]*y[-2]
    #d = d + np.abs(1-x[-2])*np.abs(1-y[-2])
    return d