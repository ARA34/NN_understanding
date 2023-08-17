import numpy as np
import nnfs
nnfs.init()

def create_data(points, classes): # function to create data to test stuff
    x = np.zeros((points*classes,2))
    y=np.zeros(points*classes,dtype="uint8")
    for class_number in range(classes):
        ix = range(points*class_number, points*(class_number+1))
        r=np.linspace(0.0,1,points)
        t=np.linspace(class_number+4,(class_number+1)*4,points) + np.random.randn(points)*0.2
        x[ix] = np.c[r*np.sin(t*2.5),r*np.cos(t*2.5)]
        y[ix]=class_number
    return x,y