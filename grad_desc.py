import numpy as np
import random

def grad_desc(x, y, t, a, m, it):
    xTrans = x.transpose()
    for i in range(0, it):
        h=np.dot(x, t)
        e=h-y
        costFun=np.sum(e**2)/(2*m)
        gradient=np.dot(xTrans,e)/m
        t=t-a*gradient
    return t


def DataGenarate(numPoints, bias, variance):
    x = np.zeros(shape=(numPoints, 2))
    y = np.zeros(shape=numPoints)
    for i in range(0, numPoints):
        x[i][0]=1
        x[i][1]=i
        y[i]=(i + bias)+random.uniform(0,1)*variance
    return x, y

x,y = DataGenarate(100, 25, 10)
m,n = np.shape(x)
it = 100000
a = 0.0005
t = np.ones(n)
t = grad_desc(x, y, t, a, m, it)
print(t)
