import numpy as np
import random
def grad_desc(x,y,t,t1,a,m,it):
    xTrans = x.transpose()
    for i in range(0, it):
        h=np.dot(x, t)
        e=h-y
        gradient=np.dot(xTrans,e)/m
        t1=t1 - a*gradient*x
        t=t-a*gradient		
    return t,t1



def DataGenarate(numPoints, bias, variance):
    x = np.zeros(shape=(numPoints, 2))
    y = np.zeros(shape=numPoints)
    for i in range(0, numPoints):
        x[i][0]=1
        x[i][1]=i
        y[i]=(i + bias)+random.uniform(0,1)*variance
    return x, y

x,y = DataGenarate(100,25,10)
m,n = np.shape(x)
it = 100000
a = 0.0005
t = np.ones(n)
t1 = np.ones(n)
t,t1 = grad_desc(x,y,t,t1,a,m,it)

print(t)
print(t1)
