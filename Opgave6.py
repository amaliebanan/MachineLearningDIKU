import numpy as np
import pandas as pd
import math as math
import matplotlib.pyplot as plt

df = pd.read_csv('PCB.dt',sep='\t', names=['X','Y'])

x = df['X'].to_numpy().reshape(-1, 1)
y = df['Y'].to_numpy()
ylog = np.log(y)
xsquared = np.sqrt(x)
ysquared = np.sqrt(y)

def calAB(x,y):
    z = np.ones((len(x),1))
    X = np.append(x,z,axis=1)
    w,b = np.dot(np.dot(np.linalg.inv(np.dot(X.T,X)),X.T),y)
    return [w,b]

a,b = calAB(x,ylog)
print("My prediction model: exp(",a,"* x +",b,") = y")

def h(x):
    return np.exp(a*x+b)

x_log_new = [i for i in range(np.min(x),np.max(x)+1)] #Ages 1-12
y_log_hat = [h(i) for i in x_log_new]    #Calculate the predicted y-label for all ages in x_new

def lossfunc(y,x,h):
    n = len(y)
    sum = 0.0
    for i in range(n):
        sum += (y[i]-h(x[i]))**2
    return sum*(1/n)


print("MSE log: ",lossfunc(y,x,h))

def R2(ylog,y,x,h):
    mean = sum(ylog)/len(ylog)
    sum1, sum2 = 0.0,0.0
    n = len(x)
    for i in range(n):
        sum1 += (y[i] - h(x[i]))**2
        sum2 += (y[i] - mean)**2
    return 1 - (sum1/sum2)

print("R2 log: ",R2(ylog,y,x,h))

a2,b2 = calAB(xsquared,ylog)

def h2(x):
    return np.exp(a2*np.sqrt(x)+b2)

x_new_squared = [i for i in range(np.min(x),np.max(x)+1)]
y_squared_hat = [h2(i) for i in x_new_squared]

print("MSE sqrt: ",lossfunc(y,x,h2))
print("R2 sqrt: ",R2(ylog,y,x,h2))

plt.plot(x,y,'o',label="measurements")
plt.plot(xsquared,ylog,'o',label="measurements, squared")
plt.plot(x_log_new,y_log_hat,color="blue",label="Linear Regression")
plt.plot(x_new_squared,y_squared_hat,color="orange",label="Linear Regression squared")
plt.legend()
plt.xlabel("Age (yrs.)")
plt.ylabel("PCB Conc. (ppm)")
plt.show()
