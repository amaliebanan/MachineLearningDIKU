import numpy as np
import pandas as pd
import math as math
import matplotlib.pyplot as plt

df = pd.read_csv('PCB.dt',sep='\t', names=['X','Y'])

x = df['X'].to_numpy().reshape(-1, 1)
y = df['Y'].to_numpy()
ylog = np.log(y)
xsquared = np.sqrt(x)

def calAB(x,y):
    z = np.ones((len(x),1))
    X = np.append(x,z,axis=1)
    w,b = np.dot(np.dot(np.linalg.inv(np.dot(X.T,X)),X.T),y)
    return [w,b]

a,b = calAB(x,ylog)
print("My prediction model: exp(",a,"* x +",b,") = y")

def h(x,a,b):
    return np.exp(a*x+b)

y_log_hat = [math.log(h(i,a,b)) for i in x]  #Calculate the predicted y-label for all ages in x_new

def lossfunc(ylog,yhat):
    n = len(ylog)
    sum = 0.0
    for i in range(n):
        sum += (ylog[i]-yhat[i])**2
    return sum*(1/n)

print("MSE log: ",lossfunc(ylog,y_log_hat))
mean = sum(ylog)/len(ylog)

def R2(ylog,yhat):
    sum1, sum2 = 0.0,0.0
    n = len(yhat)
    for i in range(n):
        sum1 += (ylog[i] - yhat[i])**2
        sum2 += (ylog[i] - mean)**2
    return 1 - (sum1/sum2)

print("R2 log: ",R2(ylog,y_log_hat))

w,b2 = calAB(xsquared,ylog)

y_squared_hat = [math.log(h(i,w,b2)) for i in xsquared]

print("MSE sqrt: ",lossfunc(ylog,y_squared_hat))
print("R2 sqrt: ",R2(ylog,y_squared_hat))

#plt.plot(x,ylog,'o',label="measurements",color="red")
#plt.plot(x,y_log_hat, label="Linear regression",color="Blue")
plt.plot(xsquared,ylog,'o')
plt.plot(xsquared,y_squared_hat,label="Linear regression, squared",color="Blue")
plt.title("Linear Regression squared")
plt.legend()
plt.xlabel("Age (yrs.)")
plt.ylabel("log(PCB Conc. (ppm))")
plt.show()
