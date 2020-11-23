import numpy as np
import pandas as pd
import math as math
import matplotlib.pyplot as plt

df = pd.read_csv('PCB.dt',sep='\t', names=['X','Y'])

x = df['X'].to_numpy().reshape(-1, 1)
y = df['Y'].to_numpy()
ylog = np.log(y)
xsquared = np.sqrt(x)

def calculateAB(x,y):
    z = np.ones((len(x),1))
    X = np.append(x,z,axis=1)
    a,b = np.dot(np.dot(np.linalg.inv(np.dot(X.T,X)),X.T),y)
    return a,b

a,b = calculateAB(x,ylog)
print("My prediction model: exp(",a,"* x +",b,") = y")

def h(x,a,b):
    return np.exp(a*x+b)

y_log_hat = [math.log(h(i,a,b)) for i in x]  #Calculate the predicted y-label for all ages in x_new

def MSE(ylog,yhat):
    n = len(ylog)
    sum = 0.0
    for i in range(n):
        sum += (ylog[i]-yhat[i])**2
    return sum*(1/n)

MSEfirst = MSE(ylog,y_log_hat)
print("MSE log: ",MSE(ylog,y_log_hat))
mean = sum(ylog)/len(ylog)

def R2(ylog,yhat):
    sum1, sum2 = 0.0,0.0
    n = len(yhat)
    for i in range(n):
        sum1 += (ylog[i] - yhat[i])**2
        sum2 += (ylog[i] - mean)**2
    return 1 - (sum1/sum2)

print("R2 log: ",R2(ylog,y_log_hat))

w,b2 = calculateAB(xsquared,ylog)
print("My prediction model: exp(",w,"* x +",b2,") = y")

y_squared_hat = [math.log(h(i,w,b2)) for i in xsquared]

my_x = np.linspace(0,14,1000)
my_y = [math.log(h(np.sqrt(i),w,b2)) for i in my_x]

print("MSE sqrt: ",MSE(ylog,y_squared_hat))
print("R2 sqrt: ",R2(ylog,y_squared_hat))

#plt.plot(x,ylog,'o',label="measurements",color="red")
plt.plot(x,y_log_hat, label="Linear regression",color="Orange")
plt.plot(x,ylog,'o')
plt.plot(my_x,my_y,label="Non-linear regression",color="Green")

plt.title("Regression")
plt.legend()
plt.xlabel("Age (yrs.)")
plt.ylabel("log(PCB Conc. (ppm))")
plt.show()
