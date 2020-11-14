import numpy as np
from scipy.stats import bernoulli
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

############## Opgave 2, Markov & Chebyshev ##############
def my_bernoulli(bias, n, alpha, expected_value):
    experiments = []

    for i in range(0,n):
        R = bernoulli.rvs(size=20,p=bias)
        experiments.append(R)

    avg_experiments = [float(sum(i))/20 for i in experiments]

    y = []
    #Find antal af tilfælde, hvor sum[Xi] >= alpha (min y-akse)
    for i in alpha:
        count = 0
        for j in avg_experiments:
            if i<=j:
                count += 1
        y.append(count)

    markov_bound = np.array([expected_value/(i) for i in alpha])
    chebyshev_bound = [1]

    #Calculate chebyshev bound depending on bias:
    for i in alpha[1:]:
        if expected_value == 1/2:
            if (20*(2*i-1)**2) == 0:
                bound = 1
            else:
                bound = 1/float(20*(2*i-1)**2)
        if expected_value == 0.1:
            if ((9/5)/float((20*i-2)**2)) == 0:
                bound = 1
            else:
                bound = (9/5)/float((20*i-2)**2)
        if bound >= 1:
            bound = 1
        chebyshev_bound.append(bound)

    plt.plot(alpha,np.array(y)/n,label="Emperical frequency")
    plt.plot(alpha,markov_bound, label="Markov bound")
    plt.plot(alpha,chebyshev_bound, label="Chebyshev bound")
    plt.legend()
    plt.title("Emperical frequency with bias "+str(expected_value))
    plt.xlabel("Alpha")
    plt.ylabel("Normalized frequency")

    plt.show()


alphaA = np.arange(0.5,1.05,0.05)
#my_bernoulli(0.5,10000,alphaA,1/2)
alphaB = np.arange(0.1,1.05,0.05)
my_bernoulli(0.1,10000,alphaB,0.1)



