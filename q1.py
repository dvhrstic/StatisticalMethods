import csv
import numpy as np
import matplotlib.pyplot as plt

def readData():
    y = np.zeros(500)
    params = np.zeros(3)
    with open('mydata.csv', 'r',encoding='ascii') as datafile:
        reader = csv.reader(datafile)
        for i,row in enumerate(reader):
            y[i] = row[0]
    datafile.close()

    with open('myparameters.csv', 'r',encoding='utf8') as paramsfile:
        reader = csv.reader(paramsfile)
        for i,row in enumerate(reader):
            params[i] = row[0]
    paramsfile.close()

    return y, params

#TODO rewrite code for question 5
def generateXt():
    T = 500
    xt = np.zeros(T+1)
    xt[0] = np.random.normal(0,np.sqrt(params[1] ** 2 / (1 - params[0] ** 2)))
    for t in range(1, T + 1):
        xt[t] = np.random.normal(params[0] * xt[t - 1], params[1])
    return xt

def gauss_logL(y, V, n, sigma, mu):
    """Equation 5.57: gaussian likelihood"""
    return (-(n + 1) * np.log(sigma)
            - 0.5 * n * ((y - mu) ** 2 + V) / sigma ** 2)

def logLikelihood2(xt,yt, params):
    wt = np.zeros(len(yt))
    y_var = params[2]**2 * np.exp(xt[0])
    #y_mu = 0
    wt[0] = -1/2 * len(yt) * np.log(2 * np.pi * y_var) - ((yt[0])**2)/(2 * y_var)
    for t in range(1,len(yt)):
        y_var = params[2]**2 * np.exp(xt[t])
        wt[t] = (term1 - term2)*wt[t-1]
        print(wt[t])
    return wt


def logLikelihood(xt,yt, params):
    wt = np.zeros(len(yt))
    y_var = params[2]**2 * np.exp(xt[0])
    #y_mu = 0
    #Mean mu is equal to zero and that is why it is not included in wt[]
    #wt[0] = -1/2 * np.log(2 * np.pi * y_var) - ((yt[0])**2)/(2 * y_var)
    wt[0] = 1
    for t in range(1,len(yt)):
        y_var = params[2]*params[2] * np.exp(xt[t])
        term1 = -1/2 * len(yt[:t]) * np.log(2 * np.pi * y_var)
        sum_term = 0
        for i in range(t+1):
            sum_term = sum_term + (yt[i]**2)/(2*y_var)
        #normalizing weights
        wt[t] = (term1 - sum_term)
        #print(term1 - sum_term)
        #wt[t] = wt[t] / np.sum(wt)
    #wt += 1.e-300 # avoid round-off to zero
    wt /= np.sum(wt)
    return wt

yt, params = readData()
xt = generateXt()
print(xt)
print(yt)
wt = logLikelihood(xt, yt, params)
print(np.sum(wt))
plt.plot(np.linspace(0,1,500), wt,'*')
plt.show()
