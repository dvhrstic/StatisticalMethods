import csv
import numpy as np
import matplotlib.pyplot as plt


#Read the data from the file and put it in the datastructures
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

def generateXt():
    T = 500
    xt = np.zeros(T+1)
    xt[0] = np.random.normal(0,np.sqrt(params[1] ** 2 / (1 - params[0] ** 2)))
    for t in range(1, T + 1):
        xt[t] = np.random.normal(params[0] * xt[t - 1], params[1])
    return xt

def calculateWeights(xt,yt, params):
    wt = np.zeros(len(yt))
    y_var = params[2]**2 * np.exp(xt[0])
    #y_mu = 0
    #Mean equals zero and that is why it is not included in the calculations
    wt[0] = -1/2 * np.log(2 * np.pi * y_var) - ((yt[0])**2)/(2 * y_var)
    for t in range(1,len(yt)):
        y_var = params[2]*params[2] * np.exp(xt[t])
        term1 = -1/2 * len(yt[:t]) * np.log(2 * np.pi * y_var)
        sum_term = 0
        for i in range(t+1):
            sum_term = sum_term + (yt[i]**2)/(2*y_var)
        wt[t] = (term1 - sum_term)
    #normalizing weights
    wt /= np.sum(wt)
    return wt

yt, params = readData()
#We need x in order to calculate the variance for yt
xt = generateXt()
print(xt)
print(yt)
wt = calculateWeights(xt, yt, params)
print("Weights sum"np.sum(wt))
plt.plot(np.linspace(0,1,500), wt,'*')
plt.show()
