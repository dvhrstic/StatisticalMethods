import csv
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats

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

def calculateSMC(xt,yt, params, n_particles):
    wt = np.ones((len(xt), n_particles))
    wt[0] = wt[0]/n_particles
    #Instantiate weights randomly across 100 particles
    for i in range(n_particles):
        y_var = params[2]**2 * np.exp(xt[0])
        wt[0][i] = np.random.uniform(1,n_particles)
    wt[0] /= np.sum(wt[0])
    #Iterate in time t, calculate weights w.r.t previous value and likelihood
    for t in range(1,len(yt)):
        likelihood = scipy.stats.multivariate_normal.pdf(xt[t],0, y_var)
        for i in range(n_particles):
            y_var = params[2]**2 * np.exp(xt[t])
            wt[t][i] = likelihood * wt[t-1][i]
        #in order to normalize we need to divide and thereby we give som values to zeros
        wt[t] += 1.e-300
        wt[t] /= np.sum(wt[t])
    return wt

yt, params = readData()
#We need x in order to calculate the variance for yt
xt = generateXt()
n_particles = 100
wt = calculateSMC(xt, yt, params, n_particles)
max_weight = np.max(wt[2])
particle = np.where(wt[2] == max_weight)[0][0]
print("Max weight ",max_weight, " belongs to particle ", particle)
#print(wt)
print("Weights sum:",np.sum(wt[2]))
plt.plot(np.linspace(1,100,n_particles), wt[2], '*')
plt.show()
