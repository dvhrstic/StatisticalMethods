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

def generateXt(T, n_particles, params):
    Xt = np.zeros((T+1, n_particles))
    #Samling from stationary Xo
    Xt[0] = np.random.normal(0,np.sqrt(params[1] ** 2 / (1 - params[0] ** 2)))
    for i in range(n_particles):
        Xt[0][i] = np.random.normal(0, params[1]**2 / (1 - params[0]**2))
    for t in range(1, T + 1):
        for i in range(n_particles):
            Xt[t][i] = np.random.normal(params[0] * Xt[t-1][i], params[1]**2)
    return Xt

def calculateSMC(Xt,yt, params, n_particles):
    wt = np.ones((len(Xt), n_particles))
    #wt[0] = wt[0]/n_particles
    #Instantiate weights randomly across 100 particles
    for i in range(n_particles):
        y_var = params[2]**2 * np.exp(Xt[0][i])
        wt[0][i] = np.random.uniform(1,n_particles)
    wt[0] /= np.sum(wt[0])
    #Iterate in time t, calculate weights w.r.t previous value and likelihood
    for t in range(1,len(yt)+1):
        for i in range(n_particles):
            y_var = params[2]**2 * np.exp(Xt[t][i])
            likelihood = scipy.stats.multivariate_normal.pdf(yt[t-1],0, y_var)
            wt[t][i] = likelihood * wt[t-1][i]
        #in order to normalize we need to divide and thereby we give som values to zeros
        wt[t] += 1.e-300
        wt[t] /= np.sum(wt[t])
    return wt

yt, params = readData()
#We need x in order to calculate the variance for yt
T = 500
n_particles = 100
Xt = generateXt(T, n_particles, params)
print(len(Xt))
Wt = calculateSMC(Xt, yt, params, n_particles)
#max_weight = np.max(Wt[2])
#particle = np.where(Wt[2] == max_weight)[0][0]
#print("Max weight ",max_weight, " belongs to particle ", particle)
#print(Wt)
#print("Weights sum:",np.sum(wt[2]))
plt.plot(np.linspace(1,100,n_particles), Wt[2], '*')
plt.show()
