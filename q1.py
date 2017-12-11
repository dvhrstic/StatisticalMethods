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
    Xt[0] = np.random.normal(0,params[1] ** 2 / (1 - params[0] ** 2))
    for i in range(n_particles):
        Xt[0][i] = np.random.normal(0, params[1]**2 / (1 - params[0]**2))
    for t in range(1, T + 1):
        for i in range(n_particles):
            Xt[t][i] = np.random.normal(params[0] * Xt[t-1][i], params[1]**2)
    return Xt

def sampleX(n_particles,x_prev=None):
    x_new = np.ones(n_particles)
    if x_prev is None:
        for i in range(n_particles):
            x_new[i] = np.random.normal(0,params[1] ** 2 / (1 - params[0] ** 2))
    else:
        for i in range(n_particles):
            x_new[i] = np.random.normal(params[0] * x_prev[i], params[1]**2)
    return x_new

def calculateSMC(Xt,yt, params, n_particles):
    wt = np.ones((len(Xt), n_particles))
    #wt[0] = wt[0]/n_particles
    #Instantiate weights randomly across 100 particles
    for i in range(n_particles):
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

def resample(Xt, yt):
    n_particles = len(Xt[0])
    n_observations = len(Xt)-1
    print(n_observations)
    Wt = np.ones((n_observations + 1, n_particles))
    Xt_resampled = np.ones((n_observations + 1, n_particles))
    c = np.ones(n_particles)

    #Initiate weights
    for i in range(n_particles):
        Wt[0][i] = np.random.uniform(1,n_particles)
    Wt[0] /= np.sum(Wt[0])

    Xt_resampled[0] = sampleX(n_particles)

    c[0] = Wt[0][0]
    for i in range(1,n_particles):
            c[i] = c[i-1] + Wt[0][i]
    offset = np.random.uniform(0, 1/n_particles)
    index = 0
    for p in range(n_particles):
        while(offset > c[index]):
            index += 1
        Xt_resampled[0][p] = Xt_resampled[0][index]
        offset = offset + 1/n_particles

    for t in range(1, n_observations+1):
        Xt_resampled[t] = sampleX(n_particles, Xt_resampled[t-1])
        for i in range(n_particles):
            y_var = params[2]**2 * np.exp(Xt_resampled[t][i])
            likelihood = scipy.stats.multivariate_normal.pdf(yt[t-1],0, y_var)
            Wt[t][i] = likelihood * Wt[t-1][i]
            #May be deleted eventually
        Wt[t] += 1.e-300
        Wt[t] /= np.sum(Wt[t])
        #Resampling
        c[0] = Wt[t][0]
        for i in range(1,n_particles):
            c[i] = c[i-1] + Wt[t][i]
        offset = np.random.uniform(0, 1/n_particles)
        index = 0
        for p in range(n_particles):
            while(offset > c[index]):
                index += 1
            Xt_resampled[t][p] = Xt_resampled[t][index]
            offset = offset + 1/n_particles
        #Here weights should be uniformed
        for i in range(n_particles):
            Wt[t][i] = 1/np.sum(Wt[t])
    return Xt_resampled


yt, params = readData()
#We need x in order to calculate the variance for yt
T = 500
n_particles = 100
#number of times we resample
N = 10
Xt = generateXt(T, n_particles, params)
#Wt = calculateSMC(Xt, yt, params, n_particles)
Xt_resampled = resample(Xt, yt)

variance_resampling = np.zeros(len(Xt_resampled))
variance_nores = np.zeros(len(Xt))

for t in range(len(Xt)):
    variance_nores[t] = np.var(Xt[t])
for t in range(len(Xt_resampled)):
    variance_resampling[t] = np.var(Xt_resampled[t])

plt.plot(np.linspace(-3,3,501), variance_nores, '+')
plt.plot(np.linspace(-3,3,501),variance_resampling, '*')
plt.show()
#print(Xt_resampled.)

#max_weight = np.max(Wt[2])
#particle = np.where(Wt[2] == max_weight)[0][0]
#print("Max weight ",max_weight, " belongs to particle ", particle)
#print(Wt)
#print("Weights sum:",np.sum(wt[2]))
#plt.plot(np.linspace(1,100,n_particles), Wt[2], '*')
#plt.show()
