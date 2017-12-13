import csv
import numpy as np
import matplotlib.pyplot as plt
import math
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
#Used by SMC to sample from x
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
#Used by resample() to sample a vector of x:s
def sampleX(n_particles,params,x_prev=None):
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

def resample(n_particles, yt, params):
    n_observations = len(yt)
    Wt = np.ones((n_observations + 1, n_particles))
    Xt_resampled = np.ones((n_observations + 1, n_particles))
    c = np.ones(n_particles)

    #Here we save the values of the unnormalized weights that we will need in Q5
    Wt_resampled = np.ones((n_observations + 1, n_particles))

    #Initiate weights
    for i in range(n_particles):
        Wt[0][i] = np.random.uniform(1,n_particles)
    Wt[0] /= np.sum(Wt[0])

    #Initiate the first row of th unnormalized matrix with the Wt[0]
    Wt_resampled[0] = Wt[0]

    Xt_resampled[0] = sampleX(n_particles, params)

    #initial resampling
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
        Xt_resampled[t] = sampleX(n_particles,params, Xt_resampled[t-1])
        for i in range(n_particles):
            y_var = params[2]**2 * np.exp(Xt_resampled[t][i])
            likelihood = scipy.stats.multivariate_normal.pdf(yt[t-1],0, y_var)
            Wt[t][i] = likelihood * Wt[t-1][i]
            #Saving the values of the weights
            Wt_resampled[t][i] = likelihood * Wt_resampled[t-1][i]
        #Normalize them for the calculations on resampling
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
        #for i in range(n_particles):
        #    Wt[t][i] = 1/np.sum(Wt[t])
    return Xt_resampled, Wt_resampled

def plotSMC(Xt):
    variance_nores = np.zeros(len(Xt))
    mean_nores = np.zeros(len(Xt))
    for t in range(len(Xt)):
        variance_nores[t] = np.var(Xt[t])
        mean_nores[t] = np.sum(Xt[t])/len(Xt[0])
    #plt.plot(np.linspace(-3,3,501), variance_nores)
    plt.plot(np.linspace(-1,500,501), mean_nores)
    plt.gca().fill_between(np.linspace(-1,500,501), mean_nores-np.sqrt(variance_nores), mean_nores+np.sqrt(variance_nores), color="#dddddd")
    plt.title('SMC without systematic resampling')
    plt.show()


def plotResampling(Xt_resampled):
    mean_res = np.zeros(len(Xt_resampled))
    variance_resampling = np.zeros(len(Xt_resampled))
    for t in range(len(Xt_resampled)):
        variance_resampling[t] = np.var(Xt_resampled[t])
        mean_res[t] = np.sum(Xt_resampled[t])/len(Xt_resampled[0])
    #plt.plot(np.linspace(-3,3,501), variance_resampling)
    plt.plot(np.linspace(-1,500,501), mean_res)
    plt.gca().fill_between(np.linspace(-1,500,501), mean_res-np.sqrt(variance_resampling), mean_res+np.sqrt(variance_resampling), color="#dddddd")
    plt.title('SMC with the systematic resampling')
    plt.show()

def betaLikelihood(n_particles, yt, params,SMC_count):
    T = len(yt)
    betas = np.zeros(10)
    syntetic_beta = params[2]
    print(syntetic_beta)
    betas[0] = 0.05
    for i in range(1, len(betas)):
        betas[i] = syntetic_beta*i
    for b,beta in enumerate(betas):
        #changing the value of the vector with another beta
        params[2] = beta
        _, Wt_resampled = resample(n_particles, yt,params)
        #print(Wt_resampled)
        likelihoods = np.zeros(SMC_count)
        for run_count in range(SMC_count):
            likelihoods[run_count]=0
            for t in range(T + 1):
                inner_term = 0
                for n in range(n_particles):
                    inner_term += Wt_resampled[t][n]
                likelihoods[run_count] += np.log(inner_term) - np.log(n_particles)
        variance = np.var(likelihoods, dtype=np.float64)
        mean = np.sum(likelihoods) / SMC_count
        print(beta, mean, variance)
    #lebels =
    #plt.boxplot(likelihoods, 0.75, 'rs',0, 0.25)
    #print
    #plt.show()

def main():
    yt, params = readData()
    T = 50
    n_particles = 10
    SMC_count = 10

    betaLikelihood(n_particles, yt[:T], params, SMC_count)




    #print(params)

    #Wt = calculateSMC(Xt, yt, params, n_particles)
    #for t in range(len(Xt)):
    #    mean_nores[t] = np.sum(Xt[t])/len(Xt[0])
    #plt.plot(np.linspace(-1,500, 501),mean_nores)
    #plt.show()
    #Xt = generateXt(T, n_particles, params)
    #plotSMC(Xt)

    #Xt_resampled, Wt_resampled = resample(n_particles, yt, params)
    # w = np.ones(501)
    # print(Wt_resampled)
    # for i in range(501):
    #     w[i] = np.sum(Wt_resampled[i])
    # plt.plot(np.linspace(-1, 500, 501), w)
    #plotResampling(Xt_resampled)
    #plt.show()
main()
