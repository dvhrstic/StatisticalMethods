############################
# ----------Author---------#
# ---Dusan Viktor Hrstic---#
# ----------2017-----------#
############################
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
    #Xt[0] = np.random.normal(0,params[1] ** 2 / (1 - params[0] ** 2))
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

def calculateSMC(Xt,yt, params, n_particles, beta=None):
    wt = np.ones((len(Xt), n_particles))
    #wt[0] = wt[0]/n_particles
    #Instantiate weights with the likelihood of each X0 ~ N(0, sigma^2/(1-phi^2))
    for i in range(n_particles):
        wt[0][i] = scipy.stats.multivariate_normal.pdf(Xt[0][i], 0, params[1]**2/(1 - params[0]**2))
    wt[0] /= np.sum(wt[0])
    #Iterate in time t, calculate weights w.r.t previous value and likelihood
    for t in range(1,len(yt)+1):
        for i in range(n_particles):
            #Beta is unknown and we are sampling from uniform distribution (0, 2)
            params[2] = np.random.uniform(0.0001, 2)
            y_var = params[2]**2 * np.exp(Xt[t][i])
            likelihood = scipy.stats.multivariate_normal.pdf(yt[t-1],0, y_var)
            wt[t][i] = likelihood * wt[t-1][i]
        #in order to normalize we need to divide and thereby we give som values to zeros
        wt[t] += 1.e-300
        wt[t] /= np.sum(wt[t])
    return wt

def resample(n_particles, yt, params, beta=None, betaLogLikelihood=False):
    n_observations = len(yt)
    Wt = np.ones((n_observations + 1, n_particles))
    Xt_resampled = np.ones((n_observations + 1, n_particles))
    c = np.ones(n_particles)

    if type(beta).__name__ == 'float64':
        params[2] = beta
    else:
        params[2] = np.random.uniform(0.001, 2)

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
        if type(beta).__name__ == 'float64':
            params[2] = beta
        else:
            params[2] = np.random.uniform(0.001, 2)

        Xt_resampled[t] = sampleX(n_particles,params, Xt_resampled[t-1])

        for i in range(n_particles):

            y_var = params[2]**2 * np.exp(Xt_resampled[t][i])
            likelihood = scipy.stats.multivariate_normal.pdf(yt[t-1],0, y_var)
            Wt[t][i] = likelihood * Wt[t-1][i]
            #Saving the values of the weights
            Wt_resampled[t][i] = likelihood #* Wt_resampled[t-1][i]
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
        Wt[t].fill(1/np.sum(Wt[t]))
        #Here weights should be uniformed
        #for i in range(n_particles):
        #    Wt[t][i] = 1/np.sum(Wt[t])
    if betaLogLikelihood == True:
        return Xt_resampled, Wt_resampled
    else:
        return Xt_resampled, Wt

def plotSMC(Xt, Wt):
    variance_nores = np.zeros(len(Xt))
    estimator = np.zeros(len(Xt))
    for t in range(len(Xt)):
        variance_nores[t] = np.var(Xt[t], ddof=1)
        estimator_index = np.argmax(Wt[t])
        estimator[t] = Xt[t][estimator_index]
    print("SMC no resampling, average of variances", np.mean(variance_nores))
    plt.plot(np.linspace(-1,len(Xt), len(Xt)), estimator)
    plt.gca().fill_between(np.linspace(-1,len(Xt),len(Xt)),estimator - np.sqrt(variance_nores),estimator + np.sqrt(variance_nores), color="#dddddd")
    plt.title('SMC no resampling, average of variances')
    plt.show()


def plotResampling(Xt_resampled, Wt_resampled):
    estimator = np.zeros(len(Xt_resampled))
    variance_resampling = np.zeros(len(Xt_resampled))
    for t in range(len(Xt_resampled)):
        variance_resampling[t] = np.var(Xt_resampled[t], ddof=1)
        estimator_index = np.argmax(Wt_resampled[t])
        estimator[t] = Xt_resampled[t][estimator_index]
        #plt.plot(np.linspace(-3,3,501), variance_resampling)
    print("Resampling SMC, average of variances", np.mean(variance_resampling))
    plt.plot(np.linspace(-1,len(Xt_resampled), len(Xt_resampled)), estimator)
    plt.gca().fill_between(np.linspace(-1,len(Xt_resampled),len(Xt_resampled)), estimator-np.sqrt(variance_resampling), estimator+np.sqrt(variance_resampling), color="#dddddd")
    plt.title('Resampling SMC, average of variances')
    plt.show()

def betaLikelihood(n_particles, yt, params,SMC_count):
    T = len(yt)
    betas = sampleBetas(10)
    likelihoods = np.zeros((len(betas), SMC_count))
    for b,beta in enumerate(betas):
        params[2] = beta
        for run_count in range(SMC_count):
            _, Wt_resampled = resample(n_particles, yt,params, beta, True)
            likelihoods[b][run_count]=0
            print("Beta ", b, "Round, ", run_count)
            for t in range(T + 1):
                inner_term = 0
                for n in range(n_particles):
                    inner_term += Wt_resampled[t][n]
                likelihoods[b][run_count] += np.log(inner_term) - np.log(n_particles)
        variance = np.var(likelihoods[b], ddof=1)
        #mean = np.sum(likelihoods) / SMC_count
    list = []
    for r in range(len(betas)):
        list.append(likelihoods[r])
    #print(list)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.boxplot(list,positions=betas,widths=(0.15), showfliers='false')
    ax.set_xlabel("beta")
    ax.set_ylabel("log-likelihood")

    plt.show()

def sampleBetas(N, syntetic_beta=None):
    beta = np.zeros(N)
    for n in range(N):
        beta[n] =  np.random.uniform(0.00001, 2)
    return beta

def main():
    yt, params = readData()
    T = 500
    # Data depends on how many observationes we observe
    yt = yt[:T]
    n_particles = 10
    SMC_count = 10

    Xt = generateXt(T, n_particles, params)
    Wt = calculateSMC(Xt, yt, params, n_particles)
    plotSMC(Xt, Wt)

    Xt_resampled, Wt_resampled = resample(n_particles, yt, params)
    plotResampling(Xt_resampled, Wt_resampled)

    betaLikelihood(n_particles, yt, params, SMC_count)

main()
