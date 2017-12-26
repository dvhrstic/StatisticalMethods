import csv
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.stats import norm
from sampleFunc import sample_xt_prim

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

def generate_xt(T, params):
    xt = np.zeros(T+1)
    #Samling from stationary Xo
    xt[0] = np.random.normal(0, params[1]**2 / (1 - params[0]**2))
    #Sampling from ~ N(phi*x_t-1 , sigma ^2)
    for t in range(1, T + 1):
        xt[t] = np.random.normal(params[0] * xt[t-1], params[1]**2)
    return xt

def sampleBetas(N, syntetic_beta=None):
    beta = np.zeros(N)
    beta[0] =  0.01

    for n in range(1, N):
        beta[n] = 0.05*(n*4)
    return beta

def mcmc(xt,yt,t):
    return 0

yt, params = readData()
T = len(yt)
print(params)
#maybe this should be done for every new beta
xt = generate_xt(T, params)
t = np.random.randint(0, 500)
N = 10
betas = sampleBetas(N)
beta_occ = np.zeros(N)
x_curr = xt[t]
for si in range(1000):
    for b,beta in enumerate(betas):
        params = np.append(params[:2], beta)
        x_prime, gt_prime,gt_curr = sample_xt_prim(xt, yt, params, t)
        prior_curr = norm(params[0]*xt[t-1], params[1]**2).pdf(xt[t])
        prior_xt_prime = norm(params[0]*x_prime, params[1]**2).pdf(x_prime)
        term1 = prior_xt_prime * gt_curr
        term2 = prior_curr * gt_prime
        acc_prob = term1 / term2
        r = min(1.0, acc_prob)
        u = np.random.uniform()
        if(u < r):
            x_curr = x_curr
            beta_occ[b] += 1
print(betas)
print(beta_occ)
