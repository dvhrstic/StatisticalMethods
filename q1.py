import csv
import numpy as np
import matplotlib.pyplot as plt

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

print(params)

#beta = np.random.uniform(0,2,100)
#max_it = 10
#print(beta)
print(len(y)
T = 500
xt = np.zeros(T+1)
print(len(xt))
xt[0] = np.random.normal(0,np.sqrt(params[1] ** 2 / (1 - params[0] ** 2)))
for t in range(1, T + 1):
    xt[t] = np.random.normal(params[0] * xt[t - 1], params[1])
#plt.plot(x[1:501],y)

wt = np.zeros(500)
wt = y/xt[1:501]
print(wt)

t = np.linspace(1,501, 500)
plt.plot(t, y,'+')
plt.show()
