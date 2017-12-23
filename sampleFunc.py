import numpy as np

def sample_xt(xt, yt, params, t):
    T = len(yt)
    rand = np.random()
    x_new = gt = 1
    if(rand <= 0.33):
        if(t > 0):
            vn = np.random.normal(0,1)
            x_new = params[0]*xt[t-1] + params[1]*vn
        else:
            x_new = np.random.normal(0, params[1]^2/(1 - params[0]^2))
        gt = (1/params[1]) * np.sqrt(2/np.pi) * (-1/(2*params[1]^2))
    elif(rand > 0.33 and rand <= 0.66)
        if (t < T):
            vn = np.random.normal(0,1)
            x_new = (xt[t+1] - params[1]*vn)/params[0]
            gt = (params[0]/params[1]) * np.sqrt(2/np.pi) * (-params[0]^2/(2*params[1]^2))
        else:
            x_new = xt[T]
    else:
        if(t > 0):
            wn = np.random.normal(0,1)
            x_new = 2 * np.log(np.abs(y[t-1]) / (params[2] * np.abs(wn)))
        else:
            x_new = xt[0]
        gt = np.abs(yt[t-1])/(np.sqrt(2*np.pi)*params[2]) * np.exp(
            (yt[t-1]^2/(2*params[2]^2)) * np.exp(x_new) - x_new/2   )
    return x_new, gt
