# signal_kharmdb_pink.m
import numpy as np # math pack!
import pandas as pd # subplots?
import scipy.signal as sg # DSP
import time # tic toc
import scipy.optimize as optm # optimization pack
import scipy.io as sio # load save matlab files
from scipy import interpolate # interpolate lib
from scipy import integrate # integration pack
import vibration_toolbox as vtb # vibration model
# %% Functions to make the numerical signals
def signal_kharmdb_noise(A,f_0,K,fs,sigmadb,zeta= 0.05,color='white'):
# color='white' or color='pink'
    # N length of the signal
    # f_0 IF profile
    # sigmadb snr in db
    
    w_0= np.cumsum(f_0) / fs
    N  = w_0.size
    x = np.zeros(w_0.size)
    if A.size >= K:
        for k in np.arange(0,K):
            x = x + A[:,k]*np.cos(2*np.pi*k*w_0)
    else:
        for k in np.arange(0,K):
            x = x + A[k]*np.cos(2*np.pi*k*w_0)
           
    # zeta= 0.05
    m     = 1
    omega_d = .25*fs*2*np.pi
    omega   = omega_d*np.sqrt(1 - zeta ** 2)
    c     = zeta*m*omega*2 
    k     = omega**2*m
    x0    = 1
    v0    =-1
    max_time=N/fs

    _, hs, *_ = vtb.sdof.free_response(m, c, k, x0, v0, max_time,fs) 
    hs = hs[:,0]
    # k**2 it is the natural undamped frequency given m =1
    # omega = np.sqrt(k / m)
    # zeta = c / 2 / omega / m
    # 250 Hz as fs default
    
    x = sg.fftconvolve(x, hs, mode='full')
    x = x[0:w_0.size]
    x = noise_snr(x,sigmadb,color)
    return x,hs

def noise_snr(sig,reqSNR,color):
    sigPower = np.sum(np.abs(sig**2))/sig.size
    reqSNR   = np.power(10,reqSNR/10)
    noisePower=sigPower/reqSNR
    if color=='pink':
        noise=np.sqrt(noisePower)*pink(sig.size)
    if color=='white':
        noise=np.sqrt(noisePower)*np.random.randn(sig.size)
    sig=sig + noise
    return sig
# % Noise
def pink(nrows, ncols=16):
    """Generates pink noise using the Voss-McCartney algorithm.
    
    nrows: number of values to generate
    rcols: number of random sources to add
    
    returns: NumPy array
    """
    array = np.empty((nrows, ncols))
    array.fill(np.nan)
    array[0, :] = np.random.random(ncols)
    array[:, 0] = np.random.random(nrows)
    
    # the total number of changes is nrows
    n = nrows
    cols = np.random.geometric(0.5, n)
    cols[cols >= ncols] = 0
    rows = np.random.randint(nrows, size=n)
    array[rows, cols] = np.random.random(n)

    df = pd.DataFrame(array)
    df.fillna(method='ffill', axis=0, inplace=True)
    total = df.sum(axis=1)
    
    # zscore
    x = total.values
    x = x - np.mean(x)
    x = x/np.std(x)
    #x = x/np.trapz(np.abs(x)**2)/x.size
    return x
# %% IAS optimization functions
def cost_fun_1d(param=np.r_[240],x=np.r_[:100],L=8):#(np.array([1,2]),array,N,L)
    N=x.size
    Z= vanderZ(0,param[0],N,L) # alpha,w_0,N,L
    P= np.var(x-Z@np.linalg.pinv(Z)@x)
    C= N*np.log(P) # only necessary for order estiamtion
    return C

def cost_fun_2d(param=np.r_[0,240],x=np.r_[:100],L=8):#(np.array([1,2]),array,N,L)
    N=x.size
    Z= vanderZ(param[0],param[1],N,L) # alpha,w_0,N,L
    P= np.var(x-Z@np.linalg.pinv(Z)@x)
    C= N*np.log(P)  # only necessary for order estiamtion
    return C

# a.shape size of a matrix
# a.size nuel of a matrix
# @ matrix multiplication
def vanderZ(alpha=0,w_0=2*np.pi,N=100,L=8):
    N = np.arange(0,N)
    z = np.zeros((N.size,L))+1j*np.zeros((N.size,L))
    for l in np.arange(0,L):
        z[:,l] = np.exp(1j*(l+1)*(0.5*alpha*N**2 + w_0*N)*2*np.pi)
    return z

def cost_func_grid(alpha_g=np.r_[0],w_g=np.r_[2*np.pi],xw=np.r_[:100],L=8):
    C = np.zeros((w_g.size,alpha_g.size))
    for lm in np.r_[:alpha_g.size]:
        for m in np.r_[:w_g.size]:
            param = np.r_[alpha_g[lm],w_g[m]]
            C[m,lm]  = cost_fun_2d(param,xw,L)
    idx = np.where(C == np.min(C))
    #idx = np.unravel_index(np.argmin(C, axis=None), C.shape)
    return C,idx

def iaslinearapproxv2(data_r,Nw,K,fmaxmin,delta,chirp,savename):
    # data_r data in real domain
    # Nw window length for method
    # K=np.r[1,2,3] amount of orders to study from 0 to K
    # fmaxmin=[170,260]/fs, alpmaxmin=[-1e-3,1e-3]
    # delta normalized delta for adaptive search interval
    # chirp==0 baseline no chirp rate
    # chirp==1 linear approximation
    # savename='savedir'
    
    f0_max = fmaxmin[1]
    f0_min = fmaxmin[0]
    
    H     = int(Nw/2) # step
    L     = int(np.round((data_r.size-Nw)/H))
    pos   = np.r_[:Nw]
    w_alph_e = np.zeros([L,2*K.size])
    elapsed = np.zeros(L*K.size)
    kount = 0
    
    Nr    = 2**8 # resolution of w_g
    delta_w = (f0_max-f0_min)/2
    w_c   = f0_min+delta_w
    w_g   = np.linspace(w_c-delta_w,w_c+delta_w,int(Nr))
    delta_w = delta_w
    flag = 0
    
    f0_min2 = np.zeros([L,K.size])
    f0_max2 = np.zeros(f0_min2.shape)

    for k in np.r_[:K.size]:
        for l in np.arange(0,L):
            t = time.time() 
            xw = sg.hilbert(data_r[pos+l*H])
            
            if l==0 or flag==1:                
                C,idx = cost_func_grid(np.r_[w_alph_e[l,2*k+1]],w_g,xw,K[k])
                w_alph_e[l,2*k] = w_g[idx[0]]
                
                f0_max2[l,k] = w_alph_e[l,2*k]+delta
                f0_min2[l,k] = f0_max2[l,k]-2*delta
                flag = 0
            elif chirp==1:
                x0   = np.r_[w_alph_e[l-1,2*k+1],w_alph_e[l-1,2*k]]
                args = (xw,K[k])
                Xc = optm.minimize(cost_fun_2d, x0, args, 
                               method='Nelder-Mead',options={'xatol':1e-6})# default 1e-4
                w_alph_e[l,2*k+1] = np.r_[Xc['x'][0]]
                w_alph_e[l,2*k] = np.r_[Xc['x'][1]]
            
                f0_max2[l,k] = w_alph_e[l-1,2*k]+delta
                f0_min2[l,k] = f0_max2[l,k]-2*delta
            elif chirp==0:
                x0   = np.r_[w_alph_e[l-1,2*k]]
                args = (xw,K[k])
                Xc = optm.minimize(cost_fun_1d, x0, args, 
                               method='Nelder-Mead',options={'xatol':1e-6})# default 1e-4
                w_alph_e[l,2*k] = np.r_[Xc['x'][0]]
                
                f0_max2[l,k] = w_alph_e[l-1,2*k]+delta
                f0_min2[l,k] = f0_max2[l,k]-2*delta
            
            if (w_alph_e[l,2*k]>f0_max2[l,k] or w_alph_e[l,2*k]<f0_min2[l,k])\
            or (w_alph_e[l,2*k]>f0_max or w_alph_e[l,2*k]<f0_min):
                w_alph_e[l,2*k] = w_alph_e[l-1,2*k]
                w_alph_e[l,2*k+1] = w_alph_e[l-1,2*k+1]
                
                flag = 1
            
            elapsed[kount] = time.time() - t    
            
            print('elapsed=',round(elapsed[kount],4),'[sec], restante=', 
                  round((L*K.size-kount)*np.mean(elapsed[kount-l:kount])/60,4),'[Min]')
            kount+=1
            print('iteration',kount,'d', L*K.size)
            
        result_dict = {'f_alpha_est': w_alph_e,'data_r': data_r,\
                       'H':H, 'K':K}
        sio.savemat('Dataz/'+savename+'.mat', 
                result_dict,do_compression=True)
    return w_alph_e

def f_est_linear(alpha_est,f_est,Nw=512,H=256):
    # function to find the mean frequency given the linear approximation
    # x original signal
    # alpha_est chirp rate
    # f_est intial frequency
    # Nw window length
    # H overlap
    N = np.arange(0,Nw)
    f_est_lin = np.zeros(H*(f_est.size-2)+2*H)
    f_est_lin2 = np.zeros((3,Nw))
    for k in np.r_[:f_est.size-2]:
        f_est_lin2[0,:] = alpha_est[k]*N + f_est[k]
        f_est_lin2[1,:] = alpha_est[k+1]*N + f_est[k+1]
        f_est_lin2[2,:] = alpha_est[k+2]*N + f_est[k+2]
        if k==0:
            f_est_1 = np.r_[f_est_lin2[0,:H]]
            f_est_2 = np.r_[[f_est_lin2[0,H:]],[f_est_lin2[1,:H]]]
            f_est_2 = np.mean(f_est_2,axis=0)
            f_est_3 = np.r_[[f_est_lin2[1,H:]],[f_est_lin2[2,:H]]]
            f_est_3 = np.mean(f_est_3,axis=0)
            f_est_lin[k*H:k*H+3*H] = np.r_[f_est_1,f_est_2,f_est_3]
        else:
            f_est_1 = np.r_[f_est_2]
            f_est_2 = np.r_[[f_est_lin2[0,H:]],[f_est_lin2[1,:H]]]
            f_est_2 = np.mean(f_est_2,axis=0)
            f_est_3 = np.r_[[f_est_lin2[1,H:]],[f_est_lin2[2,:H]]]
            f_est_3 = np.mean(f_est_3,axis=0)
            f_est_lin[k*H:k*H+3*H] = np.r_[f_est_1,f_est_2,f_est_3]
    return f_est_lin
# encoder signal to IAS
def tTacho_fsigLVA(top,fs,TPT=44):
    top = top/max(abs(top))
    
    t = np.r_[:len(top)]/fs
    seuil=0;
    ifm=np.where((top[:-1]<=seuil) & (top[1:]>seuil))[0]
    tfm = (t[ifm]*top[ifm+1]-t[ifm+1]*top[ifm])/(top[ifm+1] -top[ifm])
    ifm = tfm*fs;
    
    W =np.array([])
    tW=np.array([])
    for ii in np.r_[:TPT]:
        tfmii = tfm[ii::TPT];
        W=np.append(W,[1/np.diff(tfmii)])
        tW=np.append(tW,[tfmii[:-1]/2 + tfmii[1:]/2])
    
    ordre=np.argsort(tW)
    tW=np.sort(tW)
    W = W[ordre]
    
    W=interpolate.InterpolatedUnivariateSpline\
    (tW,W,w=None, bbox=[None, None])
    
    W = W(t)
    return t,W
