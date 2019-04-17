import os # change dir
Wdir= os.getcwd()
import sys 
sys.path.append(Wdir+'\functions')# add to path

import numpy as np
import scipy.io as sio # load save matlab files
import scipy.signal as sg # signal pack
import matplotlib.pyplot as plt # plotting
from matplotlib.gridspec import GridSpec # easy subplots
from functions.IAS_functions import signal_kharmdb_noise,iaslinearapprox,\
iasbaseline,f_est_linear

# %% Numerical signal
lfile = 'Dataz\\dummy_6_christansen_base_linear_engine.mat'
vnames = ['f_estS', 'fs', 't_est']
mat_contents = sio.loadmat(lfile, mdict=None, appendmat=True,variable_names=vnames)
for k,v in mat_contents.items():
    globals()[k]=v
f_0 = f_estS
fs2 = fs
t_est = t_est.reshape(-1,1) # force it to be a column vector
del k,v,mat_contents,f_estS,vnames


fs = 2**12
K = fs/2/max(f_0); K = int(K)
f_0 = sg.resample(f_0,int(t_est[-1]/4*fs/fs2))# for computational reasons 4 times faster the IAS profile f(4t)
f_0 = f_0[fs:-fs]

Nwk = 2**int(np.log2(.5*fs)) # window length
f_0= sg.savgol_filter(f_0[:,0],int(Nwk-1),3)# 2del
# A with a slice
A  = np.ones((f_0.size,K));
pK = 1 # pk cantidad de segmentos aleatorios
Nw = int(.1*fs/2) # Zeors size ________________________________________________
for m in np.arange(0,K):
    #if m==0:
    pos = np.random.randint(Nw,f_0.size-Nw,pK)
    for k in np.arange(0,pK):
        A[pos[k]:Nw+pos[k],m] = np.zeros((1,Nw))


# equal length signal as windows integer amount of slices
Nw = 2**int(np.log2(.2*fs)) # window length
H  = int(Nw/2)                # step
L     = int(np.round((f_0.size-Nw)/H))
pos   = np.r_[:Nw]
f_02  = f_0[0:int(pos[-1]+L*H)]
A     = A[0:f_02.size,:]
del f_0,Nwk

snrdb = np.r_[100]
x_f_0 = np.zeros((f_02.size,2))
x_f_0[:,1] = f_02; del f_02
x_f_0[:,0],_ = signal_kharmdb_noise(A,x_f_0[:,1],K,fs,\
     sigmadb=snrdb[snrdb.size-1],zeta= 0.05,color='white')

snrdb = np.r_[-20:21]/2
datanoisy_w = np.zeros([x_f_0.shape[0],snrdb.size])
datanoisy_p = np.zeros(datanoisy_w.shape)
for k in np.r_[:snrdb.size]:
    datanoisy_w[:,k],_ = signal_kharmdb_noise(A,x_f_0[:,1],K,fs,\
     sigmadb=snrdb[k],zeta= 0.05,color='white')
    datanoisy_p[:,k],_ = signal_kharmdb_noise(A,x_f_0[:,1],K,fs,\
     sigmadb=snrdb[k],zeta= 0.05,color='pink')

result_dict = {'x_f_0': x_f_0,'fs':fs,'datanoisy_w':datanoisy_w,\
                           'datanoisy_p':datanoisy_p,'H':H,'Nw':Nw,\
                           'K':K,'snrdb':snrdb}
sio.savemat('Dataz/'+'Numerical_signal'+'.mat',result_dict,do_compression=True)
