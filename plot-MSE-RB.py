import os # change dir
import sys 
Wdir= os.getcwd()
sys.path.append(Wdir+'\functions')# add to path

import numpy as np
import scipy.io as sio # load save matlab files
import scipy.signal as sg # signal pack
import matplotlib.pyplot as plt # plotting
from scipy.io import wavfile # read wav files
from matplotlib.gridspec import GridSpec # easy subplots
from functions.IAS_functions import f_est_linear,tTacho_fsig,tTacho_fsigLVA

# function load matfile
def loading_matfile(filed='none'):
    mat_contents = sio.loadmat(filed, mdict=None, appendmat=True)
    for k, v in mat_contents.items():
        try:
            if v.size==sum(v.shape)-1:
                globals()[k]=v.flatten()
            else:
                globals()[k]=v
        except:
            globals()[k]=v
    return globals()
plt.rc('font', size=14) # default text size
# %% ACC1
lfile = (Wdir+'\Surveillance8 Contest')
fs, tacho = wavfile.read(lfile+'\Tacho.wav')

t,f_0 = tTacho_fsigLVA(tacho,fs,TPT=44)
f_0 = f_0*62/61 # visual examining
dfac = 2**4 # decimate factor
fs2    = fs/dfac # new sampling frequency
f_0r = sg.decimate(f_0,dfac) # Referencing to L5
f_0r = f_0r[int(fs2/2):int(-fs2/2)]
t = np.r_[:len(f_0)]/fs2

lfile = ['D16_ias_baseline_accx','D16_ias_linear_approx_accx'\
         ,'D16_ias_baseline_acc2_nodown_dfacf0'\
         ,'D16_ias_linear_acc2_nodown_dfacf0'\
         ,'D16_ias_baseline_acc1_nodown_dfacf0'\
         ,'D16_ias_linear_acc1_nodown_dfacf0']# ACC1 y 2 downsampled


n = 0 # amount of orders used in the estimation 0 for all 1 for all-1
lfileK = len(lfile)
f_0mse = np.zeros([3,lfileK])# ACC0 ACC1 ACC2
for lfilek in np.r_[:lfileK]:
    loading_matfile('Dataz/'+lfile[lfilek]); # same parameters all tests
    Nw = int(Nw)
    H  = int(Nw/2)
    for Accx in np.r_[:3]:
        if f_alpha_est[-1,Accx,2*n]!=0:
            f_est = f_alpha_est[:,Accx,2*n]
            alpha_est = f_alpha_est[:,Accx,2*n+1]
            f_est_lin = f_est_linear(alpha_est,f_est,Nw,H)*fs
            if lfilek<2:
                f_02= f_0r[:f_est_lin.shape[0]]*183.135/181.438
                f_0mse[Accx,lfilek]=np.mean(np.abs(f_est_lin-f_02)**2)
            elif lfilek==2 or lfilek==3:
                f_est_lin = f_est_lin/31
                f_02= f_0[:f_est_lin.shape[0]]*5599.78/5691.86
                f_0mse[Accx,lfilek]=np.mean(np.abs(f_est_lin[int(fs2/2):int(-fs2/2)]-f_02[int(fs2/2):int(-fs2/2)])**2)
            elif lfilek>3:
                f_02= f_0[:f_est_lin.shape[0]]*183.114/183.543 # small adjustments usuall when dealing with real signals, the pearsons correlation index could be another metric
                f_0mse[Accx,lfilek]=np.mean(np.abs(f_est_lin[int(fs2/2):int(-fs2/2)]-f_02[int(fs2/2):int(-fs2/2)])**2)
