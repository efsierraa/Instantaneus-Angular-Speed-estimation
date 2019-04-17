import os
import sys
Wdir  = os.getcwd() 
sys.path.append(Wdir+'\functions')# add to path
import numpy as np
import scipy.io as sio # load save matlab files
from scipy.io import wavfile # read wav files
import scipy.signal as sg # signal pack
import matplotlib.pyplot as plt # plotting
from matplotlib.gridspec import GridSpec # easy subplots
from functions.func_sig_cost import iaslinearapproxv2,f_est_linear,tTacho_fsig,tTacho_fsigLVA

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
# %% loading wav file
lfile = (Wdir+'\Surveillance8 Contest')
data = np.zeros([10240000,3])
fs, data[:,0] = wavfile.read(lfile+'\Acc0.wav')
fs, data[:,1] = wavfile.read(lfile+'\Acc1.wav')
fs, data[:,2] = wavfile.read(lfile+'\Acc2.wav')
fs, tacho = wavfile.read(lfile+'\Tacho.wav')

#data = data[140*fs:160*fs,:]
#tacho = tacho[140*fs:160*fs]

#f_0,_,_ = tTacho_fsig(tacho,fs,PPR=44,isencod=1); del tacho
#f_0 = f_0/60
_,f_0=tTacho_fsigLVA(tacho,fs,TPT=44)
f_0 = f_0*62/61
#%% ---------------------------------------------------------------------------
# No downsampling

fmaxmin = np.r_[170,260]/fs#fmaxmin = np.r_[170,260]/fs
Nw = 2**np.ceil(np.log2(1/fmaxmin[0]*2))
Nw = int(Nw) # 1024 default important parameterS and K
H  = Nw/2
# --------------dfac order best identifyed but initial visual exam is requiered
dfac = 62/2 # frequence d'engranement 62
fmaxmin = fmaxmin*dfac
delta   = 5/fs*dfac
K = fs/2/max(f_0*dfac); K = np.r_[int(K)] # amount of orders

# No downsampling fliplr ------------------------------------------------------
savename = 'D16_ias_baseline_acc2_nodown_fliplr_dfacf0'
f_alpha_est = np.zeros(data.shape+(K.size*2,))
L = np.r_[2]
for l in L:
    tmp = iaslinearapproxv2(np.flipud(data[:,l]),Nw,K,fmaxmin,delta,0,'tmpaccxv3')
    if l==L[0]:
        f_alpha_est = f_alpha_est[:tmp.shape[0],:,:]
    f_alpha_est[:,l,:] = tmp

result_dict = {'f_alpha_est': f_alpha_est,'fs':fs,\
                           'Nw':Nw, 'K':K}
sio.savemat('Dataz/'+savename+'.mat',result_dict,do_compression=True)


savename = 'D16_ias_linear_acc2_nodown_fliplr_dfacf0'
f_alpha_est = np.zeros(data.shape+(K.size*2,))
L = np.r_[2]
for l in L:
    tmp = iaslinearapproxv2(np.flipud(data[:,l]),Nw,K,fmaxmin,delta,1,'tmpaccxv3')
    if l==L[0]:
        f_alpha_est = f_alpha_est[:tmp.shape[0],:,:]
    f_alpha_est[:,l,:] = tmp

result_dict = {'f_alpha_est': f_alpha_est,'fs':fs,\
                           'Nw':Nw, 'K':K}
sio.savemat('Dataz/'+savename+'.mat',result_dict,do_compression=True)

# ----------------------------------------------------------------------------
savename = 'D16_ias_baseline_acc2_nodown_dfacf0'
f_alpha_est = np.zeros(data.shape+(K.size*2,))
L = np.r_[2]
for l in L:
    tmp = iaslinearapproxv2(data[:,l],Nw,K,fmaxmin,delta,0,'tmpaccxv3')
    if l==L[0]:
        f_alpha_est = f_alpha_est[:tmp.shape[0],:,:]
    f_alpha_est[:,l,:] = tmp

result_dict = {'f_alpha_est': f_alpha_est,'fs':fs,\
                           'Nw':Nw, 'K':K}
sio.savemat('Dataz/'+savename+'.mat',result_dict,do_compression=True)


savename = 'D16_ias_linear_acc2_nodown_dfacf0'
f_alpha_est = np.zeros(data.shape+(K.size*2,))
L = np.r_[2]
for l in L:
    tmp = iaslinearapproxv2(data[:,l],Nw,K,fmaxmin,delta,1,'tmpaccxv3')
    if l==L[0]:
        f_alpha_est = f_alpha_est[:tmp.shape[0],:,:]
    f_alpha_est[:,l,:] = tmp

result_dict = {'f_alpha_est': f_alpha_est,'fs':fs,\
                           'Nw':Nw, 'K':K}
sio.savemat('Dataz/'+savename+'.mat',result_dict,do_compression=True)
#%% No downsampling ACC1
fmaxmin = np.r_[170,260]/fs#fmaxmin = np.r_[170,260]/fs
Nw = 2**np.ceil(np.log2(1/fmaxmin[0]*2))
Nw = int(Nw) # 1024 default important parameterS and K
H  = Nw/2
# --------------dfac order best identifyed but initial visual exam is requiered
dfac = 1 # premier harmonic
fmaxmin = fmaxmin*dfac
delta   = 5/fs*dfac
K = fs/2/max(f_0*dfac); K = np.r_[int(K)] # amount of orders

savename = 'D16_ias_baseline_acc1_nodown_dfacf0'
f_alpha_est = np.zeros(data.shape+(K.size*2,))
L = np.r_[1]
for l in L:
    tmp = iaslinearapproxv2(data[:,l],Nw,K,fmaxmin,delta,0,'tmpaccxv3')
    if l==L[0]:
        f_alpha_est = f_alpha_est[:tmp.shape[0],:,:]
    f_alpha_est[:,l,:] = tmp

result_dict = {'f_alpha_est': f_alpha_est,'fs':fs,\
                           'Nw':Nw, 'K':K}
sio.savemat('Dataz/'+savename+'.mat',result_dict,do_compression=True)

#%
savename = 'D16_ias_linear_acc1_nodown_dfacf0'
f_alpha_est = np.zeros(data.shape+(K.size*2,))
L = np.r_[1]
for l in L:
    tmp = iaslinearapproxv2(data[:,l],Nw,K,fmaxmin,delta,1,'tmpaccxv3')
    if l==L[0]:
        f_alpha_est = f_alpha_est[:tmp.shape[0],:,:]
    f_alpha_est[:,l,:] = tmp

result_dict = {'f_alpha_est': f_alpha_est,'fs':fs,\
                           'Nw':Nw, 'K':K}
sio.savemat('Dataz/'+savename+'.mat',result_dict,do_compression=True)
#%% No downsampling ACC2
fmaxmin = np.r_[170,260]/fs#fmaxmin = np.r_[170,260]/fs
Nw = 2**np.ceil(np.log2(1/fmaxmin[0]*2))
Nw = int(Nw) # 1024 default important parameterS and K
H  = Nw/2
# --------------dfac order best identifyed but initial visual exam is requiered
dfac = 1 # premier harmonic
fmaxmin = fmaxmin*dfac
delta   = 5/fs*dfac
K = fs/2/max(f_0*dfac); K = np.r_[int(K)] # amount of orders

savename = 'D16_ias_baseline_acc2_nodown_allorders'
f_alpha_est = np.zeros(data.shape+(K.size*2,))
L = np.r_[2]
for l in L:
    tmp = iaslinearapproxv2(data[:,l],Nw,K,fmaxmin,delta,0,'tmpaccxv3')
    if l==L[0]:
        f_alpha_est = f_alpha_est[:tmp.shape[0],:,:]
    f_alpha_est[:,l,:] = tmp

result_dict = {'f_alpha_est': f_alpha_est,'fs':fs,\
                           'Nw':Nw, 'K':K,'dfac':dfac}
sio.savemat('Dataz/'+savename+'.mat',result_dict,do_compression=True)

#%
savename = 'D16_ias_linear_acc1_nodown_allorders'
f_alpha_est = np.zeros(data.shape+(K.size*2,))
L = np.r_[2]
for l in L:
    tmp = iaslinearapproxv2(data[:,l],Nw,K,fmaxmin,delta,1,'tmpaccxv3')
    if l==L[0]:
        f_alpha_est = f_alpha_est[:tmp.shape[0],:,:]
    f_alpha_est[:,l,:] = tmp

result_dict = {'f_alpha_est': f_alpha_est,'fs':fs,\
                           'Nw':Nw, 'K':K,'dfac':dfac}
sio.savemat('Dataz/'+savename+'.mat',result_dict,do_compression=True)