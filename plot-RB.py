import os # change dir
Wdir= os.getcwd()
import sys 
sys.path.append(Wdir+'\functions')# add to path

import numpy as np
import scipy.io as sio # load save matlab files
from scipy.io import wavfile # read wav files
import scipy.signal as sg # signal pack
import matplotlib.pyplot as plt # plotting
from matplotlib.gridspec import GridSpec # easy subplots
from functions.func_sig_cost import f_est_linear,tTacho_fsig,tTacho_fsigLVA

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
# plt.rcdefaults()

# % ------------------------ loading wav file ---------------------------------
lfile = (Wdir+'\Surveillance8 Contest')
data = np.zeros([10240000,3])
fs, data[:,0] = wavfile.read(lfile+'\Acc0.wav')
fs, data[:,1] = wavfile.read(lfile+'\Acc1.wav')
fs, data[:,2] = wavfile.read(lfile+'\Acc2.wav')
fs, tacho = wavfile.read(lfile+'\Tacho.wav')
#data = data[140*fs:160*fs,:]
#tacho = tacho[140*fs:160*fs]

#f_0,_,_ = tTacho_fsig(tacho,fs,PPR=44,isencod=1); del tacho
#f_0 = f_0/60*62/61
t,f_0 = tTacho_fsigLVA(tacho,fs,TPT=44)
f_0 = f_0*62/61 # visual examining
dfac = 2**4 # decimate factor
fs2    = fs/dfac # new sampling frequency

# removing 0.5 sec of the signal to avoid interp errors
f_0r = sg.decimate(f_0,dfac)
# downsampling
data_r = np.zeros((int(data.shape[0]/dfac),data.shape[1]))
for k in np.r_[:data.shape[1]]:
    data_r[:,k] = sg.decimate(data[:,k],dfac)# for computational reasons

# cupping one second of the signal to avoid interpolation errors
f_0r = f_0r[int(fs2/2):int(-fs2/2)]
data_r = data_r[int(fs2/2):int(-fs2/2),:]
data = data[int(fs/2):int(-fs/2)]

t    = np.arange(0,data.shape[0])
t    = t/fs
t2    = np.arange(0,data_r.shape[0])
t2    = t2/fs2
#%% plotting Real Db
plt.close('all')
NFFT = 2**8  # the length of the windowing segments
for k in np.r_[1:3]:
    fig = plt.figure(constrained_layout=True,figsize=(10,6))
    gs = GridSpec(5, 5, figure=fig)
    ax1 = fig.add_subplot(gs[0, 0:])
    ax2 = fig.add_subplot(gs[1:, 0:])
    
    ax1.plot(t,data[:,k]/np.std(data[:,k]))
    ax1.set_xlim([t2[0],t2[-1]])
    
    # spectrogram
    Pxx, freqs, bins, im = ax2.specgram(data[:,k], NFFT=NFFT, Fs=fs, noverlap=NFFT/2,cmap='viridis')
    ax2.plot(t2,62*f_0r,'r')
    ax2.set_xlim([t2[0],t2[-1]])
    ax2.set_ylim([0,fs/2])
    ax2.set_xlabel('Time [sec]')
    ax2.set_ylabel('Frequency [Hz]')
    
    plt.show()
    ax2.legend(['$f_{62}[n]$'])
    plt.savefig('Figures\exp03_fig_spec_engine_Acc'+str(k)+'_original.pdf',bbox_inches='tight')
    
for k in np.r_[1:3]:
    fig = plt.figure(constrained_layout=True,figsize=(10,6))
    gs = GridSpec(5, 5, figure=fig)
    ax1 = fig.add_subplot(gs[0, 0:])
    ax2 = fig.add_subplot(gs[1:, 0:])
    
    ax1.plot(t,data[:,k]/np.std(data[:,k]))
    ax1.set_xlim([t2[0],t2[-1]])
    
    # spectrogram
    Pxx, freqs, bins, im = ax2.specgram(data_r[:,k], NFFT=NFFT, Fs=fs2, noverlap=NFFT/2,cmap='viridis')
    ax2.plot(t2,f_0r,t2,f_0r*3)
    ax2.set_xlim([t2[0],t2[-1]])
    ax2.set_ylim([0,fs2/2])
    ax2.set_xlabel('Time [sec]')
    ax2.set_ylabel('Frequency [Hz]')
    
    plt.show()
    ax2.legend(['$f_{1}[n]$','$f_{3}[n]$'])
    plt.savefig('Figures\exp03_fig_spec_engine_Acc'+str(k)+'_downsampled_2pow4.pdf',bbox_inches='tight')
#%% ploting results
lfile = ['D16_ias_linear_approx_accx','D16_ias_baseline_accx']
sfile = ['_linear_app','_baseline']
snrdb = np.r_[-20:21]/2
fmaxmin = np.r_[170,260]

loading_matfile('Dataz/'+lfile[0]); # same parameters all tests
#n  = f_alpha_est.shape[1]/2;n = int(n); n = np.r_[:n]
n = np.r_[0]
kest  = np.where(snrdb==-10)[0][0]# -7.5

for k1 in np.r_[:2]:
    loading_matfile('Dataz/'+lfile[k1]);
    H = Nw/2
    H = int(H)
    Nw = int(Nw)
    t2 = np.r_[:len(f_alpha_est)]/fs*H

    for k2 in np.r_[:3]:
        f_est = f_alpha_est[:,k2,2*n]*fs
        alpha_est = f_alpha_est[:,k2,2*n+1]*fs
        
        f_est_lin = np.zeros(((f_est.shape[0]-2)*H+2*H,f_est.shape[1]))
        for k in np.r_[kest:alpha_est.shape[1]]:
            f_est_lin[:,k] = f_est_linear(alpha_est[:,k],f_est[:,k],Nw,H)
        
        t = np.r_[:len(f_est_lin)]/fs
        
        fig = plt.figure(constrained_layout=True,figsize=(10,6))
        gs = GridSpec(5, 5, figure=fig);ax1 = fig.add_subplot(gs[0, 0:])
        ax2 = fig.add_subplot(gs[1:, 0:])
        
        ax1.plot(t2,alpha_est[:,kest:]);ax2.plot(t2,f_est[:,kest:])
        ax1.grid(True)
        ax1.set_xlim([t2[0],t2[-1]]);ax2.set_xlim([t2[0],t2[-1]])
        
        ax1.set_ylabel("Chirp rate");ax2.set_ylabel('Frequency [Hz]')
        ax2.set_xlabel('Time [sec]')
        ax2.grid(True)
        ax1.legend([r'$\hat{\beta}_{1}[n]$']);ax2.legend([r'$\hat{\beta}_{2}[n]$'])
        plt.show()
        plt.savefig('Figures\exp03_results_Acc'+str(k2)+sfile[k1]+'.pdf',bbox_inches='tight')
        
        if k1==1:
            fig = plt.figure(constrained_layout=True,figsize=(10,6))
            t = t[:f_est_lin.shape[0]]
            f_02 = f_0r[:f_est_lin.shape[0]]*183.135/181.438
            plt.plot(t,f_est_lin)
            plt.grid(True)
            #plt.plot(t,f_02)
            plt.plot(t,f_02)
            plt.axis((t[0],t[-1],fmaxmin[0],fmaxmin[1]))
            plt.xlabel('Time [sec]')
            plt.ylabel('Frequency [Hz]')
            plt.legend([r'Piecewise approx $\hat{f}_0[n]$','IAS from tachometer'])
            plt.show()
            plt.savefig('Figures\exp03_results_Acc'+str(k2)+sfile[k1]+\
                        'f_est'+'.pdf',bbox_inches='tight')
#%% plotting ACC2 flipud
plt.close('all')
lfile = ['D16_ias_baseline_acc2_nodown_fliplr_dfacf0','D16_ias_linear_acc2_nodown_fliplr_dfacf0',\
         'D16_ias_baseline_acc2_nodown_dfacf0','D16_ias_linear_acc2_nodown_dfacf0',\
         'D16_ias_baseline_acc1_nodown_dfacf0','D16_ias_linear_acc1_nodown_dfacf0']
sfile = ['_baseline_fliplr','_linear_app_fliplr','_baseline','_linear_app']

dfac = 62/2
fmaxmin = np.r_[170,260]*dfac

dfac = 1
n =np.r_[0]
for k1 in np.r_[:2]+2:
    if k1==2 and k1==3:
        L=2
    elif k1>3:
        L=1
    loading_matfile('Dataz/'+lfile[k1])
    Nw = int(Nw)
    H  = int(Nw/2)
    if k1<2:
        f_alpha_est = np.flipud(f_alpha_est)
    f_est = f_alpha_est[:,L,2*n]*fs/dfac
    alpha_est = f_alpha_est[:,L,2*n+1]*fs/dfac
    
    f_est_lin = np.zeros(((f_est.shape[0]-2)*H+2*H,f_est.shape[1]))
    for k in np.r_[:alpha_est.shape[1]]:
        f_est_lin[:,k] = f_est_linear(alpha_est[:,k],f_est[:,k],Nw,H)
    
    t = np.r_[:len(f_est_lin)]/fs
    t2 = np.r_[:len(f_alpha_est)]/fs*H
    
    fig = plt.figure(constrained_layout=True,figsize=(10,6))
    gs = GridSpec(5, 5, figure=fig);ax1 = fig.add_subplot(gs[0, 0:])
    ax2 = fig.add_subplot(gs[1:, 0:])
    
    ax1.plot(t2,alpha_est);ax2.plot(t2,f_est)
    ax1.grid(True)
    ax2.grid(True)
    ax1.set_xlim([t2[0],t2[-1]]);ax2.set_xlim([t2[0],t2[-1]])
    ax1.set_ylabel("Chirp rate");ax2.set_ylabel('Frequency [Hz]')
    ax2.set_xlabel('Time [sec]')
    ax1.legend([r'$\hat{\beta}_{1}[n]$']);ax2.legend([r'$\hat{\beta}_{2}[n]$'])
    plt.show()
    plt.savefig('Figures\exp04_results_Acc'+str(L)+sfile[k1]+'.pdf',bbox_inches='tight')
    
    fig = plt.figure(constrained_layout=True,figsize=(10,6))
    f_02 = f_0[:f_est_lin.shape[0]]*31*5599.78/5691.86
    plt.plot(t,f_est_lin)
    plt.plot(t,f_02)
    plt.grid(True)
    plt.axis((t2[0],t2[-1],fmaxmin[0],fmaxmin[1]))
    plt.xlabel('Time [sec]')
    plt.ylabel('Frequency [Hz]')
    plt.legend([r'Piecewise approx $\hat{f}_{31}[n]$','IAS from tachometer'])
    plt.show()
    plt.savefig('Figures\exp04_results_Acc'+str(L)+sfile[k1]+\
                'f_est'+'.pdf',bbox_inches='tight')