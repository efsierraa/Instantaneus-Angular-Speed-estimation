import os # change dir
Wdir= os.getcwd()
import sys 
sys.path.append(Wdir+'\functions')# add to path

import numpy as np
import scipy.io as sio # load save matlab files
import matplotlib.pyplot as plt # plotting
from matplotlib.gridspec import GridSpec # easy subplots
from functions.allfunctions import f_est_linear,iaslinearapproxv2
#%% Basreline snr test
mat_contents = sio.loadmat('Dataz/'+'Numerical_signal', mdict=None, appendmat=True)
for k, v in mat_contents.items():
    try:
        if v.size==sum(v.shape)-1:
            globals()[k]=v.flatten()
        else:
            globals()[k]=v
    except:
        globals()[k]=v
del k,v,mat_contents

Nw = 2**int(np.log2(.2*fs)) # window length
fmaxmin = np.r_[170,260]/fs
delta   = 3/fs 
K = np.r_[K]

savename  = 'tmpd15'
savename1 = 'D15_ias_numerical_baseline_white'
savename2 = 'D15_ias_linear_approx_numerical_white'
savename3 = 'D15_ias_numerical_baseline_pink'
savename4 = 'D15_ias_linear_approx_numerical_pink'

f_alpha_est = np.zeros([x_f_0.shape[0],snrdb.size*2])
chirp = 0
for k in np.r_[:snrdb.size]:
    tmp = iaslinearapproxv2(datanoisy_w[:,k],Nw,K,fmaxmin,delta,chirp,savename)
    if k==0:
        f_alpha_est = f_alpha_est[:tmp.shape[0],:]
    f_alpha_est[:,2*k:2*k+2] = tmp
        
result_dict = {'f_alpha_est': f_alpha_est,'fs':fs,\
                           'Nw':Nw, 'K':K}
sio.savemat('Dataz/'+savename1+'.mat',result_dict,do_compression=True)


f_alpha_est = np.zeros([x_f_0.shape[0],snrdb.size*2])
chirp = 1
for k in np.r_[:snrdb.size]:
    tmp = iaslinearapproxv2(datanoisy_w[:,k],Nw,K,fmaxmin,delta,chirp,savename)
    if k==0:
        f_alpha_est = f_alpha_est[:tmp.shape[0],:]
    f_alpha_est[:,2*k:2*k+2] = tmp
        
result_dict = {'f_alpha_est': f_alpha_est,'fs':fs,\
                           'Nw':Nw, 'K':K}
sio.savemat('Dataz/'+savename2+'.mat',result_dict,do_compression=True)



f_alpha_est = np.zeros([x_f_0.shape[0],snrdb.size*2])
chirp = 0
for k in np.r_[:snrdb.size]:
    tmp = iaslinearapproxv2(datanoisy_p[:,k],Nw,K,fmaxmin,delta,chirp,savename)
    if k==0:
        f_alpha_est = f_alpha_est[:tmp.shape[0],:]
    f_alpha_est[:,2*k:2*k+2] = tmp
        
result_dict = {'f_alpha_est': f_alpha_est,'fs':fs,\
                           'Nw':Nw, 'K':K}
sio.savemat('Dataz/'+savename3+'.mat',result_dict,do_compression=True)


f_alpha_est = np.zeros([x_f_0.shape[0],snrdb.size*2])
chirp = 1
for k in np.r_[:snrdb.size]:
    tmp = iaslinearapproxv2(datanoisy_p[:,k],Nw,K,fmaxmin,delta,chirp,savename)
    if k==0:
        f_alpha_est = f_alpha_est[:tmp.shape[0],:]
    f_alpha_est[:,2*k:2*k+2] = tmp
        
result_dict = {'f_alpha_est': f_alpha_est,'fs':fs,\
                           'Nw':Nw, 'K':K}
sio.savemat('Dataz/'+savename4+'.mat',result_dict,do_compression=True)

# %% ploting data
plt.rc('font', size=14) # default text size
savename1 = 'D15_ias_numerical_baseline_white'
savename2 = 'D15_ias_linear_approx_numerical_white'
savename3 = 'D15_ias_numerical_baseline_pink'
savename4 = 'D15_ias_linear_approx_numerical_pink'
lfile = [savename1,savename2,savename3,savename4]

for lfilek in np.r_[:4]:
    
    mat_contents = sio.loadmat('Dataz/'+'Numerical_signal', mdict=None, appendmat=True)
    for k, v in mat_contents.items():
        try:
            if v.size==sum(v.shape)-1:
                globals()[k]=v.flatten()
            else:
                globals()[k]=v
        except:
            globals()[k]=v
    del k,v,mat_contents
    
    mat_contents = sio.loadmat('Dataz/'+lfile[lfilek], mdict=None, appendmat=True)
    for k, v in mat_contents.items():
        try:
            if v.size==sum(v.shape)-1:
                globals()[k]=v.flatten()
            else:
                globals()[k]=v
        except:
            globals()[k]=v
    del k,v,mat_contents
    fmaxmin = np.r_[170,260]
    # -----------------------------------------------------------------------------
    Nw = 2*H # for this case Nw = 2*
    Nw = int(Nw)
    H = int(H)
    n = f_alpha_est.shape[1]/2;n = int(n); n = np.r_[:n]
    f_est = f_alpha_est[:,2*n]*fs
    alpha_est = f_alpha_est[:,2*n+1]*fs
    
    
    
    kest  = np.where(snrdb==-10)[0][0]# -7.5
    
    f_est_lin = np.zeros(((f_est.shape[0]-2)*H+2*H,f_est.shape[1]))
    for k in np.r_[kest:alpha_est.shape[1]]:
        f_est_lin[:,k] = f_est_linear(alpha_est[:,k],f_est[:,k],Nw,H)
        
    x_f_0 = x_f_0[:f_est_lin.shape[0],:]
    t = np.r_[:len(x_f_0[:,1])]/fs
    t2 = np.r_[:len(f_alpha_est)]/fs*H
    #--------------------------------------------
    plt.close('all')
    fig = plt.figure(constrained_layout=True)
    gs = GridSpec(5, 5, figure=fig);ax1 = fig.add_subplot(gs[0, 0:])
    ax2 = fig.add_subplot(gs[1:, 0:])
    
    ax1.plot(t2,alpha_est);ax2.plot(t2,f_est)
    ax1.set_xlim([t2[0],t2[-1]]);ax2.set_xlim([t2[0],t2[-1]])
    ax1.set_ylabel("Chirp rate");ax2.set_ylabel('Frequency [Hz]')
    ax2.set_xlabel('Time [sec]')
    ax1.legend([r'$\hat{\beta}_{1}[n]$']);ax2.legend([r'$\hat{\beta}_{2}[n]$'])
    plt.tight_layout()
    plt.show()
    if lfilek==0:
        plt.savefig(Wdir+'/Figures/exp02_awgn_parameters_all_baseline.pdf',bbox_inches='tight')
    elif lfilek==1:
        plt.savefig(Wdir+'/Figures/exp02_awgn_parameters_all_proposal.pdf',bbox_inches='tight')
    elif lfilek==2:
        plt.savefig(Wdir+'/Figures/exp02_pink_parameters_all_baseline.pdf',bbox_inches='tight')
    elif lfilek==3:
        plt.savefig(Wdir+'/Figures/exp02_pink_parameters_all_proposal.pdf',bbox_inches='tight')
    #--------------------------------------------
    plt.figure()
    plt.plot(t,f_est_lin)
    plt.plot(t,x_f_0[:,1])
    plt.axis((t[0],t[-1],fmaxmin[0],fmaxmin[1]))
    plt.xlabel('Time [sec]')
    plt.ylabel('Frequency [Hz]')
    plt.legend([r'Piecewise approx $\hat{f}_0[n]$','Theoretical IAS'])
    plt.tight_layout()
    plt.show()
    if lfilek==0:
        plt.savefig(Wdir+'/Figures/exp02_awgn_f_est_baseline.pdf',bbox_inches='tight')
    elif lfilek==1:
        plt.savefig(Wdir+'/Figures/exp02_awgn_f_est_proposal.pdf',bbox_inches='tight')
    elif lfilek==2:
        plt.savefig(Wdir+'/Figures/exp02_pink_f_est_baseline.pdf',bbox_inches='tight')
    elif lfilek==3:
        plt.savefig(Wdir+'/Figures/exp02_pink_f_est_proposal.pdf',bbox_inches='tight')