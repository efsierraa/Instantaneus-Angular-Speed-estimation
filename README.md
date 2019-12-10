# Instantaneus Angular Speed short-time least squares estimation

This repository has the functions used in the paper "Cyclo-non-stationary analysis for bearing fault identification based on instantaneous angular speed estimation" submitted to the conference https://survishno.sciencesconf.org/, for the Instantaneous Angular Speed Estimation. The database is available as supplementary material of the article, "Feedback on the Surveillance 8 challenge: Vibration-based diagnosis of a Safran aircraft engine", founded in https://doi.org/10.1016/j.ymssp.2017.01.037. If you find this material useful, please give a citation to the papers mentioned above.

The list of the functions:

  - signal_kharmdb_noise # numerical signal computation
  - noise_snr # add noise for a given signal to accomplish the desired signal-to-noise ratio
  - pink # pink noise generator using the Voss-McCartney algorithm. Taken from some place on the internet
  - cost_fun_1d # cost function for the optimization model assuming null qudratic term
  - cost_fun_2d # cost function given the two parameters for the linear approximation of the IAS for a given segment
  - vanderZ # Z vandermonde matrix for the model used in the cost function
  - cost_func_grid # returns several values of the cost function given a search grid (brute force to find the optimum) used as initialization
  - iaslinearapproxv2 # returns the quadratic term \alpha and w the cut frequency (constant term) for the piece-wise approwiamtion using the function f_est_linear
  - f_est_linear # piece-wise approximation given the computed parameters
  - tTacho_fsigLVA # converts the tachometer signal to the IAS, Matlab version made on the http://lva.insa-lyon.fr/en/

Functions translated to Python from the book Noise and Vibration Analysis Signal Analysis and Experimental Procedures by Anders Brandt

  - tTacho_fsig # Time indexes for angular resampling, and the Instantaneous Angular Profile (IAS), uses as input the tachometer signal
  - COT_intp # interpolates a vibration signal with given time indexes in tTacho_fsig
  - COT_intp2 # Interpolates the signal given an IAS profile, useful for numerical tests
