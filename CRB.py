import numpy as np
import multiprocessing
from joblib import Parallel, delayed
from scipy.optimize import minimize
from scipy.ndimage import gaussian_filter

def CRB(PSF, SBR0, pxsize, N, detNoise = [], center = []):
    """
    Calculate Cramer-Rao Bound
    (S26, 10.1126/science.aak9913)
    ===========================================================================
    Input       Meaning
    ---------------------------------------------------------------------------
    PSF         np.array(K, Ny, Nx) with experimental or simulated PSFs
    SBR0        Signal to background ratio at the position specified by center
    pxsize      number or np.array(PxsX, PxsY) [nm]
    N           number or np.array(Ny, Nx). Total number of photons per position
    detNoise    np.array(K) with dark noise countings per pixel. Default is 
                flat noise
    center      np.array([row,col]) of the reference pixel for the SBR. Default
                is the center of the ROI
    ===========================================================================
    Output      Meaning
    ---------------------------------------------------------------------------
    sigma_CRB   np.array(Ny, Nx) with the CRB in 2D
    sigmax      np.array(Ny, Nx) with the uncertainty in the x direction
    sigmay      np.array(Ny, Nx) with the uncertainty in the y direction
    ===========================================================================
    """
    
    psfsize = np.shape(PSF)
    K = psfsize[0]
    sizex = psfsize[2]
    sizey = psfsize[1]
    d = 2
    
    normPSF = np.sum(PSF, axis = 0)
    
    #check optional inputs
    if np.shape(center)[0] == 0:
        rowc = int(sizey/2)
        colc = int(sizex/2)
    else:
        rowc = center[0]
        colc = center[1]
        
    if np.shape(detNoise)[0] == 0:
        detNoise = np.ones(K)
    
    # size of the (x,y) grid
    if np.isscalar(pxsize):
        dx = pxsize
        dy = pxsize
    else:
        dx = pxsize[0]
        dy = pxsize[1]
    
    # define different arrays needed to compute CR
    # p = np.zeros((K, sizey, sizex), dtype=np.float32)
    #lambd = np.zeros((K, sizey, sizex), dtype=np.float32)
    #dpdx = np.zeros((K, sizey, sizex), dtype=np.float32)
    #dpdy = np.zeros((K, sizey, sizex), dtype=np.float32)
    #A = np.zeros((K, sizey, sizex), dtype=np.float32)
    #B = np.zeros((K, sizey, sizex), dtype=np.float32)
    #C = np.zeros((K, sizey, sizex), dtype=np.float32)
    #D = np.zeros((K, sizey, sizex), dtype=np.float32)
    
    p, lambd, dpdx, dpdy, A, B, C, D = (np.zeros((K, sizey, sizex)) for i in range(8))
    
    # calculate the (different) SBR in the whole ROI
    if type(SBR0) == np.ndarray:
        SBR = SBR0
    else:
        SBR = SBR0 * normPSF / normPSF[rowc,colc]
    
    # probability arrays
    for i in np.arange(K):
        p[i,:,:] = (SBR/(SBR + 1)) * PSF[i,:,:]/normPSF + (1/(SBR + 1)) * detNoise[i]/np.sum(detNoise)
        
    # probabilities in each (x,y)
    for i in np.arange(K):
        # gradient of ps in each (x,y)
        dpdy[i, :, :], dpdx[i, :, :] = np.gradient(p[i, :, :], -dy, dx)
       
        # terms needed to compute CR bound in each (x,y)
        A[i, :, :] = (1/p[i, :, :]) * dpdx[i, :, :]**2
        B[i, :, :] = (1/p[i, :, :]) * dpdy[i, :, :]**2
        C[i, :, :] = (1/p[i, :, :]) *(dpdx[i, :, :] * dpdy[i, :, :])
        D[i, :, :] = (1/p[i, :, :]) * (dpdx[i, :, :]**2 + dpdy[i, :, :]**2)

    # sigma Cramer-Rao numerator and denominator
    E = np.sum(D, axis=0)
    F = (np.sum(A, axis=0) * np.sum(B, axis=0)) - np.sum(C, axis=0)**2 # matrix determinant
    
    # make sure CRB is not 0 or NaN
    E[E <= 0] = np.min(E[E>0])
    F[F <= 0] = np.min(F[F>0])
    
    sigma_CRB = np.sqrt(1/(d*N))*np.sqrt(E/F)
    sigmay = np.sqrt(1/(d*N)) * np.sqrt(np.sum(A, axis=0) / F)
    sigmax = np.sqrt(1/(d*N)) * np.sqrt(np.sum(B, axis=0) / F)
    
    return sigma_CRB, sigmax, sigmay

def CRB_Ns(PSF, SBR0, pxsize, N, detNoise = [], center = [], returnN = False):
    """
    Calculate Cramer-Rao Bound assuming a different number of signal photons
    depending on the (x,y) position and a constant number of background counts.
    Can be used in experiments in which the frame rate is constant
        --> number of signal photons depends on the excitation intensity at
        position (x,y)
        --> the SBR depends on the (x,y) position as well
    Note: this is an approximation of the 'real' CRB, but much faster than the
    exact algorithm
    ===========================================================================
    Input       Meaning
    ---------------------------------------------------------------------------
    PSF         np.array(K, Ny, Nx) with experimental or simulated PSFs
    SBR0        Signal to background ratio at the position specified by center
    pxsize      number or np.array(PxsX, PxsY) [nm]
    N           number. Total number of photons at the reference position
    detNoise    np.array(K) with dark counts per pixel. Default is flat noise
    center      np.array([row,col]) of the reference pixel for the SBR. Default
                    is the center of the ROI
    returnN     Boolean. If True, return the np.array(Ny, Nx) with the number
                of photons
    ===========================================================================
    Output      Meaning
    ---------------------------------------------------------------------------
    sigma_CRB   np.array(Ny, Nx) with the CRB in 2D
    sigmax      np.array(Ny, Nx) with the uncertainty in the x direction
    sigmay      np.array(Ny, Nx) with the uncertainty in the y direction
    ===========================================================================
    """
    psfsize = np.shape(PSF)
    K = psfsize[0]
    sizex = psfsize[2]
    sizey = psfsize[1]
    
    normPSF = np.sum(PSF, axis = 0)
    
    ##check optional inputs
    if np.shape(center)[0] == 0:
        rowc = int(sizey/2)
        colc = int(sizex/2)
    else:
        rowc = center[0]
        colc = center[1]
        
    if np.shape(detNoise)[0] == 0:
        detNoise = np.ones(K)
    
    # calculate the (different) number of photons in the whole ROI
    if type(N) == np.ndarray:
        N_array = N
    else:
        N_array = N * normPSF / normPSF[rowc,colc] # correct for excitation and detection efficiency
    
    sigma_CRB, sigmax, sigmay = CRB(PSF, SBR0, pxsize, N_array, detNoise, np.array([rowc,colc]))
    
    if returnN:
        return sigma_CRB, sigmax, sigmay, N_array
    
    return sigma_CRB, sigmax, sigmay

def CRB_Loc(PSF, SBR0, pxsize, N, xloc, yloc, detNoise = [], center = []):
    """
    Calculate Cramer-Rao Bound for a series of points in space specified by 
    xloc and yloc, assuming a different number of signal photons depending on 
    the (x,y) position and a constant number of background counts.
    Can be used as an estimator for the MLE uncertainty
    ===========================================================================
    Input       Meaning
    ---------------------------------------------------------------------------
    PSF         np.array(K, Ny, Nx) with experimental or simulated PSFs
    SBR0        Signal to background ratio at the position specified by center
    pxsize      number or np.array(PxsX, PxsY) [nm]
    N           np.array(Nloc) with photon counts for each localization
    xMLE        np.array(Nloc). Estimation of the x position [nm]
    yMLE        np.array(Nloc). Estimation of the y position [nm]
    detNoise    np.array(K) with dark counts per pixel. Default is flat noise
    center      np.array([row,col]) of the reference pixel for the SBR. Default
                    is the center of the ROI
    ===========================================================================
    Output      Meaning
    ---------------------------------------------------------------------------
    sigma_loc   np.array(Nloc) with the CRB for the list of localizations
    sigma_locx  np.array(Nloc) with the uncertainty in the x direction
    sigma_locy  np.array(Nloc) with the uncertainty in the y direction
    ===========================================================================
    """
    psfsize = np.shape(PSF)
    K = psfsize[0]
    sizex = psfsize[2]
    sizey = psfsize[1]
    
    #check optional inputs
    if np.shape(center)[0] == 0:
        rowc = int(sizey/2)
        colc = int(sizex/2)
    else:
        rowc = center[0]
        colc = center[1]
        
    if np.shape(detNoise)[0] == 0:
        detNoise = np.ones(K)  
        
    if np.shape(pxsize)[0] == 0:
        psx = 1
        psy = 1
    else:
        psx = pxsize[0]
        psy = pxsize[1]
        
    sigma_loc, sigma_locx, sigma_locy = (np.zeros(N.shape) for i in range(3))
    
    # calculate the CRB for a fiduciary number of photons of 100
    Nph = 100
    
    sigma_CRB, sigmax, sigmay = CRB(PSF, SBR0, pxsize, Nph, detNoise, np.array([rowc,colc]))
    
    # transform localizations in pixel coordinates
    jj = np.round(xloc/psx + sizex/2).astype(int)
    ii = np.round(yloc/psy + sizey/2).astype(int)
    
    #pick the coordinates of interest and rescale the uncertainty
    sigma_loc = sigma_CRB[ii,jj]*np.sqrt(Nph/N)
    sigma_locx = sigmax[ii,jj]*np.sqrt(Nph/N)
    sigma_locy = sigmay[ii,jj]*np.sqrt(Nph/N)
    
    return sigma_loc, sigma_locx, sigma_locy
    

def loclikelihood(PSF, n, SBR0, detNoise = [], center = [], SBRvar = True):
    """
    Calculate likelihood function of the localization
        Based on tools_simulation from
        Luciano Masullo, Lucía López and Lars Richter
    ===========================================================================
    Input       Meaning
    ---------------------------------------------------------------------------
    PSF         np.array(K, Ny, Nx) with K the number of PSFs
    n           [K elements] list with the number of photons collected for each PSF
    SBR0        Signal to background ratio at the position specified by center
    detNoise    np.array(K) with dark noise countings per pixel. Default is 
                flat noise
    center      np.array([row,col]) of the reference pixel for the SBR. Default
                is the center of the ROI
    ===========================================================================
    Output      Meaning
    ---------------------------------------------------------------------------
    likelihood  np.array(Ny, Nx) with the log likelihood function for the
                localization
    ===========================================================================
    """
    psfsize = np.shape(PSF)
    K = psfsize[0]
    sizex = psfsize[2]
    sizey = psfsize[1]
    
    normPSF = np.sum(PSF, axis = 0)
    
    #check optional inputs
    if np.shape(center)[0] == 0:
        rowc = int(sizey/2)
        colc = int(sizex/2)
    else:
        rowc = center[0]
        colc = center[1]
        
    if np.shape(detNoise)[0] == 0:
        detNoise = np.ones(K)
    
    # calculate the (different) SBR in the whole ROI
    if SBRvar:
        SBR = SBR0 * normPSF / normPSF[rowc,colc]
    else:
        SBR = SBR0
    
    # probability vector 
    p = np.zeros((K, sizey, sizex))
    for i in np.arange(K):
        p[i,:,:] = (SBR/(SBR + 1)) * PSF[i,:,:]/normPSF + (1/(SBR + 1)) * detNoise[i]/np.sum(detNoise)
        
    # log-likelihood function
    l_aux = np.zeros((K, sizey, sizex))
    for i in np.arange(K):
        l_aux[i, :, :] = n[i] * np.log(p[i, : , :])
        
    likelihood = np.sum(l_aux, axis = 0)
    
    likelihood -= np.max(likelihood)
    likelihood = np.exp(likelihood)
    #NOTE: may produce an error if likelihood is so small the sum is 0
    likelihood /= np.sum(likelihood)
    
    return likelihood

def loclikelihood2Part(PSF, n, SBR, Nf=2, x0=[256, 256, 260, 260]):
    """
    Calculate likelihood function of the localization of multiple fluorophores
    simultaneously
    ===========================================================================
    Input       Meaning
    ---------------------------------------------------------------------------
    PSF         np.array(K, Ny, Nx) with K the number of PSFs
    n           [K elements] list with the number of photons collected for each PSF
    SBR         Overall signal to background ratio
    Nf          Number of active fluorophores
    detNoise    np.array(K) with dark noise countings per pixel. Default is 
                flat noise
    center      np.array([row,col]) of the reference pixel for the SBR. Default
                is the center of the ROI
    ===========================================================================
    Output      Meaning
    ---------------------------------------------------------------------------
    likelihood  np.array(Ny, Nx) with the log likelihood function for the
                localization
    ===========================================================================
    """
    
    res = minimize(likelihoodForOptimization, x0, args=(PSF, SBR, n))
    return res
    

def likelihoodForOptimization(x, PSF, SBR, n):
    """
    Calculate likelihood of the localization of multiple active fluorophores
    simultaneously. The number of fluorophores can be anything.
    Used to find the most likely position of the fluorophores in an
    optimization approach.    
    ===========================================================================
    Input       Meaning
    ---------------------------------------------------------------------------
    x           np.array([y0, x0, y1, x1, ..., yp, xp]) with the 2D coordinates
                    of the p active fluorophores
    PSF         np.array(K, Ny, Nx) with K the number of PSFs
    SBR         Overall signal to background ratio
    n           [K elements] list with the number of photons collected for each PSF
    ===========================================================================
    Output      Meaning
    ---------------------------------------------------------------------------
    likelihood  np.array(Ny, Nx) with the log likelihood function for the
                localization
    ===========================================================================
    """
    normPSF = np.sum(PSF, axis = 0)
    
    Ns = SBR / (SBR + 1) # relative number of signal counts
    Nb = 1 / (SBR + 1) # relative number of noise counts
    
    psfsize = np.shape(PSF)
    K = psfsize[0]
    
    # fluorophore positions
    Nf = int(len(x) / 2)
    yc = []
    xc = []
    for i in range(Nf):
        yc.append(int(x[2*i]))
        xc.append(int(x[2*i+1]))
    
    # expected number of counts for p particles at position x0, x1,...x(p-1)
    p = np.zeros(K)
    
    Normfactor = 0
    for j in range(Nf):
        Normfactor += normPSF[yc[j], xc[j]]
    
    for i in np.arange(K):
        for j in range(Nf):
            p[i] += Ns * PSF[i,yc[j],xc[j]] / Normfactor
        p[i] += Nb / K
        
    # log-likelihood value
    l_aux = np.zeros(K)
    for i in np.arange(K):
        l_aux[i] = n[i] * np.log(p[i])
    likelihood = -np.sum(l_aux)
    
    return likelihood

def MLE(PSF, stack, SBR0, detNoise = [], center = [], pxsize = [], smooth = False, s = [], mode = 'nearest'):
    """
    Calculate the maximum likelihood estimation of the localization of the 
    emitter using variable SBR and dark counts
    ===========================================================================
    Input       Meaning
    ---------------------------------------------------------------------------
    PSF         np.array(K, Ny, Nx) with K the number of PSFs
    stack       np.array(N, sqrt(K), sqrt(K)) or np.array(N, K).
                Sequence of N arrays or frames containing the number of photons
                collected per pixel for each acquistion
    SBR0        Signal to background ratio at the position specified by center
    detNoise    np.array(K) with dark noise countings per pixel. Default is 
                flat noise
    center      np.array([row,col]) of the reference pixel for the SBR. Default
                is the center of the ROI
    pxsize      np.array([psx,psy]) pixel size of the PSF
    smooth      bool for activating the gaussian smoothing of the likelihood
    s           np.array([sx,sy]) containing the sigma of the gaussian in
                metric units
    mode        mode for ndimage gaussian_filter
    ===========================================================================
    Output      Meaning
    ---------------------------------------------------------------------------
    xMLE        np.array(N). Maximum likelihood estimation of the x position
    yMLE        np.array(N). Maximum likelihood estimation of the y position
    ===========================================================================
    """
    psfsize = np.shape(PSF)
    K = psfsize[0]
    sizex = psfsize[2]
    sizey = psfsize[1]
    
    normPSF = np.sum(PSF, axis = 0)
    
    #eventually reshape input
    if stack.shape[1] == K:
        #stack = np.array(N,K)
        n = stack
    else:
        #stack = np.array(N,sqrt(k),sqrt(K))
        n = np.reshape(stack,[-1,K])
    
    #check optional inputs
    if np.shape(center)[0] == 0:
        rowc = int(sizey/2)
        colc = int(sizex/2)
    else:
        rowc = center[0]
        colc = center[1]
        
    if np.shape(detNoise)[0] == 0:
        detNoise = np.ones(K)
        
    if np.shape(pxsize)[0] == 0:
        psx = 1
        psy = 1
    else:
        psx = pxsize[0]
        psy = pxsize[1]
    
    # calculate the (different) SBR in the whole ROI
    SBR = SBR0 * normPSF / normPSF[rowc,colc]
    
    # probability vector 
    p = np.zeros((K, sizey, sizex))
    for i in np.arange(K):
        p[i,:,:] = (SBR/(SBR + 1)) * PSF[i,:,:]/normPSF + (1/(SBR + 1)) * detNoise[i]/np.sum(detNoise)
    
    xMLE = np.zeros(n.shape[0])
    yMLE = np.zeros(n.shape[0])
    
    for h in range(n.shape[0]):
        if h%1000 == 0: print(str(h)+'\\'+str(n.shape[0]))
        l_aux = np.expand_dims(n[h,:],axis=(1,2)) * np.log(p)
        
        likelihood = np.sum(l_aux, axis = 0)
        
        if smooth:
            likelihood = gaussian_filter(likelihood,sigma=s[::-1],mode=mode)
        
        ii, jj = np.unravel_index(np.argmax(likelihood),likelihood.shape)
        xMLE[h] = jj - sizex/2
        yMLE[h] = ii - sizey/2
     
    xMLE *= psx
    yMLE *= psy
    
    return xMLE, yMLE

def MLE3D(PSF, stack, SBR0, detNoise = [], center = [], pxsize = [], smooth = False, s = [], mode='nearest'):
    """
    Calculate the maximum likelihood estimation of the localization of the 
    emitter using variable SBR and dark counts in 3D
    ===========================================================================
    Input       Meaning
    ---------------------------------------------------------------------------
    PSF         np.array(K, Nz, Ny, Nx) with K the number of PSFs
    stack       np.array(N, sqrt(K), sqrt(K)) or np.array(N, K).
                Sequence of N arrays or frames containing the number of photons
                collected per pixel for each acquistion
    SBR0        Signal to background ratio at the position specified by center
    detNoise    np.array(K) with dark noise countings per pixel. Default is 
                flat noise
    center      np.array([page,row,col]) of the reference pixel for the SBR. Default
                is the center of the ROI
    pxsize      np.array([psx,psy,psz]) pixel size of the PSF
    smooth      bool for activating the gaussian smoothing of the likelihood
    s           np.array([sx,sy,sz]) containing the sigma of the gaussian in
                metric units
    mode        mode for ndimage gaussian_filter
    ===========================================================================
    Output      Meaning
    ---------------------------------------------------------------------------
    xMLE        np.array(N). Maximum likelihood estimation of the x position
    yMLE        np.array(N). Maximum likelihood estimation of the y position
    ===========================================================================
    """
    psfsize = np.shape(PSF)
    K = psfsize[0]
    sizex = psfsize[3]
    sizey = psfsize[2]
    sizez = psfsize[1]

    normPSF = np.sum(PSF, axis = 0)

    #eventually reshape input
    if stack.shape[1] == K:
        #stack = np.array(N,K)
        n = stack
    else:
        #stack = np.array(N,sqrt(k),sqrt(K))
        n = np.reshape(stack,[-1,K])

    #check optional inputs
    if np.shape(center)[0] == 0:
        page = int(sizez/2)
        rowc = int(sizey/2)
        colc = int(sizex/2)
    else:
        page = center[0]
        rowc = center[1]
        colc = center[2]

    if np.shape(detNoise)[0] == 0:
        detNoise = np.ones(K)
        
    if np.shape(pxsize)[0] == 0:
        psx = 1
        psy = 1
        psz = 1
    else:
        psx = pxsize[0]
        psy = pxsize[1]
        psz = pxsize[2]

    # calculate the (different) SBR in the whole ROI
    SBR = SBR0 * normPSF / normPSF[page,rowc,colc]

    # probability vector 
    p = np.zeros((K, sizez, sizey, sizex))
    for i in np.arange(K):
        p[i, :, :, :] = (SBR/(SBR + 1)) * PSF[i,:,:,:]/normPSF + (1/(SBR + 1)) * detNoise[i]/np.sum(detNoise)

    xMLE = np.zeros(n.shape[0])
    yMLE = np.zeros(n.shape[0])
    zMLE = np.zeros(n.shape[0])

    for h in range(n.shape[0]):
        if h%100 == 0: print(str(h)+'\\'+str(n.shape[0]))
        l_aux = np.expand_dims(n[h,:],axis=(1,2,3)) * np.log(p)

        likelihood = np.sum(l_aux, axis = 0)
        
        if smooth:
            likelihood = gaussian_filter(likelihood,sigma=s[::-1],mode=mode)
        
        hh, ii, jj = np.unravel_index(np.argmax(likelihood),likelihood.shape)
        xMLE[h] = jj - sizex/2
        yMLE[h] = ii - sizey/2
        zMLE[h] = hh - sizez/2

    xMLE *= psx
    yMLE *= psy
    zMLE *= psz

    return xMLE, yMLE, zMLE