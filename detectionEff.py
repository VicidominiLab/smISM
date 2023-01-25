import numpy as np
from shift2Darray import shift2Darray
from scipy.ndimage.interpolation import zoom

def detectionEff(emPSF, detSize, compress=1):
    """
    Calculate photon percentage that falls within detector area as a function 
    of the position of the fluorophore wrt the array detector
    ===========================================================================
    Input       Meaning
    ---------------------------------------------------------------------------
    emPSF       np.array(Nx x Nx) with the emission PSF
    SPADsize    size of the array detector (in pixels) assuming a square
    compress    decrease size of the emPSF array by this factor at the start of
                the calculation to speed up the calculation, then increase
                the size again at the end of the calculation
    ===========================================================================
    Output      Meaning
    ---------------------------------------------------------------------------
    pde         np.array(Nx x Nx) with the relative number of photons that
                falls on the detector for a molecule at a given position
    ===========================================================================
    """
    
    # decrease emPSF size
    emPSF = zoom(emPSF, 1/compress)
    detSize = int(detSize / compress)
    
    Nx = np.shape(emPSF)[0]
    
    SPAD = np.zeros((Nx, Nx))
    
    center = int(Nx / 2 - 1)
    
    if np.mod(detSize, 2) == 0:
        detSize -= 1 # let this be odd
    detSize = np.max((detSize, 1))
    
    startcoord = int(np.ceil(center - 0.5 * detSize))
    stopcoord = startcoord + detSize
    
    SPAD[startcoord:stopcoord, startcoord:stopcoord] = 1
    
    pde = np.zeros((Nx, Nx))
    
    for i in range(Nx):
        for j in range(Nx):
            pde[i,j] = np.sum(SPAD * shift2Darray(emPSF, [i-Nx/2, j-Nx/2], bc=None))
    
    pde = zoom(pde, compress)
    
    return pde