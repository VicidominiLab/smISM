import numpy as np

def shift2Darray(array, shift, bc="periodic"):
    """
    Shift 2D array with either periodic boundary conditions or by appending
    zeros
    ===========================================================================
    Input       Meaning
    ---------------------------------------------------------------------------
    array       2D numpy array
    shift       Vector with 2 elements with shift in y and x [shiftx, shifty]
    bc          Boundary conditions, either "periodic" or None
    ===========================================================================
    Output      Meaning
    ---------------------------------------------------------------------------
    arrayOut    2D array with all values shifted
    ===========================================================================
    """
    
    shiftx = int(shift[0])
    shifty = int(shift[1])
    
    if bc == "periodic":
        # no periodic boundary conditions, instead append zeros
        arrayOut = np.roll(array, shifty, axis=0)
        arrayOut = np.roll(arrayOut, shiftx, axis=1)
    
    else:
        
        # get size original image
        arraySize = np.shape(array)
        Ny = arraySize[0]
        Nx = arraySize[1]
        
        # store image in 3x larger matrix
        arrayLarge = np.zeros((3*Ny, 3*Nx))
        arrayLarge[Ny:2*Ny, Nx:2*Nx] = array
        
        # get shifted array with periodic boundary conditions
        arrayOut = arrayLarge[Ny-shifty:2*Ny-shifty, Nx-shiftx:2*Nx-shiftx]
    
    return arrayOut