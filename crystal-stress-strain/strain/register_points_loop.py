# Name:
#       register points loop
# Function:
#       Register points in A with those in B
# Inputs:
#       A, B - Arrays with row vectors A.size <= B.size
#       Dimension (2D/3D)
#       Indices- Contains premade index pairs and groups for faster performance
# Outputs:
#       Ar, Br - Registered A and B points
#       Fit - Mean Sq distance between registered points. If -1 registration failed
#


import numpy as np
import scipy.special as spsc
import rmsd
import warnings
warnings.filterwarnings("ignore")

def reg_points_loop(A, B, dim, ind):

    nPointsA = A.shape[0]
    nPointsB = B.shape[0]

    indicesA = ind[0] # All combinations of the indices
    indicesB = ind[1] # All permutations of the indices
    indicesC = np.array(ind[2]) # Distance matrix index if coords are in a dxn matrix


    a = 0
    pointsA = A[indicesA[a*dim],:] # Get coords of 3 points for A that will be registered

    regSize = int(spsc.binom(nPointsA, dim)*spsc.factorial(dim))
    minSqD = np.ones(regSize) * -1 # Create array of regSize -1s
    regis1 = [None for _ in range(regSize)] # Create list of regSize empty lists

    for b in range(regSize):

        pointsB = B[indicesB[b],:]

        rot = rmsd.kabsch(pointsA, pointsB) # Get rotation matrix that maps A onto B
        rotA = np.zeros((B.shape[0], dim)) # Generate empty A array to hold rotated points of A

        ASz = A.shape[0]

        for c in range(ASz):
            rotA[c,:] = np.reshape(np.matmul(rot, np.reshape(A[c,:], [dim,1])), [1,dim]) # Add rotated points of A to matrix

        diff = (rotA[indicesC[:,0],:] - B[indicesC[:,1],:])**2 # Square distances x^2 and y^2 between point and all other points
        diffSq = np.sum(diff, axis=1) # Squared distances r^2
        dSqI = np.argsort(diffSq) # Sort in acending order
        indCSorted = indicesC[dSqI[:nPointsA],:]

        ud1 = np.unique(indCSorted[:,0]).size
        ud2 = np.unique(indCSorted[:,1]).size

        if((ud1 == nPointsA) & (ud2 == nPointsA)):
            minSqD[b] = np.sum(diffSq[dSqI][:nPointsA]) # Add sum of smallest ones to overall array
            regis1[b] = indCSorted # Add Registration indices to overall array


    posSqD = np.where(minSqD > 0)[0] # Where registration is 1:1
    pminSqD = minSqD[posSqD] # Distance values for 1:1
    pregis1 = np.array(regis1)[posSqD] # Index values for 1:1
    szSqD = pminSqD.size

    if szSqD > 0:
        minSqD = np.amin(pminSqD) # Smallest square distance
        iMSqD = np.where(pminSqD == minSqD)[0] # Index of smallest sq dist
        if iMSqD.shape[0] > 1: # Multiple permutations of same smallest sq dist
            iMSqD = iMSqD[0]
        iMSqD = int(iMSqD)
        fit = minSqD / nPointsA
        Ar = pregis1[iMSqD][:,0]
        Br = pregis1[iMSqD][:,1]
    else: # If no 1:1
        fit = -1
        Ar = np.arange(nPointsA)
        Br = np.arange(nPointsB)

    return Ar, Br, fit
