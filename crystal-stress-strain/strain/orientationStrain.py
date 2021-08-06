# Name: orientation
# Function:
#   Calculate orientation of particles
# Inputs:
#       pos- Positions of all the particles
#       nPos- List of lists containing particle neighbours' positions
#       Ref- Reference structure
#       minN - Min number of neighbours per particle
#
# Output:
#       Orientation Array- List of arrays containing the orientation matrices
#       fit - fit of each of the point set registrations, -1 if it fails
#


import numpy as np
from generate_indices import gen_indices
from register_points_loop import reg_points_loop
import rmsd



def orientS(pos, nPos, ref, minN):

    dim = ref.shape[1] # Dimension
    maxN = ref.shape[0] # Max num elements
    n = len(pos) # Size of data set
    orAr = [None for _ in range(n)] # Orientational array
    rotAr = [None for _ in range(n)]
    fitT = np.zeros(n)

    iden = np.identity(dim) # Identity matrix

    indicesT = gen_indices(minN, maxN, dim)

    for a in range(n):

        #if a % 1000 == 0:
        #    print(str(a))

        cent = pos[a] # Particle coordinates
        neighs = nPos[a] # Particle neighbour coords
        rPoints = neighs - cent # Make neighbours relative to particle
        nPointsA = neighs.shape[0] # Num of neighs
        if nPointsA > maxN:
            rPoints = rPoints[:maxN,:]
            nPointsA = maxN

        index = nPointsA - minN
        indices = (indicesT[0][index], indicesT[1][index], indicesT[2][index]) # Get all arrays of indices for this num. neighbours

        Ar, Br, fit = reg_points_loop(rPoints, ref, dim, indices)
        sA = rPoints[Ar,:] # ORdered Points of data
        sB = ref[Br,:] # Reference points
        fitT[a] = fit

        U = rmsd.kabsch(sA, sB[:Ar.shape[0],:]) # Rotation matrix for these

        A = np.zeros((dim,dim)) # A Matrix
        D = np.zeros((dim,dim)) # D Matrix
        sBLen = sB.shape[0]
        # Rotate reference to position
        for p in range(sBLen):
            if p > sA.shape[0]-1:
                continue
            rRef = sB[p,:]
            A += np.outer(sA[p,:],rRef)
            D += np.outer(rRef,rRef)

        F = np.dot(A, np.linalg.inv(D))

        stn = 0.5 * (np.inner(F,F)-iden)

        orAr[a] = F
        rotAr[a] = U

        refTens = np.zeros((6,2))
        ref2 = np.zeros((6,2))
        for i in range(6):
            refTens[i,:] += F @ ref[i,:]
            if i < 4:
                ref2[i,:] = stn @ sA[i,:]

    return orAr, fitT, rotAr
