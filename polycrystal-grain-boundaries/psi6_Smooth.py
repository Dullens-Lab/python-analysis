## Purpose: Find the neighbour averaged bond-orientational order from coordinates
##          Outlined in Lavergne et al. PRX 7 041064 (2017)
##
##          Input: Particle coordinate data: x, y, frame
##          Output: Particle coordiante data, |Psi6|, arg(Psi6)
import numpy as np
import scipy.spatial as sps

def arr1Dto2D(arr):
    return np.resize(arr, (arr.shape[0], 1))

def getSShell(pNNAr, nNList):
    szList = pNNAr.shape[0]
    shellOut = np.empty((0,1))
    for pNN in range(szList):
        indx = pNNAr[pNN]
        nNN = nNList[1][nNList[0][indx]:nNList[0][indx+1]]
        shellOut = np.row_stack((shellOut, arr1Dto2D(nNN)))
    return shellOut

def psi6S(pData, pDistSq, fileLoc):
    tMax = int(np.amax(pData[:,2])) # Max time to go through all frames

    psiList = np.empty((0,5)) # x, y, t, mod, arg

    for i in range(0, tMax+1):
        print("Frame: " + str(i))

        frData = pData[np.where(pData[:,2] == i)] # Get particle coordinates for this frame

        vertices = sps.Delaunay(frData[:,:2]) # Delaunay triangluation of points
        nrNghbs = vertices.vertex_neighbor_vertices  # Nearest neighbours of each point -> index i: nn = nghb[1][nghb[0][i]:nghb[0][i+1]]

        pNum = frData.shape[0]
        framePsiList = np.empty((0,1)) # psi6
        psiTempList = np.empty((0,5)) # x, y, t, mod, arg
        # Psi6 Calculation
        for j in range(pNum):
            nNum = 6
            pNghbs = nrNghbs[1][nrNghbs[0][j]:nrNghbs[0][j+1]] # Get the neighbours of particle j
            cNum = pNghbs.shape[0] # Number of nearest neighbours
            pCoord = frData[j,:2] # Particle coordinates
            nCoord = frData[pNghbs,:2] # Neighbour coordinates
            diffAr = nCoord - pCoord # Differences to turn into psi6
            diff2 = np.sum(diffAr**2, axis=1) # Distances between particles
            dClos = np.where(diff2 < pDistSq)
            angles = np.arctan2(diffAr[dClos,1], diffAr[dClos,0]) # Angle between particle and neighbours
            ltZ = np.where(angles < 0)
            angles[ltZ] = (2*np.pi) + angles[ltZ]
            pPsi6 = np.average(np.exp(1j*6*angles)) # Psi6 formula
            framePsiList = np.row_stack((framePsiList, pPsi6)) # add to overall Array
            p6a = np.angle(pPsi6)
            if p6a < 0: # Angle between 0 and 2pi
                p6a = (2*np.pi) + p6a

        for k in range(pNum):
            smPNghbs = nrNghbs[1][nrNghbs[0][k]:nrNghbs[0][k+1]] # Get particle's neighbours
            smPNghbs = np.row_stack((arr1Dto2D(smPNghbs), getSShell(smPNghbs, nrNghbs))) # Get second shell
            smPNghbs = np.unique(smPNghbs) # Remove duplicates due to shared neighbours
            smPNghbs = smPNghbs.astype(int) # Turn floats to ints to allow indexing
            coordPsi = np.average(framePsiList[smPNghbs]) # Gets average psi6 over these indices
            p6a = np.angle(coordPsi)
            if p6a < 0: # Angle between 0 and 2pi
                p6a = (2*np.pi) + p6a
            psiTempList = np.row_stack((psiTempList, [frData[k,0], frData[k,1], i, np.absolute(coordPsi), p6a])) # Add to overall array

        psiList = np.row_stack((psiList, psiTempList)) # Add to overall array
        print("Finished k: " + str(i))


    ofile = fileLoc + "psi6smooth.dat"
    np.savetxt(ofile, psiList, fmt = "%.4f\t%.4f\t%.4f\t%.4f\t%.4f\r")
    print("Finished Psi6")
    return psiList
