#   Purpose: Find the strain of particles in a colloidal crystal
#            Method detailed in the appendix of my thesis
#
#   Input: Particle coordinates: x, y, frame
#   Output: x, y, frame, E_xx, E_xy/E_yx, E_yy

import numpy as np
import scipy.spatial as sps
from orientationStrain import orientS
from scipy.spatial import Voronoi, cKDTree

## Generates the local ideal model for a 2D crystal- ie a regular hexagon with lattice spacing as side length centred around 0
## Input: Lattice spacing
def gen_model(r):
    A = np.reshape(np.array([[1,	0],
              [0.5,	-0.866025],
              [-0.5,	-0.866025],
              [-1	, 0],
              [-0.5,	0.866025],
              [0.5,	0.866025]]), [6,2])
    return A*r


fileLoc = 'F:\\Data\\2021-05-20\\GB 1\\'

dircS = fileLoc + 'small_coords.dat'
sData = np.loadtxt(dircS, dtype = float) # x, y, t

tMax = int(np.amax(sData[:,2]))


#### Generate a model for the strain
#### Hexagonal unit cell with inter particle spacing from average lattice spacing
meanDist = 0
frNum = 100
for x in range(0,tMax,frNum):
    if x % 100 == 0:
        print('Dist frame ' + str(x))
    xData = sData[sData[:,2] == x][:,:2]
    tree = cKDTree(xData)
    dists, neighs = tree.query(xData,6+1) # 6 nearest neighbour distances
    meanDist += np.mean(dists[:,1:])

meanDist /= (tMax/frNum)

r2 = meanDist**2

mdl = gen_model(r) # Model hexagon


pStrain = np.column_stack((sData, np.zeros((sData.shape[0],3)) # x, y, frame, E_xx, E_xy/E_yx, E_yy
for t in range(0,tMax,1):

    print("Frame: " + str(t))
    frameWhr = np.where(sData[:,2] == t)
    stData = sData[frameWhr]
    sNum = stData.shape[0]

    vertices = sps.Delaunay(stData[:,:2])
    nghb = vertices.vertex_neighbor_vertices

    neighCrd = []
    pCrd = []
    pNum = []

    # Get particles' coordinates and their neighbours' coordinates
    for i in range(sNum):
        nNs = nghb[1][nghb[0][i]:nghb[0][i+1]] # neighbours of i
        iPos = stData[i,:2]
        nPos = stData[nS,:2]
        iDist = np.sum((nPos - iPos)**2, axis=1)
        iClose = np.where(iDist < 2.25*r2)[0] # Are the neighbours close enough for a unit cell?
        nPos1 = nPos[iClose]
        pN = pNum[i]
        if nPos1.shape[0] > 3:
            neighCrd.append(nPos1)
            pCrd.append(iPos)
            pNum.append(i)

    pOr, pFit, pRot = orientS(pCrd, neighCrd, mdl, minN) # Produces the deformation gradient of each particle

    # Turn deformation gradient -> strain tensor
    for p in range(len(pOr)):
        pOr[p] = pOr[p] @ np.linalg.inv(pRot[po]) # Rotate in terms of x,y axes
        pOr[p] = 0.5*(pOr[p]+pOr[p].T) - np.array([[1,0],[0,1]]) # Strain tensor

        pStrain[frameWhr[0][pNum[p]],4] = pOr[0,0]
        pStrain[frameWhr[0][pNum[p]],5] = pOr[0,1]
        pStrain[frameWhr[0][pNum[p]],6] = pOr[1,1]

np.savetxt(fileLoc+"small_strain_components.dat", pStress, fmt = "%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\r")
