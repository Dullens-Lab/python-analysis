## Purpose: Sort particles in a particular frame into nGrains
##          Outlined in Lavergne et al. PRX 7 041064 (2017)
##

import numpy as np
import scipy.spatial as sps

##### # Joins all lists containing same particles together -> grains #####
##          Input: List of lists of particle groups in same grain
##          Output: All particles in a grain
def sortIntoGrains(l):
    out = [] # l is a list of lists
    while len(l)>0:
        first, *rest = l # first is the list of lists l, *rest is an unpacked list containing all the values from all of the lists
        first = set(first) # First is a set of all the lists (sets are unordered collections of unique (non-duplicate) items)

        lf = -1
        while len(first)>lf:
            lf = len(first) # Number of lists in set first

            rest2 = []
            for r in rest: # For each unique list
                if len(first.intersection(set(r)))>0: # How many intersections are there between values in list r and all the lists in first
                    first |= set(r) # Join lists that intersect into a list
                else:
                    rest2.append(r) # If no intersections add to 'unmatched list' list
            rest = rest2

        out.append(first)
        l = rest
    out2 = []
    for i in out:
        out2.append(list(i))
    return out2

##### Single frame particle sorting into grains #####
##      Input: psi6_Smooth frame data, Minimum grain size, Orientation difference of neighbours to be in same grain
##      Output: x, y, |Psi6|, arg(psi6), Grain number
def grCl(frData, grSz=19, commAng=0.5):

    frData[:,4] = np.rad2deg(frData[:,4])/6 # Turn absolute angle to theta 6 in degrees
    #commAng = 0.5 # Difference in angle to be considered part of the same crystal
    finalGrains = np.empty((0,6)) # x, y, frame, |P6|, tht6, Grain

    vertices = sps.Delaunay(frData[:,:2]) # Delaunay triangluation of points
    nrNghbs = vertices.vertex_neighbor_vertices  # Nearest neighbours of each point -> index i: nn = nghb[1][nghb[0][i]:nghb[0][i+1]]
    pNum = frData.shape[0]
    frameGrainList = [] # List containing lists of which particles are part of the same grain

    # Finds neighbours closeness
    for j in range(pNum):
        if frData[j,3] >= 0.7:
            pNNs = nrNghbs[1][nrNghbs[0][j]:nrNghbs[0][j+1]] # Nearest Neighbours to particle j
            angDiff = np.abs(frData[j,4] - frData[pNNs,4]) # Matrix with angle differences
            angLess = pNNs[np.where((angDiff < commAng)|(angDiff > 60-commAng))] # Indices of points where angles are closer than commAng degrees
            if angLess.shape[0] >= 3: # 3 neighbours with the same orientation
                addList = [j]
                addList.extend(np.ndarray.tolist(angLess))
                frameGrainList.append(addList) # Adds list of particles that are neighbours, crystalline and have a small diff in theta 6

    frameGrainList = sortIntoGrains(frameGrainList)

    nGrains = len(frameGrainList) # Final number of grains
    gCount = 0

    for l in range(nGrains):
        gIndices = np.array(frameGrainList[l]) # Get indices of all grains
        if gIndices.shape[0] >= 19: # Grain has a size greater than 19 particles
            frDataTake = frData[gIndices,:]
            frDataTake = np.column_stack((frDataTake, (np.ones((gIndices.shape[0],1))*gCount))) # Add grain value to other data
            finalGrains = np.row_stack((finalGrains, frDataTake)) # Add to overall array
            gCount += 1

    return finalGrains

##### Take paricle coordinates and sort particles into their respective grains #####
##      Input: psi6_Smooth output, Minimum grain size, Orientation difference of neighbours to be in same grain, folder to Output to
##      Output: x, y, frame, |Psi6|, arg(psi6), Grain number

def findG(pData, fileLoc, grSz=19, commAng=0.5):
    tMax = int(np.amax(pData[:,2]))
    grains = np.empty((0,6)) # x, y, frame, |P6|, tht6, Grain
    for i in range(tMax+1):
        if i % 10 == 0:
            print("Finding grain frame " + str(i))
        frD = pData[np.where(pData[:,2] == i)]
        iGr = grCl(frD, grSz, commAng)
        grains = np.row_stack((grains, iGr))

    ofile = fileLoc + "grainSort.dat"
    np.savetxt(ofile, grains, fmt = "%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\r")
    print("Finished grain finding")
    return grains
