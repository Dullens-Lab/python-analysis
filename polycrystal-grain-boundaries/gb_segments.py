## Purpose: Take grains from snapshot and find grain boundaries
##          Outlined in Lavergne et al. PRX 7 041064 (2017)
##
##
import numpy as np
import scipy.spatial as sps

def misOr(ang1, ang2): # Returns smaller of misorientation angles
    delAngA = np.absolute(ang1 - ang2)
    delAngB = 60 - delAngA
    return delAngA if (delAngA < delAngB) else delAngB

def avMisor(mis1, mis2):
    avAngA = np.absolute((mis1 + mis2)/2)
    avAngB = 60 - avAngA
    return avAngA if (avAngA < avAngB) else avAngB

##### Purpose: For single frame find grain boundary segment positions, their lengths and their misorientations #####
## Input:   Grain frame data: x, y, frame, |P6|, tht6, Grain
## Output: Segment point 1 x, y; Segment point 2 x, y; Distance of GB to grains; Grain 1; Grain 2; Misorientation; Segment length
#####
def gbPts(frData):

    frData = np.column_stack((frData[:,:2], frData[:,3:])) # Remove frame
    gbMid = np.empty((0,8))

    vertices = sps.Delaunay(frData[:,:2]) # Delaunay triangluation of points
    nrNghbs = vertices.vertex_neighbor_vertices  # Nearest neighbours of each point -> index i: nn = nghb[1][nghb[0][i]:nghb[0][i+1]]
    pNum = frData.shape[0]
    gEdge = np.empty((0,3), dtype=int) # Contains particle index in frData, index in gNeigh, and grain number
    gNeigh = [] # Neighbour's grain
    gNeighInd = [] # Neighbour's ind
    edgeCount = 0

    for j in range(pNum): # Find particles that are on the edge of the grain

        pNNs = nrNghbs[1][nrNghbs[0][j]:nrNghbs[0][j+1]]
        gDiff = frData[j,4] - frData[pNNs,4] # Get difference of grain numbers to find whether particle is at a grain edge
        gChangeInd = pNNs[np.where(gDiff != 0)]
        gChange = frData[gChangeInd,4] # Grain number

        if gChange.shape[0] > 0: # ie borders different grains

            gEdge = np.row_stack((gEdge, np.array([j, edgeCount, frData[j,4]]))) # Add index to list of indices that are on the edge of a grain
            gNeigh.append(np.ndarray.tolist(gChange)) # Put list of neighbouring indices in a list
            gNeighInd.append(np.ndarray.tolist(gChangeInd))
            edgeCount += 1 # Counts how many particles are in the grain


    # Go through each grain, check for borders and go through each of those borders, check if already been counted

    gbMid2 = np.empty((0,8)) # Temp Holding Variable

    gMax = int(np.amax(frData[:,4])) # Max grain value

    for k in range(0,gMax+1): ## Iterating over a speicific grain ##

        gPoints = gEdge[np.where(gEdge[:,2] == k)] # Get indices of edge of grain k
        gPSz = gPoints.shape[0]
        gBData = np.empty((0,7)) # Grain 1, Grain 2, lInd, mInd, gPDist, midpoint x, midpoint y

        for l in range(gPSz): # Over edge particles in grain k

            lInd = gPoints[l,0] # Index of particle l
            lCoords = frData[int(lInd),:2] # Coordinates of particle l
            lEdgeInd = gPoints[l,1]
            spGNeigh = gNeigh[int(lEdgeInd)] # Contains bordering grains
            spGNeighInd = gNeighInd[int(lEdgeInd)] # Contains indices of opposing grain border particles
            numGrNghs = len(spGNeigh) # Number of neighbours in other grains that l has

            for m in range(numGrNghs): # Over edge particles connected to particle l

                mGrainNum = spGNeigh[m]

                if mGrainNum > k: # Only count each grain boundary point once

                    mInd = spGNeighInd[m] # Index of grain neighbour particle
                    mCoords = frData[int(mInd),:2] # Coordinates of grain neighbour particle
                    gPDist = (lCoords[0]-mCoords[0])**2 + (lCoords[1]-mCoords[1])**2
                    midPoint = ((mCoords + lCoords) / 2).tolist() # Location of the grain boundary between these points
                    gBAddIn = np.reshape(np.array([k, mGrainNum, lInd, mInd, gPDist, midPoint[0], midPoint[1]]), (1,7))
                    gBData = np.row_stack((gBData, gBAddIn))


        midCrd = np.empty((0,8), dtype=float) # Mid P 1 x, y and Mp 2 x, y; Grain 1; Grain 2; Distance from midpoint to grain; Misorientaiton
        unqGNgh = np.unique(gBData[:,1]) # Get unique neighbours of grain
        uGNLen = unqGNgh.shape[0]

        for n in range(uGNLen): # Goes over neighbouring grains

            kNeigh = unqGNgh[n] # Neighbour value
            kNArr = gBData[np.where(gBData[:,1] == kNeigh)] # gBData involving grain n
            unqParticles = np.unique(kNArr[:,2])
            unq1P = np.copy(unqParticles)
            unqParticles = np.append(unqParticles, np.unique(kNArr[:,3]))
            uPLen = unqParticles.shape[0]
            alrCt = np.empty((0,2))

            if uPLen > 2: # Need at least 3 for a grain boundary

                for o in range(uPLen): # Goes over a particle at the edge of the grain

                   part1 = unqParticles[o]
                   p1Data = frData[int(part1),:]
                   p1Ns = nrNghbs[1][nrNghbs[0][int(part1)]:nrNghbs[0][int(part1)+1]]
                   indxVal = 2
                   oppIV = 3
                   hasMatch = np.intersect1d(part1, unq1P).shape[0]
                   if hasMatch == 0: # Get the particles from the other grain
                       indxVal = 3
                       oppIV = 2

                   k1GN = kNArr[np.where(kNArr[:,indxVal] == part1)]
                   p1OpGN = k1GN[:,oppIV] # Particles on other grain that are connected
                   nbsInGB = np.intersect1d(p1Ns, kNArr[:,indxVal]) # Particles to find the length between
                   nIGBLen = nbsInGB.shape[0]

                   for p in range(nIGBLen): # Goes over neighbours on edge of same grain of particle)

                       part2 = nbsInGB[p]
                       cntd = np.where((alrCt[:,0] == part2)*(alrCt[:,1] == part1))[0] # Has already been counted

                       if cntd.shape[0] > -1: # This segment has not already been counted

                           alrCt = np.row_stack((alrCt, np.array([part1, part2])))
                           p2Data = frData[int(part2),:]
                           k2GN = kNArr[np.where(kNArr[:,indxVal] == part2)]
                           p2OpGN = k2GN[:,oppIV] # Opposing particles that p2 is connected to
                           thirdVertex = np.intersect1d(p1OpGN, p2OpGN) # Particle that forms 3rd vertex (k) of ikj

                           if thirdVertex.shape[0] == 1:

                               tVAng = frData[int(thirdVertex), 3]
                               mis1 = misOr(p1Data[3], tVAng)
                               mis2 = misOr(p2Data[3], tVAng)
                               aMis = avMisor(mis1, mis2) # Misorientation
                               ### segLen = np.linalg.norm(p1Data[:2]-p2Data[:2])/2 # Length of Segment

                               k1 = k1GN[np.where(k1GN[:,oppIV] == thirdVertex)]
                               k2 = k2GN[np.where(k2GN[:,oppIV] == thirdVertex)]
                               # Mid Coords and av distance of mid point
                               midCrd = np.row_stack((midCrd, np.array([k1[:,5][0], k1[:,6][0], k2[:,5][0], k2[:,6][0],
                                                                        (k1[:,4][0]+k2[:,4][0])/2, k1[:,0][0], k1[:,1][0], aMis]).reshape(1,8)))



        gbMid2 = np.row_stack((gbMid2, midCrd))


    gbMid = np.row_stack((gbMid, gbMid2))
    segLength = (gbMid[:,0]-gbMid[:,2])**2 + (gbMid[:,1]-gbMid[:,3])**2
    gbMid = np.column_stack((gbMid, np.sqrt(segLength))) # Segment length
    return gbMid

##### Take paricle coordinates and sort particles into their respective grains #####
##      Input: findG output, folder to save to
##      Output: Frame, Segment point 1 x, y; Segment point 2 x, y; Distance of GB to grains; Grain 1; Grain 2; Misorientation; Segment length
def gbSegFind(pData, fileLoc):
    tMax = int(np.amax(pData[:,2]))
    gbSeg = np.empty((0,10))
    for i in range(tMax+1):
        if i % 10 == 0:
            print("Finding gb frame " + str(i))
        frD = pData[np.where(pData[:,2] == i)]
        gbS = gbPts(frD)
        gbSeg = np.row_stack((gbSeg, np.column_stack((np.ones(gbS.shape[0])*i, gbS))))

    ofile = fileLoc + "gbSegments.dat"
    np.savetxt(ofile, gbSeg, fmt = "%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\r")
    print("Finished grain finding")
    return gbSeg
