# Purpose: Get RDFs for binary system.
#
# Input: Large and Small particle coordinates: x, y, frame
# Ourput: r (um), g_ss(r), g_sl(r), g_ll(r) 

import numpy as np
import scipy.spatial.distance as spd
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

def distHist(crds, bins): # Takes same particle coords, finds pairwise dists and outputs histogram
    return np.histogram(spd.pdist(crds, 'euclidean'), bins)

def binDHist(crds1, crds2, bins): # Takes opposite species coords, finds pairwise dists and outputs histogram
    return np.histogram(spd.cdist(crds1, crds2, 'euclidean'), bins)

def getrcrd(rnddat, frsz, xdim, ydim):
    rcoord = np.random.random_sample((frsz,2)) # Generate random floats between 0 and 1
    rcoord[:,0] = rcoord[:,0]*xdim # Scales random xcoords to the size of the frame
    rcoord[:,1] = rcoord[:,1]*ydim # Scales Random ycoords to the size of the frame
    return distHist(rcoord, rnddat[:,0])[0] # Add histogram of No. particles to total data of random particle images

def findgr(grdat, rnddat, maxd, dr, umppx):
    grdat[-1,1], rnddat[-1,1] = 1, 1 # No division by 0
    mxar = maxd**2 * np.pi
    rhogr, rhornd = np.sum(grdat[:,1])/mxar, np.sum(rnddat[:,1])/mxar
    grdat[:,3] = (grdat[:,1] / grdat[:,2]) / rhogr # Get unnormalised g(r)
    rnddat[:,3] = (rnddat[:,1] / rnddat[:,2]) / rhornd
    
    grdat[:,3] /= rnddat[:,3] # Normalise for box
    grdat[:,0] += dr/2 # Centre bins
    grdat[:,0] *= umppx # Convert pixels to um
    grdat[0,3] = 0
    return np.column_stack((grdat[:,0], grdat[:,3]))

## Format graph font and axes ##
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('legend', fancybox=False, edgecolor='k', shadow=False)
plt.rc('axes', linewidth=3)

# Import large and small coordinates
floc = 'G:\\Joe\\2019-11-21\\Both gr\\'
fltotal = floc + 'large_coords.dat' # File name to import
fstotal = floc + 'small_coords.dat' # File name to import
grPair = 'sl'
nout = 'gr-' + grPair + '.dat'

lBData = np.loadtxt(fltotal, dtype = float)
sBData = np.loadtxt(fstotal, dtype = float)

# Distances #
umppx = 0.2746 # Micrometres per pixel
dr = 0.1 # Step size in pixels
diam = 3.83 # Approx. particle diameter
dpx = diam // umppx # Diameter in pixels
maxd = dpx * 15 # Max g(r) distance in multiples of diameter
binnum = int(maxd / dr) # Number of bins
xdim, ydim = 2048, 2048

## Format array to hold small-large g(r) ##
grdat = np.arange(0, maxd, dr) # Get array for r values with steps of dr
grdat = np.reshape(grdat, (grdat.size, -1)) # 1D array to 2D to allow for concat
grdat = np.column_stack((grdat, np.zeros((binnum,3)))) # Number of particles, Area, g(r)
grdat[:,2] =  grdat[:,0] * dr * 2 * np.pi # Area for each shell
rnddat = np.copy(grdat) # Array for random particles -> b = a means b is a pointer to a

## Large and small particle g(r)s ##
grdatL = np.copy(grdat)
grdatS = np.copy(grdat)
rnddatL = np.copy(grdat)
rnddatS = np.copy(grdat)

dt = 1 # Time step
mintime = int(np.amin(sBData[:,2])) # First frame timestamp
maxtime = int(np.amax(sBData[:,2])) # Last frame timestamp

xdim = np.ceil(np.amax(sBData[:,0]) - np.amin(sBData[:,0])) # x-dimension
ydim = np.ceil(np.amax(sBData[:,1]) - np.amin(sBData[:,1])) # y-dimension

for i in range(mintime, maxtime+1, dt):
    scoord = sBData[np.where(sBData[:,2] == i)][:,:-1] # Get coordinates for frame at time i
    sSz = scoord.shape[0] # The number of particles in frame i - Used for generating random particle image
    lcoord = lBData[np.where(lBData[:,2] == i)][:,:-1] # Get coordinates for frame at time i
    lSz = lcoord.shape[0] # The number of particles in frame i - Used for generating random particle image
    
    rnddat[:-1, 1] += getrcrd(rnddat, sSz+lSz, xdim, ydim)
    grdat[:-1, 1] += binDHist(scoord, lcoord, grdat[:,0])[0] # Add histogram of No. particles to total data
    rnddatL[:-1, 1] += getrcrd(rnddatL, lSz, xdim, ydim)
    grdatL[:-1,1] += distHist(lcoord, grdatL[:,0])[0]
    rnddatS[:-1, 1] += getrcrd(rnddatS, sSz, xdim, ydim)
    grdatS[:-1,1] += distHist(scoord, grdatS[:,0])[0]
    
    print('Finished frame ' + str(i) + '\n')

slgr = findgr(grdat, rnddat, maxd, dr, umppx)
llgr = findgr(grdatL, rnddatL, maxd, dr, umppx)
ssgr = findgr(grdatS, rnddatS, maxd, dr, umppx)

bnGr = np.column_stack((ssgr[:,:2], slgr[:,1], llgr[:,1]))

ofile = floc + 'binned_gr.dat'
np.savetxt(ofile, bnGr,  fmt = "%.4f\t%.4f\t%.4f\t%.4f\r")

plt.figure()
plt.plot(bnGr[:,0], bnGr[:,1], color='royalblue')
plt.plot(bnGr[:,0], bnGr[:,2]+4, color='darkviolet')
plt.plot(bnGr[:,0], bnGr[:,1]+8, color='crimson')
plt.xlabel(r'$r (\mu m)$', size=16)
plt.ylabel(r'$g_{ij}(r)$', size=16)

print('Finished')