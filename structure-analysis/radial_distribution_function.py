# Purpose: Get average RDF from file containing particle coordinates with parallelisation.
#
# Input:   Particle coordinates: x, y, frame
# Output:  r, particles in shell, shell area, g(r)
# Tutorial http://www.physics.emory.edu/faculty/weeks//idl/gofr2.html

import numpy as np
import scipy.spatial.distance as spd
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

def distHist(crds, bins): # Takes particle coords, finds pairwise dists and outputs histogram
    return np.histogram(spd.pdist(crds, 'euclidean'), bins)

def parG(i, pdata, grdat):
    fcoord = pdata[np.where(pdata[:,2] == i)][:,:-1] # Get coordinates for frame at time i
    return distHist(fcoord, grdat[:,0])[0] # Add histogram of No. particles to total data


## Latex Fonts ##
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('legend', fancybox=False, edgecolor='k', shadow=False)

### Import file ###
floc = 'C:\\Users\\rdgroup\\Downloads\\'
ftotal = floc + 'AF0,56_trk' # File name to import
nout = floc + 'AF0,56_trk' + '_g_r.dat' # File name of output

pdata = np.loadtxt(ftotal, delimiter=',')

### Distances ###
umppx = 0.2735 # Micrometres per pixel
dr = 0.01 # Step size (dr) in pixels
diam = 2.95 # Particle diameter in um
dpx = diam // umppx # Diameter in pixels
maxd = dpx * 10 # Max g(r) distance (in multiples of diameter)
binnum = int(maxd / dr) # Number of bins
xdim, ydim = 2048, 2048 # x and y dimensions in pixels

### Generate array to contain g(r): r, Particle N in bin, Shell area, g(r) ###
grdat = np.arange(0, maxd, dr) # Get array for r values with steps of dr
grdat = np.reshape(grdat, (grdat.size, -1)) # 1D array to 2D to allow for concat
grdat = np.column_stack((grdat, np.zeros((binnum,3)))) # Number of particles, Area, g(r)
grdat[:,2] =  grdat[:,0] * dr * 2 * np.pi # Area for each shell
rnddat = np.copy(grdat) # Array for random particles

dt = 1 # Which frames to examine
maxtime = int(np.amax(pdata[:,2])) # Last frame timestamp

### Get g(r) for data ###
a = Parallel(n_jobs=2, backend='threading', verbose=10)(delayed(parG)(i, pdata, grdat) for i in range(0,maxtime+1,dt))
for b in a:
    grdat[:-1, 1] += b

### Get g(r) for ideal gas to account for box size ###
rcoord = np.random.random_sample((pdata.shape[0],2)) # Generate random floats between 0 and 1
rcoord[:,0] = rcoord[:,0]*xdim # Scales random xcoords to the size of the frame
rcoord[:,1] = rcoord[:,1]*ydim # Scales Random ycoords to the size of the frame
rcoord = np.column_stack((rcoord, pdata[:,2]))

c = Parallel(n_jobs=2, backend='threading', verbose=10)(delayed(parG)(i, rcoord, rnddat) for i in range(0,maxtime+1,dt))
for d in c:
    rnddat[:-1,1] += d

### Normalise data by ideal gas ###
grdat[:,3] = (grdat[:,1] / grdat[:,2]) / np.sum(grdat[:,1]) # Get unnormalised g(r)
rnddat[:,3] = (rnddat[:,1] / rnddat[:,2]) / np.sum(rnddat[:,1]) # Ideal gas g(r)

grdat[:,3] /= rnddat[:,3] # Normalise for box
grdat[:,0] += dr/2 # Centre bins
grdat[:,0] *= umppx # Convert pixels to um

ofile = floc + nout
np.savetxt(nout, grdat,  fmt = "%.4f\t%.2f\t%.3f\t%.5f\r")

plt.figure()
plt.plot(grdat[:,0], grdat[:,3], color='royalblue')
plt.xlabel(r"$r/\sigma$", size=24)
plt.ylabel(r"$g(r)$", size=24)
plt.tick_params(which='major', labelsize=24)

print('Finished')
