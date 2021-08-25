import numpy as np
from psi6_Smooth import psi6S
from find_Grains import findG
from gb_segments import gbSegFind
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.cm as cm

## Colourmaps for plotting ## 
mapper = cm.ScalarMappable(Normalize(0,60, clip=True), cm.hsv)
gbMapper = cm.ScalarMappable(Normalize(0,30, clip=True), cm.viridis)

##### Change lattice spacing for your crystal ###
a0 = 2.95 # Lattice spacing in um
pxtoum = 0.1004 # um per pixel
pDistSq = (a0*2/pxtoum)**2 # Neighbour cutoff distance^2 (for calculating psi6)

### Coordinates file ###
fileLoc = 'F:\\Backup\\Coarsening\\Binary Coars 5\\Binary Coars 5-2\\'
dircL = fileLoc + 'Bin_Coars_5-2_small_coords.dat'
pData = np.loadtxt(dircL)

## Get bond-orientational-order parameter ##
p6 = psi6S(pData, pDistSq, fileLoc) # x, y, frame, |Psi6|, arg(psi6)
print("Finshed Psi6")
## Find grains ##
grains = findG(p6, fileLoc)
print("Finshed grain finding")
## Find GB segments ## 
gbSegments = gbSegFind(grains, fileLoc)
print("Finshed GB segment finding")

### Plot a snapshot of the grains and GBs ####
plt.figure()
frame = 0
frGrain = grains[grains[:,2] == frame]
frSeg = gbSegments[gbSegments[:,0] == frame]
plt.scatter(frGrain[:,0], frGrain[:,1], color=mapper.to_rgba(frGrain[:,4]), fc='none')
for l in range(frSeg.shape[0]):
    lX = [frSeg[l,1], frSeg[l,3]]
    lY = [frSeg[l,2], frSeg[l,4]]
    plt.plot(lX, lY, c=gbMapper.to_rgba(frSeg[l,8]))
    
plt.axis('equal')
plt.axis('off')