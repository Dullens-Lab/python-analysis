Using `find_GB.py`, input your particle coordinates in a time series: `x, y, frame`.
The output of each function is saved as all are useful for other purposes.

### Firstly, find the bond-orientational order parameter Psi6.

This has a nearest neighbour cut-off to help with particles at the edge and adjacent to vacancies/impurities.
This is smoothed over two shells of nearest neighbours to reduce noise in GB finding.
Returns `x`; `y`; `frame`; `modulus of psi6` (i.e. crystallinity); `argument of psi6` (i.e. orientation).
  
### Next, find the grains using a cluster algorithm.

Goes particle by particle, checks if crystalline (`|psi6|` > 0.7), has 3 neighbours of close orientation.
Gives lists of particles and neighbours that are part of the same grain, overlapping lists are joined to create a list of the paritcles in a grain.
Returns same as `Psi6`, with extra column in which each particle is labelled with their grain number.
N.B. Each grain number is regnerated per frame, so no correlation across frames (you can track centre-of-masses with Crocker and Grier although not perfect).

### Finally, find the grain boundary segments.

A grain boundary segment is found by doing a Delaunay triangulation of all the particles and finding the connections between particles in different grains.
These form triangles A0->B->A1, of particles in grains A and B, where A0 and A1 are neighbours in the same grain.
Segment is the joined midpoints of A0->B and B->A1.
Returns the coordiantes of midpoints A0->B and B->A1, as well as the IDs of grains A and B.
Also returns distance of segment from grains A and B. Useful for deleting spurious GBs that aren't really GBs due to frustrated crystallisation etc.

Finally returns misorientation and length of this segment.
```
  0   1   2   3   4      5       6        7      8     9
  t   x0  y0  x1  y1  Grain A Grain B Distance Misor Length
 ```
