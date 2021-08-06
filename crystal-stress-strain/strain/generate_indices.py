# Name:
#       generate_indices
# Function:
#       Takes points A and 1-1 registers them with points in B
# Inputs:
#       A & B: Sets of points. A less or equal size to B.
#       Dim: 3D / 2D
#
# Outputs:
#       Tuple containing possible iA, iB and iC arrays
#       Tuple should have structure:
#       Index 0, 1, 2 for A, B, C
#       Then 0, 1, 2, 3, .., (maxN - minN) A-> minN, minN+1, minN+2, minN+3.....+ maxN
#

import numpy as np
import itertools as it

def gen_indices(minN, maxN, dim):

    # Generate indices
    indA = []
    indB = []
    indC = []

    nPointsB = maxN # Points to be registered to
    indicesB = list(it.permutations(range(nPointsB), dim)) # List of tuples containing all (dim) permutations of the indices

    for i in range(minN, maxN+1):
        nPointsA = i # Points to register

        # Generate all possible combinations of indices for the n points
        indicesA = list(it.combinations(range(nPointsA), dim)) # List of tuples containing all (dim) combinations of the indices

        # Create distance matrix indices for easy distance calculation
        # C is if 3 coord columns and each row is a particle

        indicesC = np.column_stack((np.repeat(range(nPointsA), nPointsB),
                                    np.tile(range(nPointsB), nPointsA)))

        indA.append(tuple(indicesA))
        indB.append(tuple(indicesB))
        indC.append(tuple(indicesC))

    # Turn into tuple of tuples
    indA = tuple(indA)
    indB = tuple(indB)
    indC = tuple(indC)

    return (indA, indB, indC)
