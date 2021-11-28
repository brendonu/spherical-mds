import graph_tool.all as gt
import numpy as np

#From tsNET implementation

def get_shortest_path_distance_matrix(g, weights=None):
    # Used to find which vertices are not connected. This has to be this weird,
    # since graph_tool uses maxint for the shortest path distance between
    # unconnected vertices.

    # Get the value (usually maxint) that graph_tool uses for distances between
    # unconnected vertices.

    # Get shortest distances for all pairs of vertices in a NumPy array.
    X = gt.shortest_distance(g, weights=weights).get_2d_array(range(g.num_vertices()))

    return X


# Return the distance matrix of g, with the specified metric.
def get_distance_matrix(g, verbose=True, weights=None):
    if verbose:
        print('[distance_matrix] Computing distance matrix')

    X = get_shortest_path_distance_matrix(g, weights=weights)

    # Just to make sure, symmetrize the matrix.
    X = (X + X.T) / 2

    # Force diagonal to zero
    X[range(X.shape[0]), range(X.shape[1])] = 0

    if verbose:
        print('[distance_matrix] Done!')

    return X
