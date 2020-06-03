import networkx as nx
import numpy as np
import gzip
from pickle import load


def unpack(ipFile):
    """
    Unpack nodes, input and ouput adjacency matrices
    from the pickle.
    """
    f = gzip.open(ipFile, 'rb')
    nodes, ipAdj, opAdj = load(f)
    return nodes, ipAdj, opAdj


def plot3dGraph(adj, nodes):
    """
    Plot graph based on adjacency matrix 'adj' and 'nodes'.
    Uses Mayavi and Networkx packages
    """
    from mayavi import mlab
    xyz = nodes[0:3, :].T
    G = nx.from_numpy_matrix(np.tril(adj))
    scalars = np.array(G.nodes())
    mlab.figure(1, bgcolor=(1, 1, 1))
    mlab.clf()
    pts = mlab.points3d(xyz[:, 0], xyz[:, 1], xyz[:, 2], scalars,
                        scale_factor=1, scale_mode='none',
                        color=(0.5, 0.5, 0.75), resolution=20)
    pts.mlab_source.dataset.lines = np.array(G.edges())
    tube = mlab.pipeline.tube(pts, tube_radius=0.4)
    mlab.pipeline.surface(tube, color=(1, 0, 0))
    pts.mlab_source.update()
    mlab.show()
