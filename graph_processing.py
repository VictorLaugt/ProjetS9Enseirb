import networkx as nx
import numpy as np
from tulip import tlp
import random
from scipy.sparse.csgraph import floyd_warshall
from matplotlib import pyplot as plt
from matplotlib.collections import *
import matplotlib.patches as patches
from matplotlib import transforms

def load_graph(filepath):
    g = tlp.loadGraph(filepath)
    if(not tlp.ConnectedTest.isConnected(g)):
        params = tlp.getDefaultPluginParameters('Make Connected', g)
        success = g.applyAlgorithm('Make Connected', params)
    return g

def save_graph(tlpg, path):
    success = tlp.saveGraph(tlpg, path)
    return success

def gen_one_graph(graphGen, params, N_min, N_max):
    g = tlp.importGraph(graphGen, params)
    while(not graphIsValid(g, N_min, N_max)):
        g = tlp.importGraph(graphGen, params)  
    return g

def tlp2AM(g, increased=-1):
    AM_size = max(g.numberOfNodes(), int(increased))
    AM = np.zeros((AM_size, AM_size))
    for n in g.getNodes():
        for neighbor in g.getInOutNodes(n):
            AM[n.id][neighbor.id] = 1
    return AM

def writeTLPlayout(tlpg, propname, propvalues):
    assert len(propvalues) == tlpg.numberOfNodes()
    layout_prop = tlpg.getLayoutProperty(propname)
    i = 0
    for n in tlpg.nodes():
        layout_prop.setNodeValue(n, tuple(propvalues[i]))
        i+=1
    return

def readTLPlayout(tlpg, propname):
    N = tlpg.numberOfNodes()
    node_pos = np.empty((N, 2))
    layoutProp = tlpg.getLayoutProperty(propname)
    for i, n in enumerate(tlpg.nodes()):
        node_pos[i] = [layoutProp.getNodeValue(n)[0], layoutProp.getNodeValue(n)[1]]
    return node_pos
    

def graph2edges(gname, gpath="/data/lgiovannange/DATA/snap/graph_tlp"):
    edges = []
    try:
        g = tlp.loadGraph(f"{gpath}/{gname}.tlpb.gz")
        for e in g.getEdges():
            edges.append((g.source(e).id, g.target(e).id))
    except:
        print("no graph found")
    return edges
    
def AM2DM(AM, fill_value=0):
    DM = floyd_warshall(AM)
    DM[DM == np.inf] = fill_value
    return DM

def applyTlpLayoutAlgorithm(g, algo, propertyName, params_modifs={}):
    params = tlp.getDefaultPluginParameters(algo, g)
    for k, v in params_modifs.items():
        params[k] = v
    res = g.getLayoutProperty(propertyName)
    success, string = g.applyLayoutAlgorithm(algo, res, params)
    if(not success):
        return "fail layout"
    return res

def applyTlpAlgorithm(g, algo, propertyName, params_modifs={}):
    params = tlp.getDefaultPluginParameters(algo, g)
    for k, v in params_modifs.items():
        params[k] = v
    res = g.getDoubleProperty(propertyName)
    success = g.applyDoubleAlgorithm(algo, res, params)
    return res


def graphIsValid(g, N_min, N_max):
    if(N_min is not None and g.numberOfNodes() < N_min):
        return False
    if(N_max is not None and g.numberOfNodes() > N_max):
        return False
    if(not tlp.ConnectedTest.isConnected(g)):
        return False
    return True


def AM2nx(AM):
    G = nx.Graph()
    N = AM.shape[0]
    for i in range(N):
        for j in range(i+1, N, 1):
            if(AM[i][j] == 1):
                G.add_edge(i,j)
    return G    


def AM2tlp(AM, mask=None):
    N = AM.shape[0]
    if(mask is None):
        mask = np.ones((N, 1))
    g = tlp.newGraph()
    real_n = int(np.sum(mask))
    nodes = g.addNodes(real_n)
    real_i, real_j = 0, 0
    id_mapping = {}
    for i in range(N):
        if(mask[i] == 1):
            id_mapping[real_i] = i
            for j in range(i, N, 1):
                if(AM[i][j] == 1):
                    g.addEdge(nodes[real_i], nodes[real_j])
                if(mask[j] == 1):
                    real_j += 1
            real_i += 1
            real_j = real_i
    return g, id_mapping


def AM2edgesIdx(AM):
    return np.nonzero(AM == 1)

def shrinkDM(DM, mask):
    real_n = int(np.sum(mask))
    res = np.zeros((real_n, real_n))
    real_i, real_j = 0,0
    for i in range(DM.shape[0]):
        if(mask[i] == 1):
            for j in range(DM.shape[1]):
                if(mask[j] == 1):
                    res[real_i][real_j] = DM[i][j]
                    real_j +=1
            real_i +=1
            real_j = 0
    return res

def pos2DM(pos2d, mask=None):
    diff = np.expand_dims(pos2d, 0) - np.expand_dims(pos2d, 1)
    squared_dist = np.sum(np.square(diff), axis=-1)
    DM = np.sqrt(squared_dist)
    if(mask is not None):
        matrice_mask = mask*mask.T
        DM*= matrice_mask
    return DM

def getBB(pos, rad):
    l = (pos[:, 0] - rad).min()
    r = (pos[:, 0] + rad).max()
    b = (pos[:, 1] - rad).min()
    t = (pos[:, 1] + rad).max()
    return l, r, b, t


def draw_graph(tlpg, node_radiuses=[], node_colors=None, ax=None, layoutPropName="sgd2_layout", edge_color=(0,0,0,0.5), edge_width=1, axis_off=False):    
    
    if(ax is None):
        ax = plt.gca()

    N = tlpg.numberOfNodes()
    node_pos = readTLPlayout(tlpg, layoutPropName)
    
    line_patches = []
    for e in tlpg.getEdges():
        src = tlpg.source(e).id
        tgt = tlpg.target(e).id
        (x1, y1) = node_pos[src]
        (x2, y2) = node_pos[tgt]
        line = ((x1, y1),(x2, y2))
        line_patches.append(line)
    ax.add_collection(LineCollection(line_patches, linewidths=edge_width, colors=edge_color))

    if(type(node_radiuses) == int or type(node_radiuses) == float):
        node_radiuses = np.full((N,1), node_radiuses)

    if(node_colors is None):
        node_colors = "red"

    nodes_col = EllipseCollection(widths=node_radiuses/2, heights=node_radiuses/2, angles=np.zeros(N), offsets=node_pos,transOffset=ax.transData, units="x",facecolors=node_colors, antialiased=False, linewidth=0, zorder=2)
    ax.add_collection(nodes_col)

    l, r, b, t = getBB(node_pos, node_radiuses)
    ax.set_xlim(l, r)
    ax.set_ylim(b, t)
    if(axis_off):
        ax.axis("off")
    return ax