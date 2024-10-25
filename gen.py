import graph_processing as gp
import os, glob, sys
import s_gd2  # graph drawing algorithm
from tqdm import tqdm  # progress bar

def sgd2_layout(tlp_graph, seed):
    AM = gp.tlp2AM(tlp_graph)
    I, J = gp.AM2edgesIdx(AM)
    I = I.astype("int32")
    J = J.astype("int32")
    pos2d = s_gd2.layout(I, J, random_seed=seed)
    return pos2d

def load_and_layout(seed_for_all, data_path="./data/rome_graphs"):
    files = glob.glob(data_path+"/*.tlpb.gz")
    for f in tqdm(files):
        tlpg = gp.load_graph(f)
        pos = sgd2_layout(tlpg, seed_for_all)
        gp.writeTLPlayout(tlpg, "sgd2_layout", pos)
        gp.save_graph(tlpg, f)

if(__name__ == "__main__"):
    seed = int(sys.argv[1])
    load_and_layout(seed)
