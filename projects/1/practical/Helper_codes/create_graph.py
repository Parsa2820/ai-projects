import networkx as nx
import os
import numpy as np


def safe_dir_creator(dirn='tests4VC'):
    try:
        os.mkdir('./'+dirn,)
    except:
        pass

def adj2text(path2file, fn, n, adj_mat):
    f = open(path2file + fn, 'w')
    f.write(str(n) + '\n')
    for i in range(len(adj_mat)):
        s = ""
        for j in range(len(adj_mat)):
            adj_cell = adj_mat[i, j]
            s += str(int(adj_cell)) + ', '
        s += '\n'
        f.write(s)
    f.close()

def create_graph(mode='BA', n=100, m2n = 2):
    """
    mode: {BA: Barabasi-Albert scale free network, ER: Erdos-Renyi random network}
    n : number of nodes

    output -> adj_mat
    """
    if mode == 'BA':
        G = nx.generators.random_graphs.barabasi_albert_graph(n=n, m=m2n)
    if mode == 'ER':
        m = n * m2n
        G = nx.gnm_random_graph(n, m)

    ## Check how many edges does the network have if you want ##
    print(len(G.edges))
  
    adj_mat = nx.convert_matrix.to_numpy_matrix(G,)
    return adj_mat

def main(path2file='./tests4VC/', n = 100, m2n = 2):
    for i in range(10):
        fn = 'test'+str(i+1)
        if i % 2 == 0:
            adj_mat = create_graph(n = n, mode='ER', m2n=m2n)
        else:
            adj_mat = create_graph(n = n, m2n=m2n)
        adj2text(path2file, fn, adj_mat.shape[0], adj_mat)


if __name__ == '__main__':
    m2n = 2
    n = 100
    safe_dir_creator()
    main(n=n, m2n=m2n)