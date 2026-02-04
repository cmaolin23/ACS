import networkx as nx
import pickle
import os, sys
import pathlib


def main():
    p = os.path.dirname(os.path.dirname((os.path.abspath('__file__'))))
    if p not in sys.path:
        sys.path.append(p)
    path=os.getcwd()+".\\CAI\\data\\amazon"

    print(f"Load com_amazon edges")
    if(os.path.exists(path + '//edges.pkl') == False):
        untar_amazon_data('amazon')
    with open(path+'//edges.pkl', 'rb') as file:
        new_edge = pickle.load(file)
    graph = nx.from_edgelist(new_edge)

    print(f"Load com_amazon cmty")
    with open(path+'//comms.pkl', 'rb') as file:
        com_list = pickle.load(file)
    
    print("------Community List:------")
    print(f"# of nodes: {graph.number_of_nodes()}, # of edges: {graph.number_of_edges()}")
    print([len(com) for com in com_list if len(com) >= 100])
    for com in com_list: print(f"min: {min(com)}, max: {max(com)}, size: {len(com)}")


def untar_amazon_data(name):
    """Load the snap comm datasets."""
    print(f'Untar {name} edge')
    root = pathlib.Path(os.getcwd()+'.\\CAI\\data\\amazon')
    with open(root / f'com-{name}.ungraph.txt', 'rt') as fh:
        edges = fh.read().strip().split('\n')[4:]
    edges = [[int(i) for i in e.split()] for e in edges]
    nodes = {i for x in edges for i in x}
    mapping = {u: i for i, u in enumerate(sorted(nodes))}
    edges = [[mapping[u], mapping[v]] for u, v in edges]
    print(f'Untar {name} cmty')
    with open(root / f'com-{name}.top5000.cmty.txt', 'rt') as fh:
        comms = fh.readlines()
    comms = [[mapping[int(i)] for i in x.split()] for x in comms]
    print(comms)
    with open(root/'edges.pkl', 'wb') as file:
        pickle.dump(edges, file)
    with open(root/'comms.pkl', 'wb') as file:
        pickle.dump(comms, file)
    with open(root/'mapping.pkl', 'wb') as file:
        pickle.dump(mapping, file)
    

if __name__ == "__main__":
    main()