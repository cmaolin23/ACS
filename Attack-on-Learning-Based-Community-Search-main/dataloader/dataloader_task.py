import networkx as nx
import random
import numpy as np
import pickle
import os
import sys

from utils.utils import RandomWalker,common_k_hop_neighbors
from data.preprocess_dblp import untar_dblp_data
from data.preprocess_football import untar_football_data
from data.preprocess_facebook import untar_facebook_data
from data.preprocess_dblp import untar_dblp_data
from data.preprocess_email import untar_email_data
from data.preprocess_amazon import untar_amazon_data
from data.preprocess_lj import untar_lj_data
from data.preprocess_cora import untar_cora_data
from data.preprocess_citeseer import untar_citeseer_data
from data.preprocess_youtube import untar_youtube_data

from sample.RawData import RawGraphWithCommunity
from sample.RawData_cgnp import get_CGNP_Rawdata


def sampgraph_BFS(source_ls,max_size,graph,node_list_all,index):
        # graph
        sub = nx.Graph()
        sub_new = nx.Graph()
        node_id_dict = dict()
        edge_list = []
        h_hops_neighbor = []

        max_size = min(max_size, len(node_list_all))
        node_cnt = 0
        source = source_ls[index]
        index = index + 1
        h_hops_neighbor = []
        h_hops_neighbor.append(source)
        node_id_dict[int(source)] = node_cnt
        pos = 0
        # pos is a pointer used to track the current node traversal position
        while (pos < len(h_hops_neighbor)) and (pos < max_size) and (
                len(h_hops_neighbor) < max_size):
            cnode = h_hops_neighbor[pos]

            nei=graph[cnode]
            if len(nei)>200:
                nei=random.sample(nei.keys(),200)
            for nb in nei:
                # Add unvisited nodes
                if (nb not in h_hops_neighbor) and (nb in node_list_all): 
                    node_cnt = node_cnt + 1
                    h_hops_neighbor.append(nb)
                    node_id_dict[int(nb)] = node_cnt
            pos = pos + 1
        # form a subgraph
        sub = graph.subgraph(h_hops_neighbor)
        sub_node_list = sub.nodes()

        subedge_list = sub.edges()
        for idx, ege in enumerate(subedge_list):
            src = ege[0]
            dst = ege[1]
            edge_list.append((node_id_dict[src], node_id_dict[dst]))
        # subgraph with new num from 1 to len(sub)
        sub_new.add_edges_from(edge_list)

        # keys = list(node_id_dict.keys())
        # values = list(node_id_dict.values())
        # # shuffle
        # random.shuffle(values)
        # node_id_dict = dict(zip(keys, values))
        return sub_node_list,node_id_dict,sub_new,index,len(h_hops_neighbor)


def load_manul(args):
    p = os.path.dirname(os.path.dirname((os.path.abspath('__file__'))))
    if p not in sys.path:
        sys.path.append(p)
    path = os.path.join(args.data_dir, "manul")
    print(f"Load manul edges")
    if(os.path.exists(os.path.join(path, 'edges.pkl')) == False):
        untar_football_data('manul')
    with open(os.path.join(path, 'edges.pkl'), 'rb') as file:
        new_edge = pickle.load(file)
    graph = nx.from_edgelist(new_edge)
 
    print("Load {} com cmty".format(args.data_set))
    with open(os.path.join(path, 'comms.pkl'), 'rb') as file:
        com_list = pickle.load(file)
    return graph,com_list

def load_football(args):
    p = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if p not in sys.path:
        sys.path.append(p)
    path = os.path.join(args.data_dir, "Football")
    print("Load football edges")
    if not os.path.exists(os.path.join(path, 'edges.pkl')):
        untar_football_data('football')
    with open(os.path.join(path, 'edges.pkl'), 'rb') as file:
        new_edge = pickle.load(file)
    graph = nx.from_edgelist(new_edge)
    print("Load {} com cmty".format(args.data_set))
    with open(os.path.join(path, 'comms.pkl'), 'rb') as file:
        com_list = pickle.load(file)
    return graph, com_list

def load_email(args):
    p = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if p not in sys.path:
        sys.path.append(p)
    path = os.path.join(args.data_dir, "email-Eu")
    print("Load email-Eu edges")
    if not os.path.exists(os.path.join(path, 'edges.pkl')):
        untar_email_data('email-Eu-core')
    with open(os.path.join(path, 'edges.pkl'), 'rb') as file:
        new_edge = pickle.load(file)
    graph = nx.from_edgelist(new_edge)
    print("Load email-Eu cmty")
    with open(os.path.join(path, 'comms.pkl'), 'rb') as file:
        com_list = pickle.load(file)
    return graph, com_list

def load_facebook(args):
    p = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if p not in sys.path:
        sys.path.append(p)
    path = os.path.join(args.data_dir, "Facebook")
    print("Load facebook edges")
    if not os.path.exists(os.path.join(path, 'edges.pkl')):
        untar_facebook_data('facebook_all')
    with open(os.path.join(path, 'edges.pkl'), 'rb') as file:
        new_edge = pickle.load(file)
    graph = nx.from_edgelist(new_edge)
    print("Load facebook cmty")
    with open(os.path.join(path, 'comms.pkl'), 'rb') as file:
        com_list = pickle.load(file)
    return graph, com_list

def load_dblp(args):
    p = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if p not in sys.path:
        sys.path.append(p)
    path = os.path.join(args.data_dir, "DBLP")
    print("Load com_dblp edges")
    if not os.path.exists(os.path.join(path, 'edges.pkl')):
        untar_dblp_data('dblp')
    with open(os.path.join(path, 'edges.pkl'), 'rb') as file:
        new_edge = pickle.load(file)
    graph = nx.from_edgelist(new_edge)
    print("Load com_dblp cmty")
    with open(os.path.join(path, 'comms.pkl'), 'rb') as file:
        com_list = pickle.load(file)
    return graph, com_list

def load_amazon(args):
    p = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if p not in sys.path:
        sys.path.append(p)
    path = os.path.join(args.data_dir, "amazon")
    print("Load com_amazon edges")
    if not os.path.exists(os.path.join(path, 'edges.pkl')):
        untar_amazon_data('amazon')
    with open(os.path.join(path, 'edges.pkl'), 'rb') as file:
        new_edge = pickle.load(file)
    graph = nx.from_edgelist(new_edge)
    print("Load com_amazon cmty")
    with open(os.path.join(path, 'comms.pkl'), 'rb') as file:
        com_list = pickle.load(file)
    return graph, com_list

def load_lj(args):
    p = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if p not in sys.path:
        sys.path.append(p)
    path = os.path.join(args.data_dir, "lj")
    print("Load com_lj edges")
    if not os.path.exists(os.path.join(path, 'edges.pkl')):
        untar_lj_data('lj')
    with open(os.path.join(path, 'edges.pkl'), 'rb') as file:
        new_edge = pickle.load(file)
    graph = nx.from_edgelist(new_edge)
    print("Load com_lj cmty")
    with open(os.path.join(path, 'comms.pkl'), 'rb') as file:
        com_list = pickle.load(file)

    return graph, com_list


def load_youtube(args):
    p = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if p not in sys.path:
        sys.path.append(p)
    path = os.path.join(args.data_dir, "youtube")
    print("Load com_youtube edges")
    if not os.path.exists(os.path.join(path, 'edges.pkl')):
        untar_youtube_data('youtube')
    with open(os.path.join(path, 'edges.pkl'), 'rb') as file:
        new_edge = pickle.load(file)
    graph = nx.from_edgelist(new_edge)
    print("Load com_youtube cmty")
    with open(os.path.join(path, 'comms.pkl'), 'rb') as file:
        com_list = pickle.load(file)

    return graph, com_list

def load_cora(args):
    p = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if p not in sys.path:
        sys.path.append(p)
    path = os.path.join(args.data_dir, "cora")
    print("Load cora edges")
    if not os.path.exists(os.path.join(path, 'edges.pkl')):
        untar_cora_data('cora')
    with open(os.path.join(path, 'edges.pkl'), 'rb') as file:
        new_edge = pickle.load(file)
    graph = nx.from_edgelist(new_edge)
    print("Load {} com cmty".format(args.data_set))
    with open(os.path.join(path, 'comms.pkl'), 'rb') as file:
        com_list = pickle.load(file)
    return graph, com_list


def load_citeseer(args):
    p = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if p not in sys.path:
        sys.path.append(p)
    path = os.path.join(args.data_dir, "citeseer")
    print("Load citeseer edges")
    if not os.path.exists(os.path.join(path, 'edges.pkl')):
        untar_citeseer_data('citeseer')
    with open(os.path.join(path, 'edges.pkl'), 'rb') as file:
        new_edge = pickle.load(file)
    graph = nx.from_edgelist(new_edge)
    print("Load {} com cmty".format(args.data_set))
    with open(os.path.join(path, 'comms.pkl'), 'rb') as file:
        com_list = pickle.load(file)
    return graph, com_list


def sub_com(com_list,node_list,node_id_dict):
    com_list_ = [[] for _ in range(len(com_list))]
    for idx, com in enumerate(com_list):
        for node in com:
            if node in node_list:
                com_list_[idx].append(node_id_dict[node])
    while ([] in com_list_): com_list_.remove([])

    query_index = dict()
    for community in com_list_:
        for node in community:
            if node not in query_index:
                query_index[node] = set(community)
            else:
                query_index[node] = set(community)

    return com_list_,query_index


def add_noise_to_graph(graph, delete_ratio, add_ratio):
    graph1=graph.copy()
    num_edges = len(graph1.edges)
    num_nodes = len(graph1.nodes)

    # compute the number of edges to delete and add
    num_delete = int(num_edges * delete_ratio)
    num_add = int(num_edges * add_ratio)

    # randomly delete edges
    edges = list(graph1.edges)
    if(delete_ratio>0):
        for _ in range(num_delete):
            edge_to_delete = random.choice(edges)
            graph1.remove_edge(*edge_to_delete)
            edges.remove(edge_to_delete)

           
            # print("remove edge: ", str(edge_to_delete))
            # # connected graph
            # if not nx.is_connected(graph):
            #     graph.add_edge(*edge_to_delete)
            # else:
            #     edges.remove(edge_to_delete)

    # randomly add edges
    if(add_ratio>0):
        nodes = list(graph1.nodes)
        pos=0
        while pos<num_add:
            node1, node2 = random.sample(nodes, 2)
            # print(node1,node2)
            if not graph1.has_edge(node1, node2):
                graph1.add_edge(node1, node2)
                graph1.add_edge(node2, node1)
                pos=pos+1

    return graph1


def remove_boundary_edge(graph,q,query_index, delete_ratio):
    import random
    com_q = query_index[q]
    # print(com_q)
    boundary_edges = set()
    for node in com_q:
        for neighbor in graph.neighbors(node):
            if neighbor not in com_q:
                boundary_edges.add((node, neighbor))
        
    num_edges = len(boundary_edges)
    # compute the number of edges to delete and add
    num_delete = int(num_edges * delete_ratio)
    boundary_edges = list(boundary_edges)

    for _ in range(num_delete):
        edge_to_delete = random.choice(boundary_edges)
        graph.remove_edge(*edge_to_delete)
        boundary_edges.remove(edge_to_delete)
    return graph



def remove_relaxed_boundary_edge(graph,q,query_index, delete_ratio):
    import random
    from collections import deque
    def get_k_hop_neighbors(graph, com_q, k):
        k_hop_neighbors = set()
        
        for node in com_q:
            visited = set([node])
            queue = deque([(node, 0)])
            
            while queue:
                current_node, current_distance = queue.popleft()
                
                if current_distance < k:
                    for neighbor in graph.neighbors(current_node):
                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append((neighbor, current_distance + 1))
                            # if neighbor not in com_q and current_distance==k-1:
                            if neighbor not in com_q:
                                k_hop_neighbors.add(neighbor)
        return random.sample(k_hop_neighbors, min(int(len(com_q)*0),len(k_hop_neighbors)))
        # return k_hop_neighbors


    # com_q_ = query_index[q]
    # com_q=(get_k_hop_neighbors(graph, com_q_, 1))

    com_q = query_index[q]
    com_q=set(random.sample(com_q,int(len(com_q)*1))).union(get_k_hop_neighbors(graph, com_q, 2))

    boundary_edges = set()
    for node in com_q:
        for neighbor in graph.neighbors(node):
            # if neighbor not in com_q and neighbor not in com_q_:
            if neighbor not in com_q:
                boundary_edges.add((node, neighbor))
        
    num_edges = len(boundary_edges)
    # compute the number of edges to delete and add
    num_delete = int(num_edges * delete_ratio)
    boundary_edges = list(boundary_edges)

    for _ in range(num_delete):
        edge_to_delete = random.choice(boundary_edges)
        graph.remove_edge(*edge_to_delete)
        boundary_edges.remove(edge_to_delete)
    return graph



def load_graphs(args, num_tasks, label_mode, mode, train_ratio=0.6, valid_ratio=0.2):
    print(mode + " dataset generation...")
    if(args.data_set=="football"):
        graph,com_list=load_football(args)
    elif(args.data_set=="email"):
        graph,com_list=load_email(args)
    elif(args.data_set=="facebook"):
        graph,com_list=load_facebook(args)
    elif(args.data_set=="dblp"):
        graph,com_list=load_dblp(args)
    elif(args.data_set=="amazon"):
        graph,com_list=load_amazon(args)
    elif(args.data_set=="lj"):
        graph,com_list=load_lj(args)
    elif(args.data_set=="cora"):
        graph,com_list=load_cora(args)
    elif(args.data_set=="citeseer"):
        graph,com_list=load_citeseer(args)
    elif(args.data_set=="youtube"):
        graph,com_list=load_youtube(args)
    else:
        graph,com_list=load_manul(args)

   
    # # print some statistics
    # lengths = [len(com) for com in com_list]
    # average_length = sum(lengths) / len(lengths)
    # max_length = max(lengths)

    # print(f"average_length: {average_length}")
    # print(f"max_length: {max_length}")
    
    # print("------Community List:------")
    node_list_all = range(len(graph))
    node_list=node_list_all 
    raw_data_list = list()

    # node from 0 to len(graph)
    node_list_all,node_id_dict,graph_new=Form_graph(graph,node_list_all)


    com_list_ = [[] for _ in range(len(com_list))]
    for idx, com in enumerate(com_list):
        for node in com:
            com_list_[idx].append(node_id_dict[node])
    while ([] in com_list_): com_list_.remove([])
    com_list=com_list_
    graph_new_=graph_new

    # Whether the training set, validation set and test set contain the same community
    if label_mode == "disjoint":
        random.shuffle(com_list)
        if mode=='train':
            com_list = com_list[0:int(train_ratio * len(com_list))]
        elif mode=='valid':
            com_list = com_list[int(train_ratio * len(com_list)):int((train_ratio+valid_ratio) * len(com_list))]
        elif mode=='test':
            com_list = com_list[int((train_ratio+valid_ratio) * len(com_list)):]

    elif label_mode=='shared':
        random.shuffle(node_list_all)
        if mode=='train':
            node_list_=list(node_list_all)[0:int(train_ratio * len(node_list_all))]
        elif mode=='valid':
            node_list_=list(node_list_all)[int(train_ratio*len(node_list_all)):int((train_ratio+valid_ratio)*len(node_list_all))]
        elif mode=='test':
            node_list_=list(node_list_all)[int((train_ratio+valid_ratio)*len(node_list_all)):]

    #  sample queries
    #  Filter the number of nodes in the community
    communities = [com for com in com_list if len(com)>3]
    queries=[]
    max_size = args.subgraph_size
    while len(queries)<num_tasks:
        # print("sample queries...")
        if(args.label_mode=='disjoint'):
            communitiy= random.sample(communities, k=1)
            query= random.sample(communitiy[0], k=1)[0]
        else:
            #   sample from node_list
            query= random.sample(node_list_, k=1)[0]

        if(args.subgraph_size != -1):
                index=0
                # print("sample subgraph...")
                sub_node_list,node_id_dict,sub_new,index,num_g=sampgraph_BFS([query],max_size,graph_new_,node_list_all,index)
                if(num_g<max_size):
                    # print("resample...")
                    continue
                node_list=sub_node_list
                graph_new=sub_new
        
        communities_,query_index=sub_com(communities,node_list,node_id_dict)

        if(node_id_dict[query] in query_index.keys() and len(query_index[node_id_dict[query]])>3):
            queries.append(node_id_dict[query])
            # print(query_index[node_id_dict[query]])
            feats = np.array([[] for _ in node_list])
            if args.noise:
                # print("adding noise to graph...")
                # common_k_hop_neighbors(graph_new,query_index[node_id_dict[query]],1)
                graph_n=add_noise_to_graph(graph_new, args.delete_ratio, args.add_ratio)
                # graph_n=remove_boundary_edge(graph_n,node_id_dict[query],query_index,1)
                # graph_n=remove_relaxed_boundary_edge(graph_n,node_id_dict[query],query_index,0.5)
                raw_data=RawGraphWithCommunity(args,graph_n, communities_, feats, query_index ,node_list,mode)
            else:
                # print(graph_new.number_of_nodes(),graph_new.number_of_edges())
                raw_data=RawGraphWithCommunity(args,graph_new, communities_, feats, query_index ,node_list,mode)
            raw_data_list.append(raw_data)

    num_feat = 0
    # print(queries)
    return raw_data_list, num_feat,queries



def Form_graph(graph,node_list_all):
        # graph
        sub = nx.Graph()
        sub_new = nx.Graph()
        node_id_dict = dict()
        edge_list = []

        for idx,node in enumerate(node_list_all):
            node_id_dict[node]=idx
        # form a graph
        sub = graph.subgraph(node_list_all)
        sub_node_list = sub.nodes()
        subedge_list = sub.edges()
        for idx, ege in enumerate(subedge_list):
                src = ege[0]
                dst = ege[1]
                edge_list.append((node_id_dict[src], node_id_dict[dst]))
        # graph with new num from 1 to len(sub)
        sub_new.add_edges_from(edge_list)

        return sub_node_list,node_id_dict,sub_new


from utils.utils import TaskData
'''load data, get tasks and get queries'''
def load_data_and_get_tasks(args):
    num_pos, num_neg= args.num_pos,args.num_neg
    batch_size=args.batch_size
    
    raw_data_train, node_feat,queries_list_train = load_graphs(args, args.task_num, args.label_mode,'train')
    raw_data_valid, node_feat, queries_list_valid= load_graphs(args, args.valid_task_num,  args.label_mode,'valid')
    raw_data_test, node_feat, queries_list_test= load_graphs(args, args.test_task_num, args.label_mode,'test')

    train_query_data = [raw_data.get_query_data(query, num_pos, num_neg) for raw_data,query in zip(raw_data_train,queries_list_train)]
    valid_query_data = [raw_data.get_query_data(query, -1, -1)for raw_data,query in zip(raw_data_valid,queries_list_valid)]
    test_query_data = [raw_data.get_query_data(query, -1, -1)for raw_data,query in zip(raw_data_test,queries_list_test)]

    train_tasks=TaskData(batch_size,train_query_data)
    valid_tasks=TaskData(batch_size,valid_query_data)
    test_tasks=TaskData(batch_size,test_query_data) 

    return train_tasks, valid_tasks, test_tasks, node_feat


