from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import torch
import random
import numpy as np
from torch_geometric.utils import from_networkx

from scipy import sparse as sp
import os
import hashlib
from torch.utils.data import Dataset
from model.Vmodel.CAF import Graph,GraphConv_m

try:
    import torch_cluster  # noqa
    random_walk = torch.ops.torch_cluster.random_walk
except ImportError:
    random_walk = None


class CSQueryData(Data):
    def __init__(self, x, edge_index, y=torch.tensor([]), query=None, pos=None, neg=None, raw_feature=None):
        super(CSQueryData, self).__init__()
        # self.graph = graph
        self.x = x  # feature
        self.edge_index = edge_index
        self.raw_feature = raw_feature
        self.y = y  # ground truth
        self.query = query
        self.pos = pos
        self.neg = neg
        self.mask = torch.zeros(self.y.shape) # used as the loss weight
        self.mask[pos] = 1.0
        self.mask[neg] = 1.0
        self.mask_pos = torch.zeros(self.y.shape) 
        self.mask_neg = torch.zeros(self.y.shape) 
        self.mask_unl = torch.ones(self.y.shape) 
        self.mask_pos[pos] = 1
        self.mask_neg[neg] = 1
        self.mask_unl[pos] = 0
        self.mask_unl[neg] = 0
        # self.query_index = query_index


class CSQueryData_ics(Data):
    def __init__(self,graph,query_index, x, edge_index, y=torch.tensor([]), query=None, pos=None, neg=None, raw_feature=None):
        super(CSQueryData_ics, self).__init__()
        self.graph = graph
        self.x = x  # feature
        self.edge_index = edge_index
        self.raw_feature = raw_feature
        self.y = y  # ground truth
        self.query = query
        self.pos = pos
        self.neg = neg
        self.mask = torch.zeros(self.y.shape) # used as the loss weight
        self.mask[pos] = 1.0
        self.mask[neg] = 1.0
        self.mask_pos = torch.zeros(self.y.shape) 
        self.mask_neg = torch.zeros(self.y.shape) 
        self.mask_unl = torch.ones(self.y.shape) 
        self.mask_pos[pos] = 1
        self.mask_neg[neg] = 1
        self.mask_unl[pos] = 0
        self.mask_unl[neg] = 0
        self.query_index = query_index

class CSQueryData_aug(Data):
    def __init__(self, x, edge_index,edge_index_aug, y=torch.tensor([]), query=None, pos=None, neg=None, raw_feature=None):
        super(CSQueryData_aug, self).__init__()
        self.x = x  # feature
        self.edge_index = edge_index
        self.edge_index_aug = edge_index_aug

        self.raw_feature = raw_feature
        self.y = y  # ground truth
        self.query = query
        self.pos = pos
        self.neg = neg
        self.mask = torch.zeros(self.y.shape) # used as the loss weight
        self.mask[pos] = 1.0
        self.mask[neg] = 1.0
        # self.query_index = query_index
        self.mask_pos = torch.zeros(self.y.shape) 
        self.mask_neg = torch.zeros(self.y.shape) 
        self.mask_unl = torch.ones(self.y.shape) 
        self.mask_pos[pos] = 1
        self.mask_neg[neg] = 1
        self.mask_unl[pos] = 0
        self.mask_unl[neg] = 0


class CSQueryData_caf(Data):
    def __init__(self,query_index, x, edge_index, y=torch.tensor([]), query=None, pos=None, neg=None, raw_feature=None):
        super(CSQueryData_caf, self).__init__()
        # self.graph = graph
        self.x = x  # feature
        self.edge_index = edge_index
        self.raw_feature = raw_feature
        self.y = y  # ground truth
        self.query = query
        self.pos = pos
        self.neg = neg
        self.mask = torch.zeros(self.y.shape) # used as the loss weight
        self.mask[pos] = 1.0
        self.mask[neg] = 1.0
        self.mask_pos = torch.zeros(self.y.shape) 
        self.mask_neg = torch.zeros(self.y.shape) 
        self.mask_unl = torch.ones(self.y.shape) 
        self.mask_pos[pos] = 1
        self.mask_neg[neg] = 1
        self.mask_unl[pos] = 0
        self.mask_unl[neg] = 0
        self.query_index = query_index


class TaskData(object):
    def __init__(self,batch_size, all_queries_data, seed=4):
        self.all_queries_data = all_queries_data
        self.seed = seed
        self.batch_size=batch_size

    def get_batch(self):
        batch=[]
        loader = DataLoader(self.all_queries_data, batch_size=1, shuffle=False,collate_fn=self.my_collate_fn)
        for load in loader:
            batch.append(load)
        return batch
    
    def my_collate_fn(self,batch):
        return batch
    

class TaskData_cgnp(object):
    def __init__(self, all_queries_data, num_shots, seed=4):
        self.all_queries_data = all_queries_data
        self.num_shots = num_shots
        self.seed = seed
        self.support_data, self.query_data = \
            self._support_query_split()
        self.num_support, self.num_query = len(self.support_data), len(self.query_data)

    def _support_query_split(self):
        random.seed(20)
        random.shuffle(self.all_queries_data)
        support_data, query_data = self.all_queries_data[: self.num_shots], self.all_queries_data[self.num_shots:]
        return support_data, query_data

    def get_batch(self):
        loader = DataLoader(self.all_queries_data, batch_size=len(self.all_queries_data), shuffle=False)
        return next(iter(loader))

    def custom_collate_fn(self,batch):
            return batch  # 直接返回批处理数据

    
    def get_support_batch(self, isMaml=False, isSup=False,isRep=False):
        if isMaml:
            support_loader = DataLoader(self.support_data, batch_size=self.num_support, shuffle=False)
        elif isSup:
            support_loader = DataLoader(self.support_data, batch_size=self.num_support, shuffle=False)
        elif isRep:
            support_loader = DataLoader(self.support_data, batch_size=self.num_support, shuffle=False)
        else:
            support_loader = DataLoader(self.support_data, batch_size=self.num_support, shuffle=False)
        return next(iter(support_loader))

    def get_query_batch(self, isMaml=False, isSup=False, isRep=False):
        if isMaml:
            query_loader = DataLoader(self.query_data, batch_size=self.num_query, shuffle=False)
        elif isSup:
            query_loader = DataLoader(self.query_data, batch_size=self.num_query, shuffle=False)
        elif isRep:
            query_loader = DataLoader(self.query_data, batch_size=self.num_query, shuffle=False)
        else:
            query_loader = DataLoader(self.query_data, batch_size=self.num_query, shuffle=False)
        return next(iter(query_loader))
    

def create_alias_table(area_ratio):
    l = len(area_ratio)
    accept, alias = [0] * l, [0] * l
    small, large = [], []
    area_ratio_ = np.array(area_ratio) * l
    for i, prob in enumerate(area_ratio_):
        if prob < 1.0:
            small.append(i)
        else:
            large.append(i)

    while small and large:
        small_idx, large_idx = small.pop(), large.pop()
        accept[small_idx] = area_ratio_[small_idx]
        alias[small_idx] = large_idx
        area_ratio_[large_idx] = area_ratio_[large_idx] - \
                                 (1 - area_ratio_[small_idx])
        if area_ratio_[large_idx] < 1.0:
            small.append(large_idx)
        else:
            large.append(large_idx)

    while large:
        large_idx = large.pop()
        accept[large_idx] = 1
    while small:
        small_idx = small.pop()
        accept[small_idx] = 1

    return accept, alias


def alias_sample(accept, alias):
    N = len(accept)
    i = int(np.random.random() * N)
    r = np.random.random()
    if r < accept[i]:
        return i
    else:
        return alias[i]


class RandomWalker:
    def __init__(self, G, p=1, q=1, use_rejection_sampling=False):

        self.G = G
        self.p = p
        self.q = q
        self.walks_per_node = 1
        self.context_size=10
        self.use_rejection_sampling = use_rejection_sampling


    def node2vec_walk(self, walk_length, start_node):
        
        G = self.G
        alias_nodes = self.alias_nodes
        alias_edges = self.alias_edges
        walk = [start_node]
        node_set=set()
        while (len(walk) < walk_length and len(node_set)<walk_length/2):
            cur = walk[-1]
            cur_nbrs = list(G.neighbors(cur))
            if len(cur_nbrs) > 0:
                if len(walk) == 1:
                    temp=cur_nbrs[alias_sample(alias_nodes[cur][0], alias_nodes[cur][1])]
                    node_set.add(temp)
                    walk.append(temp)
                else:
                    prev = walk[-2]
                    edge = (prev, cur)
                    next_node = cur_nbrs[alias_sample(alias_edges[edge][0],
                                                      alias_edges[edge][1])]
                    walk.append(next_node)
                    node_set.add(next_node)
            else:
                break

        return walk,node_set

    def node2vec_rejection_walk(self, walk_length, start_node):

        def rejection_sample(inv_p, inv_q, nbrs_num):
            upper_bound = max(1.0, max(inv_p, inv_q))
            lower_bound = min(1.0, min(inv_p, inv_q))
            shatter = 0
            second_upper_bound = max(1.0, inv_q)
            if (inv_p > second_upper_bound):
                shatter = second_upper_bound / nbrs_num
                upper_bound = second_upper_bound + shatter
            return upper_bound, lower_bound, shatter

        G = self.G
        alias_nodes = self.alias_nodes
        inv_p = 1.0 / self.p
        inv_q = 1.0 / self.q
        walk = [start_node]
        node_set=set()
        while len(walk) < walk_length:
            cur = walk[-1]
            cur_nbrs = list(G.neighbors(cur))
            if len(cur_nbrs) > 0:
                if len(walk) == 1:
                    temp=cur_nbrs[alias_sample(alias_nodes[cur][0], alias_nodes[cur][1])]
                    walk.append(temp)
                    node_set.add(temp)
                else:
                    upper_bound, lower_bound, shatter = rejection_sample(
                        inv_p, inv_q, len(cur_nbrs))
                    prev = walk[-2]
                    prev_nbrs = set(G.neighbors(prev))
                    while True:
                        prob = random.random() * upper_bound
                        if (prob + shatter >= upper_bound):
                            next_node = prev
                            break
                        next_node = cur_nbrs[alias_sample(
                            alias_nodes[cur][0], alias_nodes[cur][1])]
                        if (prob < lower_bound):
                            break
                        if (prob < inv_p and next_node == prev):
                            break
                        _prob = 1.0 if next_node in prev_nbrs else inv_q
                        if (prob < _prob):
                            break
                    walk.append(next_node)
                    node_set.add(next_node)
            else:
                break
        return walk,node_set

    def get_alias_edge(self, t, v):
        G = self.G
        p = self.p
        q = self.q

        unnormalized_probs = []
        for x in G.neighbors(v):
            weight = G[v][x].get('weight', 1.0)  # w_vx
            if x == t:  # d_tx == 0
                unnormalized_probs.append(weight / p)
            elif G.has_edge(x, t):  # d_tx == 1
                unnormalized_probs.append(weight)
            else:  # d_tx > 1
                unnormalized_probs.append(weight / q)
        norm_const = sum(unnormalized_probs)
        normalized_probs = [
            float(u_prob) / norm_const for u_prob in unnormalized_probs]

        return create_alias_table(normalized_probs)

    def preprocess_transition_probs(self):
        G = self.G
        alias_nodes = {}
        for node in G.nodes():
            unnormalized_probs = [G[node][nbr].get('weight', 1.0)
                                  for nbr in G.neighbors(node)]
            norm_const = sum(unnormalized_probs)
            normalized_probs = [
                float(u_prob) / norm_const for u_prob in unnormalized_probs]
            alias_nodes[node] = create_alias_table(normalized_probs)

        if not self.use_rejection_sampling:
            alias_edges = {}

            for edge in G.edges():
                alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
                if not G.is_directed():
                    alias_edges[(edge[1], edge[0])] = self.get_alias_edge(edge[1], edge[0])
                self.alias_edges = alias_edges

        self.alias_nodes = alias_nodes
        return


    def pos_sample(self, walk_length, query):

        start_nodes = torch.tensor(query)
        start_nodes=start_nodes.repeat(self.walks_per_node) 
        pyg_graph = from_networkx(self.G)

        rowptr, col, _ = pyg_graph.csr()
        rowptr=torch.cat(list(rowptr.values())).clone().detach()
        col= torch.cat(list(col.values())).clone().detach()


        rw = random_walk(rowptr, col, start_nodes, walk_length, self.p, self.q)
        if not isinstance(rw, torch.Tensor):
            rw = rw[0]
        flattened_rw = rw.view(-1)
        unique_nodes_set = set(flattened_rw.tolist())
        return unique_nodes_set


    def neg_sample(self, walk_length, query):
        start_nodes = torch.tensor(query)
        start_nodes=start_nodes.repeat(self.walks_per_node) 
        rw = torch.randint(self.G.number_of_nodes(),(start_nodes.size(0), walk_length))
        rw = torch.cat([start_nodes.view(-1, 1), rw], dim=-1)
        flattened_rw = rw.view(-1)
        unique_nodes_set = set(flattened_rw.tolist()).difference([query])
        return unique_nodes_set


    def sample(self, walk_length, query):
        pos=self.pos_sample(walk_length, query)
        neg=self.neg_sample(walk_length, query).difference(pos)
        return list(pos), list(neg)
    

def percd(input_tensor, k):
    count = torch.sum(input_tensor > input_tensor[k]).item()
    return count / input_tensor.size(0)

def perc(input_list, k):
    count = torch.sum(input_list < input_list[k])
    return count.item() / len(input_list)

def kmeans(X, k, n_iters=100):
    N, D = X.shape
    centers = X[:k].clone()  
    labels = torch.zeros(N, dtype=torch.long)
    for _ in range(n_iters):
        distances = torch.cdist(X, centers)
        labels = torch.argmin(distances, dim=1)
        for i in range(k):
            centers[i] = torch.mean(X[labels == i], dim=0)
    return centers, labels

def pagerank(adj_matrix, damping_factor=0.90, max_iter=100):
    N = adj_matrix.shape[0]
    teleport_prob = (1.0 - damping_factor) / N
    pr_scores = torch.ones(N) / N
    transition_matrix = adj_matrix / adj_matrix.sum(dim=1, keepdim=True)
    transition_matrix=transition_matrix.t()
    for _ in range(max_iter):
        next_pr_scores = (damping_factor * torch.matmul(transition_matrix, pr_scores)) + teleport_prob
        if torch.sum(torch.abs(next_pr_scores - pr_scores)) < 1e-5:
            break
        pr_scores = next_pr_scores
    return pr_scores





import networkx as nx
def k_hop_neighbors(G, node, k):
    """
    Returns the k-hop neighbors of a node in graph G.
    """
    # Remove the node itself from its k-hop neighbors
    neighbors = set(nx.single_source_shortest_path_length(G, node, cutoff=k).keys())
    neighbors.remove(node)
    return neighbors

def common_k_hop_neighbors(G, S, k):

    print(f"node in S: {S}")
    core_numbers = nx.core_number(G)

    count_p_all,count_n_all=0,0
    count_p_all_,count_n_all_=0,0

    for node in S:
        
        node_k_hop_neighbors = k_hop_neighbors(G, node, k)
        count_p,count_n=0,0
        count_p_,count_n_=0,0
        count_p_nei,count_n_nei=0,0

        for neighbor in G.neighbors(node):
            neighbor_k_hop_neighbors = k_hop_neighbors(G, neighbor, k)
            common_neighbors = node_k_hop_neighbors & neighbor_k_hop_neighbors

            if(neighbor in S):
                count_p_nei+=1
                for nei in common_neighbors:
                    if nei in S:
                        count_p+=1
                    else:
                        count_n+=1
            else:
                count_n_nei+=1
                for nei in common_neighbors:
                    if nei in S:
                        count_p_+=1
                    else:
                        count_n_+=1
            print(f"node {node} and neibor {neighbor} {k}-hop neibor: {common_neighbors} number: {len(common_neighbors)}")
        print(f"node {node} neibor(P/N):{count_p_nei} {count_n_nei} core number: {core_numbers[node]}")
        count_p_all+=count_p
        count_n_all+=count_n
        count_p_all_+=count_p_
        count_n_all_+=count_n_
        print(f" node: {node}  p_rate: {count_p/(count_p+count_n+1e-5)} n_rate: {count_n/(count_p+count_n+1e-5)}")
        print(f" node: {node}  p_rate_: {count_p_/(count_p_+count_n_+1e-5)} n_rate_: {count_n_/(count_p_+count_n_+1e-5)}")

    print(f"all node  p_rate: {count_p_all/(count_p_all+count_n_all+1e-5)} n_rate: {count_n_all/(count_p_all+count_n_all+1e-5)}")
    print(f"all node  p_rate_: {count_p_all_/(count_p_all_+count_n_all_+1e-5)} n_rate_: {count_n_all_/(count_p_all_+count_n_all_+1e-5)}")

    node_color = ['red' if node in S else 'lightblue' for node in G.nodes()]




import numpy as np
from torch_geometric.data import Data
from torch_sparse import spspmm, coalesce
from torch_geometric.utils import remove_self_loops, add_remaining_self_loops
class TwoHopNeighbor(object):
    def __call__(self, data):
        edge_index, edge_attr = data.edge_index, data.edge_attr
        N = data.num_nodes

        value = edge_index.new_ones((edge_index.size(1), ), dtype=torch.float)

        index, value = spspmm(edge_index, value, edge_index, value, N, N, N, True)
        value.fill_(0)
        index, value = remove_self_loops(index, value)

        edge_index = torch.cat([edge_index, index], dim=1)
        if edge_attr is None:
            data.edge_index, _ = coalesce(edge_index, None, N, N)
        else:
            value = value.view(-1, *[1 for _ in range(edge_attr.dim() - 1)])
            value = value.expand(-1, *list(edge_attr.size())[1:])
            edge_attr = torch.cat([edge_attr, value], dim=0)
            data.edge_index, edge_attr = coalesce(edge_index, edge_attr, N, N)
            data.edge_attr = edge_attr

        return data

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)

def hypergraph_construction(edge_index, num_nodes, k=1):
    if k == 1:
        edge_index, edge_attr = add_remaining_self_loops(edge_index, num_nodes=num_nodes)
    else:
        neighbor_augment = TwoHopNeighbor()
        hop_data = Data(edge_index=edge_index, edge_attr=None)
        hop_data.num_nodes = num_nodes
        for _ in range(k - 1):
            hop_data = neighbor_augment(hop_data)
        hop_edge_index = hop_data.edge_index
        hop_edge_attr = hop_data.edge_attr
        edge_index, edge_attr = add_remaining_self_loops(hop_edge_index, hop_edge_attr, num_nodes=num_nodes)
    return edge_index, edge_attr




########################################################CAF################################################################
def margin_loss(x1,x2,target,margin,reduce):
    if reduce =='mean':
        return torch.mean(torch.sqrt(torch.relu(((x2-x1+1)**2-(1-margin)**2))))
    elif reduce=='sum':
        return torch.mean(torch.sqrt(torch.relu(((x2 - x1 + 1) ** 2 - (1 - margin) ** 2))))
    

def make_single_node_encoding(new_node, graph,conv):
    bs = len(new_node)
    n_nodes = graph.n_nodes
    ind = np.array([[v, i] for i, v in enumerate(new_node) if v is not None], dtype=np.int64).T
    if len(ind):
        data = np.ones(ind.shape[1], dtype=np.float32)
        x_nodes = conv(sp.csc_matrix((data, ind), shape=[n_nodes, bs])).T
    else:
        x_nodes = conv(sp.csc_matrix((n_nodes, bs), dtype=np.float32)).T
    return x_nodes

def make_nodes_encoding(new_node, graph, conv):
    bs = len(new_node)
    n_nodes = graph.n_nodes
    ind = [[v, i] for i, vs in enumerate(new_node) for v in vs]
    if len(ind):
        ind = np.asarray(ind, dtype=np.int64).T
        data = np.ones(ind.shape[1], dtype=np.float32)
        x_nodes = conv(sp.csc_matrix((data, ind), shape=[n_nodes, bs])).T
    else:
        x_nodes = conv(sp.csc_matrix((n_nodes, bs), dtype=np.float32)).T
    return x_nodes


def myshortest_path(nxg, seed):
    path_map = {}
    vis = [seed]
    path_map[seed] = [seed]
    cnt = 0
    while cnt < len(vis):
        u = vis[cnt]
        cnt += 1
        if len(path_map[u]) > 5:
            continue
        for v in nxg.neighbors(u):
            if v in path_map.keys():
                continue
            else:
                path_map[v] = path_map[u] + [v]
                vis.append(v)
    return path_map

def get_augment(seeds,graph,nxg,conv):
    bs = len(seeds)
    feat = np.zeros((bs, graph.n_nodes), dtype=np.float32)
    for i, seed in enumerate(seeds):
        shortest_path = myshortest_path(nxg, seed)
        len_path = [(key, 1 / len(shortest_path[key])) for key in shortest_path.keys() if len(shortest_path[key]) < 5]
        data = [v for k, v in len_path]
        ind = [k for k, v in len_path]
        feat[i][ind] = data
    return conv(sp.csr_matrix(feat.T)).T


def preprocess(com,graph,conv,nxg,idx,seed, args):
    com=list(com)
    labels = [0] * len(com)
    labels[-1] = 1
    # seed =  com[0]
    left = set(com) - set([seed])
    bfs_com = [seed]
    bfs_com_set = set(bfs_com)
    neighbor = []
    next_nodes = []
    batch_now_com = [[seed]] 
    neighbor.append(graph.neighbors[seed]) 

    while left:
        union = neighbor[-1] & left
        # print(seed,neighbor[-1],left)
        if union == set():
            break
        left -= union
        union = list(union)
        while union:
            top = union.pop()
            bfs_com.append(top)
            bfs_com_set.add(top)
            batch_now_com.append(batch_now_com[-1] + [top]) 
            neighbor.append((graph.neighbors[top] | neighbor[-1]) - bfs_com_set) 
            next_nodes.append([top]) 

    next_nodes.append([-1])


    for i in range(len(neighbor)):
        next_node = next_nodes[i]

        if (len(neighbor[i]) > args.max_neighbor):
            neighbor[i] = np.random.choice(list(neighbor[i] - set(next_node)),
                                            args.max_neighbor - len(next_node)).tolist() + next_node
        neighbor[i] = sorted(list(neighbor[i]))


    neighbor_map = [{node: i for i, node in enumerate(ner)} for ner in neighbor]
    next_onehot = np.zeros((len(next_nodes), args.max_neighbor + 1))
    # print('repeat',len(neighbor))
    for i in range(len(next_nodes) - 1):
        tmpnext =   next_nodes[i]
        for j in tmpnext:
            next_onehot[i][neighbor_map[i][j]] = 1
    next_onehot[-1][args.max_neighbor] = 1

    batch_z_seed = []
    batch_z_node = []
    z_seed = make_single_node_encoding([seed], graph,conv)
    z_node = make_nodes_encoding([[seed]] + [node for node in next_nodes[:-1]], graph,conv)

    for i in range(1, len(next_nodes)):
        z_node[i] += z_node[i - 1]
    for i in range(len(next_nodes)):
        batch_z_seed.append(torch.from_numpy(z_seed[0, (batch_now_com[i]) + neighbor[i]].todense()))
        batch_z_node.append(torch.from_numpy(z_node[i, (batch_now_com[i]) + neighbor[i]].todense()))
        assert batch_z_node[-1].shape[1] == len(batch_now_com[i]) + len(neighbor[i]) and batch_z_seed[-1].shape[1] == len(
            batch_now_com[i]) + len(neighbor[i])
    assert len(neighbor) == len(batch_now_com)
    result = {"neighbor": neighbor,
                "next_onehot": next_onehot,
                "now_com": batch_now_com,
                "z_seed": batch_z_seed,
                "z_node": batch_z_node,
                "label": labels
                }
    if args.augment:
        batch_z_augment = []
        z_augment = get_augment([seed],graph,nxg,conv)
        for i in range(len(next_nodes)):
            batch_z_augment.append(torch.from_numpy(z_augment[0, (batch_now_com[i]) + neighbor[i]].todense()))
        result["z_augment"] = batch_z_augment

    if args.rankingloss:
        neg_neighbor = neighbor[-1][:]
        if -1 in neg_neighbor:
            neg_neighbor.remove(-1)
        if len(neg_neighbor) < args.neg_num:
            neg_neighbor += np.random.choice(list(graph.nodes - set(com)), args.neg_num - len(neg_neighbor)).tolist()
        batch_neg = np.random.choice(neg_neighbor, args.neg_num).tolist()
        assert len(batch_neg) >= args.neg_num
        neg_z_node = make_nodes_encoding([[node] for node in batch_neg], graph,conv)
        for i in range(neg_z_node.shape[0]):
            neg_z_node[i] += z_node[-1]
        batch_neg_z_node = []
        batch_neg_z_seed = []
        neg_neighbors = []
        neg_batch_z_augment = []
        for i in range(len(batch_neg)):
            neg = batch_neg[i]
            neg_neighbor = list(set(neg_neighbor) | graph.neighbors[neg] - set(com + [neg]))
            if (len(neg_neighbor) > args.max_neighbor):
                neg_neighbor = np.random.choice(neg_neighbor, args.max_neighbor).tolist()

            batch_neg_z_seed.append(torch.from_numpy(z_seed[0, com + [neg] + sorted(
                neg_neighbor)].todense()))
            batch_neg_z_node.append(torch.from_numpy(neg_z_node[i, com + [neg] + sorted(neg_neighbor)].todense()))
            neg_neighbors.append(neg_neighbor)
            if args.augment:
                neg_batch_z_augment.append(torch.from_numpy(z_augment[0, com + [neg] + sorted(neg_neighbor)].todense()))

        result["neg"] = batch_neg
        result["neg_neighbors"] = neg_neighbors
        result["neg_z_seed"] = batch_neg_z_seed
        result["neg_z_node"] = batch_neg_z_node
        result["neg_z_augment"] = neg_batch_z_augment

    return result



def load_pre_dataset(args,tasks_):
    pre_dataset = []
    roll_mapper = {}
    arg_dict = vars(args)
    pretrain_related_args = ['data_set', 'task_num', 'augment',
                             'max_neighbor', 'rankingloss', 'neg_num']
    code = ' '.join([str(arg_dict[k]) for k in pretrain_related_args])
    code = hashlib.md5(code.encode('utf-8')).hexdigest().upper()
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
    rooter = os.path.join(parent_dir, "results","pre")
    path = os.path.join(rooter, code)
    name = path + '.npy'

    if os.path.exists(name) and args.cache:
        print(f'{name} exists.' )
        pre_dataset = np.load(name, allow_pickle=True).tolist()
    else:
        if os.path.exists(rooter)==False:
            os.mkdir(rooter)
        print(f'{name}  not exists.' )

        tasks= tasks_.get_batch()
        for i in range(len(tasks)):
            roll_mapper[i] = []
            batch = tasks[i]
            seed=batch.query.item()

            edges=batch.edge_index
            tr_edges = edges.transpose(0, 1)
            re_edges =tr_edges.reshape(-1, 2).numpy()
            graph=Graph(re_edges)
            conv=GraphConv_m(graph)
            nxg = nx.Graph()

            nxg.add_edges_from(re_edges)
            result = preprocess(batch.query_index[seed][0],graph,conv,nxg,i, seed,args)

            roll_mapper[i].append(len(pre_dataset))
            pre_dataset.append(result)
        pre_dataset.append(roll_mapper)
        np.save(name, pre_dataset, allow_pickle=True)
    return pre_dataset[:-1],pre_dataset[-1]


class PretrainComDataset(Dataset):
    def __init__(self, com, train_size):
        self.com = com
        self.n_com = train_size

    def __len__(self):
        return self.n_com

    def __getitem__(self, idx):
        return torch.tensor(idx, dtype=torch.long)



def get_data(idx,args,roll_mapper,pre_dataset):
    result = {
        "batch_neighbor": [],
        "batch_next_onehot": [],
        "batch_z_seed": [],
        "batch_z_node": [],
        "batch_com": [],
        "batch_labels": []
    }
    if args.rankingloss:
        result["batch_neg"] = []
        result["batch_neg_neighbors"] = []
        result["batch_neg_z_seed"] = []
        result["batch_neg_z_node"] = []
        result["batch_neg_z_augment"] = []

    if args.augment:
        result["batch_z_augment"] = []
    for j in idx:

        for i in roll_mapper[j]:
            result["batch_neighbor"].extend(pre_dataset[i]['neighbor'])
            result["batch_next_onehot"].extend(pre_dataset[i]['next_onehot'])
            result["batch_z_seed"].extend(pre_dataset[i]['z_seed'])
            result["batch_z_node"].extend(pre_dataset[i]['z_node'])
            result["batch_com"].extend(pre_dataset[i]['now_com'])
            result["batch_labels"].extend(pre_dataset[i]['label'])
            if args.rankingloss:
                result["batch_neg"].append(pre_dataset[i]['neg'])
                result["batch_neg_neighbors"].append(pre_dataset[i]['neg_neighbors'])
                result["batch_neg_z_seed"].append(pre_dataset[i]['neg_z_seed'])
                result["batch_neg_z_node"].append(pre_dataset[i]['neg_z_node'])
                result["batch_neg_z_augment"].append(pre_dataset[i]['neg_z_augment'])


            if args.augment:
                result["batch_z_augment"].extend(pre_dataset[i]['z_augment'])

    return result



def generate_com(model, args, graph, conv, attr, seeds, nxg,temperature=0.75, neribor_max=200, debug=0, stop=1, decline=2):
    old_seeds = seeds
    n = len(old_seeds)
    model.eval()
    batch_coms = []
    chunk_size = 2000
    m_e=1
    args.best_score = True
    #chunk代表着每次选择多少个seeds
    for k in range((n // chunk_size) + 1):
        seeds = old_seeds[k * chunk_size:(k + 1) * chunk_size]
        seeds = [s for s in seeds for _ in range(1)]
        n_seed = len(seeds)
        if n_seed == 0:
            continue
        with torch.no_grad():
            prior_node_dist = [torch.distributions.normal.Normal(torch.zeros([(neribor_max + 1)]),
                                                                 temperature * torch.ones((neribor_max + 1))) for i in
                               range(n_seed)]

            batch_scores = [[-1] * decline for i in range(n_seed)]
            batch_nowcom = [[seed] for seed in seeds]
            batch_neribor = [list(graph.neighbors[seed]) for seed in seeds]
            batch_bestcom = [None for i in range(n_seed)]
            bacth_bests = [-2 for i in range(n_seed)]
            z_seeds = make_single_node_encoding([seed for seed in seeds], graph,conv)
            z_nodes = make_single_node_encoding([seed for seed in seeds], graph,conv)
            batch_z_seed = [0] * n_seed
            batch_z_node = [0] * n_seed
            if args.augment:
                batch_z_augment = [0] * n_seed
                z_augment = get_augment(seeds,graph,nxg,conv)
            batch_ok = [0] * n_seed
            ok = 0
            latent_nodes = [0] * n_seed
            for size in (range(args.max_size)):
                if (ok == n_seed):
                    break
                for i in (range(n_seed)):
                    if batch_ok[i] == 1:
                        continue
                    latent_nodes[i] = prior_node_dist[i].sample().view(1, -1)
                    if len(batch_neribor[i]) > neribor_max:
                        batch_neribor[i] = np.random.choice(batch_neribor[i], neribor_max)

                    batch_neribor[i] = sorted(batch_neribor[i])
                    batch_z_seed[i] = torch.from_numpy(z_seeds[i, (batch_nowcom[i]) + batch_neribor[i]].todense())
                    batch_z_node[i] = torch.from_numpy(z_nodes[i, (batch_nowcom[i]) + batch_neribor[i]].todense())
                    if args.augment:
                        batch_z_augment[i] = torch.from_numpy(
                            z_augment[i, (batch_nowcom[i]) + batch_neribor[i]].todense())

                latent_node = torch.stack(latent_nodes).view(n_seed, -1)
                batch_data = {"batch_com": batch_nowcom,
                              "batch_neighbor": batch_neribor,
                              "batch_deq": latent_node,
                              "batch_z_seed": batch_z_seed,
                              "batch_z_node": batch_z_node
                              }
                if args.augment:
                    batch_data['batch_z_augment'] = batch_z_augment

                result = model.flow_core.batch_revser(attr, batch_data)
                out_z = result['out_z']
                newnodes = []
                for i in range(n_seed):
                    if batch_ok[i] == 1:
                        newnodes.append(None)
                        continue
                    latent_node = out_z[i]
                    neribor_num = len(batch_neribor[i])


                    if stop == 1:
                        score = result['stop_score'][i].item()
                        # if args.rule:
                        #     score = 1 - nx.conductance(nxg, batch_nowcom[i])
                        if score > bacth_bests[i]:
                            bacth_bests[i] = score
                            batch_bestcom[i] = batch_nowcom[i]
                        flag = 0
                        for d in range(decline):
                            if score < batch_scores[i][-1 - d]:
                                flag += 1
                        if flag == decline and len(batch_nowcom[i]) >= 5:
                            newnodes.append(None)
                            batch_ok[i] = 1
                            ok += 1
                        elif neribor_num == 0:
                            newnodes.append(None)
                            batch_ok[i] = 1
                            ok += 1
                        elif flag < decline:

                            index = torch.argmax(latent_node[:neribor_num]).item()
                            newnode = batch_neribor[i][index]
                            batch_nowcom[i].append(newnode)
                            batch_neribor[i].extend(list(graph.neighbors[newnode]))
                            batch_neribor[i] = list(set(batch_neribor[i]) - set(batch_nowcom[i]))
                            batch_scores[i].append(score)
                            newnodes.append(newnode)
                        else:
                            batch_scores[i].append(score)
                            newnodes.append(None)
                    else:
                        if neribor_num != 0:
                            index = torch.argmax(latent_node[:neribor_num]).item()
                            newnode = batch_neribor[i][index]
                        else:
                            index = None
                        if index == None or (
                                latent_node[-1] > latent_node[index] and len(batch_nowcom[i]) >= args.community_min):
                            newnodes.append(None)
                            batch_ok[i] = 1
                            ok += 1
                        elif latent_node[-1] <= latent_node[index]:
                            batch_nowcom[i].append(newnode)
                            batch_neribor[i].extend(list(graph.neighbors[newnode]))
                            batch_neribor[i] = list(set(batch_neribor[i]) - set(batch_nowcom[i]))
                            newnodes.append(newnode)
                        else:
                            newnodes.append(None)

                z_nodes += make_single_node_encoding(newnodes, graph,conv)
            coms = []
            for i in range(0, len(batch_nowcom), m_e):
                if stop == 1:
                    bestc = None
                    bests = -1
                    for j in range(m_e):
                        com = batch_nowcom[i + j]
                        s = bacth_bests[i + j]
                        if args.best_score == True:
                            com = batch_bestcom[i + j]
                            s = bacth_bests[i + j]
                        if s > bests:
                            bestc = com
                    coms.append(bestc)
                else:
                    coms.append(batch_nowcom[i])
        batch_coms.extend(coms)
    return batch_coms


def get_max_value_length(d):
    max_length = 0
    for value in d.values():
        if len(value[0]) > max_length:
            max_length = len(value[0])
    return max_length