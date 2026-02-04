import random
import torch
import math
import networkx as nx
import numpy as np
from utils.utils import *
from torch_geometric.utils import remove_self_loops, add_remaining_self_loops

class RawGraphWithCommunity(object):
    def __init__(self, args,graph, communities, feats,query_index,node_list,mode):
        self.args=args
        self.num_workers = 20
        self.graph = graph  # networkx graph
        self.communities = communities  # list of list
        self.feats = feats  # origin node feat
        self.x_feats = torch.from_numpy(self.feats)
        self.query_index = query_index
        self.node_list=node_list
        self.mode=mode
        self.num_tasks=args.task_num

        # get the edge index, used by all sampled
        self.edge_index = torch.ones(size=(2, self.graph.number_of_edges()*2), dtype=torch.long)
        src = []
        dst = []
    
        for id1, id2 in self.graph.edges:
            src.append(id1)
            dst.append(id2)
            src.append(id2)
            dst.append(id1)
        self.edge_index = torch.tensor([src, dst])
        self.edge_index = add_remaining_self_loops(self.edge_index, num_nodes=len(graph))[0]
        if(args.vmodel=="coclep"):
            self.edge_index_aug, self.egde_attr = hypergraph_construction(self.edge_index, len(graph.nodes()), k=2)
        


    def sample_one(self, query, num_pos, num_neg):
        pos = list(self.query_index[query].difference({query}))
        neg_set=self.query_index[query]
        neg_set.add(query)
        neg = list(set(range(self.graph.number_of_nodes())).difference(neg_set))
        if(self.mode=="train"):
            num_pos=num_pos
            num_neg=num_neg
        else:
            num_pos=1
            num_neg=1
            
        if num_pos<=1:
            masked_pos = random.sample(pos, k= min(math.ceil(num_pos*len(pos)), len(pos)))
            masked_neg = random.sample(neg, k= min(math.ceil(num_neg*len(neg)), len(neg)))
        else:
            masked_pos = random.sample(pos, k= int(min(num_pos, len(pos))))
            masked_neg = random.sample(neg, k= int(min(num_neg, len(neg))))
        # print('graph_node_num:{},num_pos:{},num_neg:{}'.format(self.graph.number_of_nodes(),len(masked_pos),len(masked_neg)))

        return query, pos, neg, masked_pos, masked_neg


    
    def get_one_query_tensor(self, query, num_pos, num_neg):
        query, pos, neg, masked_pos, masked_neg = self.sample_one(query, num_pos, num_neg)
        x_q = torch.zeros(size=(self.graph.number_of_nodes(), 1), dtype=torch.float)
        x_q[query] = 1
        x, y = self.get_elements_for_query_data(pos, neg, x_q,query)
        # print(query)
        # print(masked_pos)
        if(self.args.vmodel=="coclep"):
            query_data = CSQueryData_aug(x=x, edge_index=self.edge_index,edge_index_aug=self.edge_index_aug, y=y, query=query, pos=masked_pos,
                                 neg=masked_neg, raw_feature=self.x_feats.cpu().numpy())
        elif(self.args.vmodel=="icsgnn"):
            query_data = CSQueryData_ics(graph=self.graph,query_index=self.query_index,x=x, edge_index=self.edge_index, y=y, query=query, pos=masked_pos,
                                 neg=masked_neg, raw_feature=self.x_feats.cpu().numpy())
        elif(self.args.vmodel=="caf"):
            query_data = CSQueryData_caf(query_index=self.query_index,x=x, edge_index=self.edge_index, y=y, query=query, pos=masked_pos,
                                    neg=masked_neg, raw_feature=self.x_feats.cpu().numpy())
        else:
            query_data = CSQueryData(x=x, edge_index=self.edge_index, y=y, query=query, pos=masked_pos,
                                    neg=masked_neg, raw_feature=self.x_feats.cpu().numpy())
        return query_data


    # add node feature,core feature and cluster feature to query feature.
    def get_elements_for_query_data(self, pos, neg, x_q,q):

        def max_min_normalize(x):
            min_val = x.min()
            max_val = x.max()
            if max_val > min_val:
                return (x - min_val) / (max_val - min_val)
            
        x = torch.cat([x_q, self.x_feats], dim=-1)
        self.graph.remove_edges_from(nx.selfloop_edges(self.graph))

        # core = nx.core_number(self.graph)
        # core_feature = torch.zeros(size=(self.graph.number_of_nodes(), 1), dtype=torch.float)
        # for j in range(len(core)):
        #     core_feature[j] = core[j]
        # core_feature = max_min_normalize(core_feature)
        # x = torch.cat([x, core_feature], dim=-1)

        cluster = nx.clustering(self.graph)
        cluster_feature = torch.zeros(size=(self.graph.number_of_nodes(), 1), dtype=torch.float)
        for j in range(len(cluster)):
            cluster_feature[j] = cluster[j]
        cluster_feature = max_min_normalize(cluster_feature)
        x = torch.cat([x, cluster_feature], dim=-1)

        shortest_path_lengths = nx.single_source_shortest_path_length(self.graph, q)
        # print(shortest_path_lengths)
        distance_feature = torch.zeros(size=(self.graph.number_of_nodes(), 1), dtype=torch.float)
        for node, dist in shortest_path_lengths.items():
            distance_feature[node] = dist
        # print(shortest_path_lengths)
        distance_feature = max_min_normalize(distance_feature)
        x = torch.cat([x, distance_feature], dim=-1)

        pagerank = nx.pagerank(self.graph)
        pagerank_feature = torch.zeros(size=(self.graph.number_of_nodes(), 1), dtype=torch.float)
        for j in range(len(pagerank)):
            pagerank_feature[j] = pagerank[j]
        pagerank_feature = max_min_normalize(pagerank_feature)
        x = torch.cat([x, pagerank_feature], dim=-1)
        
        x = x.to(torch.float32)
        y = torch.zeros(size=(self.graph.number_of_nodes(),), dtype=torch.float)
        y = y
        y[pos] = 1
        return x, y


    def get_query_data(self, query,num_pos, num_neg):
        query_data = self.get_one_query_tensor(query,num_pos,num_neg)
        # task = TaskData(batch_size,query_data)
        return query_data



    






