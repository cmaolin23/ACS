import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, GINEConv, NNConv, GATConv, GraphConv
from torch import optim
from loss_criteria.loss import WeightedBinaryCrossEntropy,WeightedKLDivergence,WeightBCEWithLogitsLoss
from utils.utils import hypergraph_construction,Graph,GraphConv_m,generate_com
import networkx as nx
import numpy as np
from sklearn.metrics import precision_score, f1_score, accuracy_score, recall_score
import matplotlib.pyplot as plt
import copy
from torch_geometric.nn import  GCNConv
import time
import logging


class Policy_AE(nn.Module):
    def __init__(self,args,input_dim, hidden_dim,output_dim):
        super(Policy_AE, self).__init__()
        self.act_type = args.act_type
        self.num_layers = args.att_layer_num
        self.mlp = nn.ModuleList()
        self.mlp.append(nn.Linear(input_dim, hidden_dim))
        for l in range(self.num_layers-1):
            self.mlp.append(nn.Linear(hidden_dim, hidden_dim))
        self.mlp.append(nn.Linear(hidden_dim, output_dim))

        self.act_layer = get_act_layer(self.act_type)


    def encode(self, x):
        for l in range(self.num_layers):
            x = self.act_layer(self.mlp[l](x))
        x_hid = self.mlp[-1](x)

        return x_hid
    

    def decode(self, x, edge_index, all=True):
        sim_vec = F.cosine_similarity(x[edge_index[0]], x[edge_index[1]], dim=1)
        sim_vec =torch.sigmoid(sim_vec)
        loss_recon = -torch.log(sim_vec).mean()   

        if all:
            sim_matrix = F.cosine_similarity(x.unsqueeze(1), x.unsqueeze(0), dim=2)
            return torch.sigmoid(sim_matrix),loss_recon
        else:
            return sim_vec,loss_recon


    def forward(self, x,edge_index,all=True):
        # encoder & decoder
        x = self.encode(x)
        edge_score_matrix,loss = self.decode(x,edge_index,all=True)

        return edge_score_matrix,loss

class Policy_GAE(nn.Module):
    def __init__(self,args, input_dim, hidden_dim,output_dim):
        super(Policy_GAE, self).__init__()
        self.num_node_feat = input_dim
        self.num_layers = args.att_layer_num
        self.num_hid = hidden_dim
        self.num_out = output_dim
        self.model_type = args.att_gnn_type
        self.dropout = args.dropout
        self.convs = nn.ModuleList()
        self.act_type = args.act_type
        self.act_layer = get_act_layer(self.act_type)
        cov_layer = self.build_cov_layer(self.model_type)

        for l in range(self.num_layers):
            hidden_input_dim = self.num_node_feat if l == 0 else self.num_hid
            hidden_output_dim = self.num_out if l == self.num_layers - 1 else self.num_hid
            if self.model_type == "GAT"  or self.model_type == "GCN" or self.model_type == "SAGE":
                self.convs.append(cov_layer(hidden_input_dim, hidden_output_dim))
            else:
                assert False, "Unsupported model type!"

    def build_cov_layer(self, model_type):
        if model_type == "GAT":
            return GATConv
        elif model_type == "SAGE":
            return SAGEConv
        elif model_type == "GCN":
            return GraphConv
        else:
            assert False, "Unsupported model type!"

    def reconstruct_loss(self,x, edge_index):

        src, dst = edge_index
        edge_scores = (x[src] * x[dst]).sum(dim=-1)
        pos_loss = F.binary_cross_entropy_with_logits(edge_scores, torch.ones_like(edge_scores))

        num_nodes = x.size(0)
        existing_edges = set(zip(src.tolist(), dst.tolist()))

        neg_src, neg_dst = [], []
        while len(neg_src) < edge_index.size(1):
            a, b = torch.randint(0, num_nodes, (2,))
            if (a.item(), b.item()) not in existing_edges:
                neg_src.append(a)
                neg_dst.append(b)
        
        neg_src = torch.tensor(neg_src)
        neg_dst = torch.tensor(neg_dst)

        neg_edge_scores = (x[neg_src] * x[neg_dst]).sum(dim=-1)
        neg_loss = F.binary_cross_entropy_with_logits(neg_edge_scores, torch.zeros_like(neg_edge_scores))

        loss = pos_loss + neg_loss
        return loss

    def hard_concrete_sample(self, log_alpha, beta=0.5, training=True):
        gamma = -0.1
        zeta = 1.1
        eps = 1e-7

        if self.training:
            debug_var = eps
            bias = 0.0
            random_noise = bias + torch.rand(log_alpha.size(), dtype=log_alpha.dtype, device=log_alpha.device) * (1.0 - 2 * debug_var) + debug_var
            gate_inputs = torch.log(random_noise) - torch.log(1.0 - random_noise)
            gate_inputs = (gate_inputs + log_alpha) / beta
            gate_inputs = torch.sigmoid(gate_inputs)
        else:
            gate_inputs = torch.sigmoid(log_alpha)

        stretched_values = gate_inputs * (zeta - gamma) + gamma
        clipped = torch.clamp(stretched_values, min=1e-7, max=1)
        # print(clipped.min(),clipped.max())
        return clipped
    
    def encode(self, x,edge_index,attr):
        for i in range(self.num_layers):
            if self.model_type == "GAT" or self.model_type =="GCN" or self.model_type == "SAGE":
                x = self.convs[i](x, edge_index)
            else:
                print("Unsupported model type!")
            if i < self.num_layers - 1:
                if self.act_type != 'relu':
                    x = self.act_layer(x)
                x = F.dropout(x, p = self.dropout, training=self.training)
        return x

    def decode(self, x, edge_index,all=True):
        sim_vec = F.cosine_similarity(x[edge_index[0]], x[edge_index[1]], dim=1)
        sim_vec =torch.sigmoid(sim_vec)
        # loss_recon = -torch.log(sim_vec).mean()  
        loss_recon = self.reconstruct_loss(x, edge_index) 

        if all:
            sim_matrix = F.cosine_similarity(x.unsqueeze(1), x.unsqueeze(0), dim=2)
            return torch.sigmoid(sim_matrix),loss_recon
            # return self.hard_concrete_sample(sim_matrix, beta=0.5, training=self.training),loss_recon
        else:
            return sim_vec,loss_recon

    def forward(self,q, x,edge_index,attr=[]):
        # encoder & decoder
        x = self.encode(x,edge_index,attr)
        edge_score_matrix,loss_rec = self.decode(x,edge_index)
        return edge_score_matrix,loss_rec

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(MLP, self).__init__()
        self.fcs = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x):
        return self.fcs(x)
    
class Policy_GNN(nn.Module):
    def __init__(self,args, node_in_dim, hidden_dim, num_layers, dropout, device):

        super(Policy_GNN, self).__init__()
        self.dropout = dropout
        self.num_layers = num_layers
        self.device = device
        self.alpha = 0.3
        self.args=args

        self.linerdist = MLP(1, hidden_dim)

        # Part 1: GCN
        self.layersq = nn.ModuleList()
        self.layersq.append(GCNConv(1, hidden_dim))
        for _ in range(num_layers - 1):
            self.layersq.append(GCNConv(hidden_dim, hidden_dim))

        self.layers = nn.ModuleList()
        self.layers.append(GCNConv(node_in_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.layers.append(GCNConv(hidden_dim, hidden_dim))

        self.layersf = nn.ModuleList()
        self.layersf.append(GCNConv(hidden_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.layersf.append(GCNConv(hidden_dim, hidden_dim))

        self.mlp1 = MLP(hidden_dim, hidden_dim)
        self.mlp2 = MLP(hidden_dim, 1)

        self.layers_mlp_src = nn.ModuleList()
        self.layers_mlp_des = nn.ModuleList()
        self.layers_mlp_src.append(MLP(hidden_dim, hidden_dim))
        self.layers_mlp_des.append(MLP(hidden_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.layers_mlp_src.append(MLP(hidden_dim, hidden_dim))
            self.layers_mlp_des.append(MLP(hidden_dim, hidden_dim))

        self.linerquerys = torch.nn.Linear(1, hidden_dim)
        self.linerfeats = torch.nn.Linear(node_in_dim, hidden_dim)
        self.leaky_relu = torch.nn.LeakyReLU(0.2)
        # 调用初始化方法
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, GCNConv):
                nn.init.xavier_uniform_(m.lin.weight)
                if m.lin.bias is not None:
                    nn.init.zeros_(m.lin.bias)

    def hard_concrete_sample(self, log_alpha, beta=0.5, training=True):
        gamma = -0.1
        zeta = 1.1
        eps = 1e-7

        if training:
            debug_var = eps
            bias = 0.0
            random_noise = bias + torch.rand(log_alpha.size(), dtype=log_alpha.dtype, device=log_alpha.device) * (1.0 - 2 * debug_var) + debug_var
            gate_inputs = torch.log(random_noise) - torch.log(1.0 - random_noise)
            gate_inputs = (gate_inputs + log_alpha) / beta
            gate_inputs = torch.sigmoid(gate_inputs)
        else:
            gate_inputs = torch.sigmoid(log_alpha)

        stretched_values = gate_inputs * (zeta - gamma) + gamma
        clipped = torch.clamp(stretched_values, min=1e-7, max=1)
        # print(clipped.min(),clipped.max())
        return clipped
    
    def dis_encoding(self,q,feats,edge_index):
        G = nx.Graph()
        edge_list = edge_index.t().tolist()
        G.add_edges_from(edge_list)
        shortest_path_lengths = nx.single_source_shortest_path_length(G, source=q.item())
        # distances = torch.tensor([shortest_path_lengths[i] for i in range(feats.shape[0])], dtype=torch.float32, device=self.args.device).view(-1, 1)
        distances = []
        for i in range(feats.shape[0]):
            if i in shortest_path_lengths:
                distances.append(shortest_path_lengths[i])
            else:
                distances.append(0)
        distances = torch.tensor(distances, dtype=torch.float32).view(-1, 1).to(self.args.device)
        dists_ = self.linerdist(distances)
        return dists_

    def reconstruct_loss(self, x, q, edge_index):
        src, dst = edge_index
        edge_embeddings = (x[src] + x[dst]) / 2
        q_embedding = x[q]
        
        # print(edge_embeddings.size(),q_embedding.size())    
        edge_scores = F.cosine_similarity(edge_embeddings, q_embedding.unsqueeze(0), dim=1)
        pos_loss = F.binary_cross_entropy_with_logits(edge_scores, torch.ones_like(edge_scores))

        num_nodes = x.size(0)
        existing_edges = set(zip(src.tolist(), dst.tolist()))

        neg_src, neg_dst = [], []
        while len(neg_src) < edge_index.size(1):
            a, b = torch.randint(0, num_nodes, (2,))
            if (a.item(), b.item()) not in existing_edges:
                neg_src.append(a)
                neg_dst.append(b)
        
        neg_src = torch.tensor(neg_src)
        neg_dst = torch.tensor(neg_dst)

        neg_edge_embeddings = (x[neg_src] + x[neg_dst]) / 2
        neg_edge_scores = F.cosine_similarity(neg_edge_embeddings, q_embedding.unsqueeze(0), dim=1)
        neg_loss = F.binary_cross_entropy_with_logits(neg_edge_scores, torch.zeros_like(neg_edge_scores))

        loss = pos_loss + neg_loss
        return loss
    
    def encode(self, q, edge_index,feats):
        querys = torch.zeros(feats.shape[0], 1).to(self.device)
        querys[q] = 1.0
        querys_ = self.linerquerys(querys)
        feats_ = self.linerfeats(feats)
        edge_weight=torch.ones(edge_index.size(1)).to(self.device)

        if self.args.dis_encode:
            dis_emb=self.dis_encoding(q,feats,edge_index)
            querys_=torch.stack([querys_, dis_emb], dim=0)
            querys_=torch.sum(querys_, dim=0).to(self.device)

        h_d = torch.stack([querys_, feats_], dim=1)
        h_d = torch.sum(h_d, dim=1).to(self.device)
        edge_index_=edge_index

        # part : GCN
        hq = F.relu(self.layersq[0](querys, edge_index_,edge_weight=edge_weight)).to(self.device)
        h = F.relu(self.layers[0](feats, edge_index_,edge_weight=edge_weight)).to(self.device)
        hf = torch.stack([hq, h], dim=1)
        hf = torch.sum(hf, dim=1)

        hf_ = torch.stack([querys_, feats_], dim=1)
        hf_ = torch.sum(hf_, dim=1)
        hf = F.relu(hf + self.layersf[0](hf_, edge_index_,edge_weight=edge_weight))

        for _ in range(self.num_layers - 2):
        # part : GCN
            hq = F.dropout(hq, training=self.training, p=self.dropout)
            h = F.dropout(h, training=self.training, p=self.dropout)
            hf = F.dropout(hf, training=self.training, p=self.dropout)

            hq = F.relu(self.layersq[_+1](hq, edge_index_,edge_weight=edge_weight))
            h = F.relu(self.layers[_+1](h, edge_index_,edge_weight=edge_weight))
            hfx = torch.stack([hq, h], dim=1)
            hfx = torch.sum(hfx, dim=1)
            hf = F.relu(hfx + self.layersf[_+1](hf, edge_index_,edge_weight=edge_weight))

        hq = F.dropout(hq, training=self.training, p=self.dropout)
        h = F.dropout(h, training=self.training, p=self.dropout)
        hf = F.dropout(hf, training=self.training, p=self.dropout)

        hq = self.layersq[self.num_layers - 1](hq, edge_index_,edge_weight=edge_weight)
        h = self.layers[self.num_layers - 1](h, edge_index_,edge_weight=edge_weight)
        hfx = torch.stack([hq, h], dim=1)
        hfx = torch.sum(hfx, dim=1)

        hf = hfx + self.layersf[self.num_layers - 1](hf, edge_index_,edge_weight=edge_weight)
        h_ = self.mlp1(hf)
        return h_
    
    def decode(self, x, q, edge_index):
        loss_recon = self.reconstruct_loss(x, q, edge_index) 
        if self.args.attack_type=="proposed":
            if not self.args.chunk:
                sim_matrix_xx = F.cosine_similarity(x.unsqueeze(1), x.unsqueeze(0), dim=2).squeeze()
            else:
                x = x.half()
                batch_size = x.size(0)
                chunk_size = 16
                sim_matrix_xx = torch.zeros(batch_size, batch_size).to(x.device)

                for i in range(0, x.size(0), chunk_size):
                    for j in range(0, x.size(0), chunk_size):

                        sim_matrix_xx[i:i + chunk_size, j:j + chunk_size] = F.cosine_similarity(
                            x[i:i + chunk_size].unsqueeze(1),  
                            x[j:j + chunk_size].unsqueeze(0),  
                            dim=2
                        )
                        #torch.cuda.empty_cache()

            q_embedding = x[q]
            edge_embeddings_all = (x.unsqueeze(1) + x.unsqueeze(0)) / 2
            sim_matrix_xq = F.cosine_similarity(edge_embeddings_all, q_embedding.unsqueeze(0).unsqueeze(0), dim=-1).squeeze()


            if self.args.meth_r == "base":
                sim_matrix = sim_matrix_xx
            elif self.args.meth_r == "best":
                sim_matrix = (self.alpha * sim_matrix_xx + (1 - self.alpha) * sim_matrix_xq)*2
            return torch.sigmoid(sim_matrix), loss_recon
        else:

            q_embedding = x[q]
            node_embeddings_all = x
            sim_vec_xq = F.cosine_similarity(node_embeddings_all, q_embedding.unsqueeze(0).unsqueeze(0), dim=-1).squeeze()
            sim_vec_xx = self.mlp2(x).squeeze()
            self.alpha = 0.3
            sim_vec = ((1 - self.alpha) *sim_vec_xq + self.alpha *sim_vec_xx)*2
            # sim_vec=sim_vec_xq

            return torch.sigmoid(sim_vec),loss_recon

    def forward(self, q,feats, edge_index):
        # encoder & decoder
        x = self.encode(q, edge_index,feats)
        edge_score_matrix,loss_rec = self.decode(x,q,edge_index)
        return edge_score_matrix,loss_rec
        # return x


def evaluate_prediction(pred, targets):
    acc = accuracy_score(targets, pred)
    precision = precision_score(targets, pred)
    recall = recall_score(targets, pred)
    f1 = f1_score(targets, pred)
    return acc, precision, recall, f1

def get_act_layer(act_type: str):
    if act_type == "relu":
        return nn.ReLU()
    elif act_type == "tanh":
        return nn.Tanh()
    elif act_type == "leaky_relu":
        return nn.LeakyReLU()
    elif act_type == "prelu":
        return nn.PReLU()
    elif act_type == 'grelu':
        return nn.GELU()
    elif act_type == "none":
        return lambda x : x
    else:
        raise NotImplementedError("Error: %s activation function is not supported now." % (act_type))


        
    def forward(self,q, x,edge_index,attr=[]):
        # encoder & decoder
        x = self.encode(x,edge_index,attr)
        edge_score_matrix,loss_rec = self.decode(x,edge_index)
        return edge_score_matrix,loss_rec

class Evasion_attack(nn.Module):
    def __init__(self, args,pmodel,victim_model):
        super(Evasion_attack, self).__init__()
        self.args=args
        self.budget = args.budget   # budget for attack
        self.pmodel = pmodel
        self.vmodel = victim_model
        self.optimizer = optim.Adam(self.pmodel.parameters(), lr=args.att_learning_rate, weight_decay=args.att_weight_decay)

        # self.criterion = WeightedKLDivergence(args.L_weight)   
        self.criterion = WeightedBinaryCrossEntropy(args.L_weight)
        self.criterion1 = WeightBCEWithLogitsLoss()

    def attack(self,y,pmodel,feats,q,edge_index):
       
        # h_d=self.feat_cal(edge_index= edge_index,feats=feats,q=q)
        # embeddings = generate_node2vec_embeddings(edge_index)
        # h_d=torch.tensor([embeddings[str(node)] for node in range(len(embeddings))], device=self.args.device)
        # edge_index_att,edge_score_matrix_flat,topk_indices,loss_rec = self.generate_ptb(pmodel,feats,q,edge_index)
        h_d=feats
        edge_index_att,edge_score_matrix_flat,topk_indices,loss_rec,distance_loss = self.generate_ptb(y,pmodel,h_d,q,edge_index)

        return edge_index_att,edge_score_matrix_flat,topk_indices,loss_rec,distance_loss

    def generate_ptb(self,y,pmodel,feats,q,edge_index):
        num_edges_to_attack= self.budget

        #####################################################Random Attack#####################################################
        if self.args.attack_type == 'random':
            if self.args.att_meth =="add":
                edge_index_ = edge_index.cpu().numpy()
                num_nodes = feats.size(0)
                num_edges_to_attack = num_edges_to_attack
                
                new_edges = []
                i=0
                while len(new_edges) < num_edges_to_attack:
                    i+=1
                    if i>1000:
                        break
                    u = np.random.randint(0, num_nodes)
                    v = np.random.randint(0, num_nodes)
                    if u != v and [u, v] not in edge_index_.T.tolist() and [v, u] not in edge_index_.T.tolist():
                        new_edges.append([u, v])
                        new_edges.append([v, u])
                
                new_edges = np.array(new_edges).T
                edge_index_ = np.hstack((edge_index_, new_edges))
                edge_index_att = torch.tensor(edge_index_, dtype=torch.long).to(self.args.device)

                return edge_index_att,None,None, 0,0

            if self.args.att_meth =="del":
                edge_index_ = edge_index.cpu().numpy()
                num_edges_to_attack = num_edges_to_attack
                
                edges = edge_index_.T.tolist()
                
                non_self_loops = [edge for edge in edges if edge[0] != edge[1]]
                edges_to_delete = np.random.choice(len(non_self_loops), num_edges_to_attack, replace=False)
                
                for idx in sorted(edges_to_delete, reverse=True):
                    edge = non_self_loops[idx]
                    if edge in edges:
                        edges.remove(edge)
                    if [edge[1], edge[0]] in edges:
                        edges.remove([edge[1], edge[0]])
                
                edge_index_ = np.array(edges).T
                edge_index_att = torch.tensor(edge_index_, dtype=torch.long).to(self.args.device)
                return edge_index_att, None, None, 0,0

            if self.args.att_meth =="flip":
                # ADD
                edge_index_1 = edge_index.cpu().numpy()
                edge_index_2 = edge_index.cpu().numpy()
                num_nodes = feats.size(0)
                num_edges_to_attack = num_edges_to_attack
                
                new_edges = []
                i=0
                while len(new_edges) < num_edges_to_attack:
                    i+=1
                    if i>1000:
                        break
                    u = np.random.randint(0, num_nodes)
                    v = np.random.randint(0, num_nodes)
                    if u != v and [u, v] not in edge_index_1.T.tolist() and [v, u] not in edge_index_1.T.tolist():
                        new_edges.append([u, v])
                        new_edges.append([v, u])
                
                new_edges = np.array(new_edges).T

                # DEL
                edges = edge_index_2.T.tolist()
                non_self_loops = [edge for edge in edges if edge[0] != edge[1]]
                edges_to_delete = np.random.choice(len(non_self_loops), num_edges_to_attack, replace=False)
                
                for idx in sorted(edges_to_delete, reverse=True):
                    edge = non_self_loops[idx]
                    if edge in edges:
                        edges.remove(edge)
                    if [edge[1], edge[0]] in edges:
                        edges.remove([edge[1], edge[0]])
                
                edge_index_ = np.array(edges).T
                edge_index_ = np.hstack((edge_index_, new_edges))
                edge_index_att = torch.tensor(edge_index_, dtype=torch.long).to(self.args.device)             
                return edge_index_att, None, None, 0,0
        #####################################################Greedy Attack#####################################################
        elif self.args.attack_type in ["preference", "degree", "pagerank"]:
            # edge_index to NetworkX 
            edge_index_ = edge_index.cpu().numpy()
            # print(edge_index.size())
            G = nx.Graph()
            G.add_edges_from(edge_index_.T.tolist())
        
            if self.args.attack_type == "pagerank":
                pagerank = nx.pagerank(G)
                scores = {(min(u, v),max(u,v)): pagerank[u] + pagerank[v] for u in G.nodes() for v in G.nodes() if u != v}
            elif self.args.attack_type == "preference":
                scores = {(min(u, v),max(u,v)): G.degree[u] * G.degree[v] for u in G.nodes() for v in G.nodes() if u != v}
            elif self.args.attack_type == "degree":
                scores = {(min(u, v),max(u,v)):G.degree[u] + G.degree[v] for u in G.nodes() for v in G.nodes() if u != v}


            sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            # Add edges
            if self.args.att_meth == "add":
                new_edges = []
                for edge, _ in sorted_scores:
                    edge=list(edge)
                    if len(new_edges) < num_edges_to_attack:
                        if edge not in edge_index_.T.tolist() and [edge[1], edge[0]] not in edge_index_.T.tolist():
                            new_edges.append(edge)
                            new_edges.append([edge[1], edge[0]])
                    else:
                        break
                new_edges = np.array(new_edges).T
                edge_index_ = np.hstack((edge_index_, new_edges))
                edge_index_att = torch.tensor(edge_index_, dtype=torch.long).to(self.args.device)
                # print(edge_index_att.size())
                return edge_index_att,None,None, 0,0
            
            elif self.args.att_meth == "del":
                edges = edge_index_.T.tolist()
                num_edges=len(edges)
                for edge, _ in sorted_scores:
                    edge=list(edge)
                    if len(edges) > num_edges - num_edges_to_attack:
                        if edge in edges or [edge[1], edge[0]] in edges:
                            edges.remove(edge)
                            edges.remove([edge[1], edge[0]])
                    else:
                        break

                edge_index_ = np.array(edges).T
                edge_index_att = torch.tensor(edge_index_, dtype=torch.long).to(self.args.device)
                # print(edge_index_att.size())
                # print("=========================")
                return edge_index_att, None, None, 0,0

            elif self.args.att_meth == "flip":
                # ADD
                new_edges = []
                for edge, _ in sorted_scores:
                    edge=list(edge)
                    if len(new_edges) < num_edges_to_attack:
                        if edge not in edge_index_.T.tolist() and [edge[1], edge[0]] not in edge_index_.T.tolist():
                            new_edges.append(edge)
                            new_edges.append([edge[1], edge[0]])
                    else:
                        break
                new_edges = np.array(new_edges).T

                # DEL
                edges = edge_index_.T.tolist()
                num_edges=len(edges)
                for edge, _ in sorted_scores:
                    edge=list(edge)
                    if len(edges) > num_edges - num_edges_to_attack:
                        if edge in edges and [edge[1], edge[0]] in edges:
                            edges.remove(edge)
                            edges.remove([edge[1], edge[0]])
                    else:
                        break

                edge_index_ = np.array(edges).T
                edge_index_ = np.hstack((edge_index_, new_edges))
                edge_index_att = torch.tensor(edge_index_, dtype=torch.long).to(self.args.device)
                # print(edge_index_att.size())
                return edge_index_att, None, None, 0,0
        #####################################################Adversarial Attack#####################################################
        elif self.args.attack_type == 'proposed':
            edge_score_matrix,loss_rec= pmodel(q,feats,edge_index)
            edge_score_matrix_ = (edge_score_matrix + edge_score_matrix.t()) / 2
            edge_score_matrix_.fill_diagonal_(0)
            num_nodes = edge_score_matrix_.size(0)
            if self.args.att_meth =="add":
                all_edges = set((min(u, v), max(u, v)) for u in range(num_nodes) for v in range(num_nodes))
                edge_index_set = set((min(u, v), max(u, v)) for u, v in edge_index.t().tolist())
                non_edge_index_set = all_edges - edge_index_set


                mask1 = torch.zeros_like(edge_score_matrix_)
                mask2 = torch.ones_like(edge_score_matrix_)    
                for u, v in edge_index_set:
                    if u != v:
                        mask1[u, v] = 1

                mask2 = mask2 - mask1 - mask1.t()
                mask2.fill_diagonal_(0)
                edge_score_matrix_2 = (1-edge_score_matrix_) * mask2
                edge_score_matrix_flat2 = edge_score_matrix_2.flatten()

                if self.args.sample_ae == "probs":
                    temperature = self.args.tem_sample 
                    edge_index_set_flat2 = [u * edge_score_matrix_2.shape[1] + v for u, v in non_edge_index_set]
                    edge_score_matrix_flat_2= edge_score_matrix_flat2[edge_index_set_flat2]
                    probabilities2 = torch.softmax(edge_score_matrix_flat_2, dim=0)
                    scaled_probabilities2 = torch.pow(probabilities2, 1 / temperature)
                    scaled_probabilities2 = scaled_probabilities2 / scaled_probabilities2.sum()  # 归一化
                    topk_indices_subset2 = torch.multinomial(scaled_probabilities2, num_edges_to_attack, replacement=False)
                    topk_indices2 = torch.tensor([edge_index_set_flat2[i] for i in topk_indices_subset2])
                
                elif self.args.sample_ae == "topk":
                    edge_index_set_flat2 = [u * edge_score_matrix_2.shape[1] + v for u, v in non_edge_index_set]
                    edge_score_matrix_flat_2= edge_score_matrix_flat2[edge_index_set_flat2]
                    topk_indices_subset2= torch.topk(edge_score_matrix_flat_2, num_edges_to_attack).indices
                    topk_indices2 = torch.tensor([edge_index_set_flat2[i] for i in topk_indices_subset2])

                row_indices2 = topk_indices2 // num_nodes
                col_indices2 = topk_indices2 % num_nodes
                atk_edges2 = torch.stack([row_indices2, col_indices2], dim=0)
                new_edges2 = torch.tensor(atk_edges2, dtype=torch.long).contiguous().to(self.args.device)

                new_edges_set2 = set((min(u, v), max(u, v)) for u, v in new_edges2.t().tolist())
                remaining_edges_set = edge_index_set.union(new_edges_set2)

                bi_directional_edges = []
                for u, v in remaining_edges_set:
                    bi_directional_edges.append((u, v))
                    if u!=v:
                        bi_directional_edges.append((v, u))
                edge_index_att = torch.tensor(bi_directional_edges, dtype=torch.long).t().contiguous().to(self.args.device)

                return edge_index_att,edge_score_matrix.view(-1),None,loss_rec ,0
            
            elif self.args.att_meth == "del":
                num_nodes = edge_score_matrix_.size(0)
                edge_index_set = set((min(u, v), max(u, v)) for u, v in edge_index.t().tolist())
                mask1 = torch.zeros_like(edge_score_matrix_) 
                for u, v in edge_index_set:
                    if u != v:
                        mask1[u, v] = 1

                edge_score_matrix_1 = edge_score_matrix_ * mask1
                edge_score_matrix_flat1 = edge_score_matrix_1.flatten()

                if self.args.sample_ae == "probs":
                    temperature = self.args.tem_sample 
                    edge_index_set_flat1 = [u * edge_score_matrix_1.shape[1] + v for u, v in edge_index_set if u != v]
                    edge_score_matrix_flat_1= edge_score_matrix_flat1[edge_index_set_flat1]
                    probabilities1 = torch.softmax(edge_score_matrix_flat_1, dim=0) 
                    scaled_probabilities1 = torch.pow(probabilities1, 1 / temperature)
                    scaled_probabilities1 = scaled_probabilities1 / scaled_probabilities1.sum()  # 归一化
                    topk_indices_subset1 = torch.multinomial(scaled_probabilities1, num_edges_to_attack, replacement=False)
                    topk_indices1 = torch.tensor([edge_index_set_flat1[i] for i in topk_indices_subset1])
                
                elif self.args.sample_ae == "topk":
                    edge_index_set_flat1 = [u * edge_score_matrix_1.shape[1] + v for u, v in edge_index_set if u != v]
                    edge_score_matrix_flat_1= edge_score_matrix_flat1[edge_index_set_flat1]
                    topk_indices_subset1= torch.topk(edge_score_matrix_flat_1, num_edges_to_attack).indices
                    topk_indices1 = torch.tensor([edge_index_set_flat1[i] for i in topk_indices_subset1])

                row_indices1 = topk_indices1 // num_nodes
                col_indices1 = topk_indices1 % num_nodes
                atk_edges1 = torch.stack([row_indices1, col_indices1], dim=0)
                new_edges1 = torch.tensor(atk_edges1, dtype=torch.long).contiguous().to(self.args.device)


                new_edges_set1 = set((min(u, v), max(u, v)) for u, v in new_edges1.t().tolist())
                remaining_edges_set = edge_index_set - new_edges_set1
                bi_directional_edges = []
                for u, v in remaining_edges_set:
                    bi_directional_edges.append((u, v))
                    if u!=v:
                        bi_directional_edges.append((v, u))
                edge_index_att = torch.tensor(bi_directional_edges, dtype=torch.long).t().contiguous().to(self.args.device)

                return edge_index_att,edge_score_matrix.view(-1),None,loss_rec ,0

            elif self.args.att_meth == "flip":
                num_nodes = edge_score_matrix_.size(0)
                all_edges = set((min(u, v), max(u, v)) for u in range(num_nodes) for v in range(num_nodes))
                edge_index_set = set((min(u, v), max(u, v)) for u, v in edge_index.t().tolist())
                non_edge_index_set = all_edges - edge_index_set

                num_nodes = edge_score_matrix_.size(0)
                mask1 = torch.zeros_like(edge_score_matrix_)
                mask2 = torch.ones_like(edge_score_matrix_)    
                for u, v in edge_index_set:
                    if u != v:
                        mask1[u, v] = 1

                mask2 = mask2 - mask1 - mask1.t()
                mask2.fill_diagonal_(0)

                edge_score_matrix_1 = edge_score_matrix_ * mask1
                edge_score_matrix_2 = (1-edge_score_matrix_) * mask2
                edge_score_matrix_flat1 = edge_score_matrix_1.flatten()
                edge_score_matrix_flat2 = edge_score_matrix_2.flatten()

                if self.args.sample_ae == "probs":
                    temperature = self.args.tem_sample 
                    edge_index_set_flat1 = [u * edge_score_matrix_1.shape[1] + v for u, v in edge_index_set if u != v]
                    edge_score_matrix_flat_1= edge_score_matrix_flat1[edge_index_set_flat1]
                    probabilities1 = torch.softmax(edge_score_matrix_flat_1, dim=0) 
                    scaled_probabilities1 = torch.pow(probabilities1, 1 / temperature)
                    scaled_probabilities1 = scaled_probabilities1 / scaled_probabilities1.sum()  # 归一化
                    topk_indices_subset1 = torch.multinomial(scaled_probabilities1, num_edges_to_attack, replacement=False)
                    topk_indices1 = torch.tensor([edge_index_set_flat1[i] for i in topk_indices_subset1])

                    edge_index_set_flat2 = [u * edge_score_matrix_2.shape[1] + v for u, v in non_edge_index_set]
                    edge_score_matrix_flat_2= edge_score_matrix_flat2[edge_index_set_flat2]
                    probabilities2 = torch.softmax(edge_score_matrix_flat_2, dim=0)
                    scaled_probabilities2 = torch.pow(probabilities2, 1 / temperature)
                    scaled_probabilities2 = scaled_probabilities2 / scaled_probabilities2.sum()  # 归一化
                    topk_indices_subset2 = torch.multinomial(scaled_probabilities2, num_edges_to_attack, replacement=False)
                    topk_indices2 = torch.tensor([edge_index_set_flat2[i] for i in topk_indices_subset2])
                
                elif self.args.sample_ae == "topk":
                    edge_index_set_flat1 = [u * edge_score_matrix_1.shape[1] + v for u, v in edge_index_set if u != v]
                    edge_score_matrix_flat_1= edge_score_matrix_flat1[edge_index_set_flat1]
                    topk_indices_subset1= torch.topk(edge_score_matrix_flat_1, num_edges_to_attack).indices
                    topk_indices1 = torch.tensor([edge_index_set_flat1[i] for i in topk_indices_subset1])

                    edge_index_set_flat2 = [u * edge_score_matrix_2.shape[1] + v for u, v in non_edge_index_set]
                    edge_score_matrix_flat_2= edge_score_matrix_flat2[edge_index_set_flat2]
                    topk_indices_subset2= torch.topk(edge_score_matrix_flat_2, num_edges_to_attack).indices
                    topk_indices2 = torch.tensor([edge_index_set_flat2[i] for i in topk_indices_subset2])


                row_indices1 = topk_indices1 // num_nodes
                col_indices1 = topk_indices1 % num_nodes
                atk_edges1 = torch.stack([row_indices1, col_indices1], dim=0)
                new_edges1 = torch.tensor(atk_edges1, dtype=torch.long).contiguous().to(self.args.device)

                row_indices2 = topk_indices2 // num_nodes
                col_indices2 = topk_indices2 % num_nodes
                atk_edges2 = torch.stack([row_indices2, col_indices2], dim=0)
                new_edges2 = torch.tensor(atk_edges2, dtype=torch.long).contiguous().to(self.args.device)

                new_edges_set1 = set((min(u, v), max(u, v)) for u, v in new_edges1.t().tolist())
                new_edges_set2 = set((min(u, v), max(u, v)) for u, v in new_edges2.t().tolist())

                remaining_edges_set = (edge_index_set - new_edges_set1).union(new_edges_set2)

                bi_directional_edges = []
                for u, v in remaining_edges_set:
                    bi_directional_edges.append((u, v))
                    if u!=v:
                        bi_directional_edges.append((v, u))
                edge_index_att = torch.tensor(bi_directional_edges, dtype=torch.long).t().contiguous().to(self.args.device)

                return edge_index_att,edge_score_matrix.view(-1),topk_indices1,loss_rec ,0

        elif self.args.attack_type == 'other':
            edge_score_vec,loss_rec= pmodel(q,feats,edge_index)
            edge_score_vec_=1-edge_score_vec

            edge_index_set = set((min(u, v), max(u, v)) for u, v in edge_index.t().tolist())
            num_nodes_to_attack= self.budget
            if self.args.att_meth =="add":
                temperature = self.args.tem_sample
                if self.args.sample_ae == "probs":
                    probabilities1 = torch.softmax(edge_score_vec, dim=0)
                    scaled_probabilities1 = torch.pow(probabilities1, 1 / temperature)
                    scaled_probabilities1 = scaled_probabilities1 / scaled_probabilities1.sum()
                    topk_indices_subset1 = torch.multinomial(scaled_probabilities1, self.budget, replacement=False)
                elif self.args.sample_ae == "topk":
                    topk_indices_subset1 = torch.topk(edge_score_vec, self.budget).indices
                topk_indices1 = topk_indices_subset1

                if self.args.sample_ae == "probs":
                    probabilities2 = torch.softmax(edge_score_vec_, dim=0)
                    scaled_probabilities2 = torch.pow(probabilities2, 1 / temperature)
                    scaled_probabilities2 = scaled_probabilities2 / scaled_probabilities2.sum()
                    topk_indices_subset2 = torch.multinomial(scaled_probabilities2, self.budget, replacement=False)
                elif self.args.sample_ae == "topk":
                    topk_indices_subset2 = torch.topk(edge_score_vec_, self.budget).indices
                topk_indices2 = topk_indices_subset2

                edge_set_for_add = set()

                # 选择添加的边
                i=0
                while len(edge_set_for_add) < self.budget:
                    i+=1
                    if i>1000:
                        break
                    if self.args.add_way == "pagerank":
                        low_node_all = topk_indices1
                        low_node_pagerank = feats[:,-1][low_node_all]
                        low_node_probabilities = torch.softmax(low_node_pagerank, dim=0)
                        node1 = low_node_all[torch.multinomial(low_node_probabilities, 1).item()]
                    elif self.args.add_way == "random":
                        node1 = np.random.choice(topk_indices1.tolist())

                    node2 = np.random.choice(topk_indices2.tolist())
                    if ((node1, node2) not in edge_index_set and (node2, node1) not in edge_index_set) and node1 != node2 and (node1, node2) not in edge_set_for_add:
                        edge_set_for_add.add((min(node1, node2),max(node1, node2)))

                remaining_edges_set = edge_index_set.union(edge_set_for_add)
                bi_directional_edges = []
                for u, v in remaining_edges_set:
                    bi_directional_edges.append((u, v))
                    if u != v:
                        bi_directional_edges.append((v, u))
                edge_index_att = torch.tensor(bi_directional_edges, dtype=torch.long).t().contiguous().to(self.args.device)

                return edge_index_att, edge_score_vec.view(-1), (topk_indices1,topk_indices2), loss_rec, 0
            
            elif self.args.att_meth == "del":
                temperature = self.args.tem_sample
                if self.args.sample_ae == "probs":
                    probabilities1 = torch.softmax(edge_score_vec, dim=0)

                    scaled_probabilities1 = torch.pow(probabilities1, 1 / temperature)
                    scaled_probabilities1 = scaled_probabilities1 / scaled_probabilities1.sum()
                    topk_indices_subset1 = torch.multinomial(scaled_probabilities1, self.budget, replacement=False)
                elif self.args.sample_ae == "topk":
                    topk_indices_subset1 = torch.topk(edge_score_vec, self.budget).indices
                topk_indices1 = topk_indices_subset1

                if self.args.sample_ae == "probs":
                    probabilities2 = torch.softmax(edge_score_vec_, dim=0)
                    scaled_probabilities2 = torch.pow(probabilities2, 1 / temperature)
                    scaled_probabilities2 = scaled_probabilities2 / scaled_probabilities2.sum()
                    topk_indices_subset2 = torch.multinomial(scaled_probabilities2, self.budget, replacement=False)
                elif self.args.sample_ae == "topk":
                    topk_indices_subset2 = torch.topk(edge_score_vec_, self.budget).indices
                topk_indices2 = topk_indices_subset2
                edge_set_for_del = set()

                # 选择删除的边
                i=0
                while len(edge_set_for_del) < self.budget:
                    i+=1
                    if i>1000:
                        break
                    node1 = np.random.choice(topk_indices1.tolist())
                    neighbors = [v for u, v in edge_index.t().tolist() if u == node1 or v == node1]
                    if neighbors:
                        neighbor_scores = edge_score_vec[neighbors]
                        neighbor_probabilities = torch.softmax(neighbor_scores, dim=0)
                        node2 = neighbors[torch.multinomial(neighbor_probabilities, 1).item()]
                        # node2 = neighbors[torch.argmax(neighbor_scores).item()]
                        if node1 != node2 and (node1, node2) not in edge_set_for_del:
                            edge_set_for_del.add((min(node1, node2),max(node1, node2)))

                remaining_edges_set = edge_index_set - edge_set_for_del
                bi_directional_edges = []
                for u, v in remaining_edges_set:
                    bi_directional_edges.append((u, v))
                    if u != v:
                        bi_directional_edges.append((v, u))
                edge_index_att = torch.tensor(bi_directional_edges, dtype=torch.long).t().contiguous().to(self.args.device)

                return edge_index_att, edge_score_vec.view(-1), (topk_indices1,topk_indices2), loss_rec, 0

            elif self.args.att_meth == "flip":
                temperature = self.args.tem_sample
                if self.args.sample_ae == "probs":
                    probabilities1 = torch.softmax(edge_score_vec, dim=0)

                    scaled_probabilities1 = torch.pow(probabilities1, 1 / temperature)
                    scaled_probabilities1 = scaled_probabilities1 / scaled_probabilities1.sum()
                    topk_indices_subset1 = torch.multinomial(scaled_probabilities1, self.budget, replacement=False)
                elif self.args.sample_ae == "topk":
                    topk_indices_subset1 = torch.topk(edge_score_vec, self.budget).indices
                topk_indices1 = topk_indices_subset1

                if self.args.sample_ae == "probs":
                    probabilities2 = torch.softmax(edge_score_vec_, dim=0)
                    scaled_probabilities2 = torch.pow(probabilities2, 1 / temperature)
                    scaled_probabilities2 = scaled_probabilities2 / scaled_probabilities2.sum()
                    topk_indices_subset2 = torch.multinomial(scaled_probabilities2, self.budget, replacement=False)
                elif self.args.sample_ae == "topk":
                    topk_indices_subset2 = torch.topk(edge_score_vec_, self.budget).indices
                topk_indices2 = topk_indices_subset2

                edge_set_for_add = set()
                edge_set_for_del = set()

                # 选择删除的边
                i,j=0,0
                while len(edge_set_for_del) < self.budget:
                    i+=1
                    if i>1000:
                        break
                    node1 = np.random.choice(topk_indices1.tolist())
                    neighbors = [v for u, v in edge_index.t().tolist() if u == node1 or v == node1]
                    if neighbors:
                        neighbor_scores = edge_score_vec[neighbors]
                        neighbor_probabilities = torch.softmax(neighbor_scores, dim=0)
                        node2 = neighbors[torch.multinomial(neighbor_probabilities, 1).item()]
                        # node2 = neighbors[torch.argmax(neighbor_scores).item()]
                        if node1 != node2 and (node1, node2) not in edge_set_for_del:
                            edge_set_for_del.add((min(node1, node2),max(node1, node2)))

                # 选择添加的边
                while len(edge_set_for_add) < self.budget:
                    j+=1
                    if j>1000:
                        break
                    if self.args.add_way == "pagerank":
                        low_node_all = topk_indices1
                        low_node_pagerank = feats[:,-1][low_node_all]
                        low_node_probabilities = torch.softmax(low_node_pagerank, dim=0)
                        node1 = low_node_all[torch.multinomial(low_node_probabilities, 1).item()]
                    elif self.args.add_way == "random":
                        node1 = np.random.choice(topk_indices1.tolist())

                    node2 = np.random.choice(topk_indices2.tolist())
                    if ((node1, node2) not in edge_index_set and (node2, node1) not in edge_index_set) and node1 != node2 and (node1, node2) not in edge_set_for_add:
                        edge_set_for_add.add((min(node1, node2),max(node1, node2)))

                remaining_edges_set = (edge_index_set - edge_set_for_del).union(edge_set_for_add)
                bi_directional_edges = []
                for u, v in remaining_edges_set:
                    bi_directional_edges.append((u, v))
                    if u != v:
                        bi_directional_edges.append((v, u))
                edge_index_att = torch.tensor(bi_directional_edges, dtype=torch.long).t().contiguous().to(self.args.device)

                return edge_index_att, edge_score_vec.view(-1), (topk_indices1,topk_indices2), loss_rec, 0


    def edge_index_to_adj(self,edge_index, num_nodes):
        adj = torch.zeros((num_nodes, num_nodes), dtype=torch.float32, device=edge_index.device)
        adj[edge_index[0], edge_index[1]] = 1
        return adj
    
    def delete_k_edge_min_new(self,x, new_adj, ori_adj, k=10):
        if self.args.attack_type == 'proposed':
            device = new_adj.device
            ones = torch.ones_like(new_adj, dtype=torch.float32)
            max_value = torch.max(new_adj)
            lower_bool_label = torch.tril(ori_adj, diagonal=-1)
            upper_ori_label = ori_adj - lower_bool_label  # 没有对角线
            upper_bool_label = upper_ori_label.bool()

            new_adj_for_del_exist = torch.where(upper_bool_label, new_adj, torch.zeros_like(new_adj))
            new_adj_for_del_exist_exp = torch.exp(new_adj_for_del_exist)
            new_adj_for_del_exist_softmax = new_adj_for_del_exist_exp / torch.sum(new_adj_for_del_exist_exp)
            new_adj_for_del_exist_softmax = new_adj_for_del_exist_softmax.view(-1)
            exist_indexes = torch.multinomial(new_adj_for_del_exist_softmax, k, replacement=False)

            new_adj_for_del_nonexist = torch.where(~upper_bool_label, new_adj, ones * max_value)
            new_adj_for_del_nonexist = max_value - new_adj_for_del_nonexist
            new_adj_for_del_nonexist_exp = torch.exp(new_adj_for_del_nonexist)
            new_adj_for_del_nonexist_exp = torch.where(~upper_bool_label, new_adj_for_del_nonexist_exp, torch.zeros_like(new_adj_for_del_nonexist_exp))
            new_adj_for_del_nonexist_softmax = new_adj_for_del_nonexist_exp / torch.sum(new_adj_for_del_nonexist_exp)
            new_adj_for_del_nonexist_softmax = new_adj_for_del_nonexist_softmax.view(-1)
            nonexist_indexes = torch.multinomial(new_adj_for_del_nonexist_softmax, k, replacement=False)

            new_indexes_del = exist_indexes
            new_indexed_add = nonexist_indexes
            mask_del = upper_bool_label.view(-1)
            mask_add = ~upper_bool_label.view(-1)
            for i in range(k):
                delete_mask_idx = new_indexes_del[i].item()
                delete_onehot_mask = F.one_hot(torch.tensor(delete_mask_idx, device=device), num_classes=new_adj.size(0) * new_adj.size(0)).to(device)
                delete_onehot_mask = delete_onehot_mask.bool()
                mask_del = torch.where(delete_onehot_mask, torch.zeros_like(delete_onehot_mask), mask_del)

                add_mask_idx = new_indexed_add[i].item()
                add_onehot_mask = F.one_hot(torch.tensor(add_mask_idx, device=device), num_classes=new_adj.size(0) * new_adj.size(0)).to(device)
                add_onehot_mask = add_onehot_mask.bool()
                mask_add = torch.where(add_onehot_mask, torch.ones_like(add_onehot_mask), mask_add)

            mask_del = mask_del.view(-1)
            mask_add = mask_add.view(-1)
            if self.args.att_meth == "flip":
                new_adj_out = torch.where(mask_del, new_adj.view(-1), torch.zeros_like(new_adj.view(-1)))
                new_adj_out = torch.where(mask_add, torch.ones_like(new_adj_out.view(-1)), new_adj_out.view(-1))
                ori_adj_out = torch.where(mask_del, ori_adj.view(-1), torch.zeros_like(ori_adj.view(-1)))
                ori_adj_out = torch.where(mask_add, torch.ones_like(ori_adj.view(-1)), ori_adj_out)
            elif self.args.att_meth == "add":
                new_adj_out = torch.where(mask_add, torch.ones_like(new_adj.view(-1)), new_adj.view(-1))
                ori_adj_out = torch.where(mask_add, torch.ones_like(ori_adj.view(-1)), ori_adj.view(-1))
            elif self.args.att_meth == "del":
                new_adj_out = torch.where(mask_del, new_adj.view(-1), torch.zeros_like(new_adj.view(-1)))
                ori_adj_out = torch.where(mask_del, ori_adj.view(-1), torch.zeros_like(ori_adj.view(-1)))

           
            ori_adj_out = ori_adj_out.view(new_adj.size(0), new_adj.size(0))
            ori_adj_out = ori_adj_out + (ori_adj_out.t() - torch.diag(torch.diag(ori_adj_out)))

            new_indexes=(exist_indexes,nonexist_indexes)
            return new_adj_out.view(new_adj.size()), ori_adj_out, new_indexes
            
        elif self.args.attack_type == 'other':
            pagerank_value=x[:,-1]
            device = new_adj.device
            node_scores = new_adj
            node_scores_softmax = torch.softmax(node_scores, dim=0)
            if self.args.sample_ae == "probs":
                high_score_indexes = torch.multinomial(node_scores_softmax, k, replacement=False)
                low_score_indexes = torch.multinomial(1 - node_scores_softmax, k, replacement=False)
            elif self.args.sample_ae == "topk":
                high_score_indexes = torch.topk(node_scores_softmax, k).indices
                low_score_indexes = torch.topk(1 - node_scores_softmax,k).indices
            
            low_score_vec=[]
            high_score_vec=[]
            if self.args.att_meth == "flip":
                delete_edges = set()
                i=0
                while len(delete_edges) < self.budget:
                    i+=1
                    if i>1000:
                        break
                    node = high_score_indexes[torch.randint(len(high_score_indexes), (1,)).item()]
                    neighbors = torch.nonzero(ori_adj[node]).view(-1)
                    if len(neighbors) > 0:
                        neighbor_scores = node_scores[neighbors]
                        neighbor_probabilities = torch.softmax(neighbor_scores, dim=0)
                        # max_score_neighbor = neighbors[torch.argmax(neighbor_scores)]
                        max_score_neighbor = neighbors[torch.multinomial(neighbor_probabilities, 1).item()]
                        node1,node2=node.item(), max_score_neighbor.item()
                        if (min(node1,node2), max(node1,node2)) not in delete_edges and node1 != node2:
                            delete_edges.add((min(node1,node2), max(node1,node2)))
                            high_score_vec.append((node_scores[node1]+node_scores[node2])/2)

                add_edges = set()
                i=0
                while len(add_edges) < self.budget:
                    i+=1
                    if i>1000:
                        break
                    if self.args.add_way == "pagerank":
                        low_node_all = low_score_indexes
                        low_node_pagerank = pagerank_value[low_node_all]
                        low_node_probabilities = torch.softmax(low_node_pagerank, dim=0)
                        low_node = low_node_all[torch.multinomial(low_node_probabilities, 1).item()]
                    elif self.args.add_way == "random":
                        low_node = low_score_indexes[torch.randint(len(low_score_indexes), (1,)).item()]

                    high_node = high_score_indexes[torch.randint(len(high_score_indexes), (1,)).item()]
                    node1,node2=low_node.item(), high_node.item()
                    if ori_adj[low_node, high_node] == 0 and (min(node1,node2), max(node1,node2)) not in add_edges and node1 != node2:
                        add_edges.add((min(node1,node2), max(node1,node2)))
                        low_score_vec.append((node_scores[node1]+node_scores[node2])/2)

                mask = ori_adj.clone()
                for edge in delete_edges:
                    mask[edge[0], edge[1]] = 0
                    mask[edge[1], edge[0]] = 0

                for edge in add_edges:
                    mask[edge[0], edge[1]] = 1
                    mask[edge[1], edge[0]] = 1
            elif self.args.att_meth == "add":
                add_edges = set()
                i=0

                while len(add_edges) < self.budget:
                    i+=1
                    if i>1000:
                        break
                    if self.args.add_way == "pagerank":
                        low_node_all = low_score_indexes
                        low_node_pagerank = pagerank_value[low_node_all]
                        low_node_probabilities = torch.softmax(low_node_pagerank, dim=0)
                        low_node = low_node_all[torch.multinomial(low_node_probabilities, 1).item()]
                    elif self.args.add_way == "random":
                        low_node = low_score_indexes[torch.randint(len(low_score_indexes), (1,)).item()]

                    high_node = high_score_indexes[torch.randint(len(high_score_indexes), (1,)).item()]
                    node1,node2=low_node.item(), high_node.item()
                    if ori_adj[low_node, high_node] == 0 and (min(node1,node2), max(node1,node2)) not in add_edges and node1 != node2:
                        add_edges.add((min(node1,node2), max(node1,node2)))
                        low_score_vec.append((node_scores[node1]+node_scores[node2])/2)

                mask = ori_adj.clone()

                for edge in add_edges:
                    mask[edge[0], edge[1]] = 1
                    mask[edge[1], edge[0]] = 1
            elif self.args.att_meth == "del":
                delete_edges = set()
                i=0
                while len(delete_edges) < self.budget:
                    i+=1
                    if i>1000:
                        break
                    node = high_score_indexes[torch.randint(len(high_score_indexes), (1,)).item()]
                    neighbors = torch.nonzero(ori_adj[node]).view(-1)
                    if len(neighbors) > 0:
                        neighbor_scores = node_scores[neighbors]
                        neighbor_probabilities = torch.softmax(neighbor_scores, dim=0)
                        # max_score_neighbor = neighbors[torch.argmax(neighbor_scores)]
                        max_score_neighbor = neighbors[torch.multinomial(neighbor_probabilities, 1).item()]
                        node1,node2=node.item(), max_score_neighbor.item()
                        if (min(node1,node2), max(node1,node2)) not in delete_edges and node1 != node2:
                            delete_edges.add((min(node1,node2), max(node1,node2)))
                            high_score_vec.append((node_scores[node1]+node_scores[node2])/2)

                mask = ori_adj.clone()
                for edge in delete_edges:
                    mask[edge[0], edge[1]] = 0
                    mask[edge[1], edge[0]] = 0

            ori_adj_out = ori_adj * mask
            new_adj_out = ori_adj_out.clone()
            return new_adj_out, ori_adj_out, (torch.tensor(low_score_vec).to(device), torch.tensor(high_score_vec).to(device))

    def cal_perturbation_global(self,ori_adj,new_adj_out):

        def compute_alpha(n, S_d, d_min):

            # Ensure inputs are torch tensors for GPU support (if needed)
            n = torch.tensor(n, dtype=torch.float32)
            S_d = torch.tensor(S_d, dtype=torch.float32)
            d_min = torch.tensor(d_min, dtype=torch.float32)

            # Calculate alpha using the power law approximation
            alpha = n / (S_d - n * torch.log(d_min - 0.5)) + 1
            return alpha

        def compute_log_likelihood(n, alpha, S_d, d_min):
            # Ensure inputs are torch tensors for GPU support (if needed)
            n = torch.tensor(n, dtype=torch.float32)
            alpha = torch.tensor(alpha, dtype=torch.float32)
            S_d = torch.tensor(S_d, dtype=torch.float32)
            d_min = torch.tensor(d_min, dtype=torch.float32)

            # Calculate log likelihood using the formula
            log_likelihood = (n * torch.log(alpha) + 
                            n * alpha * torch.log(d_min) - 
                            (alpha + 1) * S_d)
            return log_likelihood
        
        d_min = 2
        degree_sequence_start = ori_adj.sum(0)
        degree_sequence_weighted = new_adj_out.sum(0)
        
        degree_sequence_start = torch.tensor(degree_sequence_start, dtype=torch.float32)
        degree_sequence_weighted = torch.tensor(degree_sequence_weighted, dtype=torch.float32)

        S_d_start = torch.sum(torch.log(degree_sequence_start[degree_sequence_start >= d_min]))
        current_S_d = torch.sum(torch.log(degree_sequence_weighted[degree_sequence_weighted >= d_min]))

        n_start = torch.sum(degree_sequence_start >= d_min).item()
        current_n = torch.sum(degree_sequence_weighted >= d_min).item()

        alpha_start = compute_alpha(n_start, S_d_start, d_min)
        log_likelihood_orig = compute_log_likelihood(n_start, alpha_start, S_d_start, d_min)

        current_alpha = compute_alpha(current_n, current_S_d, d_min)
        log_likelihood_current = compute_log_likelihood(current_n, current_alpha, current_S_d, d_min)

        log_likelihood_diff = log_likelihood_current - log_likelihood_orig

        return log_likelihood_diff

    def train_atk(self, surrogate_model,train_tasks,valid_tasks,test_tasks):
            # initial_state_dict = {key: value.clone() for key, value in self.pmodel.state_dict().items()}
            print("-----Start train & test attack model-----")
            logger=logging.getLogger('train')
            logger.info("-----Start train & test attack model-----")
            tasks=train_tasks.get_batch()
            args=self.args 
            # for epoch in range(self.args.att_epochs):
            #     self.pmodel.train()
            #     loss_c=[]
            #     for i in range(len(tasks)):
            #         self.optimizer.zero_grad()
            #         batch_ = tasks[i]
            #         # print(batch)
            #         batch = batch_.clone()
            #         if self.args.cuda:
            #             batch = batch.to(self.args.device)

            #         atk_edge_index,edge_score_matrix_flat,indice,loss_rec,distance_loss=\
            #             self.attack(batch.y,self.pmodel,batch.x,batch.query,batch.edge_index)

            #         # if(epoch>3):
            #         #     plot_graphs(batch.query,batch.edge_index,atk_edge_index,batch.y)
            #         surrogate_model.eval()
            #         loss_,h= surrogate_model(batch.query,batch.pos,atk_edge_index,batch.x)
                    
            #         numerator = torch.mm(h[batch.query], h.t())
            #         norm = torch.norm(h, dim=-1, keepdim=True)
            #         denominator = torch.mm(norm[batch.query], norm.t())
            #         sim = numerator / denominator
            #         pred = torch.sigmoid(sim).view(-1,1)


            #         loss= 0.0001*loss_rec-self.criterion(pred, batch.y)*torch.log(edge_score_matrix_flat[indice]).sum()
            #         loss_c.append(loss.item())
            #         # print("loss:{}, loss_rec:{}, loss_:{}".format(loss,0.0001*loss_rec,loss-0.0001*loss_rec))
            #         loss.backward()

            #         self.optimizer.step()

            #     print("epoch:{} loss:{}".format(epoch,np.mean(loss_c)))
            #     self.test_atk(test_tasks,epoch)
            #     # self.valid_atk(valid_tasks,epoch)
            #     # self.test_atk(train_tasks,epoch)


            if self.args.draw_loss:
                loss_epoch1=[]
                loss_epoch2=[]
                loss_epoch3=[]
                loss_epoch4=[]
            
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', factor=args.decay_factor, patience=args.decay_patience)
            train_time=0.0
            attack_time=0.0
            for epoch in range(self.args.att_epochs):
                self.pmodel.train()
                if self.args.draw_loss:
                    loss_a=[]
                    loss_b=[]
                    loss_c=[]
                    loss_d=[]
                epoch_loss= 0.0
                start_time = time.time()
                for i in range(len(tasks)):
                    self.optimizer.zero_grad()
                    batch_ = tasks[i]
                    # print(batch)
                    batch = batch_.clone()
                    if self.args.cuda:
                        batch = batch.to(self.args.device)

                    new_adj,loss_rec= self.pmodel(batch.query,batch.x,batch.edge_index)

                    new_adj=(new_adj+new_adj.t())/2 
                    ori_adj= self.edge_index_to_adj(batch.edge_index,batch.x.size(0))
                    
                    k = self.budget
                    if self.args.attack_type == 'proposed':
                        new_adj_out, _,new_indexes= self.delete_k_edge_min_new(batch.x,new_adj, ori_adj, k)
                        score_ward = torch.log(new_adj.view(-1)[new_indexes[0]]).sum()-torch.log(new_adj.view(-1)[new_indexes[1]]).sum()
                    elif self.args.attack_type == 'other':
                        # edge_index_att, edge_score_vec, indices_, loss_rec, distance_loss = self.attack(batch.y,self.pmodel,batch.x,batch.query,batch.edge_index)
                        # new_adj_out = self.edge_index_to_adj(edge_index_att, batch.x.size(0))
                        new_adj_out, _,new_indexes= self.delete_k_edge_min_new(batch.x,new_adj, ori_adj, k)
                        score_ward = torch.log(new_indexes[0]).sum()-torch.log(new_indexes[1]).sum()

                    surrogate_model.eval()
                    pred= surrogate_model(batch.query,batch.x,new_adj_out).squeeze()
                    pred2= surrogate_model(batch.query,batch.x,ori_adj).squeeze()

                    loss_1= self.criterion(pred2,batch.y)-self.criterion(pred, batch.y)
                    loss_2=new_adj.sum()
                    loss_3=self.cal_perturbation_global(ori_adj,new_adj_out)
                    loss_4=loss_rec
                    loss=(loss_1*self.args.lamd1+loss_3*self.args.lamd2)*score_ward+loss_4*self.args.lamd3+loss_2*self.args.lamd4

                    with torch.no_grad():
                        if self.args.draw_loss:
                            loss_a.append(self.criterion(pred, batch.y).item())
                            loss_b.append(loss_2.item()*0.001)
                            loss_c.append(loss_3.item()*0.01)
                            loss_d.append(loss_4.item()*0.01)
                    loss.backward()
                    self.optimizer.step()
                    #torch.cuda.empty_cache()
                #torch.cuda.empty_cache()
                epoch_loss+=loss
                if scheduler is not None:
                    scheduler.step(epoch_loss)
                    # print("epoch:{} loss:{}".format(epoch,np.mean(loss_c)))

                end_time = time.time() 
                elapsed_time1 = end_time - start_time 
                train_time+=elapsed_time1 

                start_time = time.time()
                self.test_atk(test_tasks,epoch)
                end_time = time.time() 
                self.valid_atk(valid_tasks,epoch)
                elapsed_time2 = end_time - start_time 
                attack_time+=elapsed_time2

                if self.args.draw_loss:
                    loss_epoch1.append(np.mean(loss_a))
                    loss_epoch2.append(np.mean(loss_b))
                    loss_epoch3.append(np.mean(loss_c))
                    loss_epoch4.append(np.mean(loss_d))
                    logger.info("epoch:{} loss:{} loss2:{} loss3:{} loss4:{}".format(epoch,np.mean(loss_a),np.mean(loss_b),np.mean(loss_c),np.mean(loss_d)))
            if self.args.draw_loss:
                import os
                parent_dir = os.path.abspath(os.path.dirname(os.path.abspath(__file__)))
                root_loss=os.path.join(parent_dir,"results","loss")
                # print(root_loss)
                plt.figure()
                plt.plot(range(self.args.att_epochs), loss_epoch1, label='Training Loss')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.title('Training Loss over Epochs')
                plt.legend()
                p1=os.path.join(root_loss,"loss1.png")
                plt.savefig(p1)
                plt.close()

                plt.figure()
                plt.plot(range(self.args.att_epochs), loss_epoch2, label='Training Loss')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.title('Training Loss over Epochs')
                plt.legend()
                p2=os.path.join(root_loss,"loss2.png")
                plt.savefig(p2)
                plt.close()

                plt.figure()
                plt.plot(range(self.args.att_epochs), loss_epoch3, label='Training Loss')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.title('Training Loss over Epochs')
                plt.legend()
                p3=os.path.join(root_loss,"loss3.png")
                plt.savefig(p3)
                plt.close()

                plt.figure()
                plt.plot(range(self.args.att_epochs), loss_epoch4, label='Training Loss')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.title('Training Loss over Epochs')
                plt.legend()
                p4=os.path.join(root_loss,"loss4.png")
                plt.savefig(p4)
                plt.close()
            
            print("Train time :{}".format(train_time))
            print("Attack time:{}".format(attack_time))
            logger.info("Train time :{}".format(train_time))
            logger.info("Attack time:{}".format(attack_time))
            logger.info("________________________________________________________________________________________________________________________")
            logger.info("________________________________________________________________________________________________________________________")
            
    def test_atk(self, test_tasks,epoch):
        args=self.args
        acc_all_a,acc_all_o=[],[]
        pre_all_a,pre_all_o=[],[]
        rec_all_a,rec_all_o=[],[]
        f1_all_a,f1_all_o=[],[]

        vmodel=self.vmodel
        logger=logging.getLogger('test')
        if args.vmodel == 'cgnp':
            
            self.pmodel.eval()
            for i in range(len(test_tasks)):
                task = test_tasks[i]
                support_batch = task.get_support_batch()
                query_batch = task.get_query_batch()

                # only attack the support batch
                if args.cuda:
                    support_batch = support_batch.to(args.device)
                    query_batch = query_batch.to(args.device)
                
                edge_indexs = support_batch.edge_index.clone()
                edge_indexq = query_batch.edge_index.clone()
                # print("+++++++++++++++test attack++++++++++++++++++")
                atk_edge_indexs,_,__,loss_rec,distance_loss=self.attack(support_batch.y,self.pmodel,support_batch.x,support_batch.query[0],support_batch.edge_index)

                # atk_edge_indexq,_,__=self.attack(self.pmodel,query_batch.x,query_batch.query,query_batch.edge_index)

                # print("evluate test attack")
                acc_a, precision_a, recall_a, f1_a=self.evaluate(vmodel,(support_batch,query_batch),(atk_edge_indexs,None))
                # print("evluate test origin")
                acc_o, precision_o, recall_o, f1_o=self.evaluate(vmodel,(support_batch,query_batch),(edge_indexs,None))

                #torch.cuda.empty_cache()
                acc_all_a.append(acc_a)
                acc_all_o.append(acc_o)
                pre_all_a.append(precision_a)
                pre_all_o.append(precision_o)
                rec_all_a.append(recall_a)
                rec_all_o.append(recall_o)
                f1_all_a.append(f1_a)
                f1_all_o.append(f1_o)


        else:
            tasks=test_tasks.get_batch()
            args=self.args  

            self.pmodel.eval()
            for i in range(len(tasks)):
                batch_ = tasks[i]
                # print(batch)
                batch = batch_.clone()
                if self.args.cuda:
                    batch = batch.to(self.args.device)
                edge_index=batch.edge_index.clone()
                # print(batch.query)

                atk_edge_index,_,__,loss_rec,distance_loss=self.attack(batch.y,self.pmodel,batch.x,batch.query,batch.edge_index)
                if(epoch>5) and self.args.draw_pic:
                    plot_graphs(batch.query,edge_index,atk_edge_index,batch.y)

                with torch.no_grad():
                    # print(atk_edge_index.size(),edge_index.size())
                    acc_a, precision_a, recall_a, f1_a=self.evaluate(vmodel,batch,atk_edge_index)
                    acc_o, precision_o, recall_o, f1_o=self.evaluate(vmodel,batch,edge_index)

                #torch.cuda.empty_cache()
                acc_all_a.append(acc_a)
                acc_all_o.append(acc_o)
                pre_all_a.append(precision_a)
                pre_all_o.append(precision_o)
                rec_all_a.append(recall_a)
                rec_all_o.append(recall_o)
                f1_all_a.append(f1_a)
                f1_all_o.append(f1_o)

        #torch.cuda.empty_cache()
        acc_test_a = np.mean(acc_all_a)
        precision_test_a = np.mean(pre_all_a)
        recall_test_a = np.mean(rec_all_a)
        f1_test_a = np.mean(f1_all_a)

        acc_test_o = np.mean(acc_all_o)
        precision_test_o = np.mean(pre_all_o)
        recall_test_o = np.mean(rec_all_o)
        f1_test_o = np.mean(f1_all_o)

        print("epoch:{} After attcked test Result: Acc={:.4f}, Pre={:.4f}, Recall={:.4f}, F1={:.4f}"
          .format(epoch,acc_test_a, precision_test_a, recall_test_a, f1_test_a))
        print("epoch:{} Before attcked test Result: Acc={:.4f}, Pre={:.4f}, Recall={:.4f}, F1={:.4f}"
          .format(epoch,acc_test_o, precision_test_o, recall_test_o, f1_test_o))
        logger.info("epoch:{} After attcked test Result: Acc={:.4f}, Pre={:.4f}, Recall={:.4f}, F1={:.4f}"
            .format(epoch,acc_test_a, precision_test_a, recall_test_a, f1_test_a))
        logger.info("epoch:{} Before attcked test Result: Acc={:.4f}, Pre={:.4f}, Recall={:.4f}, F1={:.4f}"
            .format(epoch,acc_test_o, precision_test_o, recall_test_o, f1_test_o))
        
    def convert_to_unidirectional(self,atk_edge_indexs):
        edge_index_np = atk_edge_indexs.cpu().numpy()

        unidirectional_edges = set()
        for u, v in edge_index_np.T:
            unidirectional_edges.add((min(u, v), max(u, v)))

        unidirectional_edges = torch.tensor(list(unidirectional_edges), dtype=torch.long).t().contiguous()
        return unidirectional_edges.to(atk_edge_indexs.device)


    def valid_atk(self, valid_tasks,epoch):
        args=self.args
        acc_all_a,acc_all_o=[],[]
        pre_all_a,pre_all_o=[],[]
        rec_all_a,rec_all_o=[],[]
        f1_all_a,f1_all_o=[],[]

        vmodel=self.vmodel
        logger=logging.getLogger('valid')   
        if args.vmodel == 'cgnp':
            
            self.pmodel.eval()
            for i in range(len(valid_tasks)):
                task = valid_tasks[i]
                support_batch = task.get_support_batch()
                query_batch = task.get_query_batch()

                # only attack the support batch
                if args.cuda:
                    support_batch = support_batch.to(args.device)
                    query_batch = query_batch.to(args.device)
                
                edge_indexs = support_batch.edge_index.clone()
                edge_indexq = query_batch.edge_index.clone()
                # print("+++++++++++++++valid attack++++++++++++++++++")
                atk_edge_indexs,_,__,loss_rec,distance_loss=self.attack(support_batch.y,self.pmodel,support_batch.x,support_batch.query[0],support_batch.edge_index)

                # atk_edge_indexq,_,__=self.attack(self.pmodel,query_batch.x,query_batch.query,query_batch.edge_index)
                # atk_edge_indexs = self.convert_to_unidirectional(atk_edge_indexs)

                # print("evluate valid attack")
                acc_a, precision_a, recall_a, f1_a=self.evaluate(vmodel,(support_batch,query_batch),(atk_edge_indexs,None))
                # print("evluate valid origin")
                acc_o, precision_o, recall_o, f1_o=self.evaluate(vmodel,(support_batch,query_batch),(edge_indexs,None))

                #torch.cuda.empty_cache()
                acc_all_a.append(acc_a)
                acc_all_o.append(acc_o)
                pre_all_a.append(precision_a)
                pre_all_o.append(precision_o)
                rec_all_a.append(recall_a)
                rec_all_o.append(recall_o)
                f1_all_a.append(f1_a)
                f1_all_o.append(f1_o)


        else:
            tasks=valid_tasks.get_batch()
            args=self.args  

            self.pmodel.eval()
            for i in range(len(tasks)):
                batch_ = tasks[i]
                # print(batch)
                batch = batch_.clone()
                if self.args.cuda:
                    batch = batch.to(self.args.device)
                edge_index=batch.edge_index.clone()
                # print(batch.query)
                atk_edge_index,_,__,loss_rec,distance_loss=self.attack(batch.y,self.pmodel,batch.x,batch.query,batch.edge_index)
                if(epoch>5) and self.args.draw_pic:
                    plot_graphs(batch.query,edge_index,atk_edge_index,batch.y)

                with torch.no_grad():
                    # print(atk_edge_index.size(),edge_index.size())
                    acc_a, precision_a, recall_a, f1_a=self.evaluate(vmodel,batch,atk_edge_index)
                    acc_o, precision_o, recall_o, f1_o=self.evaluate(vmodel,batch,edge_index)

                #torch.cuda.empty_cache()
                acc_all_a.append(acc_a)
                acc_all_o.append(acc_o)
                pre_all_a.append(precision_a)
                pre_all_o.append(precision_o)
                rec_all_a.append(recall_a)
                rec_all_o.append(recall_o)
                f1_all_a.append(f1_a)
                f1_all_o.append(f1_o)

        #torch.cuda.empty_cache()
        acc_valid_a = np.mean(acc_all_a)
        precision_valid_a = np.mean(pre_all_a)
        recall_valid_a = np.mean(rec_all_a)
        f1_valid_a = np.mean(f1_all_a)

        acc_valid_o = np.mean(acc_all_o)
        precision_valid_o = np.mean(pre_all_o)
        recall_valid_o = np.mean(rec_all_o)
        f1_valid_o = np.mean(f1_all_o)

        print("epoch:{} After attcked valid Result: Acc={:.4f}, Pre={:.4f}, Recall={:.4f}, F1={:.4f}"
          .format(epoch,acc_valid_a, precision_valid_a, recall_valid_a, f1_valid_a))
        
        print("epoch:{} Before attcked valid Result: Acc={:.4f}, Pre={:.4f}, Recall={:.4f}, F1={:.4f}"
          .format(epoch,acc_valid_o, precision_valid_o, recall_valid_o, f1_valid_o))
        
        logger.info("epoch:{} After attcked valid Result: Acc={:.4f}, Pre={:.4f}, Recall={:.4f}, F1={:.4f}"
            .format(epoch,acc_valid_a, precision_valid_a, recall_valid_a, f1_valid_a))
        logger.info("epoch:{} Before attcked valid Result: Acc={:.4f}, Pre={:.4f}, Recall={:.4f}, F1={:.4f}"
            .format(epoch,acc_valid_o, precision_valid_o, recall_valid_o, f1_valid_o))
        
    def evaluate(self, vmodel,batch,atk_edge_index):
        args=self.args
        with torch.no_grad():
            if args.vmodel == 'cgnp':
                vmodel.eval()
                support_batch,query_batch = batch

                # print("support_batch:",support_batch.edge_index)
                # print("batch:",atk_edge_index)
                atk_edge_indexs,atk_edge_indexq = atk_edge_index
                # query_batch.edge_index = atk_edge_indexq
                support_batch.edge_index = atk_edge_indexs

                output, y, mask = vmodel(support_batch, query_batch)
                pred = torch.where(output > args.thre, 1, 0)
                batch = query_batch

            else:
                vmodel.eval()
                ######################### COCLEP #########################
                if args.vmodel == 'coclep':
                    atk_edge_index_aug = hypergraph_construction(atk_edge_index,len(batch.y),k=2) # construct hypergraph
                    # atk_edge_index_aug = atk_edge_index
                    loss,h= vmodel(batch.query,batch.pos,atk_edge_index,atk_edge_index_aug,batch.x)
                    numerator = torch.mm(h[batch.query], h.t())
                    norm = torch.norm(h, dim=-1, keepdim=True)
                    denominator = torch.mm(norm[batch.query], norm.t())
                    sim = numerator / denominator
                    output = torch.sigmoid(sim).view(-1,1)
                    pred = torch.where(output > args.thre, 1, 0)

                ######################### ICSGNN #########################
                elif args.vmodel == 'icsgnn':
                    graph=batch.graph[0]
                    for src, dst in atk_edge_index.t().tolist():
                        graph.add_edge(src, dst)
                    pred,_ = vmodel([graph], batch.x, batch.query_index,batch.query,False)
                    pred = torch.tensor(pred, dtype=int).to(args.device)
                
                # ######################## QDGNN #########################
                elif args.vmodel == 'qdgnn':
                    def get_adj(edge_index):
                        num_nodes = edge_index.max().item() + 1
                        adj_matrix = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
                        adj_matrix[edge_index[0], edge_index[1]] = 1
                        adj_matrix[edge_index[1], edge_index[0]] = 1
                        adj_matrix_np = adj_matrix.numpy()
                        return adj_matrix_np
                    output, y, mask,x_emb= vmodel(batch,get_adj(atk_edge_index),args.device)
                    output = torch.sigmoid(output).view(-1,1)
                    pred = torch.where(output > args.thre, 1, 0)                

                ########################## Community-AF #########################
                elif args.vmodel == 'caf':
                    seed=batch.query.item()
                    edges=atk_edge_index
                    tr_edges = edges.transpose(0, 1)
                    re_edges =tr_edges.reshape(-1, 2).cpu().detach().numpy()
                    graph=Graph(re_edges)
                    conv=GraphConv_m(graph)
                    nxg = nx.Graph()
                    nxg.add_edges_from(re_edges)

                    args.max_size=200
                    pred_l=generate_com(vmodel, args, graph, conv, None,[seed], nxg)[0]
                    pred=np.zeros(len(batch.y),dtype=int)
                    pred[pred_l]=1
                    pred = torch.tensor(pred, dtype=int).to(args.device)

                else:
                    loss,h= vmodel(batch.query,batch.pos,atk_edge_index,batch.x)
                    numerator = torch.mm(h[batch.query], h.t())
                    norm = torch.norm(h, dim=-1, keepdim=True)
                    denominator = torch.mm(norm[batch.query], norm.t())
                    sim = numerator / denominator
                    output = torch.sigmoid(sim).view(-1,1)
                    pred = torch.where(output > args.thre, 1, 0)    

            pred = pred.cpu().detach().numpy()
            targets = batch.y.cpu().detach().numpy()
            acc_v, precision_v, recall_v, f1_v = evaluate_prediction(pred, targets)

        return acc_v, precision_v, recall_v, f1_v


