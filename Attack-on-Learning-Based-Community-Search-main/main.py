import os
import numpy as np
import torch
import time
import random
import pickle
import time
import logging
from dataloader.dataloader_task import load_data_and_get_tasks
from dataloader.dataloader_task_cgnp import load_data_and_get_tasks_cgnp
from train_model.train_coclep import train_coclep,validation_coclep,test_coclep
from train_model.train_ics import train_icsgnn,validation_icsgnn,test_icsgnn
from train_model.train_qd import train_qdgnn,validation_qdgnn,test_qdgnn
from train_model.train_cgnp import train_cgnp,validation_cgnp,test_cgnp
from train_model.train_caf import train_caf,validation_caf,test_caf
from train_model.train_csgnn import train_csgnn,validation_csgnn,test_csgnn
from train_model.train_sur import train_sur,validation_sur,test_sur

from args.args import get_args
from model.Vmodel.COCLEP import *
from model.Vmodel.CSGNN import *
from model.Vmodel.ICSGNN import *
from model.Vmodel.QDGNN import *
from model.Vmodel.CGNP import *
from model.Vmodel.CAF import *

from model.Amodel.EVA.Evasion_attack import *
# 172.20.117.16
from torch import optim
import warnings

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
warnings.filterwarnings("ignore")

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
class GCNConvWithAdj(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True):
        super(GCNConvWithAdj, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = nn.Parameter(torch.Tensor(in_channels, out_channels))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x, adj):
        deg = torch.sum(adj, dim=1)
        deg_inv_sqrt = torch.pow(deg, -0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        deg_inv_sqrt = torch.diag(deg_inv_sqrt)

        adj_norm = torch.mm(torch.mm(deg_inv_sqrt, adj), deg_inv_sqrt)
        support = torch.mm(x, self.weight)
        out = torch.mm(adj_norm, support)

        if self.bias is not None:
            out += self.bias

        return out

class sur_GNN(nn.Module):
    def __init__(self, node_in_dim, hidden_dim, num_layers, dropout, device):

        super(sur_GNN, self).__init__()
        self.dropout = dropout
        self.num_layers = num_layers
        self.device = device
        self.tau = 0.2

        # Part 1: GCN
        self.layersq = nn.ModuleList()
        self.layersq.append(GCNConvWithAdj(1, hidden_dim))
        for _ in range(num_layers - 1):
            self.layersq.append(GCNConvWithAdj(hidden_dim, hidden_dim))

        self.layers = nn.ModuleList()
        self.layers.append(GCNConvWithAdj(node_in_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.layers.append(GCNConvWithAdj(hidden_dim, hidden_dim))

        self.layersf = nn.ModuleList()
        self.layersf.append(GCNConvWithAdj(hidden_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.layersf.append(GCNConvWithAdj(hidden_dim, hidden_dim))

        self.mlp1 = MLP(hidden_dim, hidden_dim)

        self.layers_mlp_src = nn.ModuleList()
        self.layers_mlp_des = nn.ModuleList()
        self.layers_mlp_src.append(MLP(hidden_dim, hidden_dim))
        self.layers_mlp_des.append(MLP(hidden_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.layers_mlp_src.append(MLP(hidden_dim, hidden_dim))
            self.layers_mlp_des.append(MLP(hidden_dim, hidden_dim))

        self.layers_edge = nn.ModuleList()
        self.layers_edge_out = nn.ModuleList()
        self.layers_edge.append(MLP(hidden_dim*2, hidden_dim))
        self.layers_edge_out.append(MLP(hidden_dim, 1))
        for _ in range(num_layers - 1):
            self.layers_edge.append(MLP(hidden_dim*2, hidden_dim))
            self.layers_edge_out.append(MLP(hidden_dim, 1))

        self.linerquerys = torch.nn.Linear(1, hidden_dim)
        self.linerfeats = torch.nn.Linear(node_in_dim, hidden_dim)

        # 调用初始化方法
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


    def forward(self, q,feats, adj):
        querys = torch.zeros(feats.shape[0], 1).to(self.device)
        querys[q] = 1.0
        querys_ = self.linerquerys(querys)
        feats_ = self.linerfeats(feats)

        h_d = torch.stack([querys_, feats_], dim=1)
        h_d = torch.sum(h_d, dim=1).to(self.device)


        # part : GCN
        hq = F.relu(self.layersq[0](querys, adj)).to(self.device)
        h = F.relu(self.layers[0](feats, adj)).to(self.device)
        hf = torch.stack([hq, h], dim=1)
        hf = torch.sum(hf, dim=1)

        hf_ = torch.stack([querys_, feats_], dim=1)
        hf_ = torch.sum(hf_, dim=1)
        hf = F.relu(hf + self.layersf[0](hf_, adj))

        for _ in range(self.num_layers - 2):
        # part : GCN
            hq = F.dropout(hq, training=self.training, p=self.dropout)
            h = F.dropout(h, training=self.training, p=self.dropout)
            hf = F.dropout(hf, training=self.training, p=self.dropout)

            hq = F.relu(self.layersq[_+1](hq, adj))
            h = F.relu(self.layers[_+1](h, adj))
            hfx = torch.stack([hq, h], dim=1)
            hfx = torch.sum(hfx, dim=1)
            hf = F.relu(hfx + self.layersf[_+1](hf, adj))

        hq = F.dropout(hq, training=self.training, p=self.dropout)
        h = F.dropout(h, training=self.training, p=self.dropout)
        hf = F.dropout(hf, training=self.training, p=self.dropout)

        hq = self.layersq[self.num_layers - 1](hq, adj)
        h = self.layers[self.num_layers - 1](h, adj)
        hfx = torch.stack([hq, h], dim=1)
        hfx = torch.sum(hfx, dim=1)

        hf = hfx + self.layersf[self.num_layers - 1](hf, adj)
        h_ = self.mlp1(hf)

        numerator = torch.mm(h_[q], h_.t())
        norm = torch.norm(h_ , dim=-1, keepdim=True)
        denominator = torch.mm(norm[q], norm.t())
        sim = numerator / denominator
        pred = torch.sigmoid(sim).view(-1,1)

        return pred
    


parent_dir = os.path.abspath(os.path.dirname(os.path.abspath(__file__)))


def reset_args(args,subgraph_size):
    args_=args
    if args_.data_set == 'football':
        args_.subgraph_size=-1
    else:
        args_.subgraph_size=subgraph_size

    return args_

# windows
root_pkl=os.path.join(parent_dir,"data","pkl")
root_pth=os.path.join(parent_dir,"model","pth")
root_log=os.path.join(parent_dir,"results","log")

def train_test_sur_model(args):
    node_feat, train_tasks, valid_tasks, test_tasks = 0,[], [], []
    params=("Dataset-"+args.data_set+"_GraphSize-" +str(args.subgraph_size)+"_VModel-"+args.vmodel)
    params1=("Dataset-"+args.data_set+"_GraphSize-" +str(args.subgraph_size)+"_SModel-"+args.smodel)

    pkl_path=os.path.join(root_pkl , params +'.pkl')
    vmodel_path =os.path.join(root_pth , params + '.pth')
    smodel_path =os.path.join(root_pth , params1 + '.pth')
    log_path = os.path.join(root_log , params+ '.log')

    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')
    for key, value in vars(args).items():
        logging.info(f"{key}: {value}")
    logging.info("**************************************************")

    print('-----getting tasks------')
    start_time = time.time()
    
    # if(args.debug) :
    #     if os.path.exists(pkl_path):
    #         with open(pkl_path, 'rb') as f:
    #             train_tasks, valid_tasks, test_tasks, node_feat_num = pickle.load(f)
    #     else:
    #         if args.vmodel == 'cgnp':
    #             train_tasks_, valid_tasks_, test_tasks_, node_feat_num = load_data_and_get_tasks_cgnp(args)
    #             args_=reset_args(args,500)
    #             train_tasks, valid_tasks, test_tasks, node_feat_num = load_data_and_get_tasks(args_)
    #         else:
    #             train_tasks, valid_tasks, test_tasks, node_feat_num = load_data_and_get_tasks(args)
    #         with open(pkl_path, 'wb') as f:
    #             pickle.dump((train_tasks, valid_tasks, test_tasks, node_feat_num), f)
    # else:
    #     if args.vmodel == 'cgnp':
    #         train_tasks_, valid_tasks_, test_tasks_, node_feat_num = load_data_and_get_tasks_cgnp(args)
    #         args_=reset_args(args,500)
    #         train_tasks, valid_tasks, test_tasks, node_feat_num = load_data_and_get_tasks(args_)
    #     else:
    #         train_tasks, valid_tasks, test_tasks, node_feat_num= load_data_and_get_tasks(args)
    #     with open(pkl_path, 'wb') as f:
    #             pickle.dump((train_tasks, valid_tasks, test_tasks, node_feat_num), f)
    
    pkl_path_cgnp=os.path.join(root_pkl ,params +'1.pkl')
    if args.vmodel == 'cgnp':
        
        if os.path.exists(pkl_path_cgnp) and os.path.exists(pkl_path):
            with open(pkl_path_cgnp, 'rb') as f:
                train_tasks_, valid_tasks_, test_tasks_, node_feat_num = pickle.load(f)
            with open(pkl_path, 'rb') as f:
                train_tasks, valid_tasks, test_tasks, node_feat_num = pickle.load(f)
        else:
            train_tasks_, valid_tasks_, test_tasks_, node_feat_num = load_data_and_get_tasks_cgnp(args)
            # print("cgnp data loaded")
            # args_=reset_args(args,500)
            train_tasks, valid_tasks, test_tasks, node_feat_num = load_data_and_get_tasks(args)
            with open(pkl_path, 'wb') as f:
                pickle.dump((train_tasks, valid_tasks, test_tasks, node_feat_num), f)
            with open(pkl_path_cgnp, 'wb') as f:
                pickle.dump((train_tasks_, valid_tasks_, test_tasks_, node_feat_num), f)
    else:
        
        if os.path.exists(pkl_path):
            with open(pkl_path, 'rb') as f:
                train_tasks, valid_tasks, test_tasks, node_feat_num = pickle.load(f)
        else:
            train_tasks, valid_tasks, test_tasks, node_feat_num= load_data_and_get_tasks(args)
            with open(pkl_path, 'wb') as f:
                pickle.dump((train_tasks, valid_tasks, test_tasks, node_feat_num), f)

    end_time = time.time()
    runtime = end_time - start_time
    print("Data pre time cost:{}".format(runtime))
    logging.info("Data pre time cost:{}".format(runtime))

    if args.vmodel == 'coclep':
        model = COCLEP(node_in_dim=node_feat_num + 4, hidden_dim=args.gnn_out_dim,num_layers=args.num_layers,
                    dropout=args.dropout,device=args.device,tau=args.tau, alpha=args.alpha, lam=args.lam, k=args.k)
    elif args.vmodel == 'icsgnn':
        model =ICSGNN(args=args,node_feat_dim=node_feat_num + 4, edge_feat_dim=2)
    elif args.vmodel == 'qdgnn':
        model = QDGNN(nfeat=node_feat_num + 4, nhid=args.gnn_out_dim,nclass=1,dropout=args.dropout)
    elif args.vmodel == 'cgnp':
        # model = CSCNP(args=args,node_feat_dim=node_feat_num + 4, edge_feat_dim=10)
        model = CSCNPComp(args=args,node_feat_dim=node_feat_num + 4, edge_feat_dim=64)
    elif args.vmodel == 'caf':
        model = CSAFModel(args=args)
    elif args.vmodel == 'csgnn':
        model = CS_GNN(node_in_dim=node_feat_num + 4, hidden_dim=args.gnn_out_dim,num_layers=args.num_layers,
                        dropout=args.dropout,device=args.device)
    
    # smodel = CS_GNN(node_in_dim=node_feat_num + 4, hidden_dim=args.gnn_out_dim,num_layers=args.num_layers,
    #                     dropout=args.dropout,device=args.device)
    smodel=sur_GNN(node_in_dim=node_feat_num + 4, hidden_dim=args.gnn_out_dim,num_layers=args.num_layers,
                    dropout=args.dropout,device=args.device)

    model.to(args.device)
    print('-----Training and evaluating------')
    logging.info('-----Training and evaluating------')
    if args.train:
        # print('model:\n', model)
        start_time = time.time()

        if args.vmodel == 'coclep':
            model=train_coclep(args,model,train_tasks, valid_tasks, test_tasks)
        elif args.vmodel == 'icsgnn':
            model=train_icsgnn(args,model,train_tasks, valid_tasks, test_tasks)
        elif args.vmodel == 'qdgnn':
            model=train_qdgnn(args,model,train_tasks, valid_tasks, test_tasks)
        elif args.vmodel == 'cgnp':
            model=train_cgnp(args,model,train_tasks_, valid_tasks_, test_tasks_)
        elif args.vmodel == 'caf':
            model=train_caf(args,model,train_tasks, valid_tasks, test_tasks)
        else:
            model=train_csgnn(args,model,train_tasks, valid_tasks, test_tasks)
        model.eval()

        print('-----Training suro model ------')
        logging.info('-----Training suro model ------')
        smodel.to(args.device)
        # smodel=train_csgnn(args,smodel,train_tasks, valid_tasks, test_tasks)
        smodel=train_sur(args,smodel,train_tasks, valid_tasks, test_tasks)
        smodel.eval()

        torch.save(model.state_dict(), vmodel_path)
        torch.save(smodel.state_dict(), smodel_path)
        end_time = time.time()
        runtime = end_time - start_time
        print("Train model time cost:{}".format(runtime))
        logging.info("Train model time cost:{}".format(runtime))
        print('-----Model saved------')
        logging.info('-----Model saved------')

    elif args.test:
        model.load_state_dict(torch.load(vmodel_path))
        # print('model:\n', model)
        start_time = time.time()
        if args.vmodel == 'coclep':
            thre=validation_coclep(args,model,valid_tasks, epoch=args.epochs)
            test_coclep(args,model,test_tasks, epoch=args.epochs,thre=thre)
        elif args.vmodel == 'icsgnn':
            validation_icsgnn(args,model,valid_tasks, epoch=args.epochs)
            test_icsgnn(args,model,test_tasks, epoch=args.epochs)
        elif args.vmodel == 'qdgnn':
            thre=validation_qdgnn(args,model,valid_tasks, epoch=args.epochs)
            test_qdgnn(args,model,test_tasks, epoch=args.epochs,thre=thre)
        elif args.vmodel == 'cgnp':
            thre=validation_cgnp(args,model,valid_tasks_, epoch=args.epochs)
            test_cgnp(args,model,test_tasks_, epoch=args.epochs,thre=thre)
        elif args.vmodel == 'caf':
            validation_caf(args,model,valid_tasks, epoch=args.epochs)
            test_caf(args,model,test_tasks, epoch=args.epochs)
        else:
            thre=validation_coclep(args,model,valid_tasks, epoch=args.epochs)
            test_coclep(args,model,test_tasks, epoch=args.epochs,thre=thre)
        # thre=validation_csgnn(args,smodel,valid_tasks, epoch=args.epochs)
        # test_csgnn(args,smodel,test_tasks, epoch=args.epochs,thre=thre)

        thre=validation_sur(args,smodel,valid_tasks, epoch=args.epochs)
        test_sur(args,smodel,test_tasks, epoch=args.epochs,thre=thre)
        end_time = time.time()
        runtime = end_time - start_time
        print("Test model time cost:{}".format(runtime))
        logging.info("Test model time cost:{}".format(runtime))

    return model,smodel,vmodel_path,smodel_path

def train_test_att_model(args,vit_model,sur_model):
    print('-----attacking------')
    # pmodel = Policy_GAE(args, args.node_feat_dim, args.hidden_dim, args.output_dim)
    pmodel = Policy_GNN(args,node_in_dim=4, hidden_dim=128,num_layers=3,dropout=0.2,device=args.device)
    args.seed = args.seed_a

    params=("Dataset-"+args.data_set+"_GraphSize-" +str(args.subgraph_size)+"normal")
    params1=("Dataset-"+args.data_set+"_GraphSize-" +str(args.subgraph_size)+"cgnp_1")
    params2=("Dataset-"+args.data_set+"_GraphSize-" +str(args.subgraph_size)+"cgnp_2")

    pkl_path1=os.path.join(root_pkl , params1 +'.pkl')
    pkl_path2=os.path.join(root_pkl , params2 +'.pkl')
    pkl_path =os.path.join(root_pkl , params+ '.pkl')
    

    if args.vmodel == 'cgnp':
        if os.path.exists(pkl_path1) and os.path.exists(pkl_path2):
            with open(pkl_path1, 'rb') as f1:
                train_tasks_, valid_tasks_, test_tasks_, _ = pickle.load(f1)
            with open(pkl_path2, 'rb') as f2:
                train_tasks, valid_tasks, test_tasks, _ = pickle.load(f2)
        else:
            train_tasks_, valid_tasks_, test_tasks_, _ = load_data_and_get_tasks_cgnp(args)
            train_tasks, valid_tasks, test_tasks, _= load_data_and_get_tasks(args)
            with open(pkl_path1, 'wb') as f1:
                pickle.dump((train_tasks_, valid_tasks_, test_tasks_, _), f1)
            with open(pkl_path2, 'wb') as f2:
                pickle.dump((train_tasks, valid_tasks, test_tasks, _), f2)
    else:
        if os.path.exists(pkl_path):
            with open(pkl_path, 'rb') as f:
                train_tasks, valid_tasks, test_tasks, _ = pickle.load(f)
        else:
            train_tasks, valid_tasks, test_tasks, _= load_data_and_get_tasks(args)
            with open(pkl_path, 'wb') as f:
                pickle.dump((train_tasks, valid_tasks, test_tasks, _), f)

    eva_model = Evasion_attack(args=args,pmodel=pmodel,victim_model=vit_model)
    eva_model.to(args.device)

    if args.attack_type == "proposed" or args.attack_type == "other":
        if args.vmodel == 'cgnp':
            eva_model.train_atk(surrogate_model=sur_model,train_tasks=train_tasks,valid_tasks=valid_tasks_,test_tasks=test_tasks_)
        else:
            eva_model.train_atk(surrogate_model=sur_model,train_tasks=train_tasks,valid_tasks=valid_tasks,test_tasks=test_tasks)
    else:
        if args.vmodel == 'cgnp':
            for i in range(15):
                eva_model.test_atk(test_tasks=test_tasks_,epoch=i)
        else:
            for i in range(15):
                eva_model.test_atk(test_tasks=test_tasks,epoch=i)


if __name__ == "__main__":
    args = get_args()
    print(args)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # vmodel,smodel,vmodel_path,smodel_path=train_test_sur_model(args)
    # print("Models have been saved.")
    
        
    if args.read:

        params=("Dataset-"+args.data_set+"_GraphSize-" +str(args.subgraph_size)+"_VModel-"+args.vmodel)
        params1=("Dataset-"+args.data_set+"_GraphSize-" +str(args.subgraph_size)+"_SModel-"+args.smodel)
        vmodel_path =os.path.join(root_pth , params + '.pth')
        smodel_path =os.path.join(root_pth , params1 + '.pth')
        log_path = os.path.join(root_log , params+ '.log')

        logging.basicConfig(
            filename=log_path,
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S')
        for key, value in vars(args).items():
            logging.info(f"{key}: {value}")
        logging.info("**************************************************")

        smodel=sur_GNN(node_in_dim=4, hidden_dim=args.gnn_out_dim,num_layers=args.num_layers,
                        dropout=args.dropout,device=args.device)
        node_feat_num=0
        if args.vmodel == 'coclep':
            vmodel = COCLEP(node_in_dim=node_feat_num + 4, hidden_dim=args.gnn_out_dim,num_layers=args.num_layers,
                        dropout=args.dropout,device=args.device,tau=args.tau, alpha=args.alpha, lam=args.lam, k=args.k)
        elif args.vmodel == 'icsgnn':
            vmodel =ICSGNN(args=args,node_feat_dim=node_feat_num + 4, edge_feat_dim=2)
        elif args.vmodel == 'qdgnn':
            vmodel = QDGNN(nfeat=node_feat_num + 4, nhid=args.gnn_out_dim,nclass=1,dropout=args.dropout)
        elif args.vmodel == 'cgnp':
            # model = CSCNP(args=args,node_feat_dim=node_feat_num + 4, edge_feat_dim=10)
            vmodel = CSCNPComp(args=args,node_feat_dim=node_feat_num + 4, edge_feat_dim=64)
        elif args.vmodel == 'caf':
            vmodel = CSAFModel(args=args)
        elif args.vmodel == 'csgnn':
            vmodel = CS_GNN(node_in_dim=node_feat_num + 4, hidden_dim=args.gnn_out_dim,num_layers=args.num_layers,
                            dropout=args.dropout,device=args.device)
        
        smodel.to(args.device)
        vmodel.to(args.device)
        if os.path.exists(vmodel_path) and os.path.exists(smodel_path):
            vmodel.load_state_dict(torch.load(vmodel_path))
            smodel.load_state_dict(torch.load(smodel_path))
            star_time=time.time()
            train_test_att_model(args,vmodel,smodel)
            end_time=time.time()
            print("Total time cost:{}".format(end_time-star_time))
        else:
            print("Model not found.")
    else:
        vmodel,smodel,vmodel_path,smodel_path=train_test_sur_model(args)
        star_time=time.time()
        train_test_att_model(args,vmodel,smodel)
        end_time=time.time()
        print("Total time cost:{}".format(end_time-star_time))



