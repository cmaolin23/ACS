from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import torch
import os
import argparse
import json

def save_args(args, filename):
    with open(filename, 'w') as f:
        json.dump(vars(args), f, indent=4)

def load_args(filename):
    with open(filename, 'r') as f:
        args_dict = json.load(f)
    return argparse.Namespace(**args_dict)


def get_args():

     parent_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
     data_dir = os.path.join(parent_dir, "data")

     parser = ArgumentParser("SimpleCS", formatter_class=ArgumentDefaultsHelpFormatter, conflict_handler="resolve")
     # Model Settings
     parser.add_argument("--num_layers", default=3, type=int, help="number of gnn conv layers")
     parser.add_argument("--num_g_hid", default=128, type=int, help="hidden dim")
     parser.add_argument("--gnn_out_dim", default=128, type=int, help="number of output dimension")
     parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate (1 - keep probability).')

     # Learning Setting
     parser.add_argument("--vmodel", default='coclep', type=str, help='vmodel')  # coclep, csgnn
     parser.add_argument("--smodel", default='csgnn', type=str, help='model')  # csgnn only

     parser.add_argument("--data_set", default='dblp', type=str, help='dataset')  # facebook, dblp
     parser.add_argument("--data_dir", type=str, default=data_dir)
     parser.add_argument("--subgraph_size", default=10000, type=int, help='the size of subgraph sampled in large graph')
     parser.add_argument("--task_num", type=int, help='task number', default=20) # cora 50
     parser.add_argument("--valid_task_num", type=int, help='valid task number', default=20)
     parser.add_argument("--test_task_num", type=int, default=20, help='the number of test task')
     parser.add_argument("--label_mode", type=str, default='disjoint', help='shared label or disjoint label')
     parser.add_argument("--num_pos", default=5, type=float)  # (maximum) proportion of positive instances for each query node
     parser.add_argument("--num_neg", default=5, type=float)  # (maximum) proportion of negative instances in each for each query node

     # Training Settings
     parser.add_argument("--thr_pre", default=None, type=float)
     parser.add_argument("--batch_size", default=5, type=int)
     parser.add_argument("--epochs", default=10, type=int) 
     parser.add_argument("--validation", default=True, type=bool)
     parser.add_argument("--test", default=True, type=bool)
     parser.add_argument("--train", default=True, type=bool)
     parser.add_argument("--sample_method", default="BFS", type=str)  # subgraph sample method: BFS、RandomWalk
     parser.add_argument("--learning_rate", default=1e-4, type=float)#1e-4-->1e-3
     parser.add_argument('--weight_decay', type=float, default=1e-3, help='Weight decay (L2 loss on parameters).')
     parser.add_argument("--scheduler_type", default="exponential", type=str, help="the node feature encoding type")  # Exponential/exponential or Plateau/plateau
     parser.add_argument('--decay_factor', type=float, default=0.8, help='decay rate of (gamma).')
     parser.add_argument('--decay_patience', type=int, default=10, help='num of epochs for one lr decay.')
     parser.add_argument('--num_workers', type=int, default=16, help='number of workers for Dataset.')
     parser.add_argument('--seed', default=20, type=int, help='seed')
    #  parser.add_argument('--device', type=str, default=torch.device('cuda' if torch.cuda.is_available else 'cpu'))

     #noise
     parser.add_argument('--noise', type=bool, default=False)
     parser.add_argument('--delete_ratio', type=float, default=-1)
     parser.add_argument('--add_ratio', type=float, default=-1)
     parser.add_argument('--draw_graph', type=bool, default=False)
     parser.add_argument('--debug', type=bool, default=False)

     # COCLEP
     parser.add_argument('--tau', type=float, default=0.2)
     parser.add_argument('--alpha', type=float, default=0.2)
     parser.add_argument('--lam', type=float, default=0.2)
     parser.add_argument('--k', type=int, default=2)

     # ICSGNN
     parser.add_argument("--community-size",type=int,default=30,help="The size of final community. Default is 30.")
     parser.add_argument("--train-ratio",type=float,default=0.02,help="Test data ratio. Default is 0.02.")
     parser.add_argument("--layers",type=int,default=[16],nargs='+',help="The size of hidden layers. Default is [16].")
     parser.add_argument("--possize",type=int,default=1,help="Incremental train node pairs per iteration. Default is 1.")
     parser.add_argument("--round",type=int,default=5,help="The number of iteration rounds. Default is 10.")

     #CGNP-> query_node_num,subgraph_size,pos_num,neg_num,num_shots
     parser.add_argument("--num_shots", type=int, default=1, help="number of shot")
     parser.add_argument("--pool_type",default="sum",type=str) # sum,avg,att
     parser.add_argument("--query_node_num", default=16, type=int)  # total query node number
     parser.add_argument("--gnn_type", default="GCN", type=str, help="GNN type")

     #CAF
     parser.add_argument('--max_neighbor', type=int, default=200)
     parser.add_argument('--cnt_rank', type=int, default=0)
     parser.add_argument('--num_flow_layer', type=int, default=8)
     parser.add_argument('--augment', action='store_true', default=True)
     parser.add_argument('--neg_num', type=int, default=10)
     parser.add_argument('--rankingloss', action='store_false', default=True)
     parser.add_argument('--gamma', type=int, default=1)
     parser.add_argument('--margin', type=int, default=0)
     parser.add_argument('--reduce', type=str, default="sum")
     parser.add_argument('--cache',action='store_false', default=False)
     

     #Attack
     parser.add_argument('--attack', type=bool, default=True)
     parser.add_argument('--attack_type', type=str, default='random') # other, proposed, random, pagerank,degree,preference
     parser.add_argument('--act_type', type=str, default='relu')
     parser.add_argument('--att_layer_num', type=int, default=3)
     parser.add_argument('--att_gnn_type', type=str, default="GCN")
     parser.add_argument('--budget', type=int, default=20)
     parser.add_argument('--node_feat_dim', type=int, default=128)
     parser.add_argument('--hidden_dim', type=int, default=128)
     parser.add_argument('--output_dim', type=int, default=128)
     parser.add_argument("--att_learning_rate", default=1e-4, type=float)#1e-4-->1e-3
     parser.add_argument('--att_weight_decay', type=float, default=1e-3, help='Weight decay (L2 loss on parameters).')
     parser.add_argument('--L_weight',type=float,default=1)
     parser.add_argument('--att_epochs', type=int, default=10)
     parser.add_argument('--thre', type=float, default=0.65)
     parser.add_argument('--sample_ae', type=str, default="topk") # topk, probs
     parser.add_argument('--att_meth', type=str, default="flip") # add, del,flip
     parser.add_argument('--w_q', type=float, default=0.2)
     parser.add_argument('--draw_loss', type=str, default=True)
     parser.add_argument('--draw_pic', type=str, default=False)
     parser.add_argument('--add_way', type=str, default="random") 
     parser.add_argument('--tem_sample', type=float, default=0.1)
     parser.add_argument('--dis_encode', type=str, default=True)
     parser.add_argument('--seed_a', default=20, type=int, help='seed_a') #amazon200 #dblp100 #lj1000 #facebook #cora500
     parser.add_argument('--lamd1', type=float, default=1e-3)
     parser.add_argument('--lamd2', type=float, default=1e-4)
     parser.add_argument('--lamd3', type=float, default=1e-2)
     parser.add_argument('--lamd4', type=float, default=1e-3)
     parser.add_argument('--chunk', type=str, default=True)
     parser.add_argument('--meth_r', type=str, default="best")   # base,best
     parser.add_argument('--read', type=bool, default=True)

     args = parser.parse_args()
     save_args(args, os.path.join(parent_dir, "args",'args.json'))
     # set the hardware parameter
     args.cuda = torch.cuda.is_available()
     args.device = torch.device('cuda' if args.cuda else 'cpu')
     return args

# get_args()