from copy import deepcopy
from operator import index
from pyexpat import model
import torch
import torch.nn as nn
from torch import optim
from loss_criteria.loss import WeightBCEWithLogitsLoss
import random
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import precision_score, f1_score, accuracy_score, recall_score
import networkx as nx
import numpy as np
from utils.utils import percd,perc,kmeans,pagerank
import logging


def evaluate_prediction(pred, targets):
    acc = accuracy_score(targets, pred)
    precision = precision_score(targets, pred)
    recall = recall_score(targets, pred)
    f1 = f1_score(targets, pred)
    return acc, precision, recall, f1



def train_coclep(args,model,train_tasks,valid_tasks,test_tasks):
    # import tracemalloc
    # tracemalloc.start()

    logger = logging.getLogger('train')
    validation_=args.validation
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = WeightBCEWithLogitsLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=args.decay_factor, patience=args.decay_patience)
    tasks=train_tasks.get_batch()

    for epoch in range(args.epochs):
        model.train()
        # manually shuffle tasks
        # random.shuffle(batches)
        epoch_loss= 0.0
        batch_loss=0.0
        
        for i in range(len(tasks)):
            optimizer.zero_grad()
            batch = tasks[i]
            # print(batch)
            if args.cuda:
                batch = batch.to(args.device)

            if(args.vmodel == 'coclep'):
                loss,h= model(batch.query,batch.pos,batch.edge_index,batch.edge_index_aug,batch.x)
            else:
                loss_,h= model(batch.query,batch.pos,batch.edge_index,batch.x)
                numerator = torch.mm(h[batch.query], h.t())
                norm = torch.norm(h, dim=-1, keepdim=True)
                denominator = torch.mm(norm[batch.query], norm.t())
                sim = numerator / denominator
                pred = torch.sigmoid(sim)
                
                torch.set_printoptions(threshold=torch.inf)
                pred, targets = pred.view(-1,1).float(), batch.y.view(-1,1).float()

                loss_bce=criterion(pred,targets,batch.mask)
                loss= loss_bce

            batch_loss += loss
            epoch_loss += loss

            loss.backward()
            optimizer.step()

            # for name, param in model.named_parameters():
            #     if param.grad is None:
            #         print("The None grad model is:")
            #         print(name)

            i+=1
            if i+1%args.batch_size==0:
                print("epoch:{} batch:{} loss:{} avg_loss:{}"
                      .format(epoch,(i+1)/args.batch_size,batch_loss,batch_loss/args.batch_size))
                logger.info("epoch:{} batch:{} loss:{} avg_loss:{}"
                            .format(epoch,(i+1)/args.batch_size,batch_loss,batch_loss/args.batch_size))
                batch_loss=0.0


        if scheduler is not None:
            scheduler.step(epoch_loss)
            
            thre=validation_coclep(args,model,valid_tasks, epoch=epoch)
            test_coclep(args,model,test_tasks, epoch=epoch,thre=thre)

        #  # 打印内存使用情况
        # snapshot = tracemalloc.take_snapshot()
        # top_stats = snapshot.statistics('lineno')
        # print("[ Top 10 ]")
        # for stat in top_stats[:10]:
        #     print(stat)

    return model



def validation_coclep(args,model,valid_tasks,epoch=0):
    logger = logging.getLogger('valid')
    model.eval()
 
    thre = 0.1
    thre_max = thre
    f1_max,ac_max,pre_max,rec_max = 0.0,0.0,0.0,0.0
    tasks=valid_tasks.get_batch()
    while(thre<=0.9):
        acc_all=[]
        pre_all=[]
        rec_all=[]
        f1_all=[]
        f1_valid=0.0
        for batch in tasks:
            if args.cuda:
                batch = batch.to(args.device)
            if(args.vmodel == 'coclep'):
                loss,h= model(batch.query,batch.pos,batch.edge_index,batch.edge_index_aug,batch.x)
            else:
                loss_,h= model(batch.query,batch.pos,batch.edge_index,batch.x)
            # print(h)
            numerator = torch.mm(h[batch.query], h.t())
            norm = torch.norm(h, dim=-1, keepdim=True)
            denominator = torch.mm(norm[batch.query], norm.t())
            sim = numerator / denominator
            
            pred = torch.sigmoid(sim)
            pred = torch.where(pred > thre, 1, 0)
            pred, targets = pred.view(-1,1), batch.y.view(-1,1)
            pred, targets = pred.cpu().detach().numpy(), targets.cpu().detach().numpy()

            acc_v, precision_v, recall_v, f1_v = evaluate_prediction(pred, targets)
            acc_all.append(acc_v)
            pre_all.append(precision_v)
            rec_all.append(recall_v)
            f1_all.append(f1_v)
        acc_valid = np.mean(acc_all)
        precision_valid = np.mean(pre_all)
        recall_valid = np.mean(rec_all)
        f1_valid = np.mean(f1_all)

        if f1_valid>f1_max:
            ac_max = acc_valid
            pre_max = precision_valid
            rec_max = recall_valid
            f1_max = f1_valid
            thre_max = thre
        thre+=0.05

    print("epoch:{} valid Result: Acc={:.4f}, Pre={:.4f}, Recall={:.4f}, F1={:.4f}, Thre={:.2f}"
          .format(epoch,ac_max, pre_max, rec_max, f1_max, thre_max))
    logger.info(("epoch:{} valid Result: Acc={:.4f}, Pre={:.4f}, Recall={:.4f}, F1={:.4f}, Thre={:.2f}"
                 .format(epoch,ac_max, pre_max, rec_max, f1_max, thre_max)))
    return thre_max
    # return 0.7

def test_coclep(args,model,test_tasks,epoch=0,thre=0.5):
    logger = logging.getLogger('test')
    model.eval()
    i=0
    acc_all=[]
    pre_all=[]
    rec_all=[]
    f1_all=[]
    tasks=test_tasks.get_batch()

    for batch in tasks:   
        if args.cuda:
            batch = batch.to(args.device)
        if(args.vmodel == 'coclep'):
            loss,h= model(batch.query,batch.pos,batch.edge_index,batch.edge_index_aug,batch.x)
        else:
            loss_,h= model(batch.query,batch.pos,batch.edge_index,batch.x)
        numerator = torch.mm(h[batch.query], h.t())
        norm = torch.norm(h, dim=-1, keepdim=True)
        denominator = torch.mm(norm[batch.query], norm.t())
        sim = numerator / denominator
        
        pred = torch.sigmoid(sim)
        pred = torch.where(pred > thre, 1, 0)
        pred, targets = pred.view(-1,1), batch.y.view(-1,1)
        pred, targets = pred.cpu().detach().numpy(), targets.cpu().detach().numpy()

        i+=1
        
        acc_v, precision_v, recall_v, f1_v = evaluate_prediction(pred, targets)
        acc_all.append(acc_v)
        pre_all.append(precision_v)
        rec_all.append(recall_v)
        f1_all.append(f1_v)
    acc_test = np.mean(acc_all)
    precision_test = np.mean(pre_all)
    recall_test = np.mean(rec_all)
    f1_test = np.mean(f1_all)


    print("epoch:{} test Result: Acc={:.4f}, Pre={:.4f}, Recall={:.4f}, F1={:.4f}"
          .format(epoch,acc_test, precision_test, recall_test, f1_test))
    logger.info("epoch:{} test Result: Acc={:.4f}, Pre={:.4f}, Recall={:.4f}, F1={:.4f}"
                .format(epoch,acc_test, precision_test, recall_test, f1_test))
    





