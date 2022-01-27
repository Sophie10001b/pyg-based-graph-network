import os
import time
import random
import torch
import torch.nn.functional as F
import torch.optim as optm
import numpy as np
import matplotlib.pyplot as plt
from tqdm.std import trange

from graph_network import GNN

device = ('cuda:0' if torch.cuda.is_available else 'cpu')

layer_setting = {
    'cora_dataset':[128,32,10,15,1],
    'citeseer_dataset':[128,32,10,15,1],
    'pubmed_dataset':[128,32,10,15,1],
    'lastfmasia_dataset':[128,32,10,15,1],
    'facebook_dataset':[64,32,10,15,1],
    'ppi_dataset':[64,32,10,15,2],
    'enzyme_dataset':[128,32,10,15,128]
}

#--------------------图batch模块--------------------
#通过yield逐次输出多图数据集的batch数据，默认随机排序，可添加自连接以及supernode(连接所有节点，初始特征为全图平均)
#1. 依据batch长度控制每次输出的数据量，batch_x,batch_y,batch_edge均保持pyg数据集默认格式，其中边数据batch_edge通过加入偏移区分不同图节点
#2. batch_index用于标识batch_x中不同节点的所属图，图编号由batch内顺序生成，可用于global_pooling
#3. self_loop用于在batch_edge中添加每个节点与自身的边连接，也可以用于防止度为0节点的出现
#4. supernode即一类连接全图节点的虚拟节点，可用于作为图级特征表示
def graph_batch(data,batch_size=1,self_loop=True,shuffle=True,supernode=False):
    if shuffle == True:
        shuffle_index = torch.randperm(len(data))
    else :
        shuffle_index = torch.arange(0,len(data),1)
    data_len = shuffle_index.size()[0]
    for pos in range(0,data_len,batch_size):
        left = min(batch_size,data_len-pos)
        batch_x = data[shuffle_index[pos]]['x']
        batch_index = torch.zeros((len(data[shuffle_index[pos]]['x']),1),dtype=torch.int64)
        batch_y = data[shuffle_index[pos]]['y']
        batch_edge = data[shuffle_index[pos]]['edge_index']
        sp_node_index = torch.zeros(1,dtype=torch.int64)
        if self_loop == True:
            self_node = torch.arange(0,batch_x.size()[0])
            self_node = self_node.unsqueeze(0).repeat(2,1)
            batch_edge = torch.cat((batch_edge,self_node),dim=1)
        if supernode == True:
            spnode_edge = torch.cat((torch.arange(0,batch_x.size()[0],1).view(1,-1),torch.tensor(batch_x.size()[0]).repeat(batch_x.size()[0]).view(1,-1)),dim=0)
            batch_edge = torch.cat((batch_edge,spnode_edge),dim=1)
            batch_x = torch.cat((batch_x,torch.mean(batch_x,dim=0).view(1,-1)))
            batch_index = torch.cat((batch_index,batch_index[-1].view(-1,1)),dim=0)
            sp_node_index = torch.tensor(batch_x.size()[0]-1,dtype=torch.int64).view(1,-1)
        bias = batch_x.size()[0]

        if batch_size > 1 and left > 1:
            for i in range(1,left):
                batch_x = torch.cat((batch_x,data[shuffle_index[pos+i]]['x']),dim=0)
                batch_index = torch.cat((batch_index,torch.tensor(i).unsqueeze(0).repeat(data[shuffle_index[pos+i]]['x'].size()[0],1)),dim=0)
                batch_y = torch.cat((batch_y,data[shuffle_index[pos+i]]['y']),dim=0)
                batch_edge = torch.cat((batch_edge,data[shuffle_index[pos+i]]['edge_index']+bias),dim=1)
                if self_loop == True:
                    self_node = torch.arange(0,data[shuffle_index[pos+i]]['x'].size()[0])+bias
                    self_node = self_node.unsqueeze(0).repeat(2,1)
                    batch_edge = torch.cat((batch_edge,self_node),dim=1)
                if supernode == True:
                    spnode_edge = torch.cat((torch.arange(bias,data[shuffle_index[pos+i]]['x'].size()[0]+bias,1).view(1,-1),torch.tensor(data[shuffle_index[pos+i]]['x'].size()[0]+bias).repeat(data[shuffle_index[pos+i]]['x'].size()[0]).view(1,-1)),dim=0)
                    batch_edge = torch.cat((batch_edge,spnode_edge),dim=1)
                    batch_x = torch.cat((batch_x,torch.mean(data[shuffle_index[pos+i]]['x'],dim=0).view(1,-1)))
                    batch_index = torch.cat((batch_index,batch_index[-1].view(-1,1)),dim=0)
                    sp_node_index = torch.cat((sp_node_index,torch.tensor(batch_x.size()[0]-1,dtype=torch.int64).view(1,-1)),dim=1)
                bias = batch_x.size()[0]
        
        yield batch_x, batch_y, batch_edge, batch_index, sp_node_index


class module_training():
    def __init__(self,config,dataset_name,training_dataset,module_name):
        self.dataset_name=dataset_name
        self.dataset = training_dataset
        self.lr = config.lr
        self.epoch = config.epoch
        #if len(self.dataset) > 1: self.epoch = 500
        #else : self.epoch = 500   
        self.accuracy_list = []
        self.loss_list = []
        self.module_name = module_name
        self.module = GNN(training_dataset.num_node_features,layer_setting[dataset_name][0],layer_setting[dataset_name][1],layer_setting[dataset_name][2],layer_setting[dataset_name][3],training_dataset.num_classes,device,module_name,dataset_name).to(device)
        self.optim = optm.Adam(self.module.parameters(),lr=config.lr)
        self.batch_size = layer_setting[dataset_name][4]
        self.self_loop = True
        #if module_name == 'GraphSAGE': self.self_loop = False
        self.supernode = False
        if dataset_name == 'enzyme_dataset': self.supernode = True
        self.config = config

    def process(self,train_mask,test_mask,test_num):
        tepoch = trange(self.epoch)
        for epoch in tepoch:
            if len(self.dataset) > 1:
                g_batch = graph_batch(self.dataset[train_mask],self.batch_size,self.self_loop,supernode=self.supernode)
                eval_batch = graph_batch(self.dataset[test_mask],test_num,self.self_loop,supernode=self.supernode)
            else :
                g_batch = graph_batch(self.dataset,self.batch_size,self.self_loop,supernode=self.supernode)
            loss = 0
            accuracy = 0
            self.module.train()
            self.optim.zero_grad()

            batch_x, batch_y, batch_edge, batch_index, sp_node_index = next(g_batch)
            result = self.module(batch_x,batch_edge,batch_index,sp_node_index)
            #if self.module_name == 'GAT' and epoch == 19:
            #    with open(os.path.join(self.config.result_path,'1.txt'),'w') as f:
            #       for i in result.cpu().detach().numpy():
            #            f.writelines('{}\n'.format(i))
            if len(self.dataset) == 1:
                loss = F.nll_loss(result[train_mask],batch_y[train_mask].long().to(device))
            elif self.dataset_name == 'enzyme_dataset':
                loss = F.nll_loss(result,batch_y.long().to(device))
            elif self.dataset_name == 'ppi_dataset':
                loss = F.mse_loss(result,batch_y.to(device),reduction='sum')
                loss = loss / result.size()[0]
            self.loss_list.append(loss.item())
            loss.backward()
            self.optim.step()
            self.optim.zero_grad()

            self.module.eval()
            with torch.no_grad():
                if self.dataset_name == 'ppi_dataset':
                    eval_x, eval_y, eval_edge, eval_index, eval_sp_index = next(eval_batch)
                    test_result = self.module(eval_x,eval_edge,eval_index,eval_sp_index)
                    test_result = torch.where(test_result > 0.5, torch.ones(1,device=device), torch.zeros(1,device=device))
                    correct = float(test_result.eq(eval_y.to(device)).sum().item())
                    accuracy = correct / (test_result.size()[0] * test_result.size()[1])
                elif self.dataset_name == 'enzyme_dataset':
                    eval_x, eval_y, eval_edge, eval_index, eval_sp_index = next(eval_batch)
                    test_result = self.module(eval_x,eval_edge,eval_index,eval_sp_index)
                    test_result = torch.argmax(test_result,dim=1).view(-1)
                    correct = float(test_result.eq(eval_y.to(device)).sum().item())
                    accuracy = correct / test_num
                else :
                    test_result = torch.argmax(result[test_mask],dim=1).view(-1)
                    correct = float(test_result.eq(batch_y[test_mask].to(device)).sum().item())
                    accuracy = correct / test_num

                self.accuracy_list.append(accuracy)

            tepoch.set_description('module {0} is training in {1}, loss {2:.6}, accuracy {3:.6%}.'.format(self.module_name,self.dataset_name,self.loss_list[-1],self.accuracy_list[-1]))


def training(config,dataset_name,training_dataset):
    module_loss_list = []
    module_accuracy_list = []
    loss_min_index = []
    loss_min = []
    acc_max_index = []
    acc_max = []
    training_time = []
    param_num = []

    if dataset_name in ['ppi_dataset','enzyme_dataset']:
        test_index=torch.tensor(random.sample(range(len(training_dataset)),int(len(training_dataset)*0.3)))
        train_mask=torch.ones(len(training_dataset),dtype=torch.bool)
        test_mask=torch.zeros(len(training_dataset),dtype=torch.bool)
        test_num=int(len(training_dataset)*0.3)
    else:
        test_index=torch.tensor(random.sample(range(training_dataset[0].num_nodes),int(training_dataset[0].num_nodes*0.3)))
        train_mask=torch.ones(training_dataset[0].num_nodes,dtype=torch.bool)
        test_mask=torch.zeros(training_dataset[0].num_nodes,dtype=torch.bool)
        test_num=int(training_dataset[0].num_nodes*0.3)

    for i in test_index.numpy():
        train_mask[i]=False; test_mask[i]=True

    module_list=['GCN','GraphSAGE','GAT','GATs-1','GATs-2']
    #module_list=['GCN','GraphSAGE','GAT','GATs-1']
    for module_name in module_list:
        module_start_time = time.perf_counter()
        module = module_training(config,dataset_name,training_dataset,module_name)
        module.process(train_mask,test_mask,test_num)
        torch.cuda.empty_cache()
        training_time.append(time.perf_counter()-module_start_time)
        param_num.append(sum(x.numel() for x in module.module.parameters()))

        acc_max_index.append(np.argmax(module.accuracy_list))
        acc_max.append(module.accuracy_list[acc_max_index[-1]])
        module_accuracy_list.append(module.accuracy_list)

        loss_min_index.append(np.argmin(module.loss_list))
        loss_min.append(module.loss_list[loss_min_index[-1]])
        module_loss_list.append(module.loss_list)

    with open(os.path.join(config.result_path,dataset_name+'_log''.txt'),'w') as f:
        for module_name,loss,loss_index,acc,acc_index,run_time,param in zip(module_list,loss_min,loss_min_index,acc_max,acc_max_index,training_time,param_num):
            info1='Module: {0}.  Dataset: {1}.  Best Loss: {2:.6} in {3} epoch.  Best Accuracy: {4:.6%} in {5} epoch.  training time: {6:.6}s'.format(module_name,dataset_name,loss,loss_index,acc,acc_index,run_time)
            f.writelines(info1+'\n')

            info2='Total parameters number: {}'.format(param)
            f.writelines(info2+'\n'*2)

    print('all module training is finish, now start plotting...')
    plt.rcParams['figure.dpi']=600
    color_list=['r','g','b','y','k']
    color_list=color_list[:len(module_list)]
    for mod,acc,color in zip(module_list,module_accuracy_list,color_list):
        plt.plot(acc,label=mod,color=color,linewidth=0.5)
    plt.title('GNN accuracy in {}.'.format(dataset_name))
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(loc='best')
    if(os.path.exists(os.path.join(config.result_path,dataset_name+'_accuracy'+'.png'))):
        os.remove(os.path.join(config.result_path,dataset_name+'_accuracy'+'.png'))
    plt.savefig(os.path.join(config.result_path,dataset_name+'_accuracy'+'.png'),format='PNG')
    plt.close()

    for mod,los,color in zip(module_list,module_loss_list,color_list):
        plt.semilogy(los,label=mod,color=color,linewidth=0.5)
    plt.title('GNN loss in {}.'.format(dataset_name))
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(loc='best')
    if(os.path.exists(os.path.join(config.result_path,dataset_name+'_loss'+'.png'))):
        os.remove(os.path.join(config.result_path,dataset_name+'_loss'+'.png'))
    plt.savefig(os.path.join(config.result_path,dataset_name+'_loss'+'.png'),format='PNG')
    plt.close()

        
        