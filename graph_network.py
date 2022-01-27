from platform import node
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn
from torch_scatter import scatter

from weight_init import weight_init

gat_heads = {
    'GAT':[1,1],
    'GATs-1':[4,2],
    'GATs-2':[16,4]
}
eps = 1e-8

class GCNConv(nn.Module):
    def __init__(self,input_channel,output_channel,device):
        super(GCNConv,self).__init__()
        self.line = nn.Linear(input_channel,output_channel,device=device)
        self.device = device

    def forward(self,node_feature,edge_indices):
        #添加自连接
        #self_loop = torch.arange(0,node_feature.size()[0],device=self.device)
        #self_loop = self_loop.unsqueeze(0).repeat(2,1)
        #edge_indices = torch.cat((edge_indices,self_loop),dim=1)

        source, target = edge_indices
        node_num, feature_num = node_feature.size()
        source_feature, target_feature = node_feature[source], node_feature[target]

        #依据节点度归一化特征
        deg = torch.ones(edge_indices.size()[1],device=self.device)
        deg = scatter(deg,target,dim=0,reduce='sum')
        deg = 1 / (torch.sqrt((deg.index_select(dim=0,index=target) * deg.index_select(dim=0,index=source))) + eps)

        #邻居聚合
        neighbour = scatter(self.line(source_feature*deg.unsqueeze(1).repeat(1,feature_num)),target,dim=0,reduce='sum')

        return neighbour

class SAGEConv(nn.Module):
    def __init__(self,input_channel,output_channel,device):
        super(SAGEConv,self).__init__()
        self.line = nn.Linear(input_channel,output_channel,device=device)
        self.self_line = nn.Linear(input_channel,output_channel,device=device)
        self.device = device

    def forward(self,node_feature,edge_indices):
        #邻居与中心使用不同MLP，因此不需要添加自连接

        source, target = edge_indices
        source_feature, target_feature = node_feature[source], node_feature[target]

        #邻居聚合
        neighbour = self.line(scatter(source_feature,target,dim=0,reduce='mean'))
        neighbour = neighbour + self.self_line(node_feature)

        return neighbour

class GATConv(nn.Module):
    def __init__(self,input_channel,output_channel,heads,device,concat=False):
        super(GATConv,self).__init__()
        self.line = nn.Linear(input_channel,output_channel*heads,device=device)
        self.att_line = nn.Parameter(torch.Tensor(1,heads,output_channel*2))
        self.concat = concat
        self.device = device
        self.heads = heads
        self.output_channel = output_channel

    #def forward(self,node_feature,edge_indices):
    #    source , target = edge_indices
    #    node_feature = self.line(node_feature)
    #    output_feature = node_feature
    #    node_num , _ = node_feature.size()
    #    for i in range(node_num):
    #        neighbour = torch.cat((node_feature[i].view(1,-1),node_feature[edge_indices[1][edge_indices[0].eq(i)]]),dim=0)
    #        alpha = self.att_line(torch.cat((node_feature[i].expand(neighbour.size()),neighbour),dim=1))
    #        alpha = torch.mean(F.softmax(F.leaky_relu(alpha),dim=0),dim=1).view(1,-1)
    #        output_feature[i] = torch.matmul(alpha,neighbour)
    #    return output_feature

    def forward(self,node_feature,edge_indices):
        #添加自连接
        #self_loop = torch.arange(0,node_feature.size()[0],device=self.device)
        #self_loop = self_loop.unsqueeze(0).repeat(2,1)
        #edge_indices = torch.cat((edge_indices,self_loop),dim=1)

        source, target = edge_indices
        node_feature = self.line(node_feature).view(-1,self.heads,self.output_channel)
        source_feature, target_feature = node_feature[source], node_feature[target]

        #计算注意力系数(减去最大值避免exp上溢)
        e = torch.sum(torch.cat((target_feature,source_feature),dim=2) * self.att_line,dim=-1)
        att = torch.exp(F.leaky_relu(e - torch.max(e)))
        att = att / (scatter(att,target,dim=0,reduce='sum').index_select(dim=0,index=target) + eps)

        #邻居聚合
        neighbour = scatter(source_feature*att.unsqueeze(-1).repeat(1,1,self.output_channel),target,dim=0,reduce='sum')
        if self.concat == False:
            neighbour = torch.mean(neighbour,dim=1).view(-1,self.output_channel)
        elif self.concat == True:
            neighbour = neighbour.view(-1,self.output_channel*self.heads)

        return neighbour


class GNN(nn.Module):
    def __init__(self,feature_num,output_channel1,output_channel2,neighbour1,neighbour2,class_num,device,module_name,dataset_name):
        super(GNN,self).__init__()
        if module_name == 'GCN':
            self.mpnn1 = GCNConv(feature_num,output_channel1,device=device)
            self.mpnn2 = GCNConv(output_channel1,output_channel2,device=device)
        elif module_name == 'GraphSAGE':
            self.mpnn1 = SAGEConv(feature_num,output_channel1,device=device)
            self.mpnn2 = SAGEConv(output_channel1,output_channel2,device=device)
        elif module_name in ['GAT','GATs-1','GATs-2']:
            self.mpnn1 = GATConv(feature_num,output_channel1,heads=gat_heads[module_name][0],concat=True,device=device)
            self.mpnn2 = GATConv(output_channel1*gat_heads[module_name][0],output_channel2,heads=gat_heads[module_name][1],concat=False,device=device)

        self.line1 = nn.Linear(output_channel2,class_num,device=device)

        self.mpnn1.apply(weight_init)
        self.mpnn2.apply(weight_init)
        self.line1.apply(weight_init)

        self.dataset_name = dataset_name
        self.device = device

    def forward(self,node_feature,edge_index,graph_index,supernode_index):
        node_feature = node_feature.to(self.device)
        edge_index = edge_index.to(self.device)
        graph_index = graph_index.to(self.device)
        supernode_index = supernode_index.squeeze(0)
        supernode_index = supernode_index.to(self.device)

        node_feature = F.leaky_relu(self.mpnn1(node_feature,edge_index))
        node_feature = F.dropout(node_feature,training=self.training)
        node_feature = F.leaky_relu(self.mpnn2(node_feature,edge_index))
        if self.dataset_name == 'ppi_dataset':
            node_feature = torch.sigmoid(self.line1(node_feature))
        elif self.dataset_name == 'enzyme_dataset':
            #node_feature = self.line1(node_feature)
            #node_feature = F.log_softmax(scatter(node_feature,graph_index,dim=0,reduce='sum')+eps,dim=1)
            node_feature = F.log_softmax(self.line1(node_feature)+eps,dim=1)
            node_feature = node_feature.index_select(dim=0,index=supernode_index)
        else :
            node_feature = F.log_softmax(self.line1(node_feature)+eps,dim=1)
        return node_feature
