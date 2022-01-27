import torch
import torch.nn as nn
import random

random.seed(1)
torch.manual_seed(1)

def weight_init(m):
    class_name = m.__class__.__name__
    if(class_name.find('GCNConv'))!=-1:
        m.line.apply(weight_init)
    elif(class_name.find('SAGEConv'))!=-1:
        m.line.apply(weight_init)
        m.self_line.apply(weight_init)
    elif(class_name.find('GATConv'))!=-1:
        m.line.apply(weight_init)
        nn.init.normal_(m.att_line.data,0.0,0.01)
    elif(class_name.find('Linear'))!=-1:
        nn.init.normal_(m.weight.data,0.0,0.01)
        if m.bias != None:
            nn.init.normal_(m.bias.data,0.0,0.01)
