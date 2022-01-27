# GCN,GraphSAGE,GAT的优化实现
本实验基于PyTorch框架，以GCN,GraphSAGE,GAT三类经典的图网络为例，对torch_geometric(pyg)中的图网络传播过程进行简易复现，其中的复现重点如下：
- 对于多图数据集，基于pyg的数据集格式实现在图上的batch操作，同时包含添加自连接以及超级节点(supernode)
- 依据pyg中消息传递网络(MPNN)的实现机制，同样采用torch_scatter.scatter()对图邻居信息进行高效聚合，达到与pyg框架相似的运行效率
- 优化GAT网络的相关操作，依据pyg框架中的GAT实现步骤，同样采用3维张量计算实现多头注意机制，以及多头注意的concat/mean操作

对于以上的复现重点，可以参考以下示例图进一步了解具体实现思路：
- batch操作
- 自连接操作
- 添加supernode
- torch_scatter.scatter()功能

代码运行于如下模块中：
```
- python 3.7.10  
- pytorch 1.9.1    
- pyg 2.0.2
```

## 数据集下载
通过torch_geometric直接下载以下数据集：
- Cora   
- Citeseer  
- PubMed  
- LastFMAsia  
- Facebook  
- Enzyme  
- PPI

其中Enzyme为多图节点分类数据集，PPI为多图图分类数据集。使用以下命令下载数据集：
```
python cmd_line.py --dataset_path *** --result_path *** --mode download
```
其中 *dataset_path* 后为源数据集下载目录， *result_path* 后为模型运行结果保存目录，推荐运行前在 *cmd_line.py* 中直接更改对应的默认设置。后续命令将省略以上的路径设置 

## 模型训练以及结果输出
本程序通过以下命令使用对应数据集与模型进行训练：
```
python cmd_line.py --mode training --lr *** --epoch ***
```
其中 *--lr* 为学习率设置(默认1e-3)， *--epoch* 为训练轮数设置(默认500)

程序训练过程中会自动按照以下顺序训练并验证模型：
- GCN
- GraphSAGE
- GAT(single-head)
- GATs-1(multi-head)
- GATs-2(multi-head)

其中所有模型均包含有2层图网络层与1层MLP分类层，每一层的隐含表示向量长度与batch长度可在 *module.py* 中自行设置，GATs-1在前两层图网络中各包含有4,2个注意头，而GATs-2则包含有16,4个注意头，首层GAT采用concat连接多头注意结果，末层GAT则对多头注意结果进行平均

此外，由于 *PPI* 与 *Enzyme* 数据集属于多图数据集，其在最后的MLP结果处理上与其它数据集存在差异： *Enzyme* 为图分类任务，因此采用假设的supernode承载图表示向量； *PPI* 为节点分类任务，但其包含有多个预测项目，因此本实验直接采用MSE作为损失函数，理想情况下应当对每一个预测项目单独执行交叉熵损失计算

最后，模型会输出运行时的loss折线图与精度折线图，同时得到一个包含有模型名称，数据集名称，最优loss，最优精度，训练耗时，总参数数量的文本文件作为运行结果
