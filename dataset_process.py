import argparse
import os
import gc
import time
import random
import torch
import numpy as np
import joblib
from torch_geometric.datasets import TUDataset
from torch_geometric.datasets import Planetoid
from torch_geometric.datasets import LastFMAsia
from torch_geometric.datasets import FacebookPagePage
from torch_geometric.datasets import PPI

from module import training


class datasets():
    def __init__(self):
        pass

    def download(self,dataset_path):
        self.dataset_list = {
            'cora_dataset':Planetoid(root=dataset_path,name='Cora'),
            'citeseer_dataset':Planetoid(root=dataset_path,name='Citeseer'),
            'pubmed_dataset':Planetoid(root=dataset_path,name='Pubmed'),
            'lastfmasia_dataset':LastFMAsia(root=os.path.join(dataset_path,'LastFMAsia')),
            'facebook_dataset':FacebookPagePage(root=os.path.join(dataset_path,'Facebook')),
            'ppi_dataset':PPI(root=os.path.join(dataset_path,'PPI')),
            'enzyme_dataset':TUDataset(root=dataset_path,name='ENZYMES')
        }
        with open(os.path.join(dataset_path,'download_raw_data.datasets'),'wb') as f:
            joblib.dump(self.__dict__,f,compress=('gzip',3))
    
    def load(self,dataset_path):
        with open(os.path.join(dataset_path,'download_raw_data.datasets'),'rb') as f:
            self.__dict__.update(joblib.load(f))


def dataset_load(config):
    data = datasets()

    if config.mode == 'download':
        data.download(config.dataset_path)
    
    elif config.mode == 'training':
        load_start_time = time.perf_counter()
        print('data is loading...')
        data.load(config.dataset_path)
        print('data load finish, spend time {:.6}s'.format(time.perf_counter()-load_start_time))
        dataset_name_list = list(data.dataset_list.keys())
        #dataset_name_list = ['ppi_dataset','enzyme_dataset']

        for dataset in dataset_name_list:
            training(config,dataset,data.dataset_list[dataset])


if __name__ == '__main__':
    config = argparse.Namespace(
        dataset_path = r'E:\吴恩达-机器学习\function_test\pyg\dataset',
        result_path = r'E:\吴恩达-机器学习\function_test\pyg\result',
        lr = 1e-3,
        epoch = 500,
        mode = 'training'
    )
    dataset_load(config)
