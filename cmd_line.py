import os
import argparse

from dataset_process import dataset_load

def main(config):
    if not os.path.exists(config.dataset_path):
        os.makedirs(config.dataset_path)
    if not os.path.exists(config.result_path):
        os.makedirs(config.result_path)

    dataset_load(config)


if __name__ == '__main__':
    parse = argparse.ArgumentParser(description='Control the dataset import and running.')

    parse.add_argument('--dataset_path',type=str,default=r'E:\吴恩达-机器学习\function_test\pyg\dataset',help='path to save raw dataset.')
    parse.add_argument('--result_path',type=str,default=r'E:\吴恩达-机器学习\function_test\pyg\result',help='path to save result.')
    
    parse.add_argument('--lr',type=float,default=1e-3,help='the learning rate for each module\'s optimizer.')
    parse.add_argument('--epoch',type=int,default=500,help='the total epochs for each module.')
    parse.add_argument('--mode',default='training',choices=['download','training'],help='download for datasets; training for module running.')

    config=parse.parse_args()
    main(config)