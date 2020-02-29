import numpy as np
import pandas as pd
import sys, os
from random import shuffle
import torch
import torch.nn as nn
from models.gat import GATNet
from models.gat_gcn import GAT_GCN
from models.gcn import GCNNet
from models.ginconv import GINConvNet
from utils import *
import pickle

from flashtorch.utils import apply_transforms, load_image
from flashtorch.saliency import Backprop

import argparse


def calculate_value_individual_drug(modeling, num_mut, cuda_name, processed_data_file, model_file):
    dataset = "GDSC"
    with open ('mut_dict', 'rb') as fp:
        mut_dict = pickle.load(fp)
        mut_arr = np.asarray([k for k, v in mut_dict.items()])
 
    if (not os.path.isfile(processed_data_file)):
        print('please run create_data.py to prepare data in pytorch format!')
    else:
        test_data = TestbedDataset(root='data', dataset=dataset+'_bortezomib')
        test_loader = DataLoader(test_data)
        model_st = modeling.__name__
        print('\npredicting for ', dataset, ' using ', model_st)
        device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
        model = modeling().to(device)
        lstY = []
        lstM = []
        lstV = []
        if os.path.isfile(model_file):
            model.load_state_dict(torch.load(model_file))
            model.eval()
            for data in test_loader:
                data = data.to(device)
                output, _ = model(data)
                data.target.retain_grad()
                output.backward()
                lstY.append(data.y.cpu().numpy()[0])
                grad = data.target.grad
                values, indexes = grad.topk(num_mut)
                lstV.append(values)
                lstM.append(mut_arr[np.squeeze(np.asarray(indexes.cpu().numpy()))])
        else:
            print('model is not available!')
        listCell = []
        with open ('cell_blind_sal', 'rb') as fp:
            listCell = pickle.load(fp)

        lstTopY = [lstY[k] for k in np.asarray(lstY).argsort()[:num_mut]]
        lstTopM = [lstM[k] for k in np.asarray(lstY).argsort()[:num_mut]]
        lstV = [lstV[k] for k in np.asarray(lstY).argsort()[:num_mut]]
        listCell = [listCell[k] for k in np.asarray(lstY).argsort()[:num_mut]]
        print(lstTopM)
        print(lstV)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='calculate saliency value')
    parser.add_argument('--model', type=int, required=False, default=0,     help='0: GINConvNet, 1: GATNet, 2: GAT_GCN, 3: GCNNet')
    parser.add_argument('--num_feature', type=int, required=False, default=10,  help='Number of important mutation')
    parser.add_argument('--processed_data_file', type=str, required=False, default="data/processed/GDSC_bortezomib.pt", help='Path to processed file')
    parser.add_argument('--model_file', type=str, required=False, default="model_GINConvNet_GDSC.model", help='Path to model file')
    parser.add_argument('--cuda_name', type=str, required=False, default="cuda:0", help='Cuda')

    args = parser.parse_args()

    modeling = [GINConvNet, GATNet, GAT_GCN, GCNNet][args.model]
    num_mut = args.num_feature
    cuda_name = args.cuda_name
    processed_data_file = args.processed_data_file
    model_file = args.model_file

    calculate_value_individual_drug(modeling, num_mut, cuda_name, processed_data_file, model_file)
