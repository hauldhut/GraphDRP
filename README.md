# Resources:
+ README.md: this file.
+ data: GDSC dataset

###  source codes:
+ preprocess.py: create data in pytorch format
+ utils.py: include TestbedDataset used by create_data.py to create data, performance measures and functions to draw loss, pearson by epoch.
+ models/ginconv.py, gat.py, gat_gcn.py, and gcn.py: proposed models GINConvNet, GATNet, GAT_GCN, and GCNNet receiving graphs as input for drugs.
+ training.py: train a GraphDRP model.
+ saliancy_map.py: run this to get saliency value.


## Dependencies
+ [Torch](https://pytorch.org/)
+ [Pytorch_geometric](https://github.com/rusty1s/pytorch_geometric)
+ [Rdkit](https://www.rdkit.org/)
+ [Matplotlib](https://matplotlib.org/)
+ [Pandas](https://pandas.pydata.org/)
+ [Numpy](https://numpy.org/)
+ [Scipy](https://docs.scipy.org/doc/)

# Step-by-step running:

## 1. Create data in pytorch format
```sh
python preprocess.py --choice 0
```
choice:
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;0: create mixed test dataset
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;1: create saliency map dataset
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;2: create blind drug dataset
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;3: create blind cell dataset

This returns file pytorch format (.pt) stored at data/processed including training, validation, test set.

## 2. Train a GraphDRP model
```sh
python training.py --model 0 --train_batch 1024 --val_batch 1024 --test_batch 1024 --lr 0.0001 --num_epoch 300 --log_interval 20 --cuda_name "cuda:0"
```
model:
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;1: GINConvNet
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;2: GATNet
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;3: GAT_GCN
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;4: GCNNet

To train a model using training data. The model is chosen if it gains the best MSE for testing data. 

This returns the model and result files for the modelling achieving the best MSE for testing data throughout the training.

## 3. Get saliency value 
```sh
python saliency_map.py --model 0 --num_feature 10 --processed_data_file "data/processed/GDSC_bortezomib.pt" --model_file "model_GINConvNet_GDSC.model" --cuda_name "cuda:0"
```
The model and model_file must be the same kind of graph neural network. This outputs most important abberations with corresponding saliency value.
