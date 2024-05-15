import warnings
import torch
import pandas as pd
import utils as u
import os
from loader import load_data, data_to_pyg, reduce_features
from train import train, test
from models import models
from argparse import ArgumentParser
from models.custom_gat.model import GAT

from torchinfo import summary


def model_analysis(name, model, data, compare_illicit, args):
    #print(summary(model))
    #print(model)
    data = data.to(args.device)
    print('-'*50)
    print(f"Training model: {name}")
    print('-'*50)
    train(args, model, data)
    print('-'*50)
    print(f"Testing model: {name}")
    print('-'*50)
    test(model, data)
    print('-'*50)
    print(f"Computing metrics for model: {name}")
    print('-'*50)
    metrics = u.compute_metrics(model, name, data)
    metrics_df = pd.DataFrame(metrics, index=[0])  # Convert the dictionary to a DataFrame
    compare_illicit = pd.concat([compare_illicit, metrics_df], ignore_index=True)
    return compare_illicit


warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

parser = ArgumentParser()
parser.add_argument("-d", "--data", dest="data_path", help="Path of data folder")
command_line_args = parser.parse_args()
data_path = command_line_args.data_path

print("Loading configuration from file...")
args = u.get_config()
print("Configuration loaded successfully")
print("="*50)
print("Loading graph data...")
data_path = args.data_path if data_path is None else data_path

features, edges = load_data(data_path)
features_noAgg, edges_noAgg = load_data(data_path, noAgg=True)

u.seed_everything(42)

data = data_to_pyg(features, edges)
data_noAgg = data_to_pyg(features_noAgg, edges_noAgg)
# SF: added to remove all aggregated features that have high correlation (if corr=0.9, we remove 22 features)
features_reduced = reduce_features(features)
data_reduced = data_to_pyg(features_reduced, edges)

print("Graph data loaded successfully")
print("="*50)
args.use_cuda = (torch.cuda.is_available() and args.use_cuda)
args.device = 'cpu'
if args.use_cuda:
    args.device = 'cuda'
print ("Using CUDA: ", args.use_cuda, "- args.device: ", args.device)

models_to_train = {
    # GCN
    'GCN (tx)': models.GCNConvolution(args, data_noAgg.num_features, args.hidden_units_noAgg).to(args.device),
    'GCN (tx+agg)': models.GCNConvolution(args, data.num_features, args.hidden_units).to(args.device),
    'GCN (tx+agg_red)': models.GCNConvolution(args, data_reduced.num_features, args.hidden_units).to(args.device),
    'GCN+Lin (tx)': models.GCNConvolutionLin(args, data_noAgg.num_features, args.hidden_units_noAgg).to(args.device),
    'GCN+Lin (tx+agg)': models.GCNConvolutionLin(args, data.num_features, args.hidden_units).to(args.device),
    'GCN+Lin (tx+agg_red)': models.GCNConvolutionLin(args, data_reduced.num_features, args.hidden_units).to(args.device),
    'GCN+Lin+Skip (tx)': models.GCNConvolutionLinSkip(args, data_noAgg.num_features, args.hidden_units_noAgg).to(args.device),
    'GCN+Lin+Skip (tx+agg)': models.GCNConvolutionLinSkip(args, data.num_features, args.hidden_units).to(args.device),
    'GCN+Lin+Skip (tx+agg_red)': models.GCNConvolutionLinSkip(args, data_reduced.num_features, args.hidden_units).to(args.device),
    # GAT
    'GAT (tx)': models.GATConvolution(args, data_noAgg.num_features, args.hidden_units_noAgg).to(args.device),
    'GAT (tx+agg)': models.GATConvolution(args, data.num_features, args.hidden_units).to(args.device),
    'GAT (tx+agg_red)': models.GATConvolution(args, data_reduced.num_features, args.hidden_units).to(args.device),
    'GAT+Lin (tx)': models.GATConvolutionLin(args, data_noAgg.num_features, args.hidden_units_noAgg).to(args.device),
    'GAT+Lin (tx+agg)': models.GATConvolutionLin(args, data.num_features, args.hidden_units).to(args.device),
    'GAT+Lin (tx+agg_red)': models.GATConvolutionLin(args, data_reduced.num_features, args.hidden_units).to(args.device),
    'GAT+Lin+Skip (tx)': models.GATConvolutionLinSkip(args, data_noAgg.num_features, args.hidden_units_noAgg).to(args.device),
    'GAT+Lin+Skip (tx+agg)': models.GATConvolutionLinSkip(args, data.num_features, args.hidden_units).to(args.device),
    'GAT+Lin+Skip (tx+agg_red)': models.GATConvolutionLinSkip(args, data_reduced.num_features, args.hidden_units).to(args.device),
     #SAGE
    'SAGE (tx)': models.SAGEConvolution(args, data_noAgg.num_features, args.hidden_units_noAgg).to(args.device),
    'SAGE (tx+agg)': models.SAGEConvolution(args, data.num_features, args.hidden_units).to(args.device),
    'SAGE (tx+agg_red)': models.SAGEConvolution(args, data_reduced.num_features, args.hidden_units).to(args.device),
    'SAGE+Lin (tx)': models.SAGEConvolutionLin(args, data_noAgg.num_features, args.hidden_units_noAgg).to(args.device),
    'SAGE+Lin (tx+agg)': models.SAGEConvolutionLin(args, data.num_features, args.hidden_units).to(args.device),
    'SAGE+Lin (tx+agg_red)': models.SAGEConvolutionLin(args, data_reduced.num_features, args.hidden_units).to(args.device),
    'SAGE+Lin+Skip (tx)': models.SAGEConvolutionLinSkip(args, data_noAgg.num_features, args.hidden_units_noAgg).to(args.device),
    'SAGE+Lin+Skip (tx+agg)': models.SAGEConvolutionLinSkip(args, data.num_features, args.hidden_units).to(args.device),
    'SAGE+Lin+Skip (tx+agg_red)': models.SAGEConvolutionLinSkip(args, data_reduced.num_features, args.hidden_units).to(args.device),
    # ChebNet
    'Chebyshev (tx)': models.ChebyshevConvolution(args, [1, 2], data_noAgg.num_features, args.hidden_units_noAgg).to(args.device),
    'Chebyshev (tx+agg)': models.ChebyshevConvolution(args, [1, 2], data.num_features, args.hidden_units).to(args.device),
    'Chebyshev (tx+agg_red)': models.ChebyshevConvolution(args, [1, 2], data_reduced.num_features, args.hidden_units).to(args.device),
    'Chebyshev+Lin (tx)': models.ChebyshevConvolutionLin(args, [1, 2], data_noAgg.num_features, args.hidden_units_noAgg).to(args.device),
    'Chebyshev+Lin (tx+agg)': models.ChebyshevConvolutionLin(args, [1, 2], data.num_features, args.hidden_units).to(args.device),
    'Chebyshev+Lin (tx+agg_red)': models.ChebyshevConvolutionLin(args, [1, 2], data_reduced.num_features, args.hidden_units).to(args.device),
    'Chebyshev+Lin+Skip (tx)': models.ChebyshevConvolutionLinSkin(args, [1, 2], data_noAgg.num_features, args.hidden_units_noAgg).to(args.device),
    'Chebyshev+Lin+Skip (tx+agg)': models.ChebyshevConvolutionLinSkin(args, [1, 2], data.num_features, args.hidden_units).to(args.device),
    'Chebyshev+Lin+Skip (tx+agg_red)': models.ChebyshevConvolutionLinSkin(args, [1, 2], data_reduced.num_features, args.hidden_units).to(args.device),
     # GATv2 
    'GATv2 (tx)': models.GATv2Convolution(args, data_noAgg.num_features, args.hidden_units_noAgg).to(args.device),
    'GATv2 (tx+agg)': models.GATv2Convolution(args, data.num_features, args.hidden_units).to(args.device),
    'GATv2 (tx+agg_red)': models.GATv2Convolution(args, data_reduced.num_features, args.hidden_units).to(args.device),
    'GATv2+Lin (tx)': models.GATv2ConvolutionLin(args, data_noAgg.num_features, args.hidden_units_noAgg).to(args.device),
    'GATv2+Lin (tx+agg)': models.GATv2ConvolutionLin(args, data.num_features, args.hidden_units).to(args.device),
    'GATv2+Lin (tx+agg_red)': models.GATv2ConvolutionLin(args, data_reduced.num_features, args.hidden_units).to(args.device),
    'GATv2+Lin+Skip (tx)': models.GATv2ConvolutionLinSkip(args, data_noAgg.num_features, args.hidden_units_noAgg).to(args.device),
    'GATv2+Lin+Skip (tx+agg)': models.GATv2ConvolutionLinSkip(args, data.num_features, args.hidden_units).to(args.device),
    'GATv2+Lin+Skip (tx+agg_red)': models.GATv2ConvolutionLinSkip(args, data_reduced.num_features, args.hidden_units).to(args.device)
}

compare_illicit = pd.DataFrame(columns=['model','Precision','Recall', 'F1', 'F1 Micro AVG'])
print("Starting training models")
print("="*50)

model_list = list(models_to_train.items())

for i in range(0, len(model_list), 3):
    # configuration (tx)
    (name, model) = model_list[i]
    compare_illicit = model_analysis(name, model, data_noAgg, compare_illicit, args)
    # configuration (tx+agg)
    (name, model) = model_list[i + 1]
    compare_illicit = model_analysis(name, model, data, compare_illicit, args)
    # configuration (tx+agg_red)
    (name, model) = model_list[i + 2]
    compare_illicit = model_analysis(name, model, data_reduced, compare_illicit, args)


compare_illicit.to_csv(os.path.join(data_path, 'metrics.csv'), index=False)
print('Results saved to metrics.csv')

print('='*50)
print('='*20 + " RESULTS "+ '='*21)
print('='*50)
print()

print(compare_illicit)

u.plot_results(compare_illicit)

#print('-'*50)
#compare_illicit = u.compute_metrics(model, name, data)

u.aggregate_plot(compare_illicit)
