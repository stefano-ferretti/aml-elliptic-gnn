import warnings
import torch
import pandas as pd
import numpy as np
import utils as u
import os
from loader import load_data, data_to_pyg, reduce_features
from train import train, test
from models import models
from argparse import ArgumentParser
#from models.custom_gat.model import GAT
from torchinfo import summary
import time
from scipy.stats import ttest_ind


# Esegui il t-test per confrontare le metriche tra i modelli
def perform_t_test(df, metric):
    models = df['model'].unique()
    p_values = pd.DataFrame(index=models, columns=models)

    for i, model1 in enumerate(models):
        for j, model2 in enumerate(models):
            if i < j:
                data1 = df[df['model'] == model1][metric].dropna()
                data2 = df[df['model'] == model2][metric].dropna()
                # Controlla che la varianza non sia zero
                if np.var(data1) == 0 or np.var(data2) == 0:
                    p_value = np.nan
                else:
                    t_stat, p_value = ttest_ind(data1, data2)

                p_values.loc[model1, model2] = p_value
                p_values.loc[model2, model1] = p_value

    return p_values


def model_analysis(name, model, data, compare_illicit, args):
    #print(summary(model))
    #print(model)
    data = data.to(args.device)
    print('-'*40)
    print(f"Training model: {name}")
    print('-'*40)
    start_time = time.time()
    train(args, model, data)
    training_time = time.time() - start_time
    print('-'*40)
    print(f"Testing model: {name}")
    print('-'*40)
    test(model, data)
    print('-'*40)
    print(f"Computing metrics for model: {name}")
    print('-'*40)
    metrics = u.compute_metrics(model, name, data, training_time, args.device)
    metrics_df = pd.DataFrame(metrics, index=[0])  # Convert the dictionary to a DataFrame
    compare_illicit = pd.concat([compare_illicit, metrics_df], ignore_index=True)
    return compare_illicit


# Helper function to create model with proper device placement
def create_model(model_class, *model_args):
    model = model_class(*model_args)
    return model.to(args.device)


###### main analysis function ######
def whole_analysis(features, edges, args): 
    compare_illicit = pd.DataFrame(columns=['model','Precision','Recall', 'F1', 'F1 Micro AVG', 'Training Time'])
    print("Starting training models")
    print("="*50)

    for iteration in range(args.num_iterations):
        print("="*60)
        print("="*60)
        print(f"Iteration {iteration + 1}/{args.num_iterations}")
        print("="*60)
        print("="*60)

        data = data_to_pyg(features, edges)
        data_noAgg = data_to_pyg(features_noAgg, edges_noAgg)
        # remove all aggregated features that have high correlation (if corr=0.9, we remove 22 features)
        features_reduced = reduce_features(features)
        data_reduced = data_to_pyg(features_reduced, edges)
        
        models_to_train = {
            'GCN (tx)': create_model(models.GCNConvolution, args, data_noAgg.num_features, args.hidden_units_noAgg),
            'GCN (tx+agg)': create_model(models.GCNConvolution, args, data.num_features, args.hidden_units),
            'GCN (tx+agg_red)': create_model(models.GCNConvolution, args, data_reduced.num_features, args.hidden_units)
        }
        """
        models_to_train = {
            'GCN (tx)': create_model(models.GCNConvolution, args, data_noAgg.num_features, args.hidden_units_noAgg),
            'GCN (tx+agg)': create_model(models.GCNConvolution, args, data.num_features, args.hidden_units),
            'GCN (tx+agg_red)': create_model(models.GCNConvolution, args, data_reduced.num_features, args.hidden_units),
            'GCN+Lin (tx)': create_model(models.GCNConvolutionLin, args, data_noAgg.num_features, args.hidden_units_noAgg),
            'GCN+Lin (tx+agg)': create_model(models.GCNConvolutionLin, args, data.num_features, args.hidden_units),
            'GCN+Lin (tx+agg_red)': create_model(models.GCNConvolutionLin, args, data_reduced.num_features, args.hidden_units),
            'GCN+Lin+Skip (tx)': create_model(models.GCNConvolutionLinSkip, args, data_noAgg.num_features, args.hidden_units_noAgg),
            'GCN+Lin+Skip (tx+agg)': create_model(models.GCNConvolutionLinSkip, args, data.num_features, args.hidden_units),
            'GCN+Lin+Skip (tx+agg_red)': create_model(models.GCNConvolutionLinSkip, args, data_reduced.num_features, args.hidden_units),
            # GAT
            'GAT (tx)': create_model(models.GATConvolution, args, data_noAgg.num_features, args.hidden_units_noAgg),
            'GAT (tx+agg)': create_model(models.GATConvolution, args, data.num_features, args.hidden_units),
            'GAT (tx+agg_red)': create_model(models.GATConvolution, args, data_reduced.num_features, args.hidden_units),
            'GAT+Lin (tx)': create_model(models.GATConvolutionLin, args, data_noAgg.num_features, args.hidden_units_noAgg),
            'GAT+Lin (tx+agg)': create_model(models.GATConvolutionLin, args, data.num_features, args.hidden_units),
            'GAT+Lin (tx+agg_red)': create_model(models.GATConvolutionLin, args, data_reduced.num_features, args.hidden_units),
            'GAT+Lin+Skip (tx)': create_model(models.GATConvolutionLinSkip, args, data_noAgg.num_features, args.hidden_units_noAgg),
            'GAT+Lin+Skip (tx+agg)': create_model(models.GATConvolutionLinSkip, args, data.num_features, args.hidden_units),
            'GAT+Lin+Skip (tx+agg_red)': create_model(models.GATConvolutionLinSkip, args, data_reduced.num_features, args.hidden_units),
            # SAGE
            'SAGE (tx)': create_model(models.SAGEConvolution, args, data_noAgg.num_features, args.hidden_units_noAgg),
            'SAGE (tx+agg)': create_model(models.SAGEConvolution, args, data.num_features, args.hidden_units),
            'SAGE (tx+agg_red)': create_model(models.SAGEConvolution, args, data_reduced.num_features, args.hidden_units),
            'SAGE+Lin (tx)': create_model(models.SAGEConvolutionLin, args, data_noAgg.num_features, args.hidden_units_noAgg),
            'SAGE+Lin (tx+agg)': create_model(models.SAGEConvolutionLin, args, data.num_features, args.hidden_units),
            'SAGE+Lin (tx+agg_red)': create_model(models.SAGEConvolutionLin, args, data_reduced.num_features, args.hidden_units),
            'SAGE+Lin+Skip (tx)': create_model(models.SAGEConvolutionLinSkip, args, data_noAgg.num_features, args.hidden_units_noAgg),
            'SAGE+Lin+Skip (tx+agg)': create_model(models.SAGEConvolutionLinSkip, args, data.num_features, args.hidden_units),
            'SAGE+Lin+Skip (tx+agg_red)': create_model(models.SAGEConvolutionLinSkip, args, data_reduced.num_features, args.hidden_units),
            # ChebNet
            'Chebyshev (tx)': create_model(models.ChebyshevConvolution, args, [1, 2], data_noAgg.num_features, args.hidden_units_noAgg),
            'Chebyshev (tx+agg)': create_model(models.ChebyshevConvolution, args, [1, 2], data.num_features, args.hidden_units),
            'Chebyshev (tx+agg_red)': create_model(models.ChebyshevConvolution, args, [1, 2], data_reduced.num_features, args.hidden_units),
            'Chebyshev+Lin (tx)': create_model(models.ChebyshevConvolutionLin, args, [1, 2], data_noAgg.num_features, args.hidden_units_noAgg),
            'Chebyshev+Lin (tx+agg)': create_model(models.ChebyshevConvolutionLin, args, [1, 2], data.num_features, args.hidden_units),
            'Chebyshev+Lin (tx+agg_red)': create_model(models.ChebyshevConvolutionLin, args, [1, 2], data_reduced.num_features, args.hidden_units),
            'Chebyshev+Lin+Skip (tx)': create_model(models.ChebyshevConvolutionLinSkin, args, [1, 2], data_noAgg.num_features, args.hidden_units_noAgg),
            'Chebyshev+Lin+Skip (tx+agg)': create_model(models.ChebyshevConvolutionLinSkin, args, [1, 2], data.num_features, args.hidden_units),
            'Chebyshev+Lin+Skip (tx+agg_red)': create_model(models.ChebyshevConvolutionLinSkin, args, [1, 2], data_reduced.num_features, args.hidden_units),
            # GATv2
            'GATv2 (tx)': create_model(models.GATv2Convolution, args, data_noAgg.num_features, args.hidden_units_noAgg),
            'GATv2 (tx+agg)': create_model(models.GATv2Convolution, args, data.num_features, args.hidden_units),
            'GATv2 (tx+agg_red)': create_model(models.GATv2Convolution, args, data_reduced.num_features, args.hidden_units),
            'GATv2+Lin (tx)': create_model(models.GATv2ConvolutionLin, args, data_noAgg.num_features, args.hidden_units_noAgg),
            'GATv2+Lin (tx+agg)': create_model(models.GATv2ConvolutionLin, args, data.num_features, args.hidden_units),
            'GATv2+Lin (tx+agg_red)': create_model(models.GATv2ConvolutionLin, args, data_reduced.num_features, args.hidden_units),
            'GATv2+Lin+Skip (tx)': create_model(models.GATv2ConvolutionLinSkip, args, data_noAgg.num_features, args.hidden_units_noAgg),
            'GATv2+Lin+Skip (tx+agg)': create_model(models.GATv2ConvolutionLinSkip, args, data.num_features, args.hidden_units),
            'GATv2+Lin+Skip (tx+agg_red)': create_model(models.GATv2ConvolutionLinSkip, args, data_reduced.num_features, args.hidden_units)
        }
        """
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

    return compare_illicit
##############################

###### main ######
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

print("Graph data loaded successfully")
print("="*50)
"""
if torch.backends.mps.is_available():
    args.device = 'mps'
    print("Using MPS (Metal Performance Shaders)")
elif torch.cuda.is_available() and args.use_cuda:
    args.device = 'cuda'
    print("Using CUDA")
else:
    args.device = 'cpu'
    print("Using CPU")
"""
args.device = 'cpu'

print("Device being used:", args.device)

compare_illicit = whole_analysis(features, edges, args)

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

# Calcola la media e la deviazione standard per ciascun modello
aggregated_results = compare_illicit.groupby('model').agg(
    Precision_mean=('Precision', 'mean'),
    Precision_std=('Precision', 'std'),
    Recall_mean=('Recall', 'mean'),
    Recall_std=('Recall', 'std'),
    F1_mean=('F1', 'mean'),
    F1_std=('F1', 'std'),
    F1_Micro_AVG_mean=('F1 Micro AVG', 'mean'),
    F1_Micro_AVG_std=('F1 Micro AVG', 'std'),
    Training_Time_mean=('Training Time', 'mean'),
    Training_Time_std=('Training Time', 'std')
).reset_index()

aggregated_results.to_csv(os.path.join(data_path, 'aggregated_metrics.csv'), index=False)
print('Aggregated results saved to aggregated_metrics.csv')

print('='*50)
print('='*20 + " RESULTS "+ '='*21)
print('='*50)
print()

print(aggregated_results)

u.stats_plot(aggregated_results)
u.stats_stacked_plot(aggregated_results)

## T-TEST ##
# Calcola i p-value per ciascuna metrica
metrics = ['Precision_mean', 'Recall_mean', 'F1_mean', 'F1_Micro_AVG_mean']
p_values_dict = {}

for metric in metrics:
    p_values_dict[metric] = perform_t_test(aggregated_results, metric)

# Stampa i risultati del t-test
for metric, p_values in p_values_dict.items():
    print(f"P-values for {metric}:")
    print(p_values)
    print()
    p_values.to_csv(os.path.join(data_path, f'p_values_{metric}.csv'))
    print(f"P-values for {metric} saved to p_values_{metric}.csv")
