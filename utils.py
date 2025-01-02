import numpy as np
import yaml
import matplotlib.pyplot as plt
import os
from sklearn.metrics import precision_recall_fscore_support, f1_score
import torch


import random
import numpy as np

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_config():
    with open(os.path.join(os.getcwd(), "config.yaml"), "r") as config:
        args = AttributeDict(yaml.safe_load(config))
    args.lr = float(args.lr)
    args.weight_decay = float(args.weight_decay)
    return args

def accuracy(pred_y, y):
    """Calculate accuracy"""
    return ((pred_y == y).sum() / len(y)).item()

def compute_metrics(model, name, data, training_time, device):
  model.eval()
  with torch.no_grad():
    y_predicted = model((data.x, data.edge_index))[0].to(device)
    y_predicted = y_predicted.argmax(dim=1)

  # Sposta i tensori su CPU prima di convertirli in array NumPy
  y_true = data.y[data.test_mask].cpu().numpy()
  y_pred = y_predicted[data.test_mask].cpu().numpy()
 
  prec_ill,rec_ill,f1_ill,_ = precision_recall_fscore_support(y_true, y_pred, average='binary', pos_label=0)
  f1_micro = f1_score(y_true, y_pred, average='micro')
  m = {'model': name, 
       'Precision': np.round(prec_ill,3), 
       'Recall': np.round(rec_ill,3), 
       'F1': np.round(f1_ill,3),
       'F1 Micro AVG':np.round(f1_micro,3),
       'Training Time': np.round(training_time,3)}
  return m

def plot_results(df):

    labels = df['model'].to_numpy()
    precision = df['Precision'].to_numpy()
    recall = df['Recall'].to_numpy()
    f1 = df['F1'].to_numpy()
    f1_micro = df['F1 Micro AVG'].to_numpy()

    x = np.arange(len(labels))
    width = 0.15
    fig, ax = plt.subplots(figsize=(20, 7))
    ax.bar(x - width/2, precision, width, label='Precision',color='#83f27b')
    ax.bar(x + width/2, recall, width, label='Recall',color='#f27b83')
    ax.bar(x - (3/2)*width, f1, width, label='F1',color='#f2b37b')
    ax.bar(x + (3/2)*width, f1_micro, width, label='Micro AVG F1',color='#7b8bf2')

    ax.set_ylabel('value')
    ax.set_title('Metrics for illicit class')
    ax.set_xticks(np.arange(0,len(labels),1))
    ax.set_yticks(np.arange(0,1,0.05))
    ax.set_xticklabels(labels=labels)
    ax.legend(loc="lower left")

    plt.grid(True)
    plt.show()
    fig.savefig('results.png')

def aggregate_plot(df):
    labels = df['model'].to_numpy()

    precision = df['Precision'].to_numpy()
    recall = df['Recall'].to_numpy()
    f1 = df['F1'].to_numpy()
    maf1 = df['F1 Micro AVG'].to_numpy()

    x = np.arange(len(labels))  # the label locations
    width = 0.55  # the width of the bars
    fig, ax = plt.subplots(figsize=(6, 10))
    ax.bar(x, f1, width, label='F1 Score',color='#f2b37b')
    ax.bar(x , maf1, width, label='M.A. F1 Score',color='#7b8bf2',bottom=f1)
    ax.bar(x, precision, width, label='Precision',color='#83f27b',bottom=maf1 + f1)
    ax.bar(x, recall, width, label='Recall',color='#f27b83',bottom=maf1 + f1 + precision)

    ax.set_ylabel('value 0-1')
    ax.set_title('Final metrics by classifier')
    ax.set_xticks(np.arange(0,len(labels),1))
    ax.set_yticks(np.arange(0,4,0.1))
    ax.set_xticklabels(labels=labels)
    ax.legend()

    plt.xticks(rotation=90)
    plt.grid(True)
    fig.tight_layout()
    plt.show()
    fig.savefig('aggregated_results.png')


def stats_plot(df):
    labels = df['model'].to_numpy()
    precision_mean = df['Precision_mean'].to_numpy()
    precision_std = df['Precision_std'].to_numpy()
    recall_mean = df['Recall_mean'].to_numpy()
    recall_std = df['Recall_std'].to_numpy()
    f1_mean = df['F1_mean'].to_numpy()
    f1_std = df['F1_std'].to_numpy()
    f1_micro_mean = df['F1_Micro_AVG_mean'].to_numpy()
    f1_micro_std = df['F1_Micro_AVG_std'].to_numpy()

    x = np.arange(len(labels))
    width = 0.2

    fig, ax = plt.subplots(figsize=(20, 7))

    ax.bar(x - 1.5*width, precision_mean, width, yerr=precision_std, label='Precision', color='#83f27b', capsize=5)
    ax.bar(x - 0.5*width, recall_mean, width, yerr=recall_std, label='Recall', color='#f2837b', capsize=5)
    ax.bar(x + 0.5*width, f1_mean, width, yerr=f1_std, label='F1', color='#7b83f2', capsize=5)
    ax.bar(x + 1.5*width, f1_micro_mean, width, yerr=f1_micro_std, label='F1 Micro AVG', color='#f2e27b', capsize=5)

    ax.set_xlabel('Models')
    ax.set_ylabel('Scores')
    ax.set_title('Performance Metrics with Standard Deviation')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.legend()

    fig.tight_layout()
    plt.show()
    fig.savefig('stats_results.png')

def stats_stacked_plot(df):
    labels = df['model'].to_numpy()

    precision_mean = df['Precision_mean'].to_numpy()
    precision_std = df['Precision_std'].to_numpy()
    recall_mean = df['Recall_mean'].to_numpy()
    recall_std = df['Recall_std'].to_numpy()
    f1_mean = df['F1_mean'].to_numpy()
    f1_std = df['F1_std'].to_numpy()
    f1_micro_mean = df['F1_Micro_AVG_mean'].to_numpy()
    f1_micro_std = df['F1_Micro_AVG_std'].to_numpy()

    x = np.arange(len(labels))  # the label locations
    width = 0.55  # the width of the bars

    fig, ax = plt.subplots(figsize=(10, 6))

    # Create stacked bar chart with error bars
    ax.bar(x, f1_mean, width, yerr=f1_std, label='F1 Score', color='#f2b37b', capsize=5)
    ax.bar(x, f1_micro_mean, width, yerr=f1_micro_std, label='M.A. F1 Score', color='#7b8bf2', bottom=f1_mean, capsize=5)
    ax.bar(x, precision_mean, width, yerr=precision_std, label='Precision', color='#83f27b', bottom=f1_micro_mean + f1_mean, capsize=5)
    ax.bar(x, recall_mean, width, yerr=recall_std, label='Recall', color='#f27b83', bottom=f1_micro_mean + f1_mean + precision_mean, capsize=5)

    ax.set_ylabel('Value (0-1)')
    ax.set_title('Stacked Bar Chart of Performance Metrics with Standard Deviation')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.legend()

    fig.tight_layout()
    plt.show()
    fig.savefig('stats_stacked.png')

class AttributeDict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


