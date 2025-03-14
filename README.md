## Anti-money Laundering on Elliptic Dataset with GNN

The goal of this work is to tackle anti-money laundering problem, trying to classify efficiently illicit transactions and create a comparison between different architectures, in particular comparing different types of Graph Neural Networks (GNNs), to identify what are the key features and approaches that enable good performances in this given context.

### Credits

This is a revised version of a project from one of my students. The original project can be found [here]
(https://github.com/simonemarasi/aml-elliptic-gnn)

### Setup

First of all, you have to clone the repository with the standard command:

`git clone https://github.com/stefano-ferretti/aml-elliptic-gnn`

#### Download the data

You can download the dataset zipped from the following [link](http://dl.dropboxusercontent.com/scl/fi/2j7nx8y3jbyypdm7r100f/dataset.zip?rlkey=veu69cngj0els6emgp549r06u&dl=0]). After done that make sure to extract the zip file into the `data` folder located at the root of the repository. However, you can put the data also in other places, making sure to change the folder in the configuration file (`config.yaml`) accordingly.

In the configuration file, it is possible also to modify some hyperparameters such as the number of epochs, the number of hidden units to use, the learning rate, etc.

### Run

It is possible to install all the packages required for the execution launching the command

`pip install -r requirements.txt`

After that, to run the script execute the command

`python main.py`

or, assuming for example you have the dataset stored in a directory 'data/'

`python main.py -d data/`

### Publication

Please cite:

S. Ferretti, G. D'Angelo, V. Ghini "Enhancing Anti-Money Laundering Frameworks: An Application of Graph Neural Networks in Cryptocurrency Transaction Classification", IEEE Access, IEEE Press, Mar 2025

The original version of the project was presented at **2024 IEEE 21st Consumer Communications & Networking Conference (CCNC)** held in Las Vagas on January 2024. This is however a revised version, where more classifiers are considered.

S. Marasi and S. Ferretti, "Anti-Money Laundering in Cryptocurrencies Through Graph Neural Networks: A Comparative Study," 2024 IEEE 21st Consumer Communications & Networking Conference (CCNC), Las Vegas, NV, USA, 2024, pp. 272-277, doi: 10.1109/CCNC51664.2024.10454631.

Bibtex:

@INPROCEEDINGS{10454631,  
  author={Marasi, Simone and Ferretti, Stefano},  
  booktitle={2024 IEEE 21st Consumer Communications & Networking Conference (CCNC)},  
  title={Anti-Money Laundering in Cryptocurrencies Through Graph Neural Networks: A Comparative Study}, 
  year={2024},  
  volume={},  
  number={},  
  pages={272-277},  
  doi={10.1109/CCNC51664.2024.10454631}  
}


