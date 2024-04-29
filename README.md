## Anti-money Laundering on Elliptic Dataset with GNN

Experimenting Graph Neural Network with Elliptic dataset (Anti-Money Laundering)
This is a revised version of a project from one of my students. The original project is [here](https://github.com/simonemarasi/aml-elliptic-gnn)

### Setup

First of all you have to clone the repository with the standard command:

`git clone https://github.com/stefano-ferretti/aml-elliptic-gnn`

#### Download the data

You can download the dataset zipped from the following [link](https://www.4sync.com/web/directDownload/fQErng3L/5YfHxh7W.cc4f36f14c07d75ced4bf1fcfa1a0772). After done that make sure to extract the zip file into the `data` folder located at the root of the repository. However you can put the data also in other places, making sure to change the folder in the configuration file (`config.yaml`) accordingly.

In the configuration file it is possible also to modify some hyperparameters such as the number of epochs, the number of hidden units to use, the learning rate, etc.

### Run

It is possible to install all the packages required for the execution launching the command
`pip install -r requirements.txt`
After that, to run the script execute the command
`python main.py`

### Cite
You can cite this work through this paper:

S. Marasi, S. Ferretti, "Anti-Money Laundering in Cryptocurrencies Through Graph Neural Networks: A Comparative Study'', in Proc. of the IEEE Consumer Communications & Networking Conference (CCNC 2024), IEEE ComSoc, January 2024, Las Vegas, USA.

@inproceedings{DBLP:conf/ccnc/MarasiF24,  
  author       = {Simone Marasi and
                  Stefano Ferretti},    
  title        = {Anti-Money Laundering in Cryptocurrencies Through Graph Neural Networks:\\
                  {A} Comparative Study},                    
  booktitle    = {21st {IEEE} Consumer Communications {\&} Networking Conference,
                  {CCNC} 2024, Las Vegas, NV, USA, January 6-9, 2024},  
  pages        = {272--277},    
  publisher    = {{IEEE}},    
  year         = {2024},    
  url          = {https://doi.org/10.1109/CCNC51664.2024.10454631},    
  doi          = {10.1109/CCNC51664.2024.10454631}    
}
