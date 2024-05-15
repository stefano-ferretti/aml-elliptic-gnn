from torch_geometric.nn import GCNConv, GATConv, SAGEConv, ChebConv, GINConv
from torch.nn import Module, Linear
from torch_geometric.nn.conv.gatv2_conv import GATv2Conv
import torch.nn.functional as F

#######################
class GCNConvolution(Module):
    def __init__(self, args, num_features, hidden_units):
        super().__init__()
        self.conv1 = GCNConv(num_features, hidden_units)
        self.conv2 = GCNConv(hidden_units, args['num_classes'])

    def forward(self, data):
        x, edge_index = data
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return x, edge_index

#######################
class GCNConvolutionLin(Module):
    def __init__(self, args, num_features, hidden_units):
        super().__init__()
        self.conv1 = GCNConv(num_features, hidden_units)
        self.conv2 = GCNConv(hidden_units, hidden_units)
        self.linear = Linear(hidden_units, args['num_classes'])

    def forward(self, data):
        x, edge_index = data
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = F.log_softmax(self.linear(x))
        return x, edge_index

#######################
class GCNConvolutionLinSkip(Module):
    def __init__(self, args, num_features, hidden_units):
        super().__init__()
        self.conv1 = GCNConv(num_features, hidden_units)
        self.conv2 = GCNConv(hidden_units, hidden_units)
        self.linear = Linear(hidden_units, args['num_classes'])

    def forward(self, data):
        x, edge_index = data
        x = F.relu(self.conv1(x, edge_index))
        x1 = F.dropout(x, training=self.training)
        x1 = F.relu(self.conv2(x1, edge_index))
        x1 = F.dropout(x1, training=self.training)
        x = x1 + x # add skip connection
        x = F.log_softmax(self.linear(x))
        return x, edge_index
    
#######################
class SAGEConvolution(Module):
    def __init__(self, args, num_features, hidden_units):
        super().__init__()
        self.conv1 = SAGEConv(num_features, hidden_units)
        self.conv2 = SAGEConv(hidden_units, args['num_classes'])

    def forward(self, data):
        x, edge_index = data
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return x, edge_index

#######################
class SAGEConvolutionLin(Module):
    def __init__(self, args, num_features, hidden_units):
        super().__init__()
        self.conv1 = SAGEConv(num_features, hidden_units)
        self.conv2 = SAGEConv(hidden_units, hidden_units)
        self.linear = Linear(hidden_units, args['num_classes'])

    def forward(self, data):
        x, edge_index = data
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.log_softmax(self.linear(x))
        return x, edge_index

#######################
class SAGEConvolutionLinSkip(Module):
    def __init__(self, args, num_features, hidden_units):
        super().__init__()
        self.conv1 = SAGEConv(num_features, hidden_units)
        self.conv2 = SAGEConv(hidden_units, hidden_units)
        self.linear = Linear(hidden_units, args['num_classes'])

    def forward(self, data):
        x, edge_index = data
        x = F.relu(self.conv1(x, edge_index))
        x1 = F.dropout(x, training=self.training)
        x1 = self.conv2(x1, edge_index)
        x = x + x1
        x = F.log_softmax(self.linear(x))
        return x, edge_index

#######################
class GATConvolution(Module):
    def __init__(self, args, num_features, hidden_units):
        super().__init__()
        self.conv1 = GATConv(num_features, hidden_units)
        self.conv2 = GATConv(hidden_units, args['num_classes'])

    def forward(self, data):
        x, edge_index = data
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return x, edge_index

#######################
class GATConvolutionLin(Module):
    def __init__(self, args, num_features, hidden_units):
        super().__init__()
        self.conv1 = GATConv(num_features, hidden_units)
        self.conv2 = GATConv(hidden_units, hidden_units)
        self.linear = Linear(hidden_units, args['num_classes'])

    def forward(self, data):
        x, edge_index = data
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.log_softmax(self.linear(x))
        return x, edge_index

#######################
class GATConvolutionLinSkip(Module):
    def __init__(self, args, num_features, hidden_units):
        super().__init__()
        self.conv1 = GATConv(num_features, hidden_units)
        self.conv2 = GATConv(hidden_units, hidden_units)
        self.linear = Linear(hidden_units, args['num_classes'])

    def forward(self, data):
        x, edge_index = data
        x = F.relu(self.conv1(x, edge_index))
        x1 = F.dropout(x, training=self.training)
        x1 = self.conv2(x1, edge_index)
        x = x + x1
        x = F.log_softmax(self.linear(x))
        return x, edge_index
    
#######################
class ChebyshevConvolution(Module):
    def __init__(self, args, kernel, num_features, hidden_units):
        super().__init__()
        self.conv1 = ChebConv(num_features, hidden_units, kernel[0])
        self.conv2 = ChebConv(hidden_units, args['num_classes'], kernel[1])

    def forward(self, data):
        x, edge_index = data
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return x, edge_index

#######################
class ChebyshevConvolutionLin(Module):
    def __init__(self, args, kernel, num_features, hidden_units):
        super().__init__()
        self.conv1 = ChebConv(num_features, hidden_units, kernel[0])
        self.conv2 = ChebConv(hidden_units, hidden_units, kernel[1])
        self.linear = Linear(hidden_units, args['num_classes'])

    def forward(self, data):
        x, edge_index = data
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.log_softmax(self.linear(x))
        return x, edge_index

#######################
class ChebyshevConvolutionLinSkin(Module):
    def __init__(self, args, kernel, num_features, hidden_units):
        super().__init__()
        self.conv1 = ChebConv(num_features, hidden_units, kernel[0])
        self.conv2 = ChebConv(hidden_units, hidden_units, kernel[1])
        self.linear = Linear(hidden_units, args['num_classes'])

    def forward(self, data):
        x, edge_index = data
        x = F.relu(self.conv1(x, edge_index))
        x1 = F.dropout(x, training=self.training)
        x1 = self.conv2(x1, edge_index)
        x = x + x1
        x = F.log_softmax(self.linear(x))
        return x, edge_index

#######################
class GATv3Convolution(Module):
    def __init__(self, args, num_features, hidden_units):
        super(GATv2Convolution, self).__init__()
        self.conv1 = GATv2Conv(num_features, hidden_units)
        self.lin1 = Linear(num_features, hidden_units)
        self.conv2 = GATv2Conv(hidden_units, args['num_classes'])
        self.lin2 = Linear(hidden_units, args['num_classes'])

    def forward(self, data):
        x, edge_index = data
        x = F.relu(self.conv1(x, edge_index) + self.lin1(x))
        x = self.conv2(x, edge_index) + self.lin2(x)
        return F.log_softmax(x, dim=1), edge_index

#######################
class GATv2Convolution(Module):
    def __init__(self, args, num_features, hidden_units):
        super().__init__()
        self.conv1 = GATv2Conv(num_features, hidden_units)
        self.conv2 = GATv2Conv(hidden_units, args['num_classes'])

    def forward(self, data):
        x, edge_index = data
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return x, edge_index


#######################
class GATv2ConvolutionLin(Module):
    def __init__(self, args, num_features, hidden_units):
        super().__init__()
        self.conv1 = GATv2Conv(num_features, hidden_units)
        self.conv2 = GATv2Conv(hidden_units, hidden_units)
        self.linear = Linear(hidden_units, args['num_classes'])

    def forward(self, data):
        x, edge_index = data
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.log_softmax(self.linear(x))
        return x, edge_index

#######################
class GATv2ConvolutionLinSkip(Module):
    def __init__(self, args, num_features, hidden_units):
        super().__init__()
        self.conv1 = GATv2Conv(num_features, hidden_units)
        self.conv2 = GATv2Conv(hidden_units, hidden_units)
        self.linear = Linear(hidden_units, args['num_classes'])

    def forward(self, data):
        x, edge_index = data
        x = F.relu(self.conv1(x, edge_index))
        x1 = F.dropout(x, training=self.training)
        x1 = self.conv2(x1, edge_index)
        x = x + x1
        x = F.log_softmax(self.linear(x))
        return x, edge_index

#######################
class GINConvolution(Module):
    def __init__(self, args, num_features, hidden_units):
        super().__init__()
        self.conv1 = GINConv(num_features, hidden_units)
        self.conv2 = GINConv(hidden_units, args['num_classes'])

    def forward(self, data):
        x, edge_index = data
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return x, edge_index

#######################
class GINConvolutionLin(Module):
    def __init__(self, args, num_features, hidden_units):
        super().__init__()
        self.conv1 = GINConv(num_features, hidden_units)
        self.conv2 = GINConv(hidden_units, hidden_units)
        self.linear = Linear(hidden_units, args['num_classes'])

    def forward(self, data):
        x, edge_index = data
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.log_softmax(self.linear(x))
        return x, edge_index

#######################
class GINConvolutionLinSkip(Module):
    def __init__(self, args, num_features, hidden_units):
        super().__init__()
        self.conv1 = GINConv(num_features, hidden_units)
        self.conv2 = GINConv(hidden_units, hidden_units)
        self.linear = Linear(hidden_units, args['num_classes'])

    def forward(self, data):
        x, edge_index = data
        x = F.relu(self.conv1(x, edge_index))
        x1 = F.dropout(x, training=self.training)
        x1 = self.conv2(x1, edge_index)
        x = x + x1
        x = F.log_softmax(self.linear(x))
        return x, edge_index
