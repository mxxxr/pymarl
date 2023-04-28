import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.utils import dense_to_sparse

class GCN(torch.nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.conv1 = GCNConv(hidden_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return x

class Encoder(nn.Module):
	def __init__(self, din=275, hidden_dim=64):
		super(Encoder, self).__init__()
		self.fc = nn.Linear(din, hidden_dim)

	def forward(self, x):
		embedding = F.relu(self.fc(x))
		return embedding

class Q_Net(nn.Module):
	def __init__(self, hidden_dim, dout):
		super(Q_Net, self).__init__()
		self.fc = nn.Linear(hidden_dim, dout)

	def forward(self, x):
		q = self.fc(x)
		return q


class TLC_GraphAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(TLC_GraphAgent, self).__init__()
        self.args = args

        self.encoder = Encoder(input_shape,args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.gcn = GCN(args.rnn_hidden_dim)
        self.q_net = Q_Net(args.rnn_hidden_dim, args.n_actions)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.encoder.fc.weight.new(1, self.args.rnn_hidden_dim).zero_()
    
    def genGraphData(self, h2, adj):
        edge_index, x_ = dense_to_sparse(adj)
        x = h2
        return x, edge_index

    def forward(self, inputs, hidden_state, adj):
        h1 = self.encoder(inputs)
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h2 = self.rnn(h1, h_in)
        # h_dim = h2.shape[-1]
        # h2 = h2.reshape(-1,self.args.n_agents,h_dim)
        x, edge_index = self.genGraphData(h2, adj)
        h3 = self.gcn(x, edge_index)
        q = self.q_net(h3)
        return q, h2
