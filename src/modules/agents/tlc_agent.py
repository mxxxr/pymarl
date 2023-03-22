import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
	def __init__(self, din=275, hidden_dim=64):
		super(Encoder, self).__init__()
		self.fc = nn.Linear(din, hidden_dim)

	def forward(self, x):
		embedding = F.relu(self.fc(x))
		return embedding

class AttModel(nn.Module):
	def __init__(self, din, hidden_dim, dout):
		super(AttModel, self).__init__()
		self.fcv = nn.Linear(din, hidden_dim)
		self.fck = nn.Linear(din, hidden_dim)
		self.fcq = nn.Linear(din, hidden_dim)
		self.fcout = nn.Linear(hidden_dim, dout)

	def forward(self, x, mask):
		v = F.relu(self.fcv(x))
		q = F.relu(self.fcq(x))
		# if len(x.shape) == 2:
		# 	mask = mask[0]
		# 	print(q.shape,v.shape,mask.shape)
		# 	k = F.relu(self.fck(x)).permute(1,0)
		# 	att = F.softmax(torch.mul(torch.mm(q,k), mask) - 9e15*(1 - mask),dim=1)
		# 	out = torch.mm(att,v)

		# else:
		k = F.relu(self.fck(x)).permute(0,2,1)
		att = F.softmax(torch.mul(torch.bmm(q,k), mask) - 9e15*(1 - mask),dim=2)
		out = torch.bmm(att,v)

		return out

class Q_Net(nn.Module):
	def __init__(self, hidden_dim, dout):
		super(Q_Net, self).__init__()
		self.fc = nn.Linear(hidden_dim, dout)

	def forward(self, x):
		q = self.fc(x)
		return q


class TLCAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(TLCAgent, self).__init__()
        self.args = args

        self.encoder = Encoder(input_shape,args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.att_1 = AttModel(args.rnn_hidden_dim,args.rnn_hidden_dim,args.rnn_hidden_dim)
        self.att_2 = AttModel(args.rnn_hidden_dim,args.rnn_hidden_dim,args.rnn_hidden_dim)
        self.q_net = Q_Net(args.rnn_hidden_dim, args.n_actions)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.encoder.fc.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state, adj):
        h1 = self.encoder(inputs)
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h2 = self.rnn(h1, h_in)
        h_dim = h2.shape[-1]
        h2 = h2.reshape(-1,self.args.n_agents,h_dim)
        h3 = self.att_1(h2, adj)
        h4 = self.att_2(h3, adj)
        q = self.q_net(h4)
        return q, h2
