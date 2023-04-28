import torch
import torch.nn as nn
import torch.nn.functional as F

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
		k = F.relu(self.fck(x)).permute(0,2,1)
		att = F.softmax(torch.mul(torch.bmm(q,k), mask) - 9e15*(1 - mask),dim=2)
		# att = F.softmax(torch.mul(torch.bmm(q,k), mask),dim=2)
		out = torch.bmm(att,v)

		return out

class AdjAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(AdjAgent, self).__init__()
        self.args = args

        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim*2, args.n_actions)

        # self.genAdj1 = nn.Linear(args.n_agents*args.rnn_hidden_dim, args.n_agents*32)
        # self.genAdj2 = nn.Linear(args.n_agents*32, args.n_agents*args.n_agents)

        self.genAdj1 = nn.Linear(args.rnn_hidden_dim, 32)
        self.genAdj2 = nn.Linear(32, 2*args.n_agents)

        self.att = AttModel(args.rnn_hidden_dim,args.rnn_hidden_dim,args.rnn_hidden_dim)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)

        h = h.reshape(-1,self.args.n_agents,self.args.rnn_hidden_dim)

        adj = torch.bmm(h, h.permute(0,2,1))
        adj = F.relu(adj)
        adj = F.softmax(adj, dim=-1)
        # print(adj)
        h2 = torch.bmm(adj,h)

        q = self.fc2(torch.cat([h,h2],dim=2))
        # q = self.fc2(h2)
        return q, h


    # def forward(self, inputs, hidden_state):
    #     x = F.relu(self.fc1(inputs))
    #     h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
    #     h = self.rnn(x, h_in)

    #     adj = F.relu(self.genAdj1(h))
    #     adj = self.genAdj2(adj)

    #     adj = adj.reshape(-1, self.args.n_agents, self.args.n_agents, 2)
    #     adj = F.gumbel_softmax(adj, hard=True, tau=0.01)
    #     # adj = adj[:,:,:,1].view(-1, self.args.n_agents, self.args.n_agents)
    #     adj = adj[:,:,:,1]
    #     # print(adj)

    #     h = h.reshape(-1,self.args.n_agents,self.args.rnn_hidden_dim)
    #     # h2 = self.att(h, adj)
    #     h2 = torch.bmm(adj,h)
    #     # print(adj)

    #     q = self.fc2(torch.cat([h,h2],dim=2))
    #     return q, h

    # def forward(self, inputs, hidden_state):
    #     x = F.relu(self.fc1(inputs))
    #     h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
    #     h = self.rnn(x, h_in)

    #     x = h.reshape(-1, self.args.n_agents*self.args.rnn_hidden_dim)
    #     x = F.relu(self.genAdj1(x))
    #     adj = self.genAdj2(x)
    #     # adj = torch.add(F.sigmoid(x),1) / 2 # 归一化到（0，1）
    #     adj = adj.reshape(-1, self.args.n_agents, self.args.n_agents)
    #     # adj = F.softmax(adj, dim=2)
    #     adj = F.gumbel_softmax(adj, hard=True, tau=0.01)
    #     h = h.reshape(-1,self.args.n_agents,self.args.rnn_hidden_dim)
    #     h2 = self.att(h, adj)

    #     q = self.fc2(h2)
    #     return q, h

