REGISTRY = {}

from .rnn_agent import RNNAgent
REGISTRY["rnn"] = RNNAgent

from .dgn_agent import DGNAgent
REGISTRY["dgn"] = DGNAgent

from .dgn_mlpagent import DGN_MLPAgent
REGISTRY["dgn_mlp"] = DGN_MLPAgent

from .tlc_agent import TLCAgent
REGISTRY["tlc"] = TLCAgent

from .tlc_mlpagent import TLC_MLPAgent
REGISTRY["tlc_mlp"] = TLC_MLPAgent

from .tlc_graphagent import TLC_GraphAgent
REGISTRY["tlc_graph"] = TLC_GraphAgent

from .commnet_agent import COMMNETAgent
REGISTRY["commnet"] = COMMNETAgent

from .tlc_agent_new import TLCAgent_NEW
REGISTRY["tlc_new"] = TLCAgent_NEW

from .adj_agent import AdjAgent
REGISTRY["adj"] = AdjAgent