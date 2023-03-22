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