from .q_learner import QLearner
from .coma_learner import COMALearner
from .qtran_learner import QLearner as QTranLearner
from .q_learner_rethink import QLearner_rethink
from .q_learner_RWD import QLearner_RWD

REGISTRY = {}

REGISTRY["q_learner"] = QLearner
REGISTRY["coma_learner"] = COMALearner
REGISTRY["qtran_learner"] = QTranLearner
REGISTRY["q_learner_rethink"] = QLearner_rethink
REGISTRY["q_learner_RWD"] = QLearner_RWD

