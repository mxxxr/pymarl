REGISTRY = {}

from .episode_runner import EpisodeRunner
REGISTRY["episode"] = EpisodeRunner

from .parallel_runner import ParallelRunner
REGISTRY["parallel"] = ParallelRunner

from .comm_runner import CommRunner
REGISTRY["comm"] = CommRunner

from .tlc_runner import TLCCommRunner
REGISTRY["tlccomm"] = TLCCommRunner

from .tlc_runner import TLC_SCommRunner
REGISTRY["tlcScomm"] = TLC_SCommRunner

from .tlc_runner import TLC_RCommRunner
REGISTRY["tlcRcomm"] = TLC_RCommRunner

from .tlc_runner import TLC_SR0CommRunner
REGISTRY["tlcSR0comm"] = TLC_SR0CommRunner

from .tlc_runner import TLC_SR1CommRunner
REGISTRY["tlcSR1comm"] = TLC_SR1CommRunner