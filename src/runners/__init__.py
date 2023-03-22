REGISTRY = {}

from .episode_runner import EpisodeRunner
REGISTRY["episode"] = EpisodeRunner

from .parallel_runner import ParallelRunner
REGISTRY["parallel"] = ParallelRunner

from .comm_runner import CommRunner
REGISTRY["comm"] = CommRunner

from .comm_runner import TLCCommRunner
REGISTRY["tlccomm"] = TLCCommRunner