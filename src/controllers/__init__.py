REGISTRY = {}

from .basic_controller import BasicMAC
REGISTRY["basic_mac"] = BasicMAC

from .comm_controller import CommMAC
REGISTRY["comm_mac"] = CommMAC
