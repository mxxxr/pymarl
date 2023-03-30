REGISTRY = {}

from .basic_controller import BasicMAC
REGISTRY["basic_mac"] = BasicMAC

from .comm_controller import CommMAC
REGISTRY["comm_mac"] = CommMAC

from .iqlS_controller import iqlSMAC
REGISTRY["iqlS_mac"] = iqlSMAC

from .iqlS_controller import iqlS2MAC
REGISTRY["iqlS2_mac"] = iqlS2MAC