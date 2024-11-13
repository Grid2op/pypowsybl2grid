import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, List, Optional

import pandas as pd
import pypowsybl as pp
from pandas import DataFrame

logger = logging.getLogger(__name__)

DEFAULT_LF_PARAMETERS = pp.loadflow.Parameters(voltage_init_mode=pp.loadflow.VoltageInitMode.DC_VALUES)

class NetworkCache(ABC):
    VOLTAGE_LEVEL_ATTRIBUTES = ['name', 'topology_kind']
    BUS_ATTRIBUTES = ['v_mag', 'synchronous_component', 'voltage_level_id']
    INJECTION_ATTRIBUTES = ['name', 'voltage_level_id', 'bus_breaker_bus_id', 'connected', 'p', 'q']
    BRANCH_ATTRIBUTES = ['name', 'voltage_level1_id', 'voltage_level2_id', 'bus_breaker_bus1_id', 'bus_breaker_bus2_id',
                         'connected1', 'connected2', 'p1', 'q1', 'i1', 'p2', 'q2', 'i2']
    SWITCH_ATTRIBUTES = ['open', 'retained']

    def __init__(self, network: pp.network.Network, lf_parameters: pp.loadflow.Parameters):
        self._network = network
        self._lf_parameters = lf_parameters

    def get_id(self) -> str:
        return self._network.id

    @abstractmethod
    def get_voltage_levels(self) -> pd.DataFrame:
        pass

    @abstractmethod
    def get_buses(self) -> Tuple[pd.DataFrame, Dict[int, str]]:
        pass

    @abstractmethod
    def get_loads(self) -> pd.DataFrame:
        pass

    @abstractmethod
    def get_generators(self) -> pd.DataFrame:
        pass

    @abstractmethod
    def get_shunts(self) -> pd.DataFrame:
        pass

    @abstractmethod
    def get_batteries(self) -> pd.DataFrame:
        pass

    @abstractmethod
    def get_branches(self) -> pd.DataFrame:
        pass

    @abstractmethod
    def get_branches_with_limits(self) -> pd.DataFrame:
        pass

    @abstractmethod
    def get_switches(self) -> pd.DataFrame:
        pass

    def run_dc_pf(self) -> List[pp.loadflow.ComponentResult]:
        return pp.loadflow.run_dc(self._network)

    def run_ac_pf(self) -> List[pp.loadflow.ComponentResult]:
        return pp.loadflow.run_ac(self._network, self._lf_parameters)

    @abstractmethod
    def create_buses(self, df: Optional[DataFrame] = None, **kwargs: Dict[str, Any]) -> None:
        pass

    @abstractmethod
    def disconnect_load(self, iidm_id: str) -> None:
        pass

    @abstractmethod
    def connected_load(self, iidm_id: str, new_bus_id: str) -> None:
        pass

    @abstractmethod
    def disconnect_generator(self, iidm_id: str) -> None:
        pass

    @abstractmethod
    def connected_generator(self, iidm_id: str, new_bus_id: str) -> None:
        pass

    @abstractmethod
    def disconnect_shunt(self, iidm_id: str) -> None:
        pass

    @abstractmethod
    def connected_shunt(self, iidm_id: str, new_bus_id: str) -> None:
        pass

    @abstractmethod
    def disconnect_branch_side1(self, iidm_id: str) -> None:
        pass

    @abstractmethod
    def connect_branch_side1(self, iidm_id: str, new_bus_id: str) -> None:
        pass

    @abstractmethod
    def disconnect_branch_side2(self, iidm_id: str) -> None:
        pass

    @abstractmethod
    def connect_branch_side2(self, iidm_id: str, new_bus_id: str) -> None:
        pass

    @abstractmethod
    def update_load_p(self, iidm_id: str, new_p: float) -> None:
        pass

    @abstractmethod
    def update_load_q(self, iidm_id: str, new_q: float) -> None:
        pass

    @abstractmethod
    def update_generator_p(self, iidm_id: str, new_p: float) -> None:
        pass

    @abstractmethod
    def update_generator_v(self, iidm_id: str, new_v: float) -> None:
        pass

    @abstractmethod
    def update_shunt_p(self, iidm_id: str, new_p: float) -> None:
        pass

    @abstractmethod
    def update_shunt_q(self, iidm_id: str, new_q: float) -> None:
        pass

    def convert_topo_to_bus_breaker(self):
        voltage_levels = self.get_voltage_levels()
        node_breaker_voltage_levels = voltage_levels[voltage_levels['topology_kind'] == 'NODE_BREAKER']
        self._network.update_voltage_levels(id=node_breaker_voltage_levels.index, topology_kind=['BUS_BREAKER'] * len(node_breaker_voltage_levels))

class NetworkCacheFactory(ABC):

    @abstractmethod
    def create_network_cache(self, network: pp.network.Network, lf_parameters: pp.loadflow.Parameters = DEFAULT_LF_PARAMETERS) -> NetworkCache:
        pass
