# Copyright (c) 2024, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

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
    def get_load_ids(self) -> List[str]:
        pass

    @abstractmethod
    def get_generator_ids(self) -> List[str]:
        pass

    @abstractmethod
    def get_shunt_ids(self) -> List[str]:
        pass

    @abstractmethod
    def get_branch_ids(self) -> List[str]:
        pass

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
    def connect_load(self, iidm_id: List[str], connected: List[bool], new_bus_id: List[str]) -> None:
        pass

    @abstractmethod
    def connect_generator(self, iidm_id: List[str], connected: List[bool], new_bus_id: List[str]) -> None:
        pass

    @abstractmethod
    def connect_shunt(self, iidm_id: List[str], connected: List[bool], new_bus_id: List[str]) -> None:
        pass

    @abstractmethod
    def connect_branch_side1(self, iidm_id: List[str], connected: List[bool], new_bus_id: List[str]) -> None:
        pass

    @abstractmethod
    def connect_branch_side2(self, iidm_id: List[str], connected: List[bool], new_bus_id: List[str]) -> None:
        pass

    @abstractmethod
    def update_load_p(self, iidm_id: List[str], new_p: List[float]) -> None:
        pass

    @abstractmethod
    def update_load_q(self, iidm_id: List[str], new_q: List[float]) -> None:
        pass

    @abstractmethod
    def update_generator_p(self, iidm_id: List[str], new_p: List[float]) -> None:
        pass

    @abstractmethod
    def update_generator_v(self, iidm_id: List[str], new_v: List[float]) -> None:
        pass

    @abstractmethod
    def update_shunt_p(self, iidm_id: List[str], new_p: List[float]) -> None:
        pass

    @abstractmethod
    def update_shunt_q(self, iidm_id: List[str], new_q: List[float]) -> None:
        pass

    def convert_topo_to_bus_breaker(self):
        voltage_levels = self.get_voltage_levels()
        node_breaker_voltage_levels = voltage_levels[voltage_levels['topology_kind'] == 'NODE_BREAKER']
        self._network.update_voltage_levels(id=node_breaker_voltage_levels.index, topology_kind=['BUS_BREAKER'] * len(node_breaker_voltage_levels))

    @staticmethod
    def _injection_new_bus_breaker_bus_id(injections: DataFrame, first_bus_by_voltage_level: DataFrame) -> List[str]:
        return injections.merge(first_bus_by_voltage_level, on='voltage_level_id', how='left', suffixes=('', '_new'))['bus_breaker_bus_id_new'].to_list()

    @staticmethod
    def _branches_new_bus_breaker_bus_id(branches: DataFrame, first_bus_by_voltage_level: DataFrame) -> Tuple[List[str], List[str]]:
        branches_tmp = branches.merge(first_bus_by_voltage_level, left_on='voltage_level1_id', right_on='voltage_level_id', how='left', suffixes=('', '_new1'))
        branches_tmp = branches_tmp.merge(first_bus_by_voltage_level, left_on='voltage_level2_id', right_on='voltage_level_id', how='left', suffixes=('', '_new2'))
        return branches_tmp['bus_breaker_bus_id'].to_list(), branches_tmp['bus_breaker_bus_id_new2'].to_list()

    def connect_all_elements_to_first_bus(self):
        buses = self.get_buses()[0]
        first_bus_by_voltage_level = buses.drop_duplicates(subset=['voltage_level_id'], keep='first')
        first_bus_by_voltage_level = first_bus_by_voltage_level[['voltage_level_id']]
        first_bus_by_voltage_level.reset_index(inplace=True)
        first_bus_by_voltage_level = first_bus_by_voltage_level.set_index('voltage_level_id')
        first_bus_by_voltage_level.rename(columns={"id": "bus_breaker_bus_id"}, inplace=True)

        loads = self.get_loads()
        load_new_bus_breaker_bus_id = self._injection_new_bus_breaker_bus_id(loads, first_bus_by_voltage_level)
        self._network.update_loads(id=loads.index, bus_breaker_bus_id=load_new_bus_breaker_bus_id)

        gens = self.get_generators()
        gen_new_bus_breaker_bus_id = self._injection_new_bus_breaker_bus_id(gens, first_bus_by_voltage_level)
        self._network.update_generators(id=gens.index, bus_breaker_bus_id=gen_new_bus_breaker_bus_id)

        shunts = self.get_shunts()
        shunts_new_bus_breaker_bus_id = self._injection_new_bus_breaker_bus_id(shunts, first_bus_by_voltage_level)
        self._network.update_shunt_compensators(id=shunts.index, bus_breaker_bus_id=shunts_new_bus_breaker_bus_id)

        branches = self.get_branches()
        branches_new_bus_breaker_bus1_id, branches_new_bus_breaker_bus2_id = self._branches_new_bus_breaker_bus_id(branches, first_bus_by_voltage_level)
        self._network.update_branches(id=branches.index, bus_breaker_bus1_id=branches_new_bus_breaker_bus1_id, bus_breaker_bus2_id=branches_new_bus_breaker_bus2_id)


class NetworkCacheFactory(ABC):

    @abstractmethod
    def create_network_cache(self, network: pp.network.Network, lf_parameters: pp.loadflow.Parameters = DEFAULT_LF_PARAMETERS) -> NetworkCache:
        pass
