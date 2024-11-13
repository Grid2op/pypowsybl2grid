# Copyright (c) 2024, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import logging
from typing import Dict, Any, Tuple, Optional

import pandas as pd
import pypowsybl as pp
from pandas import DataFrame

from pypowsybl2grid.network_cache import NetworkCache, NetworkCacheFactory, DEFAULT_LF_PARAMETERS

logger = logging.getLogger(__name__)

class SimpleNetworkCache(NetworkCache):

    def __init__(self, network: pp.network.Network, lf_parameters: pp.loadflow.Parameters):
        super().__init__(network, lf_parameters)
        # TODO provide a way to define retained switches
        self._reset_retained_switches()

    def _reset_retained_switches(self) -> None:
        logger.info("Reset all retained switches")
        switches = self._network.get_switches(attributes=NetworkCache.SWITCH_ATTRIBUTES)
        self._network.update_switches(id=switches.index, retained=[False] * len(switches))

    def get_voltage_levels(self) -> pd.DataFrame:
        voltage_levels = self._network.get_voltage_levels(attributes=NetworkCache.VOLTAGE_LEVEL_ATTRIBUTES)
        voltage_levels['num'] = range(len(voltage_levels))  # add numbering
        return voltage_levels

    def get_buses(self) -> Tuple[pd.DataFrame, Dict[int, str]]:
        buses = self._network.get_bus_breaker_view_buses(attributes=NetworkCache.BUS_ATTRIBUTES)
        # !!!! there is a precise way to create buses global numbering (see grid2op methods to convert from local to global num)
        # global_num = substation_num + local_num * bus_per_substation count
        # So given 2 voltage levels with 2 buses each, local and global number should be:
        #
        # voltage_level_id  bus_id  local_num  global_num
        # VL1               B1      0          0
        # VL1               B2      1          2
        # VL2               B3      0          1
        # VL2               B4      1          3
        voltage_levels = self.get_voltage_levels()
        buses = buses.merge(voltage_levels.rename(columns=lambda x: x + '_voltage_level'), left_on='voltage_level_id', right_index=True, how='outer')
        buses['local_num'] = buses.groupby('voltage_level_id').cumcount()
        buses['num'] = buses['num_voltage_level'] + buses['local_num'] * len(voltage_levels)
        buses_dict = buses['num'].to_dict()
        buses_dict = {v: k for k, v in buses_dict.items()}
        return buses, buses_dict

    def get_injections(self, injections: DataFrame) -> pd.DataFrame:
        injections['num'] = range(len(injections))  # add numbering
        buses = self.get_buses()[0]
        buses_min = buses[['v_mag', 'synchronous_component', 'local_num', 'num']]
        injections_merged_with_buses = injections.merge(
            buses_min.rename(columns=lambda x: x + '_bus'), right_index=True,
            left_on='bus_breaker_bus_id', how='left')
        return injections_merged_with_buses

    def get_loads(self) -> pd.DataFrame:
        return self.get_injections(self._network.get_loads(attributes=NetworkCache.INJECTION_ATTRIBUTES))

    def get_generators(self) -> pd.DataFrame:
        return self.get_injections(self._network.get_generators(attributes=NetworkCache.INJECTION_ATTRIBUTES))

    def get_shunts(self) -> pd.DataFrame:
        return self.get_injections(self._network.get_shunt_compensators(attributes=NetworkCache.INJECTION_ATTRIBUTES))

    def get_batteries(self) -> pd.DataFrame:
        return self.get_injections(self._network.get_batteries(attributes=NetworkCache.INJECTION_ATTRIBUTES))

    def get_branches(self) -> pd.DataFrame:
        lines = self._network.get_lines(attributes=NetworkCache.BRANCH_ATTRIBUTES)
        transformers = self._network.get_2_windings_transformers(attributes=NetworkCache.BRANCH_ATTRIBUTES)
        lines['num'] = range(len(lines))  # add numbering
        transformers['num'] = range(len(lines), len(lines) + len(transformers))  # numbering starting from last line num
        # FIXME support 3 windings transformers
        branches = pd.concat([lines, transformers], axis=0)
        buses = self.get_buses()[0]
        buses_min = buses[['v_mag', 'synchronous_component', 'num']]
        branches_merged_with_buses = (
            branches.merge(buses_min.rename(columns=lambda x: x + '_bus1'), right_index=True, left_on='bus_breaker_bus1_id',
                           how='left')
            .merge(buses_min.rename(columns=lambda x: x + '_bus2'), right_index=True, left_on='bus_breaker_bus2_id',
                   how='left'))
        return branches_merged_with_buses

    def get_branches_with_limits(self) -> pd.DataFrame:
        operational_limits = self._network.get_operational_limits(
            attributes=['element_type', 'type', 'value', 'acceptable_duration'])
        # FIXME also get other limit type
        current_limits = operational_limits[(operational_limits['type'] == 'CURRENT') & (
                operational_limits['acceptable_duration'] == -1)]  # only keep permanent limit
        current_limits = current_limits.groupby('element_id').agg(
            {'value': 'max'}).reset_index()  # get side 1 and 2 max one
        branches = self.get_branches()
        branches_with_limits_a = branches.merge(current_limits, left_index=True, right_on='element_id', how='outer')
        branches_with_limits_a = branches_with_limits_a.fillna(888888)  # replace missing limits by a very high one
        return branches_with_limits_a

    def get_switches(self) -> pd.DataFrame:
        switches = self._network.get_switches(attributes=NetworkCache.SWITCH_ATTRIBUTES)
        return switches[switches['retained']]

    def create_buses(self, df: Optional[DataFrame] = None, **kwargs: Dict[str, Any]) -> None:
        self._network.create_buses(df, **kwargs)

    def disconnect_load(self, iidm_id: str) -> None:
        self._network.update_loads(id=iidm_id, connected=False)

    def connected_load(self, iidm_id: str, new_bus_id: str) -> None:
        self._network.update_loads(id=iidm_id, bus_breaker_bus_id=new_bus_id, connected=True)

    def disconnect_generator(self, iidm_id: str) -> None:
        self._network.update_generators(id=iidm_id, connected=False)

    def connected_generator(self, iidm_id: str, new_bus_id: str) -> None:
        self._network.update_generators(id=iidm_id, bus_breaker_bus_id=new_bus_id, connected=True)

    def disconnect_shunt(self, iidm_id: str) -> None:
        self._network.update_shunt_compensators(id=iidm_id, connected=False)

    def connected_shunt(self, iidm_id: str, new_bus_id: str) -> None:
        self._network.update_shunt_compensators(id=iidm_id, bus_breaker_bus_id=new_bus_id, connected=True)

    def disconnect_branch_side1(self, iidm_id: str) -> None:
        self._network.update_branches(id=iidm_id, connected1=False)

    def connect_branch_side1(self, iidm_id: str, new_bus_id: str) -> None:
        self._network.update_branches(id=iidm_id, bus_breaker_bus1_id=new_bus_id, connected1=True)

    def disconnect_branch_side2(self, iidm_id: str) -> None:
        self._network.update_branches(id=iidm_id, connected2=False)

    def connect_branch_side2(self, iidm_id: str, new_bus_id: str) -> None:
        self._network.update_branches(id=iidm_id, bus_breaker_bus2_id=new_bus_id, connected2=True)

    def update_load_p(self, iidm_id: str, new_p: float) -> None:
        self._network.update_loads(id=iidm_id, p0=new_p)

    def update_load_q(self, iidm_id: str, new_q: float) -> None:
        self._network.update_loads(id=iidm_id, q0=new_q)

    def update_generator_p(self, iidm_id: str, new_p: float) -> None:
        self._network.update_generators(id=iidm_id, target_p=new_p)

    def update_generator_v(self, iidm_id: str, new_v: float) -> None:
        self._network.update_generators(id=iidm_id, target_v=new_v)

    def update_shunt_p(self, iidm_id: str, new_p: float) -> None:
        # FIXME how to deal with discrete shunts?
        pass

    def update_shunt_q(self, iidm_id: str, new_q: float) -> None:
        # FIXME how to deal with discrete shunts?
        pass

class SimpleNetworkCacheFactory(NetworkCacheFactory):

    def create_network_cache(self, network: pp.network.Network, lf_parameters: pp.loadflow.Parameters = DEFAULT_LF_PARAMETERS) -> NetworkCache:
        return SimpleNetworkCache(network, lf_parameters)
