# Copyright (c) 2024, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import logging
from typing import Dict, Any, List, Tuple, Optional

import pandas as pd
import pypowsybl as pp
from pandas import DataFrame

from pypowsybl2grid.network_cache import NetworkCache, NetworkCacheFactory, DEFAULT_LF_PARAMETERS

logger = logging.getLogger(__name__)


class FastNetworkCache(NetworkCache):
    BUS_STATE_ATTRIBUTES = ['v_mag']
    INJECTION_STATE_ATTRIBUTES = ['p', 'q']
    BRANCH_STATE_ATTRIBUTES = ['p1', 'q1', 'i1', 'p2', 'q2', 'i2']
    BUS_TOPO_ATTRIBUTES = ['synchronous_component']
    MERGE_BUS_TOPO_ATTRIBUTES = ['synchronous_component', 'num']
    INJECTION_TOPO_ATTRIBUTES = ['bus_breaker_bus_id', 'connected']
    BRANCH_TOPO_ATTRIBUTES = ['bus_breaker_bus1_id', 'connected1', 'bus_breaker_bus2_id', 'connected2']

    def __init__(self, network: pp.network.Network, lf_parameters: pp.loadflow.Parameters):
        super().__init__(network, lf_parameters)
        # everything is up-to-date
        self._loads_topo_to_update = []
        self._generators_topo_to_update = []
        self._shunts_topo_to_update = []
        self._branches_topo_to_update = []
        self._fetch_switches()
        # TODO provide a way to define retained switches
        self._reset_retained_switches()
        self._fetch_voltage_levels()
        self._fetch_buses()
        self._fetch_loads()
        self._fetch_generators()
        self._fetch_shunts()
        self._fetch_batteries()
        self._fetch_branches()
        self._fetch_branches_limits()

    def _fetch_switches(self) -> None:
        self._switches = self._network.get_switches(attributes=NetworkCache.SWITCH_ATTRIBUTES)

    def _fetch_voltage_levels(self) -> None:
        self._voltage_levels = self._network.get_voltage_levels(attributes=NetworkCache.VOLTAGE_LEVEL_ATTRIBUTES)
        self._voltage_levels['num'] = range(len(self._voltage_levels))  # add numbering

    def _number_buses(self, buses: DataFrame) -> DataFrame:
        numbered_buses = buses.merge(self._voltage_levels.rename(columns=lambda x: x + '_voltage_level'),
                                        left_on='voltage_level_id', right_index=True, how='outer')
        numbered_buses['local_num'] = numbered_buses.groupby('voltage_level_id').cumcount()
        numbered_buses['num'] = numbered_buses['num_voltage_level'] + numbered_buses['local_num'] * len(self._voltage_levels)
        return numbered_buses

    def _fetch_buses(self) -> None:
        self._buses = self._network.get_bus_breaker_view_buses(attributes=NetworkCache.BUS_ATTRIBUTES)
        # !!!! there is a precise way to create buses global numbering (see grid2op methods to convert from local to global num)
        # global_num = substation_num + local_num * bus_per_substation count
        # So given 2 voltage levels with 2 buses each, local and global number should be:
        #
        # voltage_level_id  bus_id  local_num  global_num
        # VL1               B1      0          0
        # VL1               B2      1          2
        # VL2               B3      0          1
        # VL2               B4      1          3
        self._buses = self._number_buses(self._buses)
        self._buses_dict = self._buses['num'].to_dict()
        self._buses_dict = {v: k for k, v in self._buses_dict.items()}

    def _fetch_injections(self, injections: DataFrame) -> DataFrame:
        injections['num'] = range(len(injections))  # add numbering
        buses_min = self._buses[['v_mag', 'synchronous_component', 'local_num', 'num']]
        return injections.merge(
            buses_min.rename(columns=lambda x: x + '_bus'), right_index=True,
            left_on='bus_breaker_bus_id', how='left')

    def _fetch_loads(self) -> None:
        self._loads = self._fetch_injections(self._network.get_loads(attributes=NetworkCache.INJECTION_ATTRIBUTES))

    def _fetch_generators(self) -> None:
        self._generators = self._fetch_injections(
            self._network.get_generators(attributes=NetworkCache.INJECTION_ATTRIBUTES))

    def _fetch_shunts(self) -> None:
        self._shunts = self._fetch_injections(
            self._network.get_shunt_compensators(attributes=NetworkCache.INJECTION_ATTRIBUTES))

    def _fetch_batteries(self) -> None:
        self._batteries = self._fetch_injections(
            self._network.get_batteries(attributes=NetworkCache.INJECTION_ATTRIBUTES))

    def _fetch_branches(self) -> None:
        lines = self._network.get_lines(attributes=NetworkCache.BRANCH_ATTRIBUTES)
        transformers = self._network.get_2_windings_transformers(attributes=NetworkCache.BRANCH_ATTRIBUTES)
        lines['num'] = range(len(lines))  # add numbering
        transformers['num'] = range(len(lines), len(lines) + len(transformers))  # numbering starting from last line num
        # FIXME support 3 windings transformers
        branches = pd.concat([lines, transformers], axis=0)
        buses_min = self._buses[['v_mag', 'synchronous_component', 'num']]
        self._branches = (
            branches.merge(buses_min.rename(columns=lambda x: x + '_bus1'), right_index=True,
                           left_on='bus_breaker_bus1_id',
                           how='left')
            .merge(buses_min.rename(columns=lambda x: x + '_bus2'), right_index=True, left_on='bus_breaker_bus2_id',
                   how='left'))

    def _fetch_branches_limits(self) -> None:
        operational_limits = self._network.get_operational_limits(
            attributes=['element_type', 'type', 'value', 'acceptable_duration'])
        # FIXME also get other limit type
        current_limits = operational_limits[(operational_limits['type'] == 'CURRENT') & (
                operational_limits['acceptable_duration'] == -1)]  # only keep permanent limit
        current_limits = current_limits.groupby('element_id').agg(
            {'value': 'max'}).reset_index()  # get side 1 and 2 max one
        self._branches_limits = self._branches.merge(current_limits, left_index=True, right_on='element_id',
                                                     how='outer')
        self._branches_limits = self._branches_limits.fillna(888888)  # replace missing limits by a very high one

    @staticmethod
    def _update(initial_df: DataFrame, update_df: DataFrame) -> None:
        # is this really efficient ?
        # update and combine_first does not work as it ignore nan and none of the updated dataframe
        for col in update_df.columns:
            initial_df.loc[update_df.index, col] = update_df[col]

    @staticmethod
    def _fetch_injections_state(injections: DataFrame, buses_state: DataFrame, injections_state: DataFrame) -> None:
        # update columns coming from bus state dataframe join
        injections_merged_with_buses_state = injections[['bus_breaker_bus_id']].merge(
            buses_state.rename(columns=lambda x: x + '_bus'), right_index=True,
            left_on='bus_breaker_bus_id', how='left')
        FastNetworkCache._update(injections, injections_merged_with_buses_state)

        # update injection state columns
        FastNetworkCache._update(injections, injections_state)

    def _fetch_full_topo(self) -> None:
        self._fetch_branch_topo()
        self._fetch_load_topo()
        self._fetch_generator_topo()
        self._fetch_shunt_topo()

    def _fetch_full_state(self) -> None:
        self._fetch_full_topo()

        # update buses state columns
        buses_state = self._network.get_bus_breaker_view_buses(attributes=FastNetworkCache.BUS_STATE_ATTRIBUTES)
        self._buses.update(buses_state)

        # update injections state
        self._fetch_injections_state(self._loads, buses_state,
                                     self._network.get_loads(attributes=FastNetworkCache.INJECTION_STATE_ATTRIBUTES))
        self._fetch_injections_state(self._generators, buses_state, self._network.get_generators(
            attributes=FastNetworkCache.INJECTION_STATE_ATTRIBUTES))
        self._fetch_injections_state(self._shunts, buses_state, self._network.get_shunt_compensators(
            attributes=FastNetworkCache.INJECTION_STATE_ATTRIBUTES))
        self._fetch_injections_state(self._batteries, buses_state,
                                     self._network.get_batteries(
                                         attributes=FastNetworkCache.INJECTION_STATE_ATTRIBUTES))

        # update columns coming from bus state dataframe joins
        branches_merged_with_buses_state = self._branches[['bus_breaker_bus1_id', 'bus_breaker_bus2_id']].merge(
            buses_state.rename(columns=lambda x: x + '_bus1'), right_index=True,
            left_on='bus_breaker_bus1_id', how='left').merge(
            buses_state.rename(columns=lambda x: x + '_bus2'), right_index=True,
            left_on='bus_breaker_bus2_id', how='left')
        FastNetworkCache._update(self._branches, branches_merged_with_buses_state)

        # update branches state columns
        # FIXME support 3 windings transformers
        branches_state = self._network.get_branches(attributes=FastNetworkCache.BRANCH_STATE_ATTRIBUTES)
        FastNetworkCache._update(self._branches, branches_state)

    def _reset_retained_switches(self) -> None:
        logger.info("Reset all retained switches")
        self._network.update_switches(id=self._switches.index, retained=[False] * len(self._switches))

    def get_load_ids(self) -> List[str]:
        return self._loads.index.tolist()

    def get_generator_ids(self) -> List[str]:
        return self._generators.index.tolist()

    def get_shunt_ids(self) -> List[str]:
        return self._shunts.index.tolist()

    def get_branch_ids(self) -> List[str]:
        return self._branches.index.tolist()

    def get_voltage_levels(self) -> pd.DataFrame:
        return self._voltage_levels

    def get_buses(self) -> Tuple[pd.DataFrame, Dict[int, str]]:
        self._fetch_bus_topo()
        return self._buses, self._buses_dict

    def get_loads(self) -> pd.DataFrame:
        self._fetch_branch_topo() # because a branch connection change can cause an injection connection change
        self._fetch_load_topo()
        return self._loads

    def get_generators(self) -> pd.DataFrame:
        self._fetch_branch_topo() # because a branch connection change can cause an injection connection change
        self._fetch_generator_topo()
        return self._generators

    def get_shunts(self) -> pd.DataFrame:
        self._fetch_branch_topo() # because a branch connection change can cause an injection connection change
        self._fetch_shunt_topo()
        return self._shunts

    def get_batteries(self) -> pd.DataFrame:
        return self._batteries

    def get_branches(self) -> pd.DataFrame:
        self._fetch_branch_topo()
        return self._branches

    def get_branches_with_limits(self) -> pd.DataFrame:
        return self._branches_limits

    def get_switches(self) -> pd.DataFrame:
        switches = self._network.get_switches(attributes=NetworkCache.SWITCH_ATTRIBUTES)
        return switches[switches['retained']]

    def run_dc_pf(self) -> List[pp.loadflow.ComponentResult]:
        result = super().run_dc_pf()
        self._fetch_full_state()
        return result

    def run_ac_pf(self) -> List[pp.loadflow.ComponentResult]:
        result = super().run_ac_pf()
        self._fetch_full_state()
        return result

    def create_buses(self, df: Optional[DataFrame] = None, **kwargs: Dict[str, Any]) -> None:
        self._network.create_buses(df, **kwargs)
        self._fetch_buses()

    def _fetch_injection_topo(self, injections: DataFrame, injections_topo: DataFrame, buses_topo: DataFrame) -> None:
        FastNetworkCache._update(injections, injections_topo)

        # update columns coming from bus topo dataframe join
        injections_merged_with_buses_topo = injections[['bus_breaker_bus_id']].merge(
            buses_topo.rename(columns=lambda x: x + '_bus'), right_index=True,
            left_on='bus_breaker_bus_id', how='left')
        FastNetworkCache._update(injections, injections_merged_with_buses_topo)

    def _fetch_load_topo(self) -> None:
        if len(self._loads_topo_to_update) == 0:
            return
        self._fetch_bus_topo()
        buses_topo = self._buses[FastNetworkCache.MERGE_BUS_TOPO_ATTRIBUTES]
        self._fetch_injection_topo(self._loads, self._network.get_loads(id=self._loads_topo_to_update,
                                                                        attributes=FastNetworkCache.INJECTION_TOPO_ATTRIBUTES),
                                   buses_topo)
        self._loads_topo_to_update.clear()

    def _invalidate_loads_topo(self, iidm_id: List[str]) -> None:
        self._loads_topo_to_update.extend(iidm_id)

    def connect_load(self, iidm_id: List[str], connected: List[bool], new_bus_id: List[str]) -> None:
        self._network.update_loads(id=iidm_id, bus_breaker_bus_id=new_bus_id, connected=connected)
        self._invalidate_loads_topo(iidm_id)

    def _fetch_generator_topo(self) -> None:
        if len(self._generators_topo_to_update) == 0:
            return
        self._fetch_bus_topo()
        buses_topo = self._buses[FastNetworkCache.MERGE_BUS_TOPO_ATTRIBUTES]
        self._fetch_injection_topo(self._generators, self._network.get_generators(id=self._generators_topo_to_update,
                                                                                  attributes=FastNetworkCache.INJECTION_TOPO_ATTRIBUTES),
                                   buses_topo)
        self._generators_topo_to_update.clear()

    def _invalidate_generators_topo(self, iidm_id: List[str]) -> None:
        self._generators_topo_to_update.extend(iidm_id)

    def connect_generator(self, iidm_id: List[str], connected: List[bool], new_bus_id: List[str]) -> None:
        self._network.update_generators(id=iidm_id, bus_breaker_bus_id=new_bus_id, connected=connected)
        self._invalidate_generators_topo(iidm_id)

    def _fetch_shunt_topo(self) -> None:
        if len(self._shunts_topo_to_update) == 0:
            return
        self._fetch_bus_topo()
        buses_topo = self._buses[FastNetworkCache.MERGE_BUS_TOPO_ATTRIBUTES]
        self._fetch_injection_topo(self._shunts, self._network.get_shunt_compensators(id=self._shunts_topo_to_update,
                                                                                      attributes=FastNetworkCache.INJECTION_TOPO_ATTRIBUTES),
                                   buses_topo)
        self._shunts_topo_to_update.clear()

    def _invalidate_shunts_topo(self, iidm_id: List[str]) -> None:
        self._shunts_topo_to_update.extend(iidm_id)

    def connect_shunt(self, iidm_id: List[str], connected: List[bool], new_bus_id: List[str]) -> None:
        self._network.update_shunt_compensators(id=iidm_id, bus_breaker_bus_id=new_bus_id, connected=connected)
        self._invalidate_shunts_topo(iidm_id)

    def _fetch_bus_topo(self) -> None:
        buses_topo_update = self._network.get_bus_breaker_view_buses(attributes=FastNetworkCache.BUS_TOPO_ATTRIBUTES)
        FastNetworkCache._update(self._buses, buses_topo_update)

    def _fetch_branch_topo(self) -> None:
        if len(self._branches_topo_to_update) == 0:
            return
        # update buses because of potential synchronous component update
        self._fetch_bus_topo()
        buses_topo = self._buses[FastNetworkCache.MERGE_BUS_TOPO_ATTRIBUTES]

        # we need to update connectivity of all branches and injections
        # (potentially indirectly lost by the branch topo change)
        branches_topo = self._network.get_branches(attributes=FastNetworkCache.BRANCH_TOPO_ATTRIBUTES)
        branches_merged_with_buses_topo = branches_topo.merge(
            buses_topo.rename(columns=lambda x: x + '_bus1'), right_index=True,
            left_on='bus_breaker_bus1_id', how='left')
        branches_merged_with_buses_topo = branches_merged_with_buses_topo.merge(
            buses_topo.rename(columns=lambda x: x + '_bus2'), right_index=True,
            left_on='bus_breaker_bus2_id', how='left')
        FastNetworkCache._update(self._branches, branches_merged_with_buses_topo)

        self._fetch_injection_topo(self._loads,
                                   self._network.get_loads(attributes=FastNetworkCache.INJECTION_TOPO_ATTRIBUTES),
                                   buses_topo)
        self._fetch_injection_topo(self._generators,
                                   self._network.get_generators(attributes=FastNetworkCache.INJECTION_TOPO_ATTRIBUTES),
                                   buses_topo)
        self._fetch_injection_topo(self._shunts,
                                   self._network.get_shunt_compensators(attributes=FastNetworkCache.INJECTION_TOPO_ATTRIBUTES),
                                   buses_topo)
        self._branches_topo_to_update.clear()
        self._loads_topo_to_update.clear()
        self._generators_topo_to_update.clear()
        self._shunts_topo_to_update.clear()

    def _invalidate_branches_topo(self, iidm_id: List[str]) -> None:
        self._branches_topo_to_update.extend(iidm_id)

    def connect_branch_side1(self, iidm_id: List[str], connected: List[bool], new_bus_id: List[str]) -> None:
        self._network.update_branches(id=iidm_id, bus_breaker_bus1_id=new_bus_id, connected1=connected)
        self._invalidate_branches_topo(iidm_id)

    def connect_branch_side2(self, iidm_id: List[str], connected: List[bool], new_bus_id: List[str]) -> None:
        self._network.update_branches(id=iidm_id, bus_breaker_bus2_id=new_bus_id, connected2=connected)
        self._invalidate_branches_topo(iidm_id)

    def update_load_p(self, iidm_id: List[str], new_p: List[float]) -> None:
        self._network.update_loads(id=iidm_id, p0=new_p)
        # not need to update until LF ran again

    def update_load_q(self, iidm_id: List[str], new_q: List[float]) -> None:
        self._network.update_loads(id=iidm_id, q0=new_q)
        # not need to update until LF ran again

    def update_generator_p(self, iidm_id: List[str], new_p: List[float]) -> None:
        self._network.update_generators(id=iidm_id, target_p=new_p)
        # not need to update until LF ran again

    def update_generator_v(self, iidm_id: List[str], new_v: List[float]) -> None:
        self._network.update_generators(id=iidm_id, target_v=new_v)
        # not need to update until LF ran again

    def update_shunt_p(self, iidm_id: List[str], new_p: List[float]) -> None:
        # FIXME how to deal with discrete shunts?
        pass

    def update_shunt_q(self, iidm_id: List[str], new_q: List[float]) -> None:
        # FIXME how to deal with discrete shunts?
        pass


class FastNetworkCacheFactory(NetworkCacheFactory):

    def create_network_cache(self, network: pp.network.Network, lf_parameters: pp.loadflow.Parameters = DEFAULT_LF_PARAMETERS) -> NetworkCache:
        return FastNetworkCache(network, lf_parameters)
