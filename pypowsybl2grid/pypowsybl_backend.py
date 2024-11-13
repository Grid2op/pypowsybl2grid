# Copyright (c) 2024, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import logging
import os
from typing import Optional, Tuple, Union

import grid2op
import numpy as np
import pandapower as pdp
import pandas as pd
import pypowsybl as pp
from grid2op.Backend import Backend
from grid2op.Exceptions import DivergingPowerflow, BackendError
from grid2op.Space import DEFAULT_N_BUSBAR_PER_SUB
from pandas import DataFrame

from pypowsybl2grid.fast_network_cache import FastNetworkCache
from pypowsybl2grid.network_cache import DEFAULT_LF_PARAMETERS

logger = logging.getLogger(__name__)

class PyPowSyBlBackend(Backend):

    def __init__(self,
                 detailed_infos_for_cascading_failures=False,
                 can_be_copied: bool = True,
                 check_isolated_and_disconnected_injections = True,
                 consider_open_branch_reactive_flow = False,
                 n_busbar_per_sub = DEFAULT_N_BUSBAR_PER_SUB,
                 lf_parameters: pp.loadflow.Parameters = DEFAULT_LF_PARAMETERS):
        Backend.__init__(self,
                         detailed_infos_for_cascading_failures=detailed_infos_for_cascading_failures,
                         can_be_copied=can_be_copied)
        self._check_isolated_and_disconnected_injections = check_isolated_and_disconnected_injections
        self._consider_open_branch_reactive_flow = consider_open_branch_reactive_flow
        self.n_busbar_per_sub = n_busbar_per_sub
        self._lf_parameters = lf_parameters

        self.shunts_data_available = True
        self.supported_grid_format = ("json", "xiidm", "txt")  # FIXME dynamically get supported extensions

    @staticmethod
    def create_name(df: DataFrame) -> np.ndarray:
        return np.where(df['name'].eq(''), df.index.to_series(), df['name'])

    def load_grid(self,
                  path: Union[os.PathLike, str],
                  filename: Optional[Union[os.PathLike, str]] = None) -> None:
        # load network
        full_path = self.make_complete_path(path, filename)

        if full_path.endswith('.json'):
            n_pdp = pdp.from_json(full_path)
            n = pp.network.convert_from_pandapower(n_pdp)
        else:
            n = pp.network.load(full_path)
        self._network = FastNetworkCache(n, self._lf_parameters)

        # substations mapped to IIDM voltage levels
        voltage_levels = self._network.get_voltage_levels()
        self.n_sub = len(voltage_levels)
        self.name_sub = self.create_name(voltage_levels)

        # FIXME waiting for being able to use switch actions with the backend in case of node/breaker voltage levels
        # for now we convert all voltage level to bus/breaker ones
        self._network.convert_topo_to_bus_breaker()

        # only one value for n_busbar_per_sub is allowed => use maximum one across all voltage levels
        buses, _ = self._network.get_buses()

        max_bus_count = int(buses['local_num'].max()) + 1
        if self.n_busbar_per_sub is None:
            if max_bus_count < 1:
                raise BackendError("Network does not have any bus, it is impossible to define a n_busbar_per_sub")
            self.n_busbar_per_sub = max_bus_count
        else:
            if self.n_busbar_per_sub < max_bus_count:
                raise BackendError(f"n_busbar_per_sub ({self.n_busbar_per_sub}) is lower than maximum number of bus ({max_bus_count}) of the network with the current topology")

        # create additional buses so that each voltage level to reach max_bus_count
        bus_count_by_voltage_level = buses.groupby('voltage_level_id')['local_num'].max().reset_index()
        for _, row in bus_count_by_voltage_level.iterrows():
            voltage_level_id = row['voltage_level_id']
            bus_count = row['local_num'] + 1
            bus_nums_to_create = range(bus_count, self.n_busbar_per_sub)
            bus_ids = [f"{voltage_level_id}_extra_busbar_{i}" for i in bus_nums_to_create]
            voltage_level_ids = [voltage_level_id] * len(bus_nums_to_create)
            self._network.create_buses(id=bus_ids, voltage_level_id=voltage_level_ids)

        self.can_handle_more_than_2_busbar()

        logger.info(f"{self.n_busbar_per_sub} busbars per substation")

        # loads
        loads = self._network.get_loads()
        self.n_load = len(loads)
        self.name_load = self.create_name(loads)
        self.load_to_subid = np.zeros(self.n_load, dtype=int)
        for _, row in loads.iterrows():
            self.load_to_subid[row.num] = voltage_levels.loc[row.voltage_level_id, "num"]

        # generators
        generators = self._network.get_generators()
        self.n_gen = len(generators)
        self.name_gen = self.create_name(generators)
        self.gen_to_subid = np.zeros(self.n_gen, dtype=int)
        for _, row in generators.iterrows():
            self.gen_to_subid[row.num] = voltage_levels.loc[row.voltage_level_id, "num"]

        # shunts
        shunts = self._network.get_shunts()
        self.n_shunt = len(shunts)
        self.name_shunt = self.create_name(shunts)
        self.shunt_to_subid = np.zeros(self.n_shunt, dtype=int)
        for _, row in shunts.iterrows():
            self.shunt_to_subid[row.num] = voltage_levels.loc[row.voltage_level_id, "num"]

        # batteries
        self.set_no_storage()
        # FIXME implement batteries
        # batteries = self._network.get_batteries()
        # self.n_storage = len(batteries)
        # self.name_storage = np.array(batteries.index)
        # self.storage_type = np.full(self.n_storage, fill_value="???")
        # self.storage_to_subid = np.zeros(self.n_storage, dtype=int)
        # for index, row in batteries.iterrows():
        #     self.storage_to_subid[row.num] = voltage_levels.loc[row.voltage_level_id, "num"]

        # lines and transformers
        branches = self._network.get_branches()
        self.n_line = len(branches)
        self.name_line = self.create_name(branches)
        self.line_or_to_subid = np.zeros(self.n_line, dtype=int)
        self.line_ex_to_subid = np.zeros(self.n_line, dtype=int)
        for _, row in branches.iterrows():
            self.line_or_to_subid[row.num] = voltage_levels.loc[row.voltage_level1_id, "num"]
            self.line_ex_to_subid[row.num] = voltage_levels.loc[row.voltage_level2_id, "num"]

        self._compute_pos_big_topo()

        # thermal limits
        self.thermal_limit_a = np.zeros(self.n_line, dtype=int)
        for _, row in self._network.get_branches_with_limits().iterrows():
            self.thermal_limit_a[row.num] = row.value

        switches = self._network.get_switches()
        logger.info(f"Network '{self._network.get_id()}' loaded with {len(switches)} retained switches: {len(buses)} buses, {len(branches)} branches, {len(generators)} generators, {len(loads)} loads, {len(shunts)} shunts")

    def apply_action(self, backend_action: Union["grid2op.Action._backendAction._BackendAction", None]) -> None:
        # the following few lines are highly recommended
        if backend_action is None:
            return

        logger.info("Apply action")

        # active and reactive power of loads
        loads = self._network.get_loads()
        for load_id, new_p in backend_action.load_p:
            iidm_id = str(loads.iloc[load_id].name)
            self._network.update_load_p(iidm_id, new_p)

        for load_id, new_q in backend_action.load_q:
            iidm_id = str(loads.iloc[load_id].name)
            self._network.update_load_q(iidm_id, new_q)

        # active power and voltage target of generators
        generators = self._network.get_generators()
        for gen_id, new_p in backend_action.prod_p:
            iidm_id = str(generators.iloc[gen_id].name)
            self._network.update_generator_p(iidm_id, new_p)

        for gen_id, new_v in backend_action.prod_v:
            iidm_id = str(generators.iloc[gen_id].name)
            self._network.update_generator_v(iidm_id, new_v)

        # active and reactive power of shunts
        shunts = self._network.get_shunts()
        for shunt_id, new_p in backend_action.shunt_p:
            iidm_id = str(shunts.iloc[shunt_id].name)
            self._network.update_shunt_p(iidm_id, new_p)

        for shunt_id, new_q in backend_action.shunt_q:
            iidm_id = str(shunts.iloc[shunt_id].name)
            self._network.update_shunt_q(iidm_id, new_q)

        # loads bus connection
        _, buses_dict = self._network.get_buses()
        loads_bus = backend_action.get_loads_bus_global()
        for load_id, new_bus in loads_bus:
            iidm_id = str(loads.iloc[load_id].name)
            if new_bus == -1:
                self._network.disconnect_load(iidm_id)
            else:
                new_bus_id = buses_dict[new_bus]
                self._network.connected_load(iidm_id, new_bus_id)

        # generators bus connection
        generators_bus = backend_action.get_gens_bus_global()
        for gen_id, new_bus in generators_bus:
            iidm_id = str(generators.iloc[gen_id].name)
            if new_bus == -1:
                self._network.disconnect_generator(iidm_id)
            else:
                new_bus_id = buses_dict[new_bus]
                self._network.connected_generator(iidm_id, new_bus_id)

        # shunts bus connection
        shunts_bus = backend_action.get_shunts_bus_global()
        for shunt_id, new_bus in shunts_bus:
            iidm_id = str(shunts.iloc[shunt_id].name)
            if new_bus == -1:
                self._network.disconnect_shunt(iidm_id)
            else:
                new_bus_id = buses_dict[new_bus]
                self._network.connected_shunt(iidm_id, new_bus_id)

        # lines origin bus connection
        branches = self._network.get_branches()
        lines_or_bus = backend_action.get_lines_or_bus_global()
        for line_id, new_bus in lines_or_bus:
            iidm_id = str(branches.iloc[line_id].name)
            if new_bus == -1:
                self._network.disconnect_branch_side1(iidm_id)
            else:
                new_bus_id = buses_dict[new_bus]
                self._network.connect_branch_side1(iidm_id, new_bus_id)

        # lines extremity bus connection
        lines_ex_bus = backend_action.get_lines_ex_bus_global()
        for line_id, new_bus in lines_ex_bus:
            iidm_id = str(branches.iloc[line_id].name)
            if new_bus == -1:
                self._network.disconnect_branch_side2(iidm_id)
            else:
                new_bus_id = buses_dict[new_bus]
                self._network.connect_branch_side2(iidm_id, new_bus_id)

    def _check_isolated_injections(self) -> bool:
        loads = self._network.get_loads()
        if (loads['synchronous_component_bus'] > 0).any():
            return True
        if (~loads['connected']).any():
            return True
        generators = self._network.get_generators()
        if (generators['synchronous_component_bus'] > 0).any():
            return True
        if (~generators['connected']).any():
            return True
        shunts = self._network.get_shunts()
        if (shunts['synchronous_component_bus'] > 0).any():
            return True
        return False

    @staticmethod
    def _is_converged(result: pp.loadflow.ComponentResult) -> bool:
        return result.status == pp.loadflow.ComponentStatus.CONVERGED or result.status == pp.loadflow.ComponentStatus.NO_CALCULATION

    def runpf(self, is_dc: bool = False) -> Tuple[bool, Union[Exception, None]]:
        if self._check_isolated_and_disconnected_injections and self._check_isolated_injections():
            converged = False
        else:
            if is_dc:
                results = self._network.run_dc_pf()
            else:
                results = self._network.run_ac_pf()
            converged = self._is_converged(results[0])

        return converged, None if converged else DivergingPowerflow()  # FIXME this unusual as an API to require passing an exception as a return type

    def _update_topo_vect(self, res, df: pd.DataFrame, pos_topo_vect, bus_breaker_bus_id_attr: str,
                          connected_attr: str, num_bus_attr: str) -> None:
        for _, row in df.iterrows():
            my_pos_topo_vect = pos_topo_vect[row['num']]
            if row[bus_breaker_bus_id_attr] and row[connected_attr]:
                local_bus = self.global_bus_to_local_int(row[num_bus_attr], my_pos_topo_vect)
            else:
                local_bus = -1
            res[my_pos_topo_vect] = local_bus

    def get_topo_vect(self) -> np.ndarray:
        res = np.full(self.dim_topo, fill_value=-2, dtype=int)
        self._update_topo_vect(res, self._network.get_loads(), self.load_pos_topo_vect, 'bus_breaker_bus_id',
                               'connected', 'num_bus')
        self._update_topo_vect(res, self._network.get_generators(), self.gen_pos_topo_vect, 'bus_breaker_bus_id',
                               'connected', 'num_bus')
        # FIXME why no shunt_pos_topo_vect ?
        branches = self._network.get_branches()
        self._update_topo_vect(res, branches, self.line_or_pos_topo_vect, 'bus_breaker_bus1_id', 'connected1',
                               'num_bus1')
        self._update_topo_vect(res, branches, self.line_ex_pos_topo_vect, 'bus_breaker_bus2_id', 'connected2',
                               'num_bus2')
        return res

    def _injections_info(self, df: pd.DataFrame, sign: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        p = np.nan_to_num(np.where(df['connected'], df['p'], 0)) * sign
        q = np.nan_to_num(np.where(df['connected'], df['q'], 0)) * sign
        v = np.nan_to_num(np.array(np.where(df['connected'], df['v_mag_bus'], 0)))
        bus = np.array(np.where(df['connected'], df['local_num_bus'] + 1, -1)) # local bus number should start at 1...
        return p, q, v, bus

    def generators_info(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        p, q, v, _ = self._injections_info(self._network.get_generators(), -1.0) # load convention expected
        return p, q, v

    def loads_info(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        p, q, v, _ = self._injections_info(self._network.get_loads(), 1.0)
        return p, q, v

    def shunt_info(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return self._injections_info(self._network.get_shunts(), 1.0)

    def _lines_info(self, p_attr: str, q_attr: str, a_attr: str, v_attr: str, connected_attr: str) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        branches = self._network.get_branches()
        p = np.nan_to_num(np.where(branches[connected_attr], branches[p_attr], 0))
        if self._consider_open_branch_reactive_flow:
            q = np.nan_to_num(branches[q_attr])
            a = np.nan_to_num(branches[a_attr])
        else:
            q = np.nan_to_num(np.where(branches[connected_attr], branches[q_attr], 0))
            a = np.nan_to_num(np.where(branches[connected_attr], branches[a_attr], 0))
        v = np.nan_to_num(np.where(branches[connected_attr], branches[v_attr], 0))
        return p, q, v, a

    def lines_or_info(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return self._lines_info('p1', 'q1', 'i1', 'v_mag_bus1', 'connected1')

    def lines_ex_info(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return self._lines_info('p2', 'q2', 'i2', 'v_mag_bus2', 'connected2')
