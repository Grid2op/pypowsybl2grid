# Copyright (c) 2024, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import logging
import os
import time
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
                 connect_all_elements_to_first_bus = True,
                 lf_parameters: pp.loadflow.Parameters = DEFAULT_LF_PARAMETERS):
        Backend.__init__(self,
                         detailed_infos_for_cascading_failures=detailed_infos_for_cascading_failures,
                         can_be_copied=can_be_copied)
        self._check_isolated_and_disconnected_injections = check_isolated_and_disconnected_injections
        self._consider_open_branch_reactive_flow = consider_open_branch_reactive_flow
        self.n_busbar_per_sub = n_busbar_per_sub
        self._connect_all_elements_to_first_bus = connect_all_elements_to_first_bus
        self._lf_parameters = lf_parameters

        self.shunts_data_available = True
        self.supported_grid_format = pp.network.get_import_supported_extensions()

    @staticmethod
    def create_name(df: DataFrame) -> np.ndarray:
        return np.where(df['name'].eq(''), df.index.to_series(), df['name'])

    def load_grid(self,
                  path: Union[os.PathLike, str],
                  filename: Optional[Union[os.PathLike, str]] = None) -> None:
        logger.info("Loading network")

        start_time = time.time()

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
        if self._connect_all_elements_to_first_bus:
            self._network.connect_all_elements_to_first_bus()

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

        end_time = time.time()
        elapsed_time = (end_time - start_time) * 1000
        logger.info(f"Network '{self._network.get_id()}' loaded in {elapsed_time:.2f} ms with {len(switches)} retained switches: {len(buses)} buses, "
                    f"{len(branches)} branches, {len(generators)} generators, {len(loads)} loads, {len(shunts)} shunts")

    def apply_action(self, backend_action: Union["grid2op.Action._backendAction._BackendAction", None]) -> None:
        # the following few lines are highly recommended
        if backend_action is None:
            return

        logger.info("Applying action")

        start_time = time.time()

        _, buses_dict = self._network.get_buses()
        load_ids = self._network.get_load_ids()
        generator_ids = self._network.get_generator_ids()
        shunt_ids = self._network.get_shunt_ids()
        branch_ids = self._network.get_branch_ids()

        # active and reactive power of loads
        load_ids_to_update = []
        load_p_to_update = []
        for load_id, new_p in backend_action.load_p:
            iidm_id = str(load_ids[load_id])
            load_ids_to_update.append(iidm_id)
            load_p_to_update.append(new_p)
        self._network.update_load_p(load_ids_to_update, load_p_to_update)

        load_ids_to_update.clear()
        load_q_to_update = []
        for load_id, new_q in backend_action.load_q:
            iidm_id = str(load_ids[load_id])
            load_ids_to_update.append(iidm_id)
            load_q_to_update.append(new_q)
        self._network.update_load_q(load_ids_to_update, load_q_to_update)

        # active power and voltage target of generators
        gen_ids_to_update = []
        gen_p_to_update = []
        for gen_id, new_p in backend_action.prod_p:
            iidm_id = str(generator_ids[gen_id])
            gen_ids_to_update.append(iidm_id)
            gen_p_to_update.append(new_p)
        self._network.update_generator_p(gen_ids_to_update, gen_p_to_update)

        gen_ids_to_update.clear()
        gen_v_to_update = []
        for gen_id, new_v in backend_action.prod_v:
            iidm_id = str(generator_ids[gen_id])
            gen_ids_to_update.append(iidm_id)
            gen_v_to_update.append(new_v)
        self._network.update_generator_v(gen_ids_to_update, gen_v_to_update)

        # active and reactive power of shunts
        shunt_ids_to_update = []
        shunt_p_to_update = []
        for shunt_id, new_p in backend_action.shunt_p:
            iidm_id = str(shunt_ids[shunt_id])
            shunt_ids_to_update.append(iidm_id)
            shunt_p_to_update.append(new_p)
        self._network.update_shunt_p(shunt_ids_to_update, shunt_p_to_update)

        shunt_ids_to_update.clear()
        shunt_q_to_update = []
        for shunt_id, new_q in backend_action.shunt_q:
            iidm_id = str(shunt_ids[shunt_id])
            shunt_ids_to_update.append(iidm_id)
            shunt_q_to_update.append(new_q)
        self._network.update_shunt_q(shunt_ids_to_update, shunt_q_to_update)

        # loads bus connection
        load_ids_to_update.clear()
        load_connect_to_update = []
        load_new_bus_id_to_update = []
        loads_bus = backend_action.get_loads_bus_global()
        for load_id, new_bus in loads_bus:
            iidm_id = str(load_ids[load_id])
            load_ids_to_update.append(iidm_id)
            if new_bus == -1:
                load_connect_to_update.append(False)
                load_new_bus_id_to_update.append('')
            else:
                new_bus_id = buses_dict[new_bus]
                load_connect_to_update.append(True)
                load_new_bus_id_to_update.append(new_bus_id)
        self._network.connect_load(load_ids_to_update, load_connect_to_update, load_new_bus_id_to_update)

        # generators bus connection
        gen_ids_to_update.clear()
        gen_connect_to_update = []
        gen_new_bus_id_to_update = []
        generators_bus = backend_action.get_gens_bus_global()
        for gen_id, new_bus in generators_bus:
            iidm_id = str(generator_ids[gen_id])
            gen_ids_to_update.append(iidm_id)
            if new_bus == -1:
                gen_connect_to_update.append(False)
                gen_new_bus_id_to_update.append('')
            else:
                new_bus_id = buses_dict[new_bus]
                gen_connect_to_update.append(True)
                gen_new_bus_id_to_update.append(new_bus_id)
        self._network.connect_generator(gen_ids_to_update, gen_connect_to_update, gen_new_bus_id_to_update)

        # shunts bus connection
        shunt_ids_to_update.clear()
        shunt_connect_to_update = []
        shunt_new_bus_id_to_update = []
        shunts_bus = backend_action.get_shunts_bus_global()
        for shunt_id, new_bus in shunts_bus:
            iidm_id = str(shunt_ids[shunt_id])
            shunt_ids_to_update.append(iidm_id)
            if new_bus == -1:
                shunt_connect_to_update.append(False)
                shunt_new_bus_id_to_update.append('')
            else:
                new_bus_id = buses_dict[new_bus]
                shunt_connect_to_update.append(True)
                shunt_new_bus_id_to_update.append(new_bus_id)
        self._network.connect_shunt(shunt_ids_to_update, shunt_connect_to_update, shunt_new_bus_id_to_update)

        # lines origin bus connection
        branch_ids_to_update = []
        branch_connect_to_update = []
        branch_new_bus_id_to_update = []
        lines_or_bus = backend_action.get_lines_or_bus_global()
        for line_id, new_bus in lines_or_bus:
            iidm_id = str(branch_ids[line_id])
            branch_ids_to_update.append(iidm_id)
            if new_bus == -1:
                branch_connect_to_update.append(False)
                branch_new_bus_id_to_update.append('')
            else:
                new_bus_id = buses_dict[new_bus]
                branch_connect_to_update.append(True)
                branch_new_bus_id_to_update.append(new_bus_id)
        self._network.connect_branch_side1(branch_ids_to_update, branch_connect_to_update, branch_new_bus_id_to_update)

        # lines extremity bus connection
        branch_ids_to_update.clear()
        branch_connect_to_update.clear()
        branch_new_bus_id_to_update.clear()
        lines_ex_bus = backend_action.get_lines_ex_bus_global()
        for line_id, new_bus in lines_ex_bus:
            iidm_id = str(branch_ids[line_id])
            branch_ids_to_update.append(iidm_id)
            if new_bus == -1:
                branch_connect_to_update.append(False)
                branch_new_bus_id_to_update.append('')
            else:
                new_bus_id = buses_dict[new_bus]
                branch_connect_to_update.append(True)
                branch_new_bus_id_to_update.append(new_bus_id)
        self._network.connect_branch_side2(branch_ids_to_update, branch_connect_to_update, branch_new_bus_id_to_update)

        end_time = time.time()
        elapsed_time = (end_time - start_time) * 1000
        logger.info(f"Action applied in {elapsed_time:.2f} ms")

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
        logger.info("Running powerflow")

        start_time = time.time()

        if self._check_isolated_and_disconnected_injections and self._check_isolated_injections():
            converged = False
        else:
            if is_dc:
                results = self._network.run_dc_pf()
            else:
                results = self._network.run_ac_pf()
            converged = self._is_converged(results[0])

        end_time = time.time()
        elapsed_time = (end_time - start_time) * 1000
        logger.info(f"Powerflow ran in {elapsed_time:.2f} ms")

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

    def reset(self,
              path : Union[os.PathLike, str],
              grid_filename : Optional[Union[os.PathLike, str]]=None) -> None:
        logger.info("Reset backend")
        self.load_grid(path, filename=grid_filename)
