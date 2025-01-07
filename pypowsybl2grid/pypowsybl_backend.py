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
import pypowsybl as pp
import pypowsybl.grid2op
from grid2op.Backend import Backend
from grid2op.Exceptions import DivergingPowerflow
from grid2op.Space import DEFAULT_N_BUSBAR_PER_SUB

logger = logging.getLogger(__name__)

DEFAULT_LF_PARAMETERS = pp.loadflow.Parameters(voltage_init_mode=pp.loadflow.VoltageInitMode.DC_VALUES)

class PyPowSyBlBackend(Backend):

    def __init__(self,
                 detailed_infos_for_cascading_failures=False,
                 check_isolated_and_disconnected_injections = True,
                 consider_open_branch_reactive_flow = False,
                 n_busbar_per_sub = DEFAULT_N_BUSBAR_PER_SUB,
                 connect_all_elements_to_first_bus = False,
                 lf_parameters: pp.loadflow.Parameters = DEFAULT_LF_PARAMETERS):
        Backend.__init__(self,
                         detailed_infos_for_cascading_failures=detailed_infos_for_cascading_failures,
                         can_be_copied=True)
        self._check_isolated_and_disconnected_injections = check_isolated_and_disconnected_injections
        self._consider_open_branch_reactive_flow = consider_open_branch_reactive_flow
        self.n_busbar_per_sub = n_busbar_per_sub
        self._connect_all_elements_to_first_bus = connect_all_elements_to_first_bus
        self._lf_parameters = lf_parameters

        self.shunts_data_available = True
        self.supported_grid_format = pp.network.get_import_supported_extensions()

        self._grid = None

    @property
    def network(self) -> pp.network.Network:
        return self._grid.network if self._grid else None

    def load_grid(self,
                  path: Union[os.PathLike, str],
                  filename: Optional[Union[os.PathLike, str]] = None) -> None:
        start_time = time.time()

        # load network
        full_path = self.make_complete_path(path, filename)

        logger.info(f"Loading network from '{full_path}'")

        if full_path.endswith('.json'):
            n_pdp = pdp.from_json(full_path)
            network = pp.network.convert_from_pandapower(n_pdp)
        else:
            network = pp.network.load(full_path)

        self.load_grid_from_iidm(network)

        end_time = time.time()
        elapsed_time = (end_time - start_time) * 1000
        logger.info(f"Network '{network.id}' loaded in {elapsed_time:.2f} ms")

    def load_grid_from_iidm(self, network: pp.network.Network) -> None:
        if self._grid:
            self._grid.close()
            self._grid = None

        self._grid = pp.grid2op.Backend(network,
                                        self._consider_open_branch_reactive_flow,
                                        self.n_busbar_per_sub,
                                        self._connect_all_elements_to_first_bus)

        # substations mapped to IIDM voltage levels
        self.name_sub = self._grid.get_string_value(pp.grid2op.StringValueType.VOLTAGE_LEVEL_NAME)
        self.n_sub = len(self.name_sub)

        self.can_handle_more_than_2_busbar()

        logger.info(f"{self.n_busbar_per_sub} busbars per substation")

        # loads
        self.name_load = self._grid.get_string_value(pp.grid2op.StringValueType.LOAD_NAME)
        self.n_load = len(self.name_load)
        self.load_to_subid = self._grid.get_integer_value(pp.grid2op.IntegerValueType.LOAD_VOLTAGE_LEVEL_NUM)

        # generators
        self.name_gen = self._grid.get_string_value(pp.grid2op.StringValueType.GENERATOR_NAME)
        self.n_gen = len(self.name_gen)
        self.gen_to_subid = self._grid.get_integer_value(pp.grid2op.IntegerValueType.GENERATOR_VOLTAGE_LEVEL_NUM)

        # shunts
        self.name_shunt = self._grid.get_string_value(pp.grid2op.StringValueType.SHUNT_NAME)
        self.n_shunt = len(self.name_shunt)
        self.shunt_to_subid = self._grid.get_integer_value(pp.grid2op.IntegerValueType.SHUNT_VOLTAGE_LEVEL_NUM)

        # batteries
        self.set_no_storage()
        # FIXME implement batteries
        # self.name_storage = np.array(self._grid.get_string_value(pp.grid2op.StringValueType.BATTERY_NAME))
        # self.n_storage = len(self.name_storage)
        # self.storage_type = np.full(self.n_storage, fill_value="???")
        # self.storage_to_subid = self._grid.get_integer_value(pp.grid2op.IntegerValueType.BATTERY_VOLTAGE_LEVEL_NUM).copy()

        # lines and transformers
        self.name_line = self._grid.get_string_value(pp.grid2op.StringValueType.BRANCH_NAME)
        self.n_line = len(self.name_line)
        self.line_or_to_subid = self._grid.get_integer_value(pp.grid2op.IntegerValueType.BRANCH_VOLTAGE_LEVEL_NUM_1)
        self.line_ex_to_subid = self._grid.get_integer_value(pp.grid2op.IntegerValueType.BRANCH_VOLTAGE_LEVEL_NUM_2)

        self._compute_pos_big_topo()

        # thermal limits
        self.thermal_limit_a = self._grid.get_double_value(pp.grid2op.DoubleValueType.BRANCH_PERMANENT_LIMIT_A)

    def apply_action(self, backend_action: Union["grid2op.Action._backendAction._BackendAction", None]) -> None:
        # the following few lines are highly recommended
        if backend_action is None:
            return

        logger.info("Applying action")

        start_time = time.time()

        self._grid.update_double_value(pp.grid2op.UpdateDoubleValueType.UPDATE_LOAD_P, backend_action.load_p.values, backend_action.load_p.changed)
        self._grid.update_double_value(pp.grid2op.UpdateDoubleValueType.UPDATE_LOAD_Q, backend_action.load_q.values, backend_action.load_q.changed)
        self._grid.update_double_value(pp.grid2op.UpdateDoubleValueType.UPDATE_GENERATOR_P, backend_action.prod_p.values, backend_action.prod_p.changed)
        self._grid.update_double_value(pp.grid2op.UpdateDoubleValueType.UPDATE_GENERATOR_V, backend_action.prod_v.values, backend_action.prod_v.changed)
        # TODO shunts

        loads_bus = backend_action.get_loads_bus()
        self._grid.update_integer_value(pp.grid2op.UpdateIntegerValueType.UPDATE_LOAD_BUS, loads_bus.values, loads_bus.changed)
        generators_bus = backend_action.get_gens_bus()
        self._grid.update_integer_value(pp.grid2op.UpdateIntegerValueType.UPDATE_GENERATOR_BUS, generators_bus.values, generators_bus.changed)
        shunt_bus = backend_action.shunt_bus
        self._grid.update_integer_value(pp.grid2op.UpdateIntegerValueType.UPDATE_SHUNT_BUS, shunt_bus.values, shunt_bus.changed)
        lines_or_bus = backend_action.get_lines_or_bus()
        self._grid.update_integer_value(pp.grid2op.UpdateIntegerValueType.UPDATE_BRANCH_BUS1, lines_or_bus.values, lines_or_bus.changed)
        lines_ex_bus = backend_action.get_lines_ex_bus()
        self._grid.update_integer_value(pp.grid2op.UpdateIntegerValueType.UPDATE_BRANCH_BUS2, lines_ex_bus.values, lines_ex_bus.changed)

        end_time = time.time()
        elapsed_time = (end_time - start_time) * 1000
        logger.info(f"Action applied in {elapsed_time:.2f} ms")

    @staticmethod
    def _is_converged(result: pp.loadflow.ComponentResult) -> bool:
        return result.status == pp.loadflow.ComponentStatus.CONVERGED or result.status == pp.loadflow.ComponentStatus.NO_CALCULATION

    def runpf(self, is_dc: bool = False) -> Tuple[bool, Union[Exception, None]]:
        logger.info(f"Running {'DC' if is_dc else 'AC'} powerflow")

        start_time = time.perf_counter()

        if self._check_isolated_and_disconnected_injections and self._native_backend.check_isolated_and_disconnected_injections():
            converged = False
        else:
            beg_ = time.perf_counter()
            results = self._native_backend.run_pf(is_dc, self._lf_parameters)
            end_ = time.perf_counter()
            self.comp_time += end_ - beg_
            converged = self._is_converged(results[0])

        end_time = time.perf_counter()  # changed
        elapsed_time = (end_time - start_time) * 1000
        logger.info(f"Powerflow ran in {elapsed_time:.2f} ms")

        return converged, None if converged else DivergingPowerflow()

    def get_topo_vect(self) -> np.ndarray:
        return self._grid.get_integer_value(pp.grid2op.IntegerValueType.TOPO_VECT)

    def generators_info(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        p = self._grid.get_double_value(pp.grid2op.DoubleValueType.GENERATOR_P)
        q = self._grid.get_double_value(pp.grid2op.DoubleValueType.GENERATOR_Q)
        v = self._grid.get_double_value(pp.grid2op.DoubleValueType.GENERATOR_V)
        return p, q, v

    def loads_info(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        p = self._grid.get_double_value(pp.grid2op.DoubleValueType.LOAD_P)
        q = self._grid.get_double_value(pp.grid2op.DoubleValueType.LOAD_Q)
        v = self._grid.get_double_value(pp.grid2op.DoubleValueType.LOAD_V)
        return p, q, v

    def shunt_info(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        p = self._grid.get_double_value(pp.grid2op.DoubleValueType.SHUNT_P)
        q = self._grid.get_double_value(pp.grid2op.DoubleValueType.SHUNT_Q)
        v = self._grid.get_double_value(pp.grid2op.DoubleValueType.SHUNT_V)
        bus = self._grid.get_integer_value(pp.grid2op.IntegerValueType.SHUNT_LOCAL_BUS)
        return p, q, v, bus

    def lines_or_info(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        p = self._grid.get_double_value(pp.grid2op.DoubleValueType.BRANCH_P1)
        q = self._grid.get_double_value(pp.grid2op.DoubleValueType.BRANCH_Q1)
        v = self._grid.get_double_value(pp.grid2op.DoubleValueType.BRANCH_V1)
        a = self._grid.get_double_value(pp.grid2op.DoubleValueType.BRANCH_I1)
        return p, q, v, a

    def lines_ex_info(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        p = self._grid.get_double_value(pp.grid2op.DoubleValueType.BRANCH_P2)
        q = self._grid.get_double_value(pp.grid2op.DoubleValueType.BRANCH_Q2)
        v = self._grid.get_double_value(pp.grid2op.DoubleValueType.BRANCH_V2)
        a = self._grid.get_double_value(pp.grid2op.DoubleValueType.BRANCH_I2)
        return p, q, v, a

    def reset(self,
              path : Union[os.PathLike, str],
              grid_filename : Optional[Union[os.PathLike, str]]=None) -> None:
        logger.info("Reset backend")
        self.load_grid(path, filename=grid_filename)

    def close(self) -> None:
        if self._grid:
            self._grid.close()
            self._grid = None
