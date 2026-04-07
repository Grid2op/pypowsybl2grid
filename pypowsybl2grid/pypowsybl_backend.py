# Copyright (c) 2024, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import logging
import os
import time
import warnings
from typing import Optional, Tuple, Union

import numpy as np
import pandapower as pdp
from grid2op.Action._backendAction import _BackendAction
from grid2op.Backend import Backend
from grid2op.dtypes import dt_float, dt_int
from grid2op.Exceptions import DivergingPowerflow
from grid2op.Space import DEFAULT_N_BUSBAR_PER_SUB
from pypowsybl._pypowsybl import (
    Grid2opDoubleValueType,
    Grid2opIntegerValueType,
    Grid2opStringValueType,
    Grid2opUpdateDoubleValueType,
    Grid2opUpdateIntegerValueType,
    LoadFlowComponentStatus,
    VoltageInitMode,
)
from pypowsybl.grid2op.impl.backend import Backend as PPBackend
from pypowsybl.loadflow.impl.component_result import ComponentResult
from pypowsybl.loadflow.impl.parameters import Parameters
from pypowsybl.network.impl.network import Network
from pypowsybl.network.impl.network_creation_util import load
from pypowsybl.network.impl.pandapower_converter import convert_from_pandapower
from pypowsybl.network.impl.util import get_import_supported_extensions

from pypowsybl2grid.models import (
    PhaseTapChangerUpdate,
    PhaseTapChangerUpdatePayload,
    QUpdate,
    QUpdatePayload,
    RatioTapChangerUpdate,
    RatioTapChangerUpdatePayload,
    ShuntUpdate,
    ShuntUpdatePayload,
)

logger = logging.getLogger("pypowsybl2grid")

DEFAULT_LF_PARAMETERS = Parameters(voltage_init_mode=VoltageInitMode.DC_VALUES)


class PyPowSyBlBackend(Backend):
    shunts_data_available = True

    def __init__(
        self,
        detailed_infos_for_cascading_failures: bool = False,
        can_be_copied: bool = True,
        check_isolated_and_disconnected_injections: bool | None = None,
        consider_open_branch_reactive_flow: bool = False,
        n_busbar_per_sub: int = DEFAULT_N_BUSBAR_PER_SUB,
        connect_all_elements_to_first_bus: bool = False,
        lf_parameters: Parameters | None = None,
    ):
        Backend.__init__(
            self,
            detailed_infos_for_cascading_failures=detailed_infos_for_cascading_failures,
            can_be_copied=can_be_copied,
            # save this kwargs (might be needed)
            check_isolated_and_disconnected_injections=check_isolated_and_disconnected_injections,
            consider_open_branch_reactive_flow=consider_open_branch_reactive_flow,
            connect_all_elements_to_first_bus=connect_all_elements_to_first_bus,
            lf_parameters=lf_parameters,
        )

        self._check_isolated_and_disconnected_injections = (
            check_isolated_and_disconnected_injections
        )
        self._consider_open_branch_reactive_flow = consider_open_branch_reactive_flow
        self.n_busbar_per_sub = n_busbar_per_sub  # pyright: ignore[reportAttributeAccessIssue]
        self._connect_all_elements_to_first_bus = connect_all_elements_to_first_bus
        if lf_parameters is None:
            self._lf_parameters = DEFAULT_LF_PARAMETERS
        else:
            self._lf_parameters = lf_parameters

        self.can_output_theta = True  # pyright: ignore[reportAttributeAccessIssue]

        self.supported_grid_format = get_import_supported_extensions()  # pyright: ignore[reportAttributeAccessIssue]

        self._grid = None
        self._phase_tap_changers_to_use_in_network: (
            PhaseTapChangerUpdatePayload | None
        ) = None
        self._ratio_tap_changers_to_use_in_network: (
            RatioTapChangerUpdatePayload | None
        ) = None
        self._shunt_data_to_use_in_network: ShuntUpdatePayload | None = None
        self._q_values_for_pq_gens: QUpdatePayload | None = None

        # caching of the results
        self._gen_p: np.ndarray = np.empty(0, dtype=dt_float)
        self._gen_q: np.ndarray = np.empty(0, dtype=dt_float)
        self._gen_v: np.ndarray = np.empty(0, dtype=dt_float)

        self._load_p: np.ndarray = np.empty(0, dtype=dt_float)
        self._load_q: np.ndarray = np.empty(0, dtype=dt_float)
        self._load_v: np.ndarray = np.empty(0, dtype=dt_float)

        self._por: np.ndarray = np.empty(0, dtype=dt_float)
        self._qor: np.ndarray = np.empty(0, dtype=dt_float)
        self._aor: np.ndarray = np.empty(0, dtype=dt_float)
        self._vor: np.ndarray = np.empty(0, dtype=dt_float)

        self._pex: np.ndarray = np.empty(0, dtype=dt_float)
        self._qex: np.ndarray = np.empty(0, dtype=dt_float)
        self._aex: np.ndarray = np.empty(0, dtype=dt_float)
        self._vex: np.ndarray = np.empty(0, dtype=dt_float)

        self._shunt_p: np.ndarray = np.empty(0, dtype=dt_float)
        self._shunt_q: np.ndarray = np.empty(0, dtype=dt_float)
        self._shunt_v: np.ndarray = np.empty(0, dtype=dt_float)
        self._shunt_bus: np.ndarray = np.empty(0, dtype=dt_int)

        self._gen_theta: np.ndarray = np.empty(0, dtype=dt_float)
        self._load_theta: np.ndarray = np.empty(0, dtype=dt_float)
        self._line_or_theta: np.ndarray = np.empty(0, dtype=dt_float)
        self._line_ex_theta: np.ndarray = np.empty(0, dtype=dt_float)
        self._storage_theta: np.ndarray = np.empty(0, dtype=dt_float)

        self._topo_vect: np.ndarray = np.empty(0, dtype=dt_int)

    @property
    def shunt_data_to_use_in_network(
        self,
    ) -> ShuntUpdatePayload | None:
        """Shunt compensators to apply when loading the network."""
        return self._shunt_data_to_use_in_network

    @shunt_data_to_use_in_network.setter
    def shunt_data_to_use_in_network(self, value: ShuntUpdatePayload) -> None:
        self._shunt_data_to_use_in_network = value

    @property
    def q_values_for_pq_gens(
        self,
    ) -> QUpdatePayload:
        """Target Q values for PQ generators (voltage_regulator_on=False) to apply when loading the network."""
        return self._q_values_for_pq_gens or QUpdatePayload(updates=[])

    @q_values_for_pq_gens.setter
    def q_values_for_pq_gens(self, value: QUpdatePayload) -> None:
        self._q_values_for_pq_gens = value

    @property
    def ratio_tap_changers_to_use_in_network(
        self,
    ) -> RatioTapChangerUpdatePayload | None:
        """Ratio tap changers (classic transformers) to apply when loading the network."""
        return self._ratio_tap_changers_to_use_in_network

    @ratio_tap_changers_to_use_in_network.setter
    def ratio_tap_changers_to_use_in_network(
        self, value: RatioTapChangerUpdatePayload
    ) -> None:
        self._ratio_tap_changers_to_use_in_network = value

    @property
    def phase_tap_changers_to_use_in_network(
        self,
    ) -> PhaseTapChangerUpdatePayload | None:
        """Phase tap changers (phase-shifting transformers) to apply when loading the network."""
        return self._phase_tap_changers_to_use_in_network

    @phase_tap_changers_to_use_in_network.setter
    def phase_tap_changers_to_use_in_network(
        self, value: PhaseTapChangerUpdatePayload
    ) -> None:
        self._phase_tap_changers_to_use_in_network = value

    def init_tap_changers_from_network(self, network: Network) -> None:
        """Initialise both tap changer properties from the current taps in the given network."""
        phase_tap_steps = network.get_phase_tap_changer_steps()
        ratio_tap_steps = network.get_ratio_tap_changer_steps()
        phase_updates = []
        for i, row in network.get_phase_tap_changers(all_attributes=True).iterrows():
            phase_update = PhaseTapChangerUpdate(
                id=str(i),
                tap=row["tap"],
                rho=phase_tap_steps.loc[str(i)]["rho"].values[0],
                r=phase_tap_steps.loc[str(i)]["r"].values[0],
                x=phase_tap_steps.loc[str(i)]["x"].values[0],
                g=phase_tap_steps.loc[str(i)]["g"].values[0],
                b=phase_tap_steps.loc[str(i)]["b"].values[0],
                regulating=row["regulating"],
                regulation_mode=row["regulation_mode"],
                regulation_value=row["regulation_value"],
                regulated_side=row["regulated_side"],
                target_deadband=row["target_deadband"],
            )
            phase_updates.append(phase_update)
        ratio_updates = []
        for i, row in network.get_ratio_tap_changers(all_attributes=True).iterrows():
            ratio_update = RatioTapChangerUpdate(
                id=str(i),
                tap=row["tap"],
                rho=ratio_tap_steps.loc[str(i)]["rho"].values[0],
                r=ratio_tap_steps.loc[str(i)]["r"].values[0],
                x=ratio_tap_steps.loc[str(i)]["x"].values[0],
                g=ratio_tap_steps.loc[str(i)]["g"].values[0],
                b=ratio_tap_steps.loc[str(i)]["b"].values[0],
                regulating=row["regulating"],
                oltc=row["oltc"],
                regulated_side=row["regulated_side"],
                target_deadband=row["target_deadband"],
            )
            ratio_updates.append(ratio_update)

        self.phase_tap_changers_to_use_in_network = PhaseTapChangerUpdatePayload(
            updates=phase_updates
        )
        self.ratio_tap_changers_to_use_in_network = RatioTapChangerUpdatePayload(
            updates=ratio_updates
        )

    def init_shunt_data_from_network(self, network: Network) -> None:
        """Initialise shunt properties from the current taps in the given network."""
        self.shunt_data_to_use_in_network = ShuntUpdatePayload(
            updates=[
                ShuntUpdate(
                    id=str(i),
                    **{
                        k: row[k]
                        for k in ShuntUpdate.model_fields
                        if k != "id" and k in row.index
                    },
                )
                for i, row in network.get_shunt_compensators().iterrows()
            ]
        )

    def init_pq_gen_q_from_network(self, network: Network) -> None:
        """Initialise PQ generator Q values from the current generators in the given network."""
        gens = network.get_generators(all_attributes=True)
        self.q_values_for_pq_gens = QUpdatePayload(
            updates=[
                QUpdate(
                    id=str(i),
                    target_q=row["target_q"],
                    voltage_regulator_on=row["voltage_regulator_on"],
                )
                for i, row in gens.iterrows()
            ]
        )

    def _update_backend_network_gens_q_with_pq_gen_q_values(self) -> None:
        """
        This updates the grid2op backend held Network with the
        data stored in self.q_values_for_pq_gens.
        """
        if not self.network:
            raise ValueError(
                "self.network is None, you should have a self.network before trying to update generator Q values on it."
            )
        q_data = self.q_values_for_pq_gens
        if q_data.updates:
            self.network.update_generators(df=q_data.to_df())

    def _update_backend_network_taps_with_taps_to_use_in_network(self) -> None:
        """
        This updates the grid2op backend held Network with the
        data store in self.taps_to_use_in_network.
        """
        if (
            not self.phase_tap_changers_to_use_in_network
            or not self.ratio_tap_changers_to_use_in_network
        ):
            raise ValueError(
                "You should set self.phase_tap_changers_to_use_in_network and self.ratio_tap_changers_to_use_in_network"
            )
        if self.network:
            self.network.update_phase_tap_changers(
                df=self.phase_tap_changers_to_use_in_network.to_df()[0]
            )
            self.network.update_phase_tap_changers(
                df=self.phase_tap_changers_to_use_in_network.to_df()[1]
            )
            self.network.update_ratio_tap_changers(
                df=self.ratio_tap_changers_to_use_in_network.to_df()[0]
            )
            self.network.update_ratio_tap_changers(
                df=self.ratio_tap_changers_to_use_in_network.to_df()[1]
            )
        else:
            raise ValueError(
                "self.network is None, you should have a self.network before trying to update taps on it."
            )

    def _update_backend_network_shunt_with_shunt_data_to_use_in_network(self) -> None:
        """
        This updates the grid2op backend held Network with the
        data store in self.taps_to_use_in_network.
        """
        if not self.shunt_data_to_use_in_network:
            raise ValueError("You should set self.shunt_data_to_use_in_network")
        if self.network:
            self.network.update_shunt_compensators(
                df=self.shunt_data_to_use_in_network.to_df()
            )
        else:
            raise ValueError(
                "self.network is None, you should have a self.network before trying to update shunts on it."
            )

    @property
    def network(self) -> Network | None:
        return self._grid.network if self._grid else None

    def load_grid(
        self,
        path: Union[os.PathLike, str],
        filename: Optional[Union[os.PathLike, str]] = None,
    ) -> None:
        start_time = time.perf_counter()
        full_path = self.make_complete_path(path, filename)
        logger.info(f"Loading network from path {full_path}")
        cls = type(self)
        if hasattr(cls, "can_handle_more_than_2_busbar"):
            # grid2op version >= 1.10.0 then we use this
            self.can_handle_more_than_2_busbar()

        if hasattr(cls, "can_handle_detachment"):
            # grid2op version >= 1.11.0 then we use this
            self.can_handle_detachment()
            self.check_detachment_coherent()
        else:
            if self._check_isolated_and_disconnected_injections is None:
                # default behaviour in grid2op before detachment is allowed
                self._check_isolated_and_disconnected_injections = True

        if full_path.endswith(".json"):
            n_pdp = pdp.from_json(full_path)
            network = convert_from_pandapower(n_pdp)
        else:
            network = load(full_path)

        self.load_grid_from_iidm(network)
        logger.info(
            f"Network loaded from path {full_path} in {(time.perf_counter() - start_time) * 1000}"
        )
        logger.info(
            "Network topology: %d substations, %d loads, %d generators, %d lines, %d shunts",
            self.n_sub,
            self.n_load,
            self.n_gen,
            self.n_line,
            self.n_shunt,
        )

    def check_detachment_coherent(self):
        if self._check_isolated_and_disconnected_injections is None:
            # user does not provide anything to the backend
            if self.detachment_is_allowed:
                self._check_isolated_and_disconnected_injections = False
            else:
                self._check_isolated_and_disconnected_injections = True
        else:
            # user provided something, I check if it's consistent
            if self._check_isolated_and_disconnected_injections:
                if self.detachment_is_allowed:
                    msg_ = (
                        'You initialized the pypowsybl backend with "check_isolated_and_disconnected_injections=True" '
                        'and the environment with "allow_detachment=True" which is not consistent. '
                        "Discarding the call to env.make, the detachement is NOT supported for this env. "
                        "If you want to support detachment, either initialize the pypowsybl backend with "
                        '"check_isolated_and_disconnected_injections=False" or (preferably) with '
                        '"check_isolated_and_disconnected_injections=None"'
                    )
                    warnings.warn(msg_)
                    logger.warning(msg_)
                    type(self).detachment_is_allowed = False
                    self.detachment_is_allowed = False
            else:
                if not self.detachment_is_allowed:
                    msg_ = (
                        'You initialized the pypowsybl backend with "check_isolated_and_disconnected_injections=False" '
                        'and the environment with "allow_detachment=False" which is not consistent. '
                        'The setting of "check_isolated_and_disconnected_injections=False" will have no effect. '
                        "Detachment will NOT be supported. If you want so, please consider initializing pypowsybl backend with "
                        '"check_isolated_and_disconnected_injections=True" or (preferably) with '
                        '"check_isolated_and_disconnected_injections=None"'
                    )
                    warnings.warn(msg_)
                    logger.warning(msg_)

    def load_grid_from_iidm(self, network: Network) -> None:
        if self._grid:
            self._grid.close()
            self._grid = None

        current_ratio_tap_changers, current_phase_tap_changers = (
            network.get_ratio_tap_changers(all_attributes=True),
            network.get_phase_tap_changers(all_attributes=True),
        )
        current_phase_tap_steps = network.get_phase_tap_changer_steps(
            all_attributes=True
        )
        current_ratio_tap_steps = network.get_ratio_tap_changer_steps(
            all_attributes=True
        )
        current_shunt_compensators = network.get_shunt_compensators()
        current_generators = network.get_generators(all_attributes=True)
        current_pq_generators = current_generators[
            ~current_generators["voltage_regulator_on"]
        ]

        n_phase = len(current_phase_tap_changers)
        if self.phase_tap_changers_to_use_in_network:
            phase_overrides = {
                update.id: update.model_dump(exclude_none=True, exclude={"id"})
                for update in self.phase_tap_changers_to_use_in_network.updates
            }
            phase_unchanged = [
                str(i)
                for i, _ in current_phase_tap_changers.iterrows()
                if str(i) not in phase_overrides
            ]
            n_phase_overridden = n_phase - len(phase_unchanged)
            logger.info(
                f"Phase tap changers: {n_phase_overridden}/{n_phase} taps overridden via property"
                + (f" ({100 * n_phase_overridden // n_phase}%)" if n_phase else "")
            )
            if phase_unchanged:
                logger.info(
                    f"Phase tap changers unchanged (using network values): {phase_unchanged}"
                )
        else:
            phase_overrides = {}
            logger.info(
                f"Phase tap changers: property not set, using all {n_phase} taps from network"
            )

        phase_updates = []
        for i, row in network.get_phase_tap_changers(all_attributes=True).iterrows():
            phase_update = PhaseTapChangerUpdate(
                id=str(i),
                tap=row["tap"],
                rho=current_phase_tap_steps.loc[str(i)]["rho"].values[0],
                r=current_phase_tap_steps.loc[str(i)]["r"].values[0],
                x=current_phase_tap_steps.loc[str(i)]["x"].values[0],
                g=current_phase_tap_steps.loc[str(i)]["g"].values[0],
                b=current_phase_tap_steps.loc[str(i)]["b"].values[0],
                regulating=row["regulating"],
                regulation_mode=row["regulation_mode"],
                regulation_value=row["regulation_value"],
                regulated_side=row["regulated_side"],
                target_deadband=row["target_deadband"],
            )
            phase_updates.append(phase_update)
        self.phase_tap_changers_to_use_in_network = PhaseTapChangerUpdatePayload(
            updates=phase_updates
        )

        n_ratio = len(current_ratio_tap_changers)
        if self.ratio_tap_changers_to_use_in_network:
            ratio_overrides = {
                update.id: update.model_dump(exclude_none=True, exclude={"id"})
                for update in self.ratio_tap_changers_to_use_in_network.updates
            }
            ratio_unchanged = [
                str(i)
                for i, _ in current_ratio_tap_changers.iterrows()
                if str(i) not in ratio_overrides
            ]
            n_ratio_overridden = n_ratio - len(ratio_unchanged)
            logger.info(
                f"Ratio tap changers: {n_ratio_overridden}/{n_ratio} taps overridden via property"
                + (f" ({100 * n_ratio_overridden // n_ratio}%)" if n_ratio else "")
            )
            if ratio_unchanged:
                logger.info(
                    f"Ratio tap changers unchanged (using network values): {ratio_unchanged}"
                )
        else:
            ratio_overrides = {}
            logger.info(
                f"Ratio tap changers: property not set, using all {n_ratio} taps from network"
            )

        ratio_updates = []
        for i, row in network.get_ratio_tap_changers(all_attributes=True).iterrows():
            ratio_update = RatioTapChangerUpdate(
                id=str(i),
                tap=row["tap"],
                rho=current_ratio_tap_steps.loc[str(i)]["rho"].values[0],
                r=current_ratio_tap_steps.loc[str(i)]["r"].values[0],
                x=current_ratio_tap_steps.loc[str(i)]["x"].values[0],
                g=current_ratio_tap_steps.loc[str(i)]["g"].values[0],
                b=current_ratio_tap_steps.loc[str(i)]["b"].values[0],
                regulating=row["regulating"],
                oltc=row["oltc"],
                regulated_side=row["regulated_side"],
                target_deadband=row["target_deadband"],
            )
            ratio_updates.append(ratio_update)

        self.ratio_tap_changers_to_use_in_network = RatioTapChangerUpdatePayload(
            updates=ratio_updates
        )
        n_shunt = len(current_shunt_compensators)
        if self.shunt_data_to_use_in_network:
            shunt_overrides = {
                update.id: update.model_dump(exclude={"id"})
                for update in self.shunt_data_to_use_in_network.updates
            }
            shunt_unchanged = [
                str(i)
                for i, _ in current_shunt_compensators.iterrows()
                if str(i) not in shunt_overrides
            ]
            n_shunt_overridden = n_shunt - len(shunt_unchanged)
            logger.info(
                f"Shunt compensators: {n_shunt_overridden}/{n_shunt} overridden via property"
                + (f" ({100 * n_shunt_overridden // n_shunt}%)" if n_shunt else "")
            )
            if shunt_unchanged:
                logger.info(
                    f"Shunt compensators unchanged (using network values): {shunt_unchanged}"
                )
        else:
            shunt_overrides = {}
            logger.info(
                f"Shunt compensators: property not set, using all {n_shunt} from network"
            )
        self.shunt_data_to_use_in_network = ShuntUpdatePayload(
            updates=[
                ShuntUpdate(
                    id=str(i),
                    **{
                        **{
                            k: row[k]
                            for k in ShuntUpdate.model_fields
                            if k != "id" and k in row.index
                        },
                        **shunt_overrides.get(str(i), {}),
                    },
                )
                for i, row in current_shunt_compensators.iterrows()
            ]
        )

        n_pq_gen = len(current_pq_generators)
        if self._q_values_for_pq_gens:
            pq_gen_overrides = {
                update.id: update.model_dump(exclude={"id"})
                for update in self._q_values_for_pq_gens.updates
            }
            pq_gen_unchanged = [
                str(i)
                for i, _ in current_pq_generators.iterrows()
                if str(i) not in pq_gen_overrides
            ]
            n_pq_gen_overridden = n_pq_gen - len(pq_gen_unchanged)
            logger.info(
                f"PQ generators: {n_pq_gen_overridden}/{n_pq_gen} overridden via property"
                + (f" ({100 * n_pq_gen_overridden // n_pq_gen}%)" if n_pq_gen else "")
            )
            if pq_gen_unchanged:
                logger.info(
                    f"PQ generators unchanged (using network values): {pq_gen_unchanged}"
                )
        else:
            pq_gen_overrides = {}
            logger.info(
                f"PQ generators: property not set, using all {n_pq_gen} from network"
            )
        self.q_values_for_pq_gens = QUpdatePayload(
            updates=[
                QUpdate(
                    id=str(i),
                    **{
                        **{
                            k: row[k]
                            for k in QUpdate.model_fields
                            if k != "id" and k in row.index
                        },
                        **pq_gen_overrides.get(str(i), {}),
                    },
                )
                for i, row in current_pq_generators.iterrows()
            ]
        )

        self._grid = PPBackend(
            network,
            self._consider_open_branch_reactive_flow,
            self._check_isolated_and_disconnected_injections,  # type: ignore / cannot be None here
            self.n_busbar_per_sub,
            self._connect_all_elements_to_first_bus,
        )

        phase_taps = self.phase_tap_changers_to_use_in_network
        ratio_taps = self.ratio_tap_changers_to_use_in_network
        if isinstance(phase_taps, PhaseTapChangerUpdatePayload) and isinstance(
            ratio_taps, RatioTapChangerUpdatePayload
        ):
            if len(phase_taps.updates) > 0 and len(ratio_taps.updates) > 0:
                self._update_backend_network_taps_with_taps_to_use_in_network()
        else:
            raise ValueError(
                "Tap ratio and phase to use in the network should be set by now."
            )

        shunt_data = self.shunt_data_to_use_in_network
        if isinstance(shunt_data, ShuntUpdatePayload):
            if len(shunt_data.updates) > 0:
                self._update_backend_network_shunt_with_shunt_data_to_use_in_network()
        else:
            raise ValueError("Shunt data to use in the network should be set by now.")

        self._update_backend_network_gens_q_with_pq_gen_q_values()

        # substations mapped to IIDM voltage levels
        self.name_sub = self._grid.get_string_value(  # pyright: ignore[reportAttributeAccessIssue]
            Grid2opStringValueType.VOLTAGE_LEVEL_NAME
        )
        self.n_sub = len(self.name_sub)  # pyright: ignore[reportAttributeAccessIssue]

        logger.info(f"{self.n_busbar_per_sub} busbars per substation")

        # loads
        self.name_load = self._grid.get_string_value(  # pyright: ignore[reportAttributeAccessIssue]
            Grid2opStringValueType.LOAD_NAME
        )
        self.n_load = len(self.name_load)  # pyright: ignore[reportAttributeAccessIssue]
        self.load_to_subid = self._grid.get_integer_value(  # pyright: ignore[reportAttributeAccessIssue]
            Grid2opIntegerValueType.LOAD_VOLTAGE_LEVEL_NUM
        )

        # generators
        self.name_gen = self._grid.get_string_value(  # pyright: ignore[reportAttributeAccessIssue]
            Grid2opStringValueType.GENERATOR_NAME
        )
        self.n_gen = len(self.name_gen)  # pyright: ignore[reportAttributeAccessIssue]
        self.gen_to_subid = self._grid.get_integer_value(  # pyright: ignore[reportAttributeAccessIssue]
            Grid2opIntegerValueType.GENERATOR_VOLTAGE_LEVEL_NUM
        )

        # shunts
        self.name_shunt = self._grid.get_string_value(  # pyright: ignore[reportAttributeAccessIssue]
            Grid2opStringValueType.SHUNT_NAME
        )
        self.n_shunt = len(self.name_shunt)  # pyright: ignore[reportAttributeAccessIssue]
        self.shunt_to_subid = self._grid.get_integer_value(  # pyright: ignore[reportAttributeAccessIssue]
            Grid2opIntegerValueType.SHUNT_VOLTAGE_LEVEL_NUM
        )

        # batteries
        self.set_no_storage()
        self.n_storage = 0  # pyright: ignore[reportAttributeAccessIssue]
        # FIXME implement batteries
        # self.name_storage = np.array(self._grid.get_string_value(pp.grid2op.StringValueType.BATTERY_NAME))
        # self.n_storage = len(self.name_storage)
        # self.storage_type = np.full(self.n_storage, fill_value="???")
        # self.storage_to_subid = self._grid.get_integer_value(Grid2opIntegerValueType.BATTERY_VOLTAGE_LEVEL_NUM).copy()

        # lines and transformers
        self.name_line = self._grid.get_string_value(  # pyright: ignore[reportAttributeAccessIssue]
            Grid2opStringValueType.BRANCH_NAME
        )
        self.n_line = len(self.name_line)  # pyright: ignore[reportAttributeAccessIssue]
        self.line_or_to_subid = self._grid.get_integer_value(  # pyright: ignore[reportAttributeAccessIssue]
            Grid2opIntegerValueType.BRANCH_VOLTAGE_LEVEL_NUM_1
        )
        self.line_ex_to_subid = self._grid.get_integer_value(  # pyright: ignore[reportAttributeAccessIssue]
            Grid2opIntegerValueType.BRANCH_VOLTAGE_LEVEL_NUM_2
        )

        self._compute_pos_big_topo()

        # thermal limits
        self.thermal_limit_a = self._grid.get_double_value(  # pyright: ignore[reportAttributeAccessIssue]
            Grid2opDoubleValueType.BRANCH_PERMANENT_LIMIT_A
        )

        # cached data
        self._gen_p = np.empty(self.n_gen, dtype=dt_float)
        self._gen_q = np.empty(self.n_gen, dtype=dt_float)
        self._gen_v = np.empty(self.n_gen, dtype=dt_float)

        self._load_p = np.empty(self.n_load, dtype=dt_float)
        self._load_q = np.empty(self.n_load, dtype=dt_float)
        self._load_v = np.empty(self.n_load, dtype=dt_float)

        self._por = np.empty(self.n_line, dtype=dt_float)
        self._qor = np.empty(self.n_line, dtype=dt_float)
        self._aor = np.empty(self.n_line, dtype=dt_float)
        self._vor = np.empty(self.n_line, dtype=dt_float)

        self._pex = np.empty(self.n_line, dtype=dt_float)
        self._qex = np.empty(self.n_line, dtype=dt_float)
        self._aex = np.empty(self.n_line, dtype=dt_float)
        self._vex = np.empty(self.n_line, dtype=dt_float)

        self._shunt_p = np.empty(self.n_shunt, dtype=dt_float)
        self._shunt_q = np.empty(self.n_shunt, dtype=dt_float)
        self._shunt_v = np.empty(self.n_shunt, dtype=dt_float)
        self._shunt_bus = np.empty(self.n_shunt, dtype=dt_int)

        self._gen_theta = np.empty(self.n_gen, dtype=dt_float)
        self._load_theta = np.empty(self.n_load, dtype=dt_float)
        self._line_or_theta = np.empty(self.n_line, dtype=dt_float)
        self._line_ex_theta = np.empty(self.n_line, dtype=dt_float)
        self._storage_theta = np.empty(self.n_storage, dtype=dt_float)

        self._topo_vect = np.empty(self.dim_topo, dtype=dt_int)
        self.fetch_data()

    def apply_action(
        self,
        backend_action: Union[_BackendAction, None],
    ) -> None:
        # the following few lines are highly recommended
        if backend_action is None:
            return

        logger.debug("Applying action to grid...")

        start_time = time.time()

        self._grid.update_double_value(
            Grid2opUpdateDoubleValueType.UPDATE_LOAD_P,
            backend_action.load_p.values,
            backend_action.load_p.changed,
        )
        self._grid.update_double_value(
            Grid2opUpdateDoubleValueType.UPDATE_LOAD_Q,
            backend_action.load_q.values,
            backend_action.load_q.changed,
        )
        self._grid.update_double_value(
            Grid2opUpdateDoubleValueType.UPDATE_GENERATOR_P,
            backend_action.prod_p.values,
            backend_action.prod_p.changed,
        )
        self._grid.update_double_value(
            Grid2opUpdateDoubleValueType.UPDATE_GENERATOR_V,
            backend_action.prod_v.values,
            backend_action.prod_v.changed,
        )
        # TODO shunts

        loads_bus = backend_action.get_loads_bus()
        self._grid.update_integer_value(
            Grid2opUpdateIntegerValueType.UPDATE_LOAD_BUS,
            loads_bus.values,
            loads_bus.changed,
        )
        generators_bus = backend_action.get_gens_bus()
        self._grid.update_integer_value(
            Grid2opUpdateIntegerValueType.UPDATE_GENERATOR_BUS,
            generators_bus.values,
            generators_bus.changed,
        )
        shunt_bus = backend_action.shunt_bus
        self._grid.update_integer_value(
            Grid2opUpdateIntegerValueType.UPDATE_SHUNT_BUS,
            shunt_bus.values,
            shunt_bus.changed,
        )
        lines_or_bus = backend_action.get_lines_or_bus()
        self._grid.update_integer_value(
            Grid2opUpdateIntegerValueType.UPDATE_BRANCH_BUS1,
            lines_or_bus.values,
            lines_or_bus.changed,
        )
        lines_ex_bus = backend_action.get_lines_ex_bus()
        self._grid.update_integer_value(
            Grid2opUpdateIntegerValueType.UPDATE_BRANCH_BUS2,
            lines_ex_bus.values,
            lines_ex_bus.changed,
        )

        end_time = time.time()
        elapsed_time = (end_time - start_time) * 1000
        logger.debug(f"Action applied in {elapsed_time:.2f} ms")

    @staticmethod
    def _is_converged(result: ComponentResult) -> bool:
        return (
            result.status == LoadFlowComponentStatus.CONVERGED
            or result.status == LoadFlowComponentStatus.NO_CALCULATION
        )

    def runpf(self, is_dc: bool = False) -> Tuple[bool, Union[Exception, None]]:
        logger.debug(f"Running {'DC' if is_dc else 'AC'} powerflow")

        start_time = time.perf_counter()

        if (
            self._check_isolated_and_disconnected_injections
            and self._grid.check_isolated_and_disconnected_injections()
        ):
            converged = False
            converged_msg = f"Issue with _check_isolated_and_disconnected_injections : {self._check_isolated_and_disconnected_injections} and {self._grid.check_isolated_and_disconnected_injections()}"
        else:
            beg_ = time.perf_counter()
            results = self._grid.run_pf(is_dc, self._lf_parameters)
            end_ = time.perf_counter()
            self.comp_time += end_ - beg_
            converged = self._is_converged(results[0])
            converged_msg = results[0].status_text

        if not converged:
            self.set_all_nans()
        else:
            self.fetch_data()

        end_time = time.perf_counter()
        elapsed_time = (end_time - start_time) * 1000
        logger.info(f"Powerflow ran in {elapsed_time:.2f} ms")
        return converged, None if converged else DivergingPowerflow(converged_msg)

    def fetch_data(self):
        self._fetch_topo_vect()
        self._fetch_gen()
        self._fetch_load()
        self._fetch_line_or()
        self._fetch_line_ex()
        self._fetch_shunt()

    def set_all_nans(self):
        self._gen_p[:] = np.nan
        self._gen_q[:] = np.nan
        self._gen_v[:] = np.nan

        self._load_p[:] = np.nan
        self._load_q[:] = np.nan
        self._load_v[:] = np.nan

        self._por[:] = np.nan
        self._qor[:] = np.nan
        self._aor[:] = np.nan
        self._vor[:] = np.nan

        self._pex[:] = np.nan
        self._qex[:] = np.nan
        self._aex[:] = np.nan
        self._vex[:] = np.nan

        self._shunt_p[:] = np.nan
        self._shunt_q[:] = np.nan
        self._shunt_v[:] = np.nan
        self._shunt_bus[:] = -1

        self._gen_theta[:] = np.nan
        self._load_theta[:] = np.nan
        self._line_or_theta[:] = np.nan
        self._line_ex_theta[:] = np.nan
        self._storage_theta[:] = np.nan

        self._topo_vect[:] = -1

    def get_topo_vect(self) -> np.ndarray:
        return 1 * self._topo_vect

    def _fetch_topo_vect(self):
        self._topo_vect[:] = self._grid.get_integer_value(
            Grid2opIntegerValueType.TOPO_VECT
        )

    def generators_info(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return 1 * self._gen_p, 1 * self._gen_q, 1 * self._gen_v

    def _fetch_gen(self):
        self._gen_p = self._grid.get_double_value(Grid2opDoubleValueType.GENERATOR_P)
        self._gen_q = self._grid.get_double_value(Grid2opDoubleValueType.GENERATOR_Q)
        self._gen_v = self._grid.get_double_value(Grid2opDoubleValueType.GENERATOR_V)
        self._gen_theta = self._grid.get_double_value(
            Grid2opDoubleValueType.GENERATOR_ANGLE
        )

    def loads_info(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return 1.0 * self._load_p, 1.0 * self._load_q, 1.0 * self._load_v

    def _fetch_load(self):
        self._load_p[:] = self._grid.get_double_value(Grid2opDoubleValueType.LOAD_P)
        self._load_q[:] = self._grid.get_double_value(Grid2opDoubleValueType.LOAD_Q)
        self._load_v[:] = self._grid.get_double_value(Grid2opDoubleValueType.LOAD_V)
        self._load_theta[:] = self._grid.get_double_value(
            Grid2opDoubleValueType.LOAD_ANGLE
        )

    def shunt_info(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return (
            1.0 * self._shunt_p,
            1.0 * self._shunt_q,
            1.0 * self._shunt_v,
            1 * self._shunt_bus,
        )

    def _fetch_shunt(self):
        self._shunt_p[:] = self._grid.get_double_value(Grid2opDoubleValueType.SHUNT_P)
        self._shunt_q[:] = self._grid.get_double_value(Grid2opDoubleValueType.SHUNT_Q)
        self._shunt_v[:] = self._grid.get_double_value(Grid2opDoubleValueType.SHUNT_V)
        self._shunt_bus[:] = self._grid.get_integer_value(
            Grid2opIntegerValueType.SHUNT_LOCAL_BUS
        )

    def lines_or_info(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return 1.0 * self._por, 1.0 * self._qor, 1.0 * self._vor, 1.0 * self._aor

    def _fetch_line_or(self):
        self._por[:] = self._grid.get_double_value(Grid2opDoubleValueType.BRANCH_P1)
        self._qor[:] = self._grid.get_double_value(Grid2opDoubleValueType.BRANCH_Q1)
        self._vor[:] = self._grid.get_double_value(Grid2opDoubleValueType.BRANCH_V1)
        self._aor[:] = self._grid.get_double_value(Grid2opDoubleValueType.BRANCH_I1)
        self._line_or_theta[:] = self._grid.get_double_value(
            Grid2opDoubleValueType.BRANCH_ANGLE1
        )

    def lines_ex_info(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return 1.0 * self._pex, 1.0 * self._qex, 1.0 * self._vex, 1.0 * self._aex

    def _fetch_line_ex(self):
        self._pex[:] = self._grid.get_double_value(Grid2opDoubleValueType.BRANCH_P2)
        self._qex[:] = self._grid.get_double_value(Grid2opDoubleValueType.BRANCH_Q2)
        self._vex[:] = self._grid.get_double_value(Grid2opDoubleValueType.BRANCH_V2)
        self._aex[:] = self._grid.get_double_value(Grid2opDoubleValueType.BRANCH_I2)
        self._line_ex_theta[:] = self._grid.get_double_value(
            Grid2opDoubleValueType.BRANCH_ANGLE2
        )

    def get_theta(  # type: ignore We return the storage as well
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return (
            1.0 * self._line_or_theta,
            1.0 * self._line_ex_theta,
            1.0 * self._load_theta,
            1.0 * self._gen_theta,
            1.0 * self._storage_theta,
        )

    def reset(
        self,
        path: Union[os.PathLike, str],
        grid_filename: Optional[Union[os.PathLike, str]] = None,
    ) -> None:
        logger.info(
            f"Backend is being reset and grid will be reloaded from path {path}"
        )
        self.load_grid(path, filename=grid_filename)

    def close(self) -> None:
        if self._grid:
            self._grid.close()
            self._grid = None

        self.set_all_nans()
