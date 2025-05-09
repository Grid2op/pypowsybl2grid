# Copyright (c) 2024, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import logging
import tempfile
import uuid
from typing import Dict

import numpy as np
import numpy.testing as npt
import pypowsybl as pp
import pytest
from pypowsybl.network import Network

from pypowsybl2grid.pypowsybl_backend import PyPowSyBlBackend, DEFAULT_LF_PARAMETERS
from tests.simple_node_breaker_network import create_simple_node_breaker_network

TOLERANCE = 1e-3

@pytest.fixture(autouse=True)
def setup():
    logging.basicConfig()
    logging.getLogger('powsybl').setLevel(logging.INFO)


@pytest.fixture
def backend():
    backend = create_backend()
    # we need to set a different environment name for each test to avoid side effects
    type(backend).set_env_name('backend_' + str(uuid.uuid4()))
    yield backend
    backend.close()


@pytest.fixture
def backend_advanced_lf_parameters():
    parameters = pp.loadflow.Parameters(voltage_init_mode=pp.loadflow.VoltageInitMode.DC_VALUES, balance_type=pp.loadflow.BalanceType.PROPORTIONAL_TO_LOAD)
    backend = create_backend(parameters)
    yield backend
    backend.close()


def create_backend(lf_parameters: pp.loadflow.Parameters = DEFAULT_LF_PARAMETERS):
    return PyPowSyBlBackend(check_isolated_and_disconnected_injections=False,
                            consider_open_branch_reactive_flow=True,
                            connect_all_elements_to_first_bus=False,
                            lf_parameters=lf_parameters)


def load_grid(backend: PyPowSyBlBackend, network: Network):
    assert backend.network is None

    # backend need to grid as a file, dump it in a temporary folder
    with tempfile.TemporaryDirectory() as tmp_dir_name:
        grid_file = tmp_dir_name + "grid.xiidm"
        network.save(grid_file, 'XIIDM')
        backend.load_grid(grid_file)

    assert backend.network is not None

    backend.assert_grid_correct()


def apply_action(backend: PyPowSyBlBackend, action_dict: Dict):
    action = type(backend)._complete_action_class()
    action.update(action_dict)
    bk_act = type(backend).my_bk_act_class()
    bk_act += action
    backend.apply_action(bk_act)


def test_backend_with_node_breaker_network(backend):
    n = create_simple_node_breaker_network()
    load_grid(backend, n)

    conv, _ = backend.runpf()
    assert conv

    p_or, q_or, v_or, _ = backend.lines_or_info()
    p_ex, q_ex, v_ex, _ = backend.lines_ex_info()
    assert ['LINE12', 'LINE13', 'LINE23'] == backend.name_line.tolist()
    assert [1, 1, 1, 1, 1, 1, 1, 1, 1, 1] == backend.get_topo_vect().tolist()
    npt.assert_allclose(np.array([10.532, 12.469, -0.468]), p_or, rtol=TOLERANCE, atol=TOLERANCE)
    npt.assert_allclose(np.array([4.0, 5.033, -0.015]), q_or, rtol=TOLERANCE, atol=TOLERANCE)
    npt.assert_allclose(np.array([403.0, 403.0, 402.881]), v_or, rtol=TOLERANCE, atol=TOLERANCE)
    npt.assert_allclose(np.array([-10.531, -12.468, 0.468]), p_ex, rtol=TOLERANCE, atol=TOLERANCE)
    npt.assert_allclose(np.array([-3.984, -5.015, 0.015]), q_ex, rtol=TOLERANCE, atol=TOLERANCE)
    npt.assert_allclose(np.array([402.773, 402.775, 402.775]), v_ex, rtol=TOLERANCE, atol=TOLERANCE)
    assert [1, 1, 1, 1, 1, 1, 1, 1, 1, 1] == backend.get_topo_vect().tolist()

    # disconnect line 12
    line12_num = 0
    apply_action(backend, {"set_line_status": [(line12_num, -1)]})
    backend.fetch_data()
    topo_vect = backend.get_topo_vect()
    assert topo_vect[backend.line_or_pos_topo_vect[line12_num]] == -1
    assert topo_vect[backend.line_ex_pos_topo_vect[line12_num]] == -1
    assert [1, 1, -1, 1, 1, 1, -1, 1, 1, 1] == topo_vect.tolist()

    conv, _ = backend.runpf()
    assert conv

    p_or, q_or, v_or, _ = backend.lines_or_info()
    p_ex, q_ex, v_ex, _ = backend.lines_ex_info()
    npt.assert_allclose(np.array([0.0,  23.004, -11.0]), p_or, rtol=TOLERANCE, atol=TOLERANCE)
    npt.assert_allclose(np.array([0.0,  9.079, -4.0]), q_or, rtol=TOLERANCE, atol=TOLERANCE)
    npt.assert_allclose(np.array([0.0, 403.0, 402.324693]), v_or, rtol=TOLERANCE, atol=TOLERANCE)
    npt.assert_allclose(np.array([0.0, -23.001,  11.001]), p_ex, rtol=TOLERANCE, atol=TOLERANCE)
    npt.assert_allclose(np.array([0.0, -9.019,  4.019]), q_ex, rtol=TOLERANCE, atol=TOLERANCE)
    npt.assert_allclose(np.array([0.0, 402.594, 402.594]), v_ex, rtol=TOLERANCE, atol=TOLERANCE)
    topo_vect = backend.get_topo_vect()
    assert topo_vect[backend.line_or_pos_topo_vect[line12_num]] == -1
    assert topo_vect[backend.line_ex_pos_topo_vect[line12_num]] == -1
    assert [1, 1, -1, 1, 1, 1, -1, 1, 1, 1] == topo_vect.tolist()

    # reconnect line 12
    apply_action(backend, {"set_line_status": [(line12_num, 1)]})
    backend.fetch_data()
    assert [1, 1, 1, 1, 1, 1, 1, 1, 1, 1] == backend.get_topo_vect().tolist()

    conv, _ = backend.runpf()
    assert conv

    p_or, q_or, v_or, _ = backend.lines_or_info()
    p_ex, q_ex, v_ex, _ = backend.lines_ex_info()
    npt.assert_allclose(np.array([10.532, 12.469, -0.468]), p_or, rtol=TOLERANCE, atol=TOLERANCE)
    npt.assert_allclose(np.array([4.0, 5.033, -0.015]), q_or, rtol=TOLERANCE, atol=TOLERANCE)
    npt.assert_allclose(np.array([403.0, 403.0, 402.881]), v_or, rtol=TOLERANCE, atol=TOLERANCE)
    npt.assert_allclose(np.array([-10.531, -12.468, 0.468]), p_ex, rtol=TOLERANCE, atol=TOLERANCE)
    npt.assert_allclose(np.array([-3.984, -5.015, 0.015]), q_ex, rtol=TOLERANCE, atol=TOLERANCE)
    npt.assert_allclose(np.array([402.773, 402.775, 402.775]), v_ex, rtol=TOLERANCE, atol=TOLERANCE)
    assert [1, 1, 1, 1, 1, 1, 1, 1, 1, 1] == backend.get_topo_vect().tolist()

    # connect line 13 to bbs 1 of VL1 instead of bss2
    assert ['VL1', 'VL2', 'VL3'] == backend.name_sub.tolist()
    apply_action(backend, {'set_bus': {'lines_or_id': {'LINE12': 2}}})
    backend.fetch_data()
    assert [1, 1, 2, 1, 1, 1, 1, 1, 1, 1] == backend.get_topo_vect().tolist()

    conv, _ = backend.runpf()
    assert conv

    p_or, q_or, v_or, _ = backend.lines_or_info()
    p_ex, q_ex, v_ex, _ = backend.lines_ex_info()
    npt.assert_allclose(np.array([0.0,  23.004, -11.0]), p_or, rtol=TOLERANCE, atol=TOLERANCE)
    npt.assert_allclose(np.array([0.0,  9.079, -4.0]), q_or, rtol=TOLERANCE, atol=TOLERANCE)
    npt.assert_allclose(np.array([402.324, 403.0, 402.324693]), v_or, rtol=TOLERANCE, atol=TOLERANCE)
    npt.assert_allclose(np.array([0.0, -23.001,  11.001]), p_ex, rtol=TOLERANCE, atol=TOLERANCE)
    npt.assert_allclose(np.array([0.0, -9.019,  4.019]), q_ex, rtol=TOLERANCE, atol=TOLERANCE)
    npt.assert_allclose(np.array([402.324, 402.594, 402.594]), v_ex, rtol=TOLERANCE, atol=TOLERANCE)
    assert [1, 1, 2, 1, 1, 1, 1, 1, 1, 1] == backend.get_topo_vect().tolist()


def test_backend_with_node_breaker_network_and_an_initial_topo(backend):
    n = create_simple_node_breaker_network()
    # open the coupler to get 2 buses at VL1
    n.update_switches(id='COUPLER1_BREAKER', open=True)

    load_grid(backend, n)

    conv, _ = backend.runpf()
    assert conv

    # check line 13 origin and load1 are on busbar 2
    assert ['LOAD1', 'LOAD2', 'LOAD3'] == backend.name_load.tolist()
    topo_vect = backend.get_topo_vect()
    line13_num = 1
    load1_num = 0
    assert topo_vect[backend.line_or_pos_topo_vect[line13_num]] == 2
    assert topo_vect[backend.load_pos_topo_vect[load1_num]] == 2
    assert [2, 1, 1, 2, 1, 1, 1, 1, 1, 1] == backend.get_topo_vect().tolist()


def test_backend_with_node_breaker_network_and_load_change(backend):
    n = create_simple_node_breaker_network()

    load_grid(backend, n)

    conv, _ = backend.runpf()
    assert conv

    gen_p, gen_q, gen_v = backend.generators_info()
    npt.assert_allclose(np.array([33.0]), gen_p, rtol=TOLERANCE, atol=TOLERANCE)
    npt.assert_allclose(np.array([12.033]), gen_q, rtol=TOLERANCE, atol=TOLERANCE)
    npt.assert_allclose(np.array([403.0]), gen_v, rtol=TOLERANCE, atol=TOLERANCE)

    load_p, load_q, load_v = backend.loads_info()
    npt.assert_allclose(np.array([10.0, 11.0, 12.0]), load_p, rtol=TOLERANCE, atol=TOLERANCE)
    npt.assert_allclose(np.array([3.0, 4.0, 5.0]), load_q, rtol=TOLERANCE, atol=TOLERANCE)
    npt.assert_allclose(np.array([403.0, 402.773, 402.775]), load_v, rtol=TOLERANCE, atol=TOLERANCE)

    apply_action(backend, {"injection": {"load_p": [10.0, 11.0, 30.0],
                                                   "load_q": [3.0, 4.0, 5.0]}})

    conv, _ = backend.runpf()
    assert conv

    gen_p, gen_q, gen_v = backend.generators_info()
    npt.assert_allclose(np.array([51.005]), gen_p, rtol=TOLERANCE, atol=TOLERANCE)
    npt.assert_allclose(np.array([12.101]), gen_q, rtol=TOLERANCE, atol=TOLERANCE)
    npt.assert_allclose(np.array([403.0]), gen_v, rtol=TOLERANCE, atol=TOLERANCE)

    load_p, load_q, load_v = backend.loads_info()
    npt.assert_allclose(np.array([10.0, 11.0, 30.0]), load_p, rtol=TOLERANCE, atol=TOLERANCE)
    npt.assert_allclose(np.array([3.0, 4.0, 5.0]), load_q, rtol=TOLERANCE, atol=TOLERANCE)
    npt.assert_allclose(np.array([403.0, 402.773, 402.775]), load_v, rtol=TOLERANCE, atol=TOLERANCE)


def test_backend_with_node_breaker_network_and_generation_change(backend_advanced_lf_parameters):
    n = create_simple_node_breaker_network()

    load_grid(backend_advanced_lf_parameters, n)

    conv, _ = backend_advanced_lf_parameters.runpf()
    assert conv

    gen_p, gen_q, gen_v = backend_advanced_lf_parameters.generators_info()
    npt.assert_allclose(np.array([33.0]), gen_p, rtol=TOLERANCE, atol=TOLERANCE)
    npt.assert_allclose(np.array([12.033]), gen_q, rtol=TOLERANCE, atol=TOLERANCE)
    npt.assert_allclose(np.array([403.0]), gen_v, rtol=TOLERANCE, atol=TOLERANCE)

    load_p, load_q, load_v = backend_advanced_lf_parameters.loads_info()
    npt.assert_allclose(np.array([10.0, 11.0, 12.0]), load_p, rtol=TOLERANCE, atol=TOLERANCE)
    npt.assert_allclose(np.array([3.0, 4.0, 5.0]), load_q, rtol=TOLERANCE, atol=TOLERANCE)
    npt.assert_allclose(np.array([403.0, 402.773, 402.775]), load_v, rtol=TOLERANCE, atol=TOLERANCE)

    apply_action(backend_advanced_lf_parameters, {"injection": {"prod_p": [43.0],
                                                   "prod_v": [403.0]}})

    conv, _ = backend_advanced_lf_parameters.runpf()
    assert conv

    gen_p, gen_q, gen_v = backend_advanced_lf_parameters.generators_info()
    npt.assert_allclose(np.array([43.0]), gen_p, rtol=TOLERANCE, atol=TOLERANCE)
    npt.assert_allclose(np.array([12.052]), gen_q, rtol=TOLERANCE, atol=TOLERANCE)
    npt.assert_allclose(np.array([403.0]), gen_v, rtol=TOLERANCE, atol=TOLERANCE)

    load_p, load_q, load_v = backend_advanced_lf_parameters.loads_info()
    npt.assert_allclose(np.array([13.029, 14.332, 15.635]), load_p, rtol=TOLERANCE, atol=TOLERANCE)
    npt.assert_allclose(np.array([3.0, 4.0, 5.0]), load_q, rtol=TOLERANCE, atol=TOLERANCE)
    npt.assert_allclose(np.array([403.0, 402.773, 402.775]), load_v, rtol=TOLERANCE, atol=TOLERANCE)

    apply_action(backend_advanced_lf_parameters, {"injection": {"prod_p": [33.0],
                                                   "prod_v": [410.0]}})

    conv, _ = backend_advanced_lf_parameters.runpf()
    assert conv

    gen_p, gen_q, gen_v = backend_advanced_lf_parameters.generators_info()
    npt.assert_allclose(np.array([33.0]), gen_p, rtol=TOLERANCE, atol=TOLERANCE)
    npt.assert_allclose(np.array([12.032]), gen_q, rtol=TOLERANCE, atol=TOLERANCE)
    npt.assert_allclose(np.array([410.0]), gen_v, rtol=TOLERANCE, atol=TOLERANCE)

    load_p, load_q, load_v = backend_advanced_lf_parameters.loads_info()
    npt.assert_allclose(np.array([10.0, 11.0, 12.0]), load_p, rtol=TOLERANCE, atol=TOLERANCE)
    npt.assert_allclose(np.array([3.0, 4.0, 5.0]), load_q, rtol=TOLERANCE, atol=TOLERANCE)
    npt.assert_allclose(np.array([410.0, 409.776, 409.779]), load_v, rtol=TOLERANCE, atol=TOLERANCE)


def test_iidm_network_update(backend):
    n = create_simple_node_breaker_network()

    load_grid(backend, n)

    conv, _ = backend.runpf()
    assert conv

    loads = backend.network.get_loads(attributes=['p0'])
    assert [10.0, 11.0, 12.0] == list(loads['p0'])

    load_p, _, _ = backend.loads_info()
    npt.assert_allclose(np.array([10.0, 11.0, 12.0]), load_p, rtol=TOLERANCE, atol=TOLERANCE)

    apply_action(backend, {"injection": {"load_p": [11.0, 12.0, 13.0]}})

    conv, _ = backend.runpf()
    assert conv

    load_p, _, _ = backend.loads_info()
    npt.assert_allclose(np.array([11.0, 12.0, 13.0]), load_p, rtol=TOLERANCE, atol=TOLERANCE)

    # check IIDM network is up-to-date
    loads = backend.network.get_loads(attributes=['p0'])
    assert [11.0, 12.0, 13.0] == list(loads['p0'])


@pytest.mark.skip(reason="To fix")
def test_backend_with_bus_breaker_network(backend):
    n = pp.network.create_eurostag_tutorial_example1_network()
    load_grid(backend, n)

    conv, _ = backend.runpf()
    assert conv

    apply_action(backend, {'set_bus': {'lines_or_id': {'NHV1_NHV2_1': 2}}})

    conv, _ = backend.runpf()
    assert conv

    p_or, _, _, _ = backend.lines_or_info()
    npt.assert_allclose(np.array([-0.049, 609.595, 610.329, 600.949]), p_or, rtol=TOLERANCE, atol=TOLERANCE)


def test_backend_copy(backend):
    n = pp.network.create_eurostag_tutorial_example1_network()
    load_grid(backend, n)
    backend_cpy = backend.copy()
    assert isinstance(backend_cpy, type(backend))
    backend_cpy.close()


def test_gen_detachment(backend):
    n = pp.network.create_ieee14()
    load_grid(backend, n)

    assert [1] * 56 == backend.get_topo_vect().tolist()

    conv, _ = backend.runpf()
    assert conv

    apply_action(backend, {'set_line_status': {'L7-8-1': -1}})
    backend.fetch_data()

    conv, _ = backend.runpf()
    assert conv

    assert [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] == backend.get_topo_vect().tolist()


def test_backend_with_theta(backend):
    n = pp.network.create_eurostag_tutorial_example1_network()
    load_grid(backend, n)

    conv, _ = backend.runpf()
    assert conv

    assert (np.array([ 0.0, 0.0, 0.04059613, -0.06119749]),
            np.array([-0.06119749, -0.06119749, 0.0, -0.16780443]),
            np.array([-0.16780443]),
            np.array([0.04059613, 0.04059613]),
            np.array([]))
