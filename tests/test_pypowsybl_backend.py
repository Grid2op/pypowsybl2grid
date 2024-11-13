import logging
import tempfile
import unittest
from typing import Dict

import pypowsybl as pp
import pytest
from grid2op.dtypes import dt_float
from grid2op.tests.aaa_test_backend_interface import AAATestBackendAPI

from pypowsybl2grid.pypowsybl_backend import PyPowSyBlBackend

import numpy as np
import numpy.testing as npt

from tests.simple_node_breaker_network import create_simple_node_breaker_network

# needed config to make some grid2op test passing
TEST_LOADFLOW_PARAMETERS = pp.loadflow.Parameters(distributed_slack=True,
                                                  use_reactive_limits=False,
                                                  provider_parameters={"slackBusPMaxMismatch": "1e-2",
                                                                       "newtonRaphsonConvEpsPerEq": "1e-7",
                                                                       "useActiveLimits": "false"})
TOLERANCE = 1e-3

@pytest.fixture(autouse=True)
def setup():
    logging.basicConfig()
    logging.getLogger('powsybl').setLevel(logging.INFO)


class TestBackendPyPowSyBl(AAATestBackendAPI, unittest.TestCase):

    def make_backend(self, detailed_infos_for_cascading_failures=False):
        self.tol_one = dt_float(1e-3)
        return PyPowSyBlBackend(detailed_infos_for_cascading_failures=detailed_infos_for_cascading_failures, lf_parameters=TEST_LOADFLOW_PARAMETERS)


def apply_action(backend: PyPowSyBlBackend, action_dict: Dict):
    action = type(backend)._complete_action_class()
    action.update(action_dict)
    bk_act = type(backend).my_bk_act_class()
    bk_act += action
    backend.apply_action(bk_act)


def test_backend_with_node_breaker_network():
    backend = PyPowSyBlBackend(check_isolated_and_disconnected_injections=False, consider_open_branch_reactive_flow=True)

    # backend need to grid as a file, dump it in a temporary folder
    n = create_simple_node_breaker_network()
    with tempfile.TemporaryDirectory() as tmp_dir_name:
        grid_file = tmp_dir_name + "grid.xiidm"
        n.save(grid_file, 'XIIDM')
        backend.load_grid(grid_file)

    backend.assert_grid_correct()

    conv, _ = backend.runpf()
    assert conv

    p_or, q_or, v_or, _ = backend.lines_or_info()
    p_ex, q_ex, v_ex, _ = backend.lines_ex_info()
    assert ['LINE12', 'LINE13', 'LINE23'] == backend.name_line.tolist()
    npt.assert_allclose(np.array([10.532, 12.469, -0.468]), p_or, rtol=TOLERANCE, atol=TOLERANCE)
    npt.assert_allclose(np.array([4.0, 5.033, -0.015]), q_or, rtol=TOLERANCE, atol=TOLERANCE)
    npt.assert_allclose(np.array([403.0, 403.0, 402.881]), v_or, rtol=TOLERANCE, atol=TOLERANCE)
    npt.assert_allclose(np.array([-10.531, -12.468, 0.468]), p_ex, rtol=TOLERANCE, atol=TOLERANCE)
    npt.assert_allclose(np.array([-3.984, -5.015, 0.015]), q_ex, rtol=TOLERANCE, atol=TOLERANCE)
    npt.assert_allclose(np.array([402.773, 402.775, 402.775]), v_ex, rtol=TOLERANCE, atol=TOLERANCE)

    # disconnect line 12
    apply_action(backend, {"set_line_status": [(0, -1)]})

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

    # reconnect line 12
    apply_action(backend, {"set_line_status": [(0, 1)]})

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
