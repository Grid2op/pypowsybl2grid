import logging
import unittest

import pypowsybl as pp
import pytest
from grid2op.dtypes import dt_float
from grid2op.tests.aaa_test_backend_interface import AAATestBackendAPI

from pypowsybl2grid.pypowsybl_backend import PyPowSyBlBackend

# needed config to make some grid2op test passing
TEST_LOADFLOW_PARAMETERS = pp.loadflow.Parameters(distributed_slack=True,
                                                  use_reactive_limits=False,
                                                  provider_parameters={"slackBusPMaxMismatch": "1e-2",
                                                                       "newtonRaphsonConvEpsPerEq": "1e-7",
                                                                       "useActiveLimits": "false"})

@pytest.fixture(autouse=True)
def setup():
    logging.basicConfig()
    logging.getLogger('powsybl').setLevel(logging.INFO)


class TestBackendPyPowSyBl(AAATestBackendAPI, unittest.TestCase):

    def make_backend(self, detailed_infos_for_cascading_failures=False):
        self.tol_one = dt_float(1e-3)
        return PyPowSyBlBackend(detailed_infos_for_cascading_failures=detailed_infos_for_cascading_failures, lf_parameters=TEST_LOADFLOW_PARAMETERS)
