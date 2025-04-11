# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0


import logging

import pytest
from pypowsybl2grid import PyPowSyBlBackend
import grid2op
import warnings
from grid2op.Chronics import ChangeNothing
from grid2op.Opponent import get_kwargs_no_opponent

@pytest.fixture(autouse=True)
def setup():
    logging.basicConfig()
    logging.getLogger('powsybl').setLevel(logging.ERROR)


def test_can_load_2_envs():
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        env1 = grid2op.make("l2rpn_case14_sandbox", test=True, backend=PyPowSyBlBackend())
        env2 = grid2op.make("l2rpn_neurips_2020_track2", test=True, backend=PyPowSyBlBackend(), chronics_class=ChangeNothing)
    obs1 = env1.reset(seed=0, options={"time serie id": 0})
    assert abs(obs1.a_or.sum() - 4973.0478515625) <= 1e-6
    obs2 = env2.reset(seed=0, options={"time serie id": 0})
    assert abs(obs2.a_or.sum() - 23059.88671875) <= 1e-6

def test_can_load_attached_env_nochron_noopp():
    ref_data = {
        "educ_case14_redisp" : 5529.5869140625,
        "educ_case14_storage" : 5529.5869140625,
        "l2rpn_case14_sandbox" : 5529.5869140625,
        "l2rpn_case14_sandbox_diff_grid" : 5529.5869140625,
        "l2rpn_neurips_2020_track2": 23059.887,
        "l2rpn_wcci_2020": 27368.453125,
        "l2rpn_wcci_2022": 27368.453125,
        "l2rpn_wcci_2022_dev":27368.453125,
        "rte_case118_example": 23380.45,
        "rte_case14_opponent": 5529.587,
        "rte_case14_realistic": 5529.587,
        "rte_case14_redisp": 171647.05,
        "rte_case14_test": 171647.05,

    }
    fail_ = set()
    fail_.add("l2rpn_icaps_2021")  # issue with the alarm data (line name)
    fail_.add("l2rpn_idf_2023")  # issue with the alert data (line name)
    fail_.add("l2rpn_neurips_2020_track1")  # issue with gen cost
    fail_.add("rte_case5_example")  # maximum reactive power not set
    for el_nm in grid2op.list_available_test_env():
        if el_nm in fail_:
            continue
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env = grid2op.make(el_nm,
                               test=True,
                               backend=PyPowSyBlBackend(),
                               chronics_class=ChangeNothing,  # element name might be different
                               **get_kwargs_no_opponent(),  # line name might be different
                               )
            obs = env.reset(seed=0, options={"time serie id": 0})
            assert (obs.a_or.sum() - ref_data[el_nm]) <= 1e-5, f"error for {el_nm}: {obs.a_or.sum()} vs {ref_data[el_nm]}"
            
            
if __name__ == "__main__":
    test_can_load_attached_env_nochron_noopp()
        