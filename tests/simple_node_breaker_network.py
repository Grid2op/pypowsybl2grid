# Copyright (c) 2024, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import logging

import pandas as pd
import pypowsybl as pp
import pytest
from pypowsybl.network import Network

@pytest.fixture(autouse=True)
def setup():
    logging.basicConfig()
    logging.getLogger('powsybl').setLevel(logging.INFO)

#
#    GEN1                   LOAD1
#     |                      |
#   BBS1A --- COUPLER1 --- BBS1B
#     |                      |
#     |                      |
#   LINE12                 LINE13
#    |                       |
#    |                       |
#  BBS2 ----- LINE23 ----- BBS3
#    |                       |
#  LOAD2                   LOAD3
def create_simple_node_breaker_network() -> Network:
    network = pp.network.create_empty("Simple node breaker network")

    substations = pd.DataFrame.from_records(index='id', data=[
        {'id': 'S1'},
        {'id': 'S2'},
        {'id': 'S3'}
    ])
    network.create_substations(substations)

    voltage_levels = pd.DataFrame.from_records(index='id', data=[
        {'substation_id': 'S1', 'id': 'VL1', 'topology_kind': 'NODE_BREAKER', 'nominal_v': 400},
        {'substation_id': 'S2', 'id': 'VL2', 'topology_kind': 'NODE_BREAKER', 'nominal_v': 400},
        {'substation_id': 'S3', 'id': 'VL3', 'topology_kind': 'NODE_BREAKER', 'nominal_v': 400},
    ])
    network.create_voltage_levels(voltage_levels)

    busbar_sections = pd.DataFrame.from_records(index='id', data=[
        {'voltage_level_id': 'VL1', 'id': 'BBS1A', 'node': 0},
        {'voltage_level_id': 'VL1', 'id': 'BBS1B', 'node': 1},
        {'voltage_level_id': 'VL2', 'id': 'BBS2', 'node': 0},
        {'voltage_level_id': 'VL3', 'id': 'BBS3', 'node': 0},
    ])
    network.create_busbar_sections(busbar_sections)

    pp.network.create_coupling_device(network, switch_prefix_id='COUPLER1', bus_or_busbar_section_id_1=['BBS1A'],
                                      bus_or_busbar_section_id_2=['BBS1B'])

    loads = pd.DataFrame.from_records(index='id', data=[
        {'id': 'LOAD1', 'p0': 10.0, 'q0': 3.0, 'bus_or_busbar_section_id': 'BBS1B', 'position_order': 1, 'direction': 'BOTTOM'},
        {'id': 'LOAD2', 'p0': 11.0, 'q0': 4.0, 'bus_or_busbar_section_id': 'BBS2', 'position_order': 1, 'direction': 'BOTTOM'},
        {'id': 'LOAD3', 'p0': 12.0, 'q0': 5.0, 'bus_or_busbar_section_id': 'BBS3', 'position_order': 1, 'direction': 'BOTTOM'},
    ])
    pp.network.create_load_bay(network, loads)

    generators = pd.DataFrame.from_records(index='id', data=[
        {'id': 'GEN1', 'max_p': 100, 'min_p': 0, 'voltage_regulator_on': True, 'target_p': 33, 'target_q': 0, 'target_v': 403, 'bus_or_busbar_section_id': 'BBS1A', 'position_order': 1}
    ])
    pp.network.create_generator_bay(network, generators)

    lines = pd.DataFrame.from_records(index='id', data=[
        {'id': 'LINE12', 'r': 1.1, 'x': 20, 'g1': 0, 'b1': 0, 'g2': 0, 'b2': 0,
         'bus_or_busbar_section_id_1': 'BBS1A', 'position_order_1': 2, 'direction_1': 'BOTTOM',
         'bus_or_busbar_section_id_2': 'BBS2', 'position_order_2': 2, 'direction_2': 'TOP'},
        {'id': 'LINE13', 'r': 0.8, 'x': 16, 'g1': 0, 'b1': 0, 'g2': 0, 'b2': 0,
         'bus_or_busbar_section_id_1': 'BBS1B', 'position_order_1': 2, 'direction_1': 'BOTTOM',
         'bus_or_busbar_section_id_2': 'BBS3', 'position_order_2': 2, 'direction_2': 'TOP'},
        {'id': 'LINE23', 'r': 1.5, 'x': 23, 'g1': 0, 'b1': 0, 'g2': 0, 'b2': 0,
         'bus_or_busbar_section_id_1': 'BBS2', 'position_order_1': 3, 'direction_1': 'TOP',
         'bus_or_busbar_section_id_2': 'BBS3', 'position_order_2': 3, 'direction_2': 'TOP'}
    ])
    pp.network.create_line_bays(network, lines)

    return network


def test_simple_node_breaker_network():
    network = create_simple_node_breaker_network()
#    network.save('/tmp/toto.xiidm', 'XIIDM')
    results = pp.loadflow.run_ac(network)
    assert 1 == len(results)
    assert pp.loadflow.ComponentStatus.CONVERGED == results[0].status
    assert 2 == results[0].iteration_count
