# Copyright (c) 2024, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import pandas as pd
import pypowsybl as pp
import pytest
from numpy import nan

from pypowsybl2grid.fast_network_cache import FastNetworkCacheFactory
from pypowsybl2grid.network_cache import NetworkCacheFactory
from pypowsybl2grid.simple_network_cache import SimpleNetworkCacheFactory

@pytest.fixture(autouse=True)
def setup():
    pd.options.display.max_columns = None
    pd.options.display.expand_frame_repr = False


def test_simple_network_cache():
    run_network_cache_test(SimpleNetworkCacheFactory())


def test_fast_network_cache():
    run_network_cache_test(FastNetworkCacheFactory())


def run_network_cache_test(network_cache_factory: NetworkCacheFactory):
    n = pp.network.create_eurostag_tutorial_example1_network()
    cache = network_cache_factory.create_network_cache(n)
    cache.create_buses(id='VLGEN_extra_busbar_1', voltage_level_id="VLGEN")
    cache.run_ac_pf()
    buses, buses_dict = cache.get_buses()
    expected_buses = pd.DataFrame(index=pd.Series(name='id', data=['NGEN', 'VLGEN_extra_busbar_1', 'NHV1', 'NHV2', 'NLOAD']),
                                  columns=['v_mag', 'synchronous_component', 'voltage_level_id', 'name_voltage_level', 'topology_kind_voltage_level', 'num_voltage_level', 'local_num', 'num'],
                                  data=[[24.500000,  0,      'VLGEN', '', 'BUS_BREAKER', 0, 0, 0],
                                        [nan,        -99999, 'VLGEN', '', 'BUS_BREAKER', 0, 1, 4],
                                        [402.142826, 0,      'VLHV1', '', 'BUS_BREAKER', 1, 0, 1],
                                        [389.952653, 0,      'VLHV2', '', 'BUS_BREAKER', 2, 0, 2],
                                        [147.578618, 0,      'VLLOAD','', 'BUS_BREAKER', 3, 0, 3]])
    pd.testing.assert_frame_equal(expected_buses, buses, check_dtype=False)

    assert {0: 'NGEN', 1: 'NHV1', 2: 'NHV2', 3: 'NLOAD', 4: 'VLGEN_extra_busbar_1'} == buses_dict

    voltage_levels = cache.get_voltage_levels()
    expected_voltage_levels = pd.DataFrame(index=pd.Series(name='id', data=['VLGEN', 'VLHV1', 'VLHV2', 'VLLOAD']),
                                           columns=['name', 'topology_kind', 'num'],
                                           data=[['', 'BUS_BREAKER', 0],
                                                 ['', 'BUS_BREAKER', 1],
                                                 ['', 'BUS_BREAKER', 2],
                                                 ['', 'BUS_BREAKER', 3]])
    pd.testing.assert_frame_equal(expected_voltage_levels, voltage_levels, check_dtype=False)

    loads = cache.get_loads()
    expected_loads = pd.DataFrame(index=pd.Series(name='id', data=['LOAD']),
                                  columns=['name', 'voltage_level_id', 'bus_breaker_bus_id', 'connected', 'p', 'q', 'num',
                                           'v_mag_bus', 'synchronous_component_bus', 'local_num_bus', 'num_bus'],
                                  data=[['', 'VLLOAD', 'NLOAD', True, 600.0, 200.0, 0, 147.578618, 0, 0, 3]])
    pd.testing.assert_frame_equal(expected_loads, loads, check_dtype=False)

    generators = cache.get_generators()
    expected_generators = pd.DataFrame(index=pd.Series(name='id', data=['GEN', 'GEN2']),
                                       columns=['name', 'voltage_level_id', 'bus_breaker_bus_id', 'connected', 'p', 'q', 'num',
                                           'v_mag_bus', 'synchronous_component_bus', 'local_num_bus', 'num_bus'],
                                       data=[['', 'VLGEN', 'NGEN', True, -302.780515, -112.64135, 0, 24.5, 0, 0, 0],
                                             ['', 'VLGEN', 'NGEN', True, -302.780515, -112.64135, 1, 24.5, 0, 0, 0]])
    pd.testing.assert_frame_equal(expected_generators, generators, check_dtype=False)

    shunts = cache.get_shunts()
    assert len(shunts) == 0

    branches = cache.get_branches()
    expected_branches = pd.DataFrame(index=pd.Series(name='id', data=['NHV1_NHV2_1', 'NHV1_NHV2_2', 'NGEN_NHV1', 'NHV2_NLOAD']),
                                       columns=['name', 'voltage_level1_id', 'voltage_level2_id', 'bus_breaker_bus1_id',
                                                'bus_breaker_bus2_id', 'connected1', 'connected2', 'p1', 'q1', 'i1',
                                                'p2', 'q2', 'i2', 'num', 'v_mag_bus1', 'synchronous_component_bus1', 'num_bus1', 'v_mag_bus2', 'synchronous_component_bus2', 'num_bus2'],
                                       data=[['', 'VLHV1', 'VLHV2',  'NHV1', 'NHV2',  True, True, 302.444049, 98.740275,  456.768978,   -300.433895, -137.188493, 488.992798,  0, 402.142826, 0, 1, 389.952653, 0, 2],
                                             ['', 'VLHV1', 'VLHV2',  'NHV1', 'NHV2',  True, True, 302.444049, 98.740275,  456.768978,   -300.433895, -137.188493, 488.992798,  1, 402.142826, 0, 1, 389.952653, 0, 2],
                                             ['', 'VLGEN', 'VLHV1',  'NGEN', 'NHV1',  True, True, 605.561014, 225.282699, 15225.756113, -604.893567, -197.480432, 913.545367,  2, 24.5,       0, 0, 402.142826, 0, 1],
                                             ['', 'VLHV2', 'VLLOAD', 'NHV2', 'NLOAD', True, True, 600.867790, 274.376987, 977.985596,   -600.0,      -200.0,      2474.263394, 3, 389.952653, 0, 2, 147.578618, 0, 3]])
    pd.testing.assert_frame_equal(expected_branches, branches, check_dtype=False)

    # test IDs
    assert ['LOAD'] == cache.get_load_ids()
    assert ['GEN', 'GEN2'] == cache.get_generator_ids()
    assert [] == cache.get_shunt_ids()
    assert ['NHV1_NHV2_1', 'NHV1_NHV2_2', 'NGEN_NHV1', 'NHV2_NLOAD'] == cache.get_branch_ids()

    # test load modification
    cache.update_load_p(['LOAD'], [700.0])
    cache.update_load_q(['LOAD'], [300.0])
    cache.run_ac_pf()
    loads = cache.get_loads()
    expected_loads = pd.DataFrame(index=pd.Series(name='id', data=['LOAD']),
                                  columns=['name', 'voltage_level_id', 'bus_breaker_bus_id', 'connected', 'p', 'q',
                                           'num',
                                           'v_mag_bus', 'synchronous_component_bus', 'local_num_bus', 'num_bus'],
                                  data=[['', 'VLLOAD', 'NLOAD', True, 700.0, 300.0, 0, 138.52392450257688, 0, 0, 3]])
    pd.testing.assert_frame_equal(expected_loads, loads, check_dtype=False)

    # test generator modification
    cache.update_generator_p(['GEN'], [310.0])
    cache.update_generator_v(['GEN'], [25.0])
    cache.run_ac_pf()
    generators = cache.get_generators()
    expected_generators = pd.DataFrame(index=pd.Series(name='id', data=['GEN', 'GEN2']),
                                       columns=['name', 'voltage_level_id', 'bus_breaker_bus_id', 'connected', 'p', 'q',
                                                'num', 'v_mag_bus', 'synchronous_component_bus', 'local_num_bus', 'num_bus'],
                                       data=[['', 'VLGEN', 'NGEN', True, -206.07521276657846, -202.2736774358548, 0, 25.0, 0, 0, 0],
                                             ['', 'VLGEN', 'NGEN', True, -503.07521276657843, -202.2736774358548, 1, 25.0, 0, 0, 0]])
    pd.testing.assert_frame_equal(expected_generators, generators, check_dtype=False)

    # load disconnection
    cache.connect_load('LOAD', False, None)
    cache.run_ac_pf()
    loads = cache.get_loads()
    expected_loads = pd.DataFrame(index=pd.Series(name='id', data=['LOAD']),
                                  columns=['name', 'voltage_level_id', 'bus_breaker_bus_id', 'connected', 'p', 'q','num',
                                           'v_mag_bus', 'synchronous_component_bus', 'local_num_bus', 'num_bus'],
                                  data=[['', 'VLLOAD', 'NLOAD', False, nan, nan, 0, 167.18649525262273, 0, 0, 3]])
    pd.testing.assert_frame_equal(expected_loads, loads, check_dtype=False)

    # load reconnection
    cache.connect_load('LOAD', True, 'NLOAD')
    cache.run_ac_pf()
    loads = cache.get_loads()
    expected_loads = pd.DataFrame(index=pd.Series(name='id', data=['LOAD']),
                                  columns=['name', 'voltage_level_id', 'bus_breaker_bus_id', 'connected', 'p', 'q',
                                           'num', 'v_mag_bus', 'synchronous_component_bus', 'local_num_bus', 'num_bus'],
                                  data=[['', 'VLLOAD', 'NLOAD', True, 700.0, 300.0, 0, 142.91752783756047, 0, 0, 3]])
    pd.testing.assert_frame_equal(expected_loads, loads, check_dtype=False)

    # line disconnection
    cache.connect_branch_side1('NHV1_NHV2_1', False, None)
    cache.connect_branch_side2('NHV1_NHV2_1', False, None)
    cache.run_ac_pf()
    branches = cache.get_branches()
    expected_branches = pd.DataFrame(
        index=pd.Series(name='id', data=['NHV1_NHV2_1', 'NHV1_NHV2_2', 'NGEN_NHV1', 'NHV2_NLOAD']),
        columns=['name', 'voltage_level1_id', 'voltage_level2_id', 'bus_breaker_bus1_id',
                'bus_breaker_bus2_id', 'connected1', 'connected2', 'p1', 'q1', 'i1',
                'p2', 'q2', 'i2', 'num', 'v_mag_bus1', 'synchronous_component_bus1', 'num_bus1', 'v_mag_bus2', 'synchronous_component_bus2', 'num_bus2'],
        data=[['', 'VLHV1', 'VLHV2',  'NHV1', 'NHV2',  False, False, nan,        nan,        nan,         nan,         nan,         nan,         0, 399.7154229049848, 0, 1, 348.5241869126856,  0, 2],
              ['', 'VLHV1', 'VLHV2',  'NHV1', 'NHV2',  True,  True,  718.336375, 576.329007, 1330.233688, -701.725437, -447.888303, 1379.050354, 1, 399.7154229049848, 0, 1, 348.5241869126856,  0, 2],
              ['', 'VLGEN', 'VLHV1',  'NGEN', 'NHV1',  True,  True,  720.38758,  635.341154, 22182.47534, -718.970877, -576.32886,  1330.94852,  2, 25.0,              0, 0, 399.7154229049848,  0, 1],
              ['', 'VLHV2', 'VLLOAD', 'NHV2', 'NLOAD', True,  True,  701.725392, 447.888302, 1379.05029,  -699.999911, -299.999984, 3488.940596, 3, 348.5241869126856, 0, 2, 126.02588154999955, 0, 3]])
    pd.testing.assert_frame_equal(expected_branches, branches, check_dtype=False)

    # line reconnection
    cache.connect_branch_side1('NHV1_NHV2_1', True, 'NHV1')
    cache.connect_branch_side2('NHV1_NHV2_1', True, 'NHV2')
    cache.run_ac_pf()
    branches = cache.get_branches()
    expected_branches = pd.DataFrame(
        index=pd.Series(name='id', data=['NHV1_NHV2_1', 'NHV1_NHV2_2', 'NGEN_NHV1', 'NHV2_NLOAD']),
        columns=['name', 'voltage_level1_id', 'voltage_level2_id', 'bus_breaker_bus1_id',
                 'bus_breaker_bus2_id', 'connected1', 'connected2', 'p1', 'q1', 'i1',
                 'p2', 'q2', 'i2', 'num', 'v_mag_bus1', 'synchronous_component_bus1', 'num_bus1', 'v_mag_bus2', 'synchronous_component_bus2', 'num_bus2'],
        data=[['', 'VLHV1', 'VLHV2',  'NHV1', 'NHV2',  True, True, 353.774582, 180.95677,  565.271318,   -350.670841, -207.497956, 608.029136,  0, 405.85974058065284, 0, 1, 386.90318309661694, 0, 2],
              ['', 'VLHV1', 'VLHV2',  'NHV1', 'NHV2',  True, True, 353.774582, 180.95677,  565.271318,   -350.670841, -207.497956, 608.029136,  1, 405.85974058065284, 0, 1, 386.90318309661694, 0, 2],
              ['', 'VLGEN', 'VLHV1',  'NGEN', 'NHV1',  True, True, 709.150389, 404.547355, 18854.570955, -708.126879, -361.913383, 1131.274257, 2, 25.0,               0, 0, 405.85974058065284, 0, 1],
              ['', 'VLHV2', 'VLLOAD', 'NHV2', 'NLOAD', True, True, 701.341670, 414.995911, 1216.058256,  -699.999960, -299.999993, 3076.577445, 3, 386.90318309661694, 0, 2, 142.91752783756047, 0, 3]])
    pd.testing.assert_frame_equal(expected_branches, branches, check_dtype=False)
