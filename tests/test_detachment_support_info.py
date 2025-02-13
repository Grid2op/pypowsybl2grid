# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import logging
from typing import Dict

import grid2op
import grid2op.Space
    
import pytest

from pypowsybl2grid.pypowsybl_backend import PyPowSyBlBackend

TOLERANCE = 1e-3

@pytest.fixture(autouse=True)
def setup():
    logging.basicConfig()
    logging.getLogger('powsybl').setLevel(logging.WARN)


def create_env(check_isolated_and_disconnected_injections=None, allow_detachment=None):
    if not hasattr(grid2op.Space, "DEFAULT_ALLOW_DETACHMENT"):
        return None
    backend = create_backend(check_isolated_and_disconnected_injections=check_isolated_and_disconnected_injections)
    if allow_detachment is not None:
        env = grid2op.make("l2rpn_case14_sandbox", test=True, allow_detachment=allow_detachment, backend=backend)
    else:
        env = grid2op.make("l2rpn_case14_sandbox", test=True, backend=backend)
    return env
    

def create_backend(check_isolated_and_disconnected_injections=None):
    return PyPowSyBlBackend(check_isolated_and_disconnected_injections=check_isolated_and_disconnected_injections,
                            consider_open_branch_reactive_flow=True,
                            connect_all_elements_to_first_bus=False,
                            )


def test_None_False():
    env = create_env(check_isolated_and_disconnected_injections=None, allow_detachment=False)
    if env is None:
        # wrong grid2Op version
        return
    try:
        assert not type(env.backend).detachment_is_allowed
        assert not type(env).detachment_is_allowed
    finally:
        env.close()
        
        
def test_None_default():
    env = create_env(check_isolated_and_disconnected_injections=None)
    if env is None:
        # wrong grid2Op version
        return
    try:
        assert not type(env.backend).detachment_is_allowed
        assert not type(env).detachment_is_allowed
    finally:
        env.close()
        
        
def test_None_True():
    env = create_env(check_isolated_and_disconnected_injections=None, allow_detachment=True)
    if env is None:
        # wrong grid2Op version
        return
    try:
        assert type(env.backend).detachment_is_allowed
        assert type(env).detachment_is_allowed
    finally:
        env.close()


def test_False_False():
    env = create_env(check_isolated_and_disconnected_injections=False, allow_detachment=False)
    if env is None:
        # wrong grid2Op version
        return
    try:
        assert not type(env.backend).detachment_is_allowed
        assert not type(env).detachment_is_allowed
    finally:
        env.close()
        
        
def test_False_default():
    env = create_env(check_isolated_and_disconnected_injections=False)
    if env is None:
        # wrong grid2Op version
        return
    try:
        assert not type(env.backend).detachment_is_allowed
        assert not type(env).detachment_is_allowed
    finally:
        env.close()
        
        
def test_False_True():
    env = create_env(check_isolated_and_disconnected_injections=False, allow_detachment=True)
    if env is None:
        # wrong grid2Op version
        return
    try:
        assert type(env.backend).detachment_is_allowed
        assert type(env).detachment_is_allowed
    finally:
        env.close()


def test_True_False():
    env = create_env(check_isolated_and_disconnected_injections=True, allow_detachment=False)
    if env is None:
        # wrong grid2Op version
        return
    try:
        assert not type(env.backend).detachment_is_allowed
        assert not type(env).detachment_is_allowed
    finally:
        env.close()
        
        
def test_True_default():
    env = create_env(check_isolated_and_disconnected_injections=True)
    if env is None:
        # wrong grid2Op version
        return
    try:
        assert not type(env.backend).detachment_is_allowed
        assert not type(env).detachment_is_allowed
    finally:
        env.close()
        
        
def test_True_True():
    env = create_env(check_isolated_and_disconnected_injections=True, allow_detachment=True)
    if env is None:
        # wrong grid2Op version
        return
    try:
        assert not type(env.backend).detachment_is_allowed
        assert not type(env).detachment_is_allowed
    finally:
        env.close()
