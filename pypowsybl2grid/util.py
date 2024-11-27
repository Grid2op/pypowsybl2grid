# Copyright (c) 2024, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import json

from grid2op.Observation import BaseObservation
from sortedcontainers import SortedDict


def write_obs_json(obs: BaseObservation, file_name: str):
    result = {}

    generators = SortedDict()
    for (index, name) in enumerate(obs.name_gen):
        generators[name] = {
            "p": float(obs.gen_p[index]),
            "q": float(obs.gen_q[index]),
            "v": float(obs.gen_v[index]),
            "bus": int(obs.gen_bus[index]),
        }
    result["generators"] = generators

    loads = SortedDict()
    for (index, name) in enumerate(obs.name_load):
        loads[name] = {
            "p": float(obs.load_p[index]),
            "q": float(obs.load_q[index]),
            "v": float(obs.load_v[index]),
            "bus": int(obs.load_bus[index]),
        }
    result["loads"] = loads

    shunts = SortedDict()
    for (index, name) in enumerate(obs.name_shunt):
        shunts[name] = {
            "p": float(obs._shunt_p[index]),
            "q": float(obs._shunt_q[index]),
            "v": float(obs._shunt_v[index]),
            "bus": int(obs._shunt_bus[index]),
        }
    result["shunts"] = shunts

    lines = SortedDict()
    for (index, name) in enumerate(obs.name_line):
        lines[name] = {
            "status": bool(obs.line_status[index]),
            "p_or": float(obs.p_or[index]),
            "p_ex": float(obs.p_ex[index]),
            "q_or": float(obs.q_or[index]),
            "q_ex": float(obs.q_ex[index]),
            "bus_or": int(obs.line_or_bus[index]),
            "bus_ex": int(obs.line_ex_bus[index]),
        }
    result["lines"] = lines

    with open(file_name, 'w') as f:
        json.dump(result, f, indent = 4)
