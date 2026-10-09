# Copyright 2026 Enactic, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Sync execution of the actions executor with a fake node."""

import asyncio
import time

import numpy as np
import pyarrow as pa

from dora_openarm_actions_executor.main import _main_executor

INTERVAL_NS = 33_333_333
CHUNK_SEC = 31 * INTERVAL_NS / 1e9  # 32 points


class FakeNode:
    """Records the right-arm targets the executor sends."""

    def __init__(self):
        """Start with no outputs."""
        self.sent = []

    def send_output(self, name, value, metadata):
        """Record the first joint of each right-arm target."""
        if name == "move_position_right":
            qpos = value[0]["qpos"].values.to_numpy()
            self.sent.append((metadata["timestamp"], float(qpos[0])))


def actions(values, observation_timestamp=None):
    """Build an actions event whose 16 joints all follow values."""
    positions = np.repeat(np.asarray(values, np.float32)[:, None], 16, axis=1)
    metadata = {"interval": INTERVAL_NS, "reset": False}
    if observation_timestamp is not None:
        metadata["observation_timestamp"] = observation_timestamp
    return {
        "value": pa.array(positions.tolist(), type=pa.list_(pa.float32())),
        "metadata": metadata,
    }


def run(script, duration):
    """Feed (delay, make_event) pairs to the executor and return what it sent."""

    async def scenario():
        node, events = FakeNode(), asyncio.Queue()
        task = asyncio.create_task(
            _main_executor(node, events, ["right", "left"], True, False, 250)
        )
        for delay, make in script:
            await asyncio.sleep(delay)
            await events.put(make())
        await asyncio.sleep(duration)
        task.cancel()
        return node.sent

    return asyncio.run(scenario())


RAMP = np.arange(32, dtype=np.float32)  # positions[k] = k


def test_chunk_runs_to_the_end_despite_new_actions():
    """Actions arriving mid-chunk neither cut nor blend into the running chunk."""
    sent = run(
        [(0, lambda: actions(RAMP)), (0.3, lambda: actions(np.full(32, 100.0)))],
        duration=CHUNK_SEC,
    )
    values = [value for _, value in sent]
    assert np.allclose(values, np.sort(values))
    assert abs(values[-1] - 31.0) < 0.1  # the 250 Hz grid stops just short of the end


def test_actions_after_the_stop_are_executed():
    """The first actions arriving after the stop start from their first point."""
    sent = run(
        [
            (0, lambda: actions(RAMP)),
            (0.3, lambda: actions(np.full(32, 100.0))),
            (CHUNK_SEC, lambda: actions(np.full(32, 50.0))),
        ],
        duration=0.3,
    )
    values = {value for _, value in sent}
    assert 100.0 not in values and 50.0 in values


def test_actions_observed_before_the_stop_are_dropped():
    """Actions arriving after the stop but observed before it are skipped."""
    observed = []
    sent = run(
        [
            (0, lambda: actions(RAMP)),
            (CHUNK_SEC - 0.1, lambda: observed.append(time.time_ns()) or actions(RAMP)),
            (0.25, lambda: actions(np.full(32, 100.0), observed[0])),
            (0.05, lambda: actions(np.full(32, 50.0), time.time_ns())),
        ],
        duration=0.2,
    )
    values = {value for _, value in sent}
    assert 100.0 not in values and 50.0 in values
