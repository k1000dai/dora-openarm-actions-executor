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

"""Timing behavior of the actions executor with a fake node."""

import asyncio
import time

import numpy as np
import pyarrow as pa

from dora_openarm_actions_executor.main import _main_executor

INTERVAL_NS = 33_333_333


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


def actions(values, observation_timestamp, reset=False):
    """Build an actions event whose 16 joints all follow values."""
    positions = np.repeat(np.asarray(values, np.float32)[:, None], 16, axis=1)
    metadata = {"interval": INTERVAL_NS, "reset": reset}
    if observation_timestamp is not None:
        metadata["observation_timestamp"] = observation_timestamp
    return {
        "value": pa.array(positions.tolist(), type=pa.list_(pa.float32())),
        "metadata": metadata,
    }


def run(script, duration=0.5, **options):
    """Feed (delay, make_event) pairs to the executor and return what it sent."""

    async def scenario():
        node, events = FakeNode(), asyncio.Queue()
        kwargs = {
            "arms": ["right", "left"],
            "use_upsample": True,
            "use_filter": False,
            "control_hz": 250,
            "blend_sec": 0.2,
            "latency_compensation": True,
            "log_latency": False,
        }
        kwargs.update(options)
        task = asyncio.create_task(_main_executor(node, events, **kwargs))
        for delay, make in script:
            await asyncio.sleep(delay)
            await events.put(make())
        await asyncio.sleep(duration)
        task.cancel()
        return node.sent

    return asyncio.run(scenario())


RAMP = np.arange(32, dtype=np.float32)  # positions[k] = k


def test_late_actions_start_at_the_current_time():
    """Actions 200 ms late start 6 points (200 / 33.3 ms) into the chunk."""
    sent = run([(0, lambda: actions(RAMP, time.time_ns() - 200_000_000))])
    assert abs(sent[0][1] - 6.0) < 0.5


def test_without_latency_compensation_actions_start_at_the_first_point():
    """Disabling compensation restores the old receive-time alignment."""
    sent = run(
        [(0, lambda: actions(RAMP, time.time_ns() - 200_000_000))],
        latency_compensation=False,
    )
    assert abs(sent[0][1]) < 0.1


def test_replan_with_the_same_plan_never_steps_back():
    """Replanning onto the same time-anchored plan does not rewind the target."""
    start = time.time_ns()
    sent = run(
        [(0, lambda: actions(RAMP, start)), (0.15, lambda: actions(RAMP, start))]
    )
    assert (np.diff([value for _, value in sent]) > -1e-4).all()


def test_new_plan_is_blended_in_within_the_blend_window():
    """A different plan is cross-faded in without jumps within blend_sec."""
    sent = run(
        [
            (0, lambda: actions(np.zeros(32), time.time_ns())),
            (0.1, lambda: actions(np.full(32, 10.0), time.time_ns())),
        ]
    )
    stamps = np.array([stamp for stamp, _ in sent])
    values = np.array([value for _, value in sent])
    assert np.abs(np.diff(values)).max() < 0.5
    switched = stamps[np.argmax(values > 0)]
    settled = stamps[np.argmax(values >= 10.0 - 1e-3)]
    assert (settled - switched) / 1e9 < 0.25


def test_chunk_entirely_in_the_past_is_dropped():
    """A chunk that ended before it arrived is not executed."""
    assert not run([(0, lambda: actions(RAMP, time.time_ns() - 2_000_000_000))])


def test_queued_actions_use_only_the_newest():
    """Older queued chunks are skipped in favor of the newest."""

    async def scenario():
        node, events = FakeNode(), asyncio.Queue()
        await events.put(actions(np.full(32, 1.0), time.time_ns()))
        await events.put(actions(np.full(32, 2.0), time.time_ns()))
        task = asyncio.create_task(
            _main_executor(
                node, events, ["right", "left"], True, False, 250, 0.2, True, False
            )
        )
        await asyncio.sleep(0.1)
        task.cancel()
        return node.sent

    sent = asyncio.run(scenario())
    assert {value for _, value in sent} == {2.0}


def test_reset_switches_without_blending():
    """A reset starts the new episode's chunk immediately."""
    sent = run(
        [
            (0, lambda: actions(np.zeros(32), time.time_ns())),
            (0.1, lambda: actions(np.full(32, 5.0), time.time_ns(), reset=True)),
        ]
    )
    assert np.abs(np.diff([value for _, value in sent])).max() > 4.9
