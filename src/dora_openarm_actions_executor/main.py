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

"""Node to execute timestamped actions."""

import argparse
import asyncio
import dora
import os
import numpy as np
import pyarrow as pa
import time


QPOS_TYPE = pa.struct([("qpos", pa.list_(pa.float32()))])


class ActionChunk:
    """Action chunk anchored to the time its first point refers to.

    positions[k] is the target for start_ns + k * interval_ns. Evaluating by time
    instead of index lets a chunk that arrives late start from the point that
    corresponds to "now" rather than replaying from its first point.
    """

    def __init__(self, positions, start_ns, interval_ns, interpolate):
        """Initialize the chunk; interpolate selects Hermite over sample-and-hold."""
        self.positions = np.asarray(positions, dtype=np.float32)
        self.start_ns = int(start_ns)
        self.interval_ns = int(interval_ns)
        self.end_ns = self.start_ns + (len(self.positions) - 1) * self.interval_ns
        self.slopes = None
        if interpolate and len(self.positions) > 1:
            self.slopes = self._compute_slopes(self.positions)

    @staticmethod
    def _compute_slopes(y):
        # Slopes per chunk step, flattened at local extrema to avoid overshoot.
        s = np.diff(y, axis=0)
        m = np.zeros_like(y)
        m[0] = s[0]
        m[-1] = s[-1]
        for i in range(1, len(y) - 1):
            prod = s[i - 1] * s[i]
            mask = prod <= 0.0
            mi = 0.5 * (s[i - 1] + s[i])
            mi[mask] = 0.0
            m[i] = mi
        return m

    def at(self, t_ns):
        """Return the target at time t_ns, clamped to the chunk's time span."""
        last = len(self.positions) - 1
        s = min(max((t_ns - self.start_ns) / self.interval_ns, 0.0), float(last))
        if self.slopes is None:
            return self.positions[int(s)]
        i = min(int(s), last - 1)
        u = s - i
        h00 = 2 * u**3 - 3 * u**2 + 1
        h10 = u**3 - 2 * u**2 + u
        h01 = -2 * u**3 + 3 * u**2
        h11 = u**3 - u**2
        y = self.positions
        m = self.slopes
        return h00 * y[i] + h10 * m[i] + h01 * y[i + 1] + h11 * m[i + 1]


class Hold:
    """The last commanded target, held after a chunk runs out."""

    def __init__(self, position):
        """Initialize with the held position."""
        self.position = np.asarray(position, dtype=np.float32)
        self.end_ns = 0

    def at(self, t_ns):
        """Return the held position."""
        return self.position


class Blend:
    """Cross-fade from an old trajectory to a new one over a fixed time window."""

    def __init__(self, old, new, start_ns, duration_ns):
        """Initialize a blend that starts at start_ns and lasts duration_ns."""
        self.old = old
        self.new = new
        self.start_ns = int(start_ns)
        self.settle_ns = self.start_ns + int(duration_ns)
        self.end_ns = max(new.end_ns, self.settle_ns)

    def at(self, t_ns):
        """Return the blended target at time t_ns."""
        if t_ns >= self.settle_ns:
            return self.new.at(t_ns)
        s = max((t_ns - self.start_ns) / (self.settle_ns - self.start_ns), 0.0)
        # smoothstep keeps the velocity continuous at both ends of the window
        w = s * s * (3 - 2 * s)
        return (1 - w) * self.old.at(t_ns) + w * self.new.at(t_ns)


def _prune(trajectory, t_ns):
    """Drop blends that have finished so chained replans do not grow unbounded."""
    if isinstance(trajectory, Blend):
        if t_ns >= trajectory.settle_ns:
            return trajectory.new
        trajectory.old = _prune(trajectory.old, t_ns)
    return trajectory


# Tustin bilinear transform based biquad low-pass filter
class BiquadLowpass:
    """Biquad low-pass filter for smoothing outputs."""

    def __init__(self, fs, fc, Q=0.707):
        """Initialize the biquad low-pass filter with sampling frequency, cutoff frequency, and Q factor."""
        fs = float(fs)
        fc = float(fc)
        Q = float(Q)
        w0 = 2 * np.pi * fc / fs
        cosw0 = np.cos(w0)
        alpha = np.sin(w0) / (2 * Q)
        a0 = 1 + alpha
        self.b0 = ((1 - cosw0) / 2) / a0
        self.b1 = (1 - cosw0) / a0
        self.b2 = ((1 - cosw0) / 2) / a0
        self.a1 = (-2 * cosw0) / a0
        self.a2 = (1 - alpha) / a0
        self.x1 = None
        self.x2 = None
        self.y1 = None
        self.y2 = None

    def reset_state(self, initial_x):
        """Reset the filter state with the initial input value."""
        initial_x = np.asarray(initial_x, dtype=np.float32)
        self.x1 = initial_x.copy()
        self.x2 = initial_x.copy()
        self.y1 = initial_x.copy()
        self.y2 = initial_x.copy()

    def step(self, x):
        """Apply one step of the biquad low-pass filter to the input x."""
        x = np.asarray(x, dtype=np.float32)
        if self.x1 is None:
            self.reset_state(x)
        y = (
            self.b0 * x
            + self.b1 * self.x1
            + self.b2 * self.x2
            - self.a1 * self.y1
            - self.a2 * self.y2
        )
        self.x2, self.x1 = self.x1, x
        self.y2, self.y1 = self.y1, y
        return y.astype(np.float32)


def _latest_actions(events):
    """Take the newest queued actions; older ones are stale once a newer one exists."""
    latest = None
    reset = False
    while not events.empty():
        latest = events.get_nowait()
        reset = reset or latest["metadata"].get("reset", False)
    return latest, reset


def _chunk_from_event(event, interpolate, latency_compensation):
    metadata = event["metadata"]
    n_positions = len(event["value"])
    pos_shape = len(event["value"][0])
    positions = event["value"].values.to_numpy().reshape(n_positions, pos_shape)
    # positions[0] is the target for the moment the observation was taken. Without
    # that timestamp (or when clocks are not shared), fall back to "now".
    start_ns = metadata.get("observation_timestamp") if latency_compensation else None
    if start_ns is None:
        start_ns = time.time_ns()
    return ActionChunk(positions, start_ns, metadata["interval"], interpolate)


def _send_position(node, arms, position, timestamp):
    position = np.asarray(position, dtype=np.float32)
    offset = 0
    n_elements = 8  # 7 joints + 1 gripper
    if "right" in arms:
        node.send_output(
            "move_position_right",
            pa.array(
                [{"qpos": position[offset : offset + n_elements]}], type=QPOS_TYPE
            ),
            {"timestamp": timestamp},
        )
        offset += n_elements
    if "left" in arms:
        node.send_output(
            "move_position_left",
            pa.array(
                [{"qpos": position[offset : offset + n_elements]}], type=QPOS_TYPE
            ),
            {"timestamp": timestamp},
        )


async def _main_executor(
    node,
    events,
    arms,
    use_upsample,
    use_filter,
    control_hz,
    blend_sec,
    latency_compensation,
    log_latency,
):
    if not use_upsample and use_filter:
        print(
            "Warning: upsample is False, but filter is True. Forcing filter to False."
        )
        use_filter = False

    trajectory = None
    held = None  # Hold of the last target once a chunk has run out
    lowpass = None
    period_ns = int(1e9 / control_hz) if use_upsample else None
    blend_ns = int(blend_sec * 1e9)
    next_tick_ns = None

    while True:
        if trajectory is None:
            # Idle: wait for the next chunk, then take the newest that is queued.
            first = await events.get()
            event, reset = _latest_actions(events)
            if event is None:
                event, reset = first, first["metadata"].get("reset", False)
            else:
                reset = reset or first["metadata"].get("reset", False)
            next_tick_ns = None
        else:
            sleep_ns = next_tick_ns - time.time_ns()
            if sleep_ns > 0:
                await asyncio.sleep(sleep_ns / 1e9)
            event, reset = _latest_actions(events)

        if event is not None:
            chunk = _chunk_from_event(event, use_upsample, latency_compensation)
            now = time.time_ns()
            lag_ms = (now - chunk.start_ns) / 1e6
            if log_latency:
                print(f"Actions received {lag_ms:.1f} ms after the observation")
            if period_ns is None:
                period_ns = chunk.interval_ns
            if use_filter and lowpass is None:
                # Filter cutoff frequency is 15 Hz by default, which is a common choice for robotic arm control to balance smoothness and responsiveness.
                lowpass = BiquadLowpass(
                    fs=control_hz, fc=event["metadata"].get("cutoff_hz", 15)
                )
            if reset:
                # On a reset, these actions are the first of a new episode, so drop
                # any trajectory carried over from the previous one instead of
                # blending it.
                print("Resetting trajectory, discarding any previous trajectory.")
                trajectory = held = None
            if now > chunk.end_ns:
                print(
                    f"Warning: dropping actions received {lag_ms:.0f} ms after the "
                    "observation; the whole chunk is already in the past."
                )
            elif trajectory is None and held is None:
                trajectory = chunk
                if reset and lowpass is not None:
                    # Re-initialize the low-pass filter to the new episode's first
                    # pose. Otherwise its retained state pulls the first samples
                    # toward the previous episode's final pose.
                    lowpass.reset_state(chunk.at(now))
            else:
                previous = held if trajectory is None else _prune(trajectory, now)
                trajectory = Blend(previous, chunk, now, blend_ns)
                held = None
            if trajectory is None:
                continue

        now = time.time_ns()
        if next_tick_ns is None or now - next_tick_ns > period_ns:
            # First tick, or we fell behind by more than a period: resync instead
            # of bursting out the missed ticks.
            next_tick_ns = now
        trajectory = _prune(trajectory, now)
        position = trajectory.at(now)
        if use_filter and lowpass is not None:
            position = lowpass.step(position)
        _send_position(node, arms, position, now)
        if now >= trajectory.end_ns:
            # The chunk is exhausted; hold the last target until new actions arrive
            # and blend away from it then.
            trajectory = None
            held = Hold(position)
        next_tick_ns += period_ns


async def _main_dora(node, events, executor_task):
    while True:
        if node.is_empty():
            # Poll often: actions waiting here are already late.
            await asyncio.sleep(0.001)
            continue
        event = node.next()
        if event["type"] != "INPUT":
            break

        # Main process
        await events.put(event)
    executor_task.cancel()


async def _main_async(arms, **options):
    node = dora.Node()
    events = asyncio.Queue()
    executor_task = asyncio.create_task(_main_executor(node, events, arms, **options))
    dora_task = asyncio.create_task(_main_dora(node, events, executor_task))

    try:
        await executor_task
    except asyncio.CancelledError:
        pass
    await dora_task


def main():
    """Execute timestamped actions."""
    parser = argparse.ArgumentParser(description="Execute timestamped actions")
    parser.add_argument(
        "--arms",
        default=os.getenv("ARMS", "right,left"),
        help="The used arms: 'right,left' (default), 'right' or 'left'",
        type=str,
    )
    parser.add_argument(
        "--upsample",
        action="store_true",
        help="Whether to upsample the actions",
    )
    parser.add_argument(
        "--filter",
        action="store_true",
        help="Whether to apply low-pass filter to the upsampled actions (only works if `upsample` is set)",
    )
    parser.add_argument(
        "--control-hz",
        default=250.0,
        type=float,
        help="motor control frequency (Hz)",
    )
    parser.add_argument(
        "--blend-sec",
        default=float(os.getenv("BLEND_SEC", "0.2")),
        type=float,
        help="Cross-fade duration from the running trajectory to new actions (s)",
    )
    parser.add_argument(
        "--latency-compensation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Align actions to their observation timestamp; disable when the "
        "observer and this node do not share a clock",
    )
    parser.add_argument(
        "--log-latency",
        action="store_true",
        help="Print the delay between each observation and its actions",
    )

    args = parser.parse_args()
    arms = args.arms.split(",")

    asyncio.run(
        _main_async(
            arms,
            use_upsample=args.upsample,
            use_filter=args.filter,
            control_hz=args.control_hz,
            blend_sec=args.blend_sec,
            latency_compensation=args.latency_compensation,
            log_latency=args.log_latency,
        )
    )


if __name__ == "__main__":
    main()
