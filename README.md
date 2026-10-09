# dora-openarm-actions-executor

A [Dora](https://dora-rs.ai/) node that executes timestamped actions for OpenArm.

## Timing

Each `actions` input is a chunk whose first point is the target for the moment
the observation was taken. When the metadata carries `observation_timestamp`
(nanoseconds, `time.time_ns()`), the chunk is anchored to that time, so a chunk
that arrives late starts from the point that corresponds to now instead of
replaying from its first point. Chunks that are entirely in the past are
dropped. Without the timestamp, or with `--no-latency-compensation` (use it when
the observer runs on a host whose clock is not synchronized with this one), the
chunk starts when it is received.

When new actions arrive while a chunk is running, only the newest queued chunk
is used, and the running trajectory is cross-faded into it over `--blend-sec`
(default 0.2 s, or `BLEND_SEC`) with a smoothstep weight. A `reset` in the
metadata switches immediately. `--log-latency` prints the delay between each
observation and its actions.

## License

Licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.

Copyright 2026 Enactic, Inc.

## Code of Conduct

All participation in the OpenArm project is governed by our [Code of Conduct](CODE_OF_CONDUCT.md).
