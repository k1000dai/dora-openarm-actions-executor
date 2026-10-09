# dora-openarm-actions-executor

A [Dora](https://dora-rs.ai/) node that executes timestamped actions for OpenArm.

## Execution

Execution is synchronous. Each `actions` chunk runs to its last point and the
arm then holds that target. Actions that arrive while a chunk is running are
dropped, because they were inferred while the arm was moving. The next chunk
is the first one to arrive after the stop; when its metadata carries
`observation_timestamp` (nanoseconds, `time.time_ns()`), it must also have been
observed after the stop.

## License

Licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.

Copyright 2026 Enactic, Inc.

## Code of Conduct

All participation in the OpenArm project is governed by our [Code of Conduct](CODE_OF_CONDUCT.md).
