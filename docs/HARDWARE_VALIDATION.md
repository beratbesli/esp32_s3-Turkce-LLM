# Hardware validation matrix

This matrix separates verified configurations from planned qualification. A
blank or planned cell is not evidence of support.

| Area | Configuration / procedure | Status | Evidence required |
|---|---|---|---|
| Board | ESP32-S3 rev v0.2, N16R8, CH343 | Passed on one board | Boot log, flash/PSRAM identity, commit |
| Context | 64, 48-token `Merhaba` run | Passed once | Serial transcript and timing |
| Context | 128 | Planned | Allocation margin, long prompt, repeated runs |
| Context | 256 | Unsupported / planned | Allocation, stability, performance evidence |
| Soak | 1 hour repeated generation | Planned | Error/watchdog/reset count and throughput |
| Reset | 100 warm/cold reset cycles | Planned | Successful boot/model-CRC count |
| Brownout | Supply interruption/recovery | Planned | Voltage setup, corruption and recovery result |
| Power | Idle, prefill and generation current | Planned | Instrument, supply voltage, min/mean/peak |
| Thermal | Sustained-generation temperature | Planned | Ambient, measurement point, time series |
| Boards | Additional N16R8 units/vendors | Planned | Per-board flash/PSRAM identity and result |
| Recovery | Wired reflash after invalid app/model | Planned | Recovery steps and successful boot |

The authoritative completed evidence remains in
[VALIDATION.md](VALIDATION.md). When adding a result, record the exact project
commit, ESP-IDF version, board/chip revision, flash and PSRAM identity, power
supply, model image hash, procedure, raw log location, result, and date. Do not
replace planned rows with "passed" based solely on a successful CI cross-build.
