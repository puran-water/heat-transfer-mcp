# Installed-artifact qualification

Python 3.12 on Linux x86-64. Each source distribution and wheel was installed in an isolated uv environment with shared cached dependencies. No private package or database credential was supplied. The stdio client initialized the installed entrypoint and called the tools below on synthetic inputs.

These are targeted packaging and representative domain checks, not a full numerical or cross-platform qualification. Historical broader test counts are in [HISTORICAL-QUALITY.md](HISTORICAL-QUALITY.md).

## Source installation

MCP discovery: PASS; 23 tools.

| Check | Tool | Result |
|---|---|---|
| ua-heat-duty | `calculate_heat_duty` | PASS |
| negative-area | `calculate_heat_duty` | PASS |

## Wheel installation

MCP discovery: PASS; 23 tools.

| Check | Tool | Result |
|---|---|---|
| ua-heat-duty | `calculate_heat_duty` | PASS |
| negative-area | `calculate_heat_duty` | PASS |
