# heat-transfer-mcp

Thermal engineering tools for heat exchangers, air cooling, insulation, heat loss and property calculations.

Experimental release series; `v0.1.0` is the initial pinned release and `main` follows current development. These tools return calculations and structured evidence; the calling LLM supplies explanations and reporting. Engineering review remains necessary.

## Use it in an engineering workflow

**A thermal-duty question.** Supply temperatures, flow, geometry, material properties and the intended heat-transfer model. Compare heat loss, insulation or exchanger behavior on that basis.

The public release excludes the private materials catalogue. Do not assume every material or weather-data route is available.

This is a public calculation component of [Puran Water's engineering toolchain](https://github.com/puran-water). The working system connects these tools to sizing-to-balance, proposal and BEP procedures and retained Studio review publications. Those internal workflows and project records are separate from this standalone package.

Have a project that needs engineering interpretation or an independent review? [Work with Puran Water](https://puranwater.com/contact/). To report a software defect, open a repository issue with a minimal synthetic or redacted input, package revision, expected result and actual result.

## Quick start

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and run with Python 3.12:

```sh
uvx --python 3.12 --with 'puran-mcp-support @ git+https://github.com/puran-water/fluids-mcp.git@v0.1.0#subdirectory=support/puran-mcp-support' --from 'git+https://github.com/puran-water/heat-transfer-mcp.git@v0.1.0' heat-transfer-mcp
```

The server uses local MCP stdio. It requires no Puran account or private package. uv shares its dependency cache between isolated tool environments. PyPI publishing has not been configured; the pinned public Git release is the installation path.

## MCP client configuration

Use this configuration in a client supporting local stdio MCP servers:

```json
{
  "mcpServers": {
    "heat-transfer-mcp": {
      "command": "uvx",
      "args": [
        "--python",
        "3.12",
        "--with",
        "puran-mcp-support @ git+https://github.com/puran-water/fluids-mcp.git@v0.1.0#subdirectory=support/puran-mcp-support",
        "--from",
        "git+https://github.com/puran-water/heat-transfer-mcp.git@v0.1.0",
        "heat-transfer-mcp"
      ]
    }
  }
}
```

Run the server in a user-owned working directory. Local input and result files belong to that user. Hosted HTTP authentication and managed employee access are outside this release.

## Known issues

- **Complete optional backend and platform qualification.** Property-backend combinations, all thermal correlations, optional weather integrations and platforms need broader qualification. Unsupported inputs must be checked against the selected method domain.
- **Complete comparative engineering benchmarks.** Published upstream calculations and competing wrappers need a broader common-case benchmark.
- **Supply cleared material catalog data.** The material-property catalog is not bundled because its redistribution provenance was not cleared. Catalog-dependent helpers are unavailable; callers can use tools with explicit physical inputs. No replacement material values are invented.

See [known issue details](KNOWN-ISSUES.md), [test results](TEST-RESULTS.md), [content scan](SCAN-REPORT.md), and [licensing inventory](LICENSE-TABLE.md). These limitations are disclosed for early feedback.

## License and attribution

See LICENSE, NOTICE and THIRD-PARTY-NOTICES for the wrapper and retained upstream material. External dependencies retain their own licenses.

Built by [Puran Water](https://puranwater.com).

Historical source-suite snapshot: 352 passed. These counts predate the cleaned standalone export; they are not its qualification result. Later repairs covered selected subsets only. See [historical quality scope](HISTORICAL-QUALITY.md).

## Following current development

The `main` branch receives approved development fixes through a one-way source
mirror. Publication still checks secrets, private content, redistribution rights
and package boundaries. Version tags such as `v0.1.0` remain immutable. Installed
copies do not update automatically. To upgrade the server and its public support
package together from the current branches in your existing environment:

```sh
uv pip install --upgrade --reinstall \
  "puran-mcp-support @ git+https://github.com/puran-water/fluids-mcp.git@main#subdirectory=support/puran-mcp-support" \
  "heat-transfer-mcp @ git+https://github.com/puran-water/heat-transfer-mcp.git@main"
```

Pin both Git commits when repeatability is more important than following changes.
The mirror records the source commit in `.puran-mirror.json`.
