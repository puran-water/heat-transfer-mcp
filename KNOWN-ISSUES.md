# Known issues

## Complete optional backend and platform qualification

Property-backend combinations, all thermal correlations, optional weather integrations and platforms need broader qualification. Unsupported inputs must be checked against the selected method domain.

This is an acknowledged v0.1.0 limitation. See README.md and TEST-RESULTS.md for scope and evidence.

## Complete comparative engineering benchmarks

Published upstream calculations and competing wrappers need a broader common-case benchmark.

This is an acknowledged v0.1.0 limitation. See README.md and TEST-RESULTS.md for scope and evidence.

## Supply cleared material catalog data

The material-property catalog is not bundled because its redistribution provenance was not cleared. Catalog-dependent helpers are unavailable; callers can use tools with explicit physical inputs. No replacement material values are invented.
