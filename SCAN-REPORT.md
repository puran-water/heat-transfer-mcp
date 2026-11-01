# Release content review

The new release uses a fresh root commit. No private monorepo history is imported. The separately preserved `legacy` branch and `legacy-final` tag retain the earlier public repository history.

The release export excludes private site/project data, private database adapters, fitted vendor application presets, private corpora, operational documents and uncleared datasets. Bundled input presets without cleared provenance are excluded. Required public upstream copyright notices and scientific citations are retained.

The final release asset `scan-report.json` records the source, artifact and Git-history scan results and hashes. Checks include Gitleaks 8.30.1 and custom private-content/path/identity detectors, plus review of data origins and licence notices. The scan is bounded by those rules and the retained provenance evidence.

The final report applies to the new release commit and its attached wheel/source distributions. It does not certify arbitrary user inputs or external dependencies.
