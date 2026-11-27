# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0]

This is the first Duotonics release. It provides the following capabilities:

### Features

- Reproducible batches of 100 duotone variations by default, with configurable variation counts, seeded random color generation, and explicit RGB color pairs.
- Gamma-encoded sRGB, linear-light sRGB (IEC 61966-2-1; Poynton, 2003), and Oklab (Ottosson, 2020) blend spaces.
- JPEG, PNG, and BMP input, with lossless PNG output and variation indices and RGB hex color pairs encoded in filenames.
- `colors.csv` sidecars identifying every color pair and `metrics.csv` sidecars reporting mean, median, and 95th-percentile CIEDE2000 (Sharma, Wu & Dalal, 2005) against an Oklab-rendered perceptual reference.
- Command-line and Python library interfaces, plus optional tkinter file dialogs.
- Source installation with `python -m pip install .`, a `duotonics` console command, and continued direct script execution.
- PyPI packaging and GitHub Actions publishing through PyPI Trusted Publishing.
- Transactional batch output that stages files before publication, preserves unrelated files, and leaves the previous batch unchanged after generation or publication failures.
- Early validation for unsupported, empty, oversized, and undecodable images, invalid colors, and duplicate color pairs.
- Cross-platform tests on Linux, macOS, and Windows across Python 3.9 through 3.13.
- Contribution, code-of-conduct, security, issue-template, and pull-request-template documents.
- Coverage reporting with a measured 75% minimum, static-analysis and security CI jobs, CodeQL analysis, and Dependabot security alerts.
