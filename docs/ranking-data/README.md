# Ranking data sources and license

The ranking fields in [`../papers.json`](../papers.json) are a catalog-specific
subset adapted from [ShowJCR](https://github.com/hitfyd/ShowJCR), maintained by
hitfyd and contributors, at commit
[`4687a8f608f20aaf004299992f5d347dba009361`](https://github.com/hitfyd/ShowJCR/tree/4687a8f608f20aaf004299992f5d347dba009361).

These ShowJCR-derived fields are distributed under the upstream
[GNU GPL version 3](LICENSE), separately from this repository's CC0 catalog
and independently written Explorer interface. No ShowJCR application code is
included. The editable ranking subset is provided directly in `papers.json`.

| System | Edition | Upstream source file |
| --- | --- | --- |
| CCF | 2026 | [CCF2026-UTF8.csv](https://github.com/hitfyd/ShowJCR/blob/4687a8f608f20aaf004299992f5d347dba009361/中科院分区表及JCR原始数据文件/CCF2026-UTF8.csv) |
| 中科院 | 2025 | [FQBJCR2025-UTF8.csv](https://github.com/hitfyd/ShowJCR/blob/4687a8f608f20aaf004299992f5d347dba009361/中科院分区表及JCR原始数据文件/FQBJCR2025-UTF8.csv) |
| JCR | 2025 | [JCR2025-UTF8.csv](https://github.com/hitfyd/ShowJCR/blob/4687a8f608f20aaf004299992f5d347dba009361/中科院分区表及JCR原始数据文件/JCR2025-UTF8.csv) |
| 新锐 | 2026 | [XR2026-UTF8.csv](https://github.com/hitfyd/ShowJCR/blob/4687a8f608f20aaf004299992f5d347dba009361/中科院分区表及JCR原始数据文件/XR2026-UTF8.csv) |

Adaptation date: 2026-10-24. Changes comprise venue and ISSN matching,
catalog subsetting, JSON conversion, preservation of JCR subject categories,
and paper-track checks before assigning conference CCF labels. Each published
ranking retains its source URL, file checksum, and edition. Full upstream
snapshots and local verification reports are not included in this repository.

These are reference labels, provided without warranty, rather than official
paper-level evaluations. Missing or unverified labels appear as **无标签**.
The [upstream source notes](https://github.com/hitfyd/ShowJCR/blob/4687a8f608f20aaf004299992f5d347dba009361/README.md#数据来源)
identify the organizations behind the individual ranking systems.
