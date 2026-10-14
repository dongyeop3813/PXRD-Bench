# Dataset sources and attribution

This document lists third-party provider terms and credits.
[DATASET_LICENSE.md](DATASET_LICENSE.md) separately states the license for
PXRD-Bench contributors' own covered contributions.

PXRD-Bench is distributed as source identifiers and local reconstruction tools.
Provider snapshots, structures, PXRD arrays and tensor caches are obtained or
created locally. This repository does not grant permission to redistribute a
combined payload. Our contributor license is scoped in
[DATASET_LICENSE.md](DATASET_LICENSE.md); upstream material retains its terms.

| Source | Version | Source terms |
| --- | --- | --- |
| Materials Project | 2026.04.13 | CC BY 4.0 for records marked BY-C; GNoME-derived records marked BY-NC have separate noncommercial terms. All 77,493 benchmark MP records carry BY-C in their downloaded provider metadata. |
| Alexandria | PBE 3D 2025.07.02 | CC BY 4.0; preserve attribution, citation, license and modification notices. |
| OQMD | v1.5, October 2021 | CC BY 4.0 for supplied calculated data; original ICSD input structures are outside that grant. |
| AFLOW | Frozen record fingerprints | Scientific, academic and noncommercial access conditions, with restrictions on bulk retrieval and repackaging. No CC license is asserted for this source. |

The Materials Project record-level license summary is in
[data/materials_project_licenses.json](data/materials_project_licenses.json).
It supplements the frozen source catalog without changing its identifiers,
versions or checksums. A provider BY-C flag describes the supplied record;
it is not a license for independently acquired third-party inputs.

Local reconstruction transforms source structures into primitive-cell tokens,
canonical structures and simulated diffraction representations. Preserve source
credits and identify these transformations when sharing material where permitted.
The contributor license does not remove commercial-use rights granted by an
upstream CC BY licensor. AFLOW source or derived payloads are not mirrored here;
local installation does not confer a separate redistribution grant.

## Provider references

- Materials Project: [site](https://materialsproject.org/),
  [terms](https://materialsproject.org/terms),
  [database versions and GNoME conditions](https://docs.materialsproject.org/changes/database-versions),
  [record license fields](https://github.com/materialsproject/emmet/blob/main/emmet-core/emmet/core/vasp/material.py).
- Alexandria: [license and citation](https://alexandria.icams.rub.de/about.html),
  [datasets](https://alexandria.icams.rub.de/datasets.html).
- OQMD: [license](https://oqmd.org/api/),
  [downloads](https://oqmd.org/download/),
  [ICSD scope](https://static.oqmd.org/static/docs/faqs.html).
  Cite Saal et al., JOM (2013), [DOI](https://doi.org/10.1007/s11837-013-0755-4),
  and Kirklin et al., npj Computational Materials (2015),
  [DOI](https://doi.org/10.1038/npjcompumats.2015.10).
- AFLOW: [AFLUX documentation and access conditions](https://www.aflowlib.org/documentation/),
  [material-page rights notice](https://aflow.org/material/?id=aflow%3Ab438e1a25f9c187d).
- [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) and
  [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).

Third-party paper datasets and XDecomposer source/weights retain their own
licenses and attribution requirements; they are not relicensed by Bragglite.
