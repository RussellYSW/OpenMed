# Data policy

## No patient data in this repository — ever

TrustFed ships **only synthetic data** (`trustfed/data/synthetic.py`). Real
cohorts used in the underlying research are controlled-access and/or contain
protected health information (PHI). They must never be committed here.

Blocked by default in [`.gitignore`](../.gitignore): `*.csv`, `*.xlsx`, `*.nii`,
`*.dcm`, anything matching `*PPMI*`, `*PDBP*`, `*_DATA_*`, `*mrn*`, and common
`data/` / `phi/` directories. Do not loosen these rules.

**Before your first commit**, verify nothing sensitive is staged:

```bash
git status
git ls-files | grep -iE 'csv|xlsx|nii|dcm|ppmi|pdbp|mrn|patient' || echo "clean"
```

If a real file was ever committed, scrubbing `.gitignore` is not enough — the
file stays in git history. Remove it with `git filter-repo` (or BFG) and force-push
before the repo is made public.

## Plugging in a real cohort

Keep real data **outside** the repository and load it at runtime. Replace the
synthetic generator with a loader that returns the same shape:

```python
# your_private_loader.py  (NOT committed)
def load_sites():
    # read from an out-of-tree path or a secured store
    # return: list of objects with .site_id, .X (n, d), .y (n,)
    ...
```

The controlled-access sources referenced by the research:

| Source | Access |
|-------|--------|
| PPMI (Parkinson's Progression Markers Initiative) | Data Use Agreement via ppmi-info.org |
| PDBP (Parkinson's Disease Biomarkers Program) | Data Use Agreement via pdbp.ninds.nih.gov |
| Institutional movement-disorder registry | Local IRB + DUA |

Each requires its own agreement; none may be redistributed through this project.

## Institutional review

Releasing code derived from patient data can still require sign-off. Before
publishing, confirm with your IRB and your institution's technology-transfer /
compliance office that (a) the code contains no embedded PHI (e.g. hard-coded IDs,
paths, or example rows), and (b) open-sourcing is permitted under the governing
data-use and IP agreements.
