# Corpus

The notebooks read two PMI publications from this directory. **They are not
distributed with this repository** — they are copyrighted Project Management
Institute publications, and `.gitignore` excludes `data/corpus/*.pdf`.

Obtain your own copies and place them here with these exact filenames:

| Expected filename | Publication | Used by |
|---|---|---|
| `practice-standard-project-risk-management.pdf` | PMI, *Practice Standard for Project Risk Management* (2009) | `01_risk_standard_preprocessing.ipynb`, `03_concept_definitions_and_triplets.ipynb` |
| `PMBOK-7th-Edition.pdf` | PMI, *A Guide to the Project Management Body of Knowledge (PMBOK Guide)* | see the version note below |

Both are available to PMI members from <https://www.pmi.org/> and through
institutional library subscriptions.

## Version note — PMBOK 6 vs 7

`02_pmbok_ontology_construction.ipynb` reads a file its code names
`PMBOK6-2017.pdf` (PMBOK Guide, **6th** Edition), and the presentation in
`docs/presentations/` explicitly states the 6th Edition was chosen as the PM
corpus. The PDF that was present in the working directory when this repository
was organised is the **7th** Edition.

This discrepancy is unresolved. The notebook's section-extraction logic keys off
PMBOK 6 numbering, so **running notebook 02 against the 7th Edition will not
reproduce the published results.** Supply the 6th Edition if you intend to
re-run that notebook.

## Other inputs that are not in this repository

Notebooks 03, 04 and 05 additionally read intermediate CSVs that existed only as
private Kaggle datasets (`Concat.csv`, `finalDF (1).csv`, `definitions.csv`,
`htt.csv`, `first_version.csv`, `Glossary_Of_Risk_Management.pdf`). They are not
archived anywhere in this repository — see the "Reproducibility limitations"
section of the root `README.md`.
