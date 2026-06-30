# Minting a Zenodo DOI for this repository

A Zenodo DOI gives the paper a stable, citable handle independent of
GitHub URL churn. It also satisfies the IEEE SPMB reproducibility
recommendation of a "permanent archive."

## One-time setup

1. Visit <https://zenodo.org/account/settings/github/> and log in via
   GitHub.
2. In the list of repositories, flip the toggle next to
   **h5371h/esf-recoverability** to ON.

That's it. Zenodo now watches the repository for new releases.

## Minting the DOI for the SPMB 2026 submission

1. After the submission tag is pushed, draft a GitHub release against
   that tag:
   ```bash
   gh release create v1.0.0-spmb2026-submission \
     --title "SPMB 2026 submission snapshot" \
     --notes-file CITATION.cff
   ```
2. Zenodo will detect the release within a couple of minutes and mint
   a DOI of the form `10.5281/zenodo.NNNNNNN`.
3. Update the BibTeX in `README.md` to include the DOI:
   ```bibtex
   @software{dammu2026_spmb_repro,
     author    = {Hitesh Dammu},
     title     = {esf-recoverability: code and pre-computed
                  results for IEEE SPMB 2026},
     month     = {jun},
     year      = {2026},
     publisher = {Zenodo},
     version   = {v1.0.0-spmb2026-submission},
     doi       = {10.5281/zenodo.NNNNNNN},
     url       = {https://doi.org/10.5281/zenodo.NNNNNNN},
   }
   ```
4. Cite the DOI in the paper's reproducibility statement.

## How Zenodo reads `CITATION.cff`

The `CITATION.cff` in the repo root is the source of truth for the
Zenodo metadata: it controls the title, author list, abstract,
keywords, license, and the `preferred-citation` block. Edit it before
the GitHub release and Zenodo will pick the change up automatically.

A linting check is available via `cffconvert`:

```bash
pip install cffconvert
cffconvert --validate -i CITATION.cff
```
