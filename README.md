# Stronger Baselines for Text-Guided Protein Design

Code and result for the retrieval, generation, editing, and optimization baselines used in the paper.

## Contents

```text
configs/                 Method settings
scripts/                 Retrieval, generation, scoring, and summary commands
src/rethinking_tgpd/     Shared Python code
results/                 Final aggregate and per-sample results
```

The repository does not include model weights, training data, retrieval
indexes, or credentials.

## Mol-Instructions results (N=5,876)

| Method | Mean ProTrek |
|---|---:|
| Natural reference | 26.8951 |
| Top-1 retrieval | 24.9269 |
| ProDVA | 17.0674 |
| Pinal K=50 | 19.4327 |
| ESM-2 infill | 25.4582 |
| MCMC-50 | 26.2793 |

## CAMEO results (N=507)

| Method | Mean ProTrek |
|---|---:|
| Natural reference | 21.0915 |
| Top-1 retrieval | 15.5606 |
| ProDVA | 11.8094 |
| Pinal K=50 | 17.0904 |
| ESM-2 infill | 16.9451 |
| MCMC-50 | 18.4133 |

## Results

- `results/mol_instructions/`: 5,876 per-sample records and its own summary.
- `results/cameo/`: 507 per-sample records and its own summary.

Each method record contains its sequence, ProTrek score, pLDDT. Missing measurements
are stored as `null`.

Regenerate the aggregate table with:

```bash
python scripts/summarize_results.py mol_instructions
python scripts/summarize_results.py cameo
```

## Code

Install the small local package:

```bash
python -m pip install -e .
```

The main entry points are:

```text
build_retrieval_index.py   Build the PubMedBERT/FAISS retrieval index
run_top1_retrieval.py      Run Top-1 retrieval
run_esm2_infill.py         Generate four edits and select with ProTrek
run_mcmc.py                Run 50-step ProTrek-guided MCMC
run_pinal_k50.py           Run Pinal K=50 and joint-likelihood selection
score_protrek.py           Score instruction/sequence pairs
score_esmfold2.py          Call a configured ESMFold2 scoring service
summarize_results.py       Rebuild the aggregate CSV
```

Run any command with `--help` for its arguments. External model repositories
and checkpoints are listed in `THIRD_PARTY_NOTICES.md`.

ProTrek and ESMFold2 are computational evaluators, not experimental evidence
of protein function or stability. ESM-2 and MCMC use ProTrek during inference
and should be interpreted as evaluator-aware analyses.