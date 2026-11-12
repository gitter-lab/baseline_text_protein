# Results

The two benchmarks are kept completely separate:

- `mol_instructions/per_sample.jsonl` and `mol_instructions/summary.csv`
- `cameo/per_sample.jsonl` and `cameo/summary.csv`

Natural reference sequences and instruction text are not included.

Each row has this compact shape:

```json
{
  "sample_id": 0,
  "reference": {"length": 100, "protrek": 20.0, "plddt": 85.0},
  "top1": {"sequence": "...", "protrek": 19.0, "plddt": 84.0},
  "prodva": {"sequence": "..."},
  "pinal_k50": {"sequence": "..."},
  "esm2_oracle": {"sequence": "..."},
  "mcmc": {"sequence": "..."}
}
```

The main benchmark files report ProTrek and pLDDT. Missing values are `null`.
In `summary.csv`, each `*_n` counts available scores; averages exclude missing
values. A count of 0 means no scores are available for that metric.

CAMEO ESM-2 infill has pLDDT scores for all 507 sequences: 410 reused from
identical previously scored sequences and 97 newly predicted locally with
ESMFold2-Fast using [these settings](../configs/esmfold2.json).
The original folding settings for the 410 reused scores were not recorded;
the summary combines both sets of scores.

Additional analyses for Mol-Instructions are in:

- [training_overlap](mol_instructions/training_overlap/): the 1,141-sample
  subset with training sequence identity below 50% (Table 3).
- [retrieval_quartiles](mol_instructions/retrieval_quartiles/): scores by
  retrieval difficulty (Table 4).
- [uniref50](mol_instructions/uniref50/): sequence identity and search coverage
  (Table 5).

Each includes `per_sample.csv` and `summary.csv`. See the directory READMEs
for the subset definition and handling of older or unavailable searches.
