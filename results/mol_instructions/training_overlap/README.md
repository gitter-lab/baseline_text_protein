# Training identity <50% (Table 3)

`per_sample.csv` contains the 1,141 test samples with less than 50% sequence
identity to the training set, along with their current ProTrek scores.
`summary.csv` gives each method's mean, median, sample standard deviation
(`ddof=1`), and changes relative to the full test set. The full-test scores
come from the parent `per_sample.jsonl`.

The sample IDs come from `exp_mcmc_hard50.json`. Source details are in
`metadata.json`.

ESM-2 uses the current four-rate edit-only results, with unchanged Top-1 excluded.
Its values therefore differ from the older paper row.

From the repository root:

```bash
python scripts/summarize_paper_tables.py training_overlap
```
