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

Every method also has `tm_score`, `residue_lddt`,
`long_range_contact_f1`, and `clash_penalty`. Missing values are `null`.
