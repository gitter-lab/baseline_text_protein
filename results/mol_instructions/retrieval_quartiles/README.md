# Retrieval difficulty (Table 4)

`per_sample.csv` contains all 5,876 sample IDs, the text similarity to the
nearest training instruction, quartile assignments, and ProTrek scores.
`summary.csv` gives the four group means. Q1 has the lowest text similarity
and is the hardest group; each group contains 1,469 samples.

Quartiles use linear interpolation over the sorted similarities. Values on
a boundary belong to the lower quartile. Boundaries and source details are
in `metadata.json`.

From the repository root:

```bash
python scripts/summarize_paper_tables.py retrieval_quartiles
```
