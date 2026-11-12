# UniRef50 identity (Table 5)

`per_sample.csv` records each method/sample's best hit and search status
(`hit`, `no_hit`, or `not_queried`). `sequence_matches_current` records whether
the searched sequence matched the current output when the data was exported.
Hits are ranked by bit score,
identity, E-value, query coverage, target coverage, alignment length, then
target ID. Database and search details are in `metadata.json`.

- `summary.csv` uses only searches whose input sequence exactly matches the
  current method output. References are matched to the benchmark sequences;
  Pinal K=10 is matched to its original selected outputs.
- `cached_summary.csv` includes all cached searches, including older sequence
  versions. It is provided for comparison with earlier paper results.

Report `queried_n` alongside the scores. Coverage varies by method.
`unavailable_n` counts missing searches and, in `summary.csv`, searches of
different sequences. These cases are excluded from the statistics.

Identity mean, median, and population standard deviation use hits only.
The >90%, <30%, and no-hit percentages use all included queries as the
denominator. Missing searches are not counted as no hits, and no-hit queries
are not assigned zero identity. Thresholds are strict >90 and <30.

From the repository root:

```bash
python scripts/summarize_paper_tables.py uniref50
```
