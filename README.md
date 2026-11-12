# Stronger Baselines for Text-Guided Protein Design

Code and results for the baselines in our paper. The commands below run the
methods from downloaded benchmark inputs and pretrained weights.

Use Linux, Python 3.12 and a CUDA GPU. Run commands from this repository root.
Model and dataset licenses are listed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## 1. Set up retrieval, ESM-2, MCMC and ProTrek

```bash
python3.12 -m venv .venv-baselines
source .venv-baselines/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements/baselines.txt
python -m pip install -e .

mkdir -p external
git clone https://github.com/westlake-repl/ProTrek.git external/ProTrek
git -C external/ProTrek checkout 9ad9cc5
git -C external/ProTrek apply ../../patches/protrek.patch

hf download westlake-repl/ProTrek_650M --local-dir external/models/ProTrek_650M
hf download NeuML/pubmedbert-base-embeddings --local-dir external/models/pubmedbert-base-embeddings
hf download nwliu/Molinst-SwissProtCLAP --repo-type dataset \
  --include training.json test.json --local-dir external/data/mol-instructions
```

The requirements use the package versions in our local inference environments.
The small upstream patches support these versions of PyTorch, TorchMetrics and
FAISS. Pinal and ProDVa have separate environments because their Transformers
versions differ.

The downloaded training and test files are JSON arrays containing
`instruction` and `sequence`. Keep their row order: sample IDs and the ESM-2/MCMC
random streams depend on it.

```bash
TRAINING_JSON=external/data/mol-instructions/training.json
TEST_JSON=external/data/mol-instructions/test.json
INDEX_DIR=external/data/mol-instructions/index
RUN_DIR=outputs/mol_instructions
mkdir -p "$RUN_DIR"
```

## 2. Top-1 retrieval

[Build the index](scripts/build_retrieval_index.py), then
[retrieve a protein for each test instruction](scripts/run_top1_retrieval.py):

```bash
python scripts/build_retrieval_index.py --training "$TRAINING_JSON" \
  --output "$INDEX_DIR" --embedding-model external/models/pubmedbert-base-embeddings
python scripts/run_top1_retrieval.py --training "$TRAINING_JSON" --test "$TEST_JSON" \
  --index "$INDEX_DIR" --embedding-model external/models/pubmedbert-base-embeddings \
  --allow-dangerous-deserialization --output "$RUN_DIR/top1.jsonl"
```

The deserialization flag loads the FAISS index just built above.
Each output row contains `sample_id`, `instruction` and `top1_sequence`.
This file is the input to ESM-2 and MCMC.

## 3. ESM-2 infill

[Run ESM-2](scripts/run_esm2_infill.py) from the retrieved sequences:

```bash
python scripts/run_esm2_infill.py --input "$RUN_DIR/top1.jsonl" \
  --protrek-repository external/ProTrek --checkpoint-dir external/models/ProTrek_650M \
  --output "$RUN_DIR/esm2.jsonl"
```

The ESM-2 checkpoint downloads on first use. For each masking rate
(5%, 10%, 15%, 25%), the script performs five rounds of infilling, starting
from the same retrieved sequence. ProTrek selects among the four final
candidates. Inputs longer than 1,022 residues use their first 1,022 residues.
Output includes the four `edit_candidates`, `selected_sequence` and
`selected_protrek_score`. The global random seed is 42.

## 4. MCMC

[Run MCMC](scripts/run_mcmc.py) from the same Top-1 inputs:

```bash
python scripts/run_mcmc.py --input "$RUN_DIR/top1.jsonl" --steps 50 \
  --protrek-repository external/ProTrek --checkpoint-dir external/models/ProTrek_650M \
  --output "$RUN_DIR/mcmc.jsonl"
```

This runs 50 single-residue proposals per sample, with ProTrek as the objective
and temperature decreasing from 1.0 toward 0.01. Output includes
`best_sequence`, `best_score` and the score trajectory. The default uses one
random stream seeded with 42 across samples.

## CAMEO inputs for these three methods

CAMEO uses function keywords with the fixed instruction template removed.
Prepare these files and build a keyword index:

```bash
hf download nwliu/CAMEO --repo-type dataset --include training.json test.json \
  --local-dir external/data/CAMEO
python scripts/prepare_cameo_keywords.py --data-dir external/data/CAMEO

TRAINING_JSON=external/data/CAMEO/training_keywords.json
TEST_JSON=external/data/CAMEO/test_keywords.json
INDEX_DIR=external/data/CAMEO/index_keywords
RUN_DIR=outputs/cameo
mkdir -p "$RUN_DIR"
python scripts/build_retrieval_index.py --training "$TRAINING_JSON" \
  --output "$INDEX_DIR" --embedding-model external/models/pubmedbert-base-embeddings \
  --unique-instructions
```

Then repeat the `run_top1_retrieval.py`, `run_esm2_infill.py` and `run_mcmc.py`
commands above. Their outputs now go to `outputs/cameo/`.

## 5. Pinal K=50

Use the upstream version expected by [our Pinal runner](scripts/run_pinal_k50.py):

```bash
python3.12 -m venv .venv-pinal
source .venv-pinal/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements/pinal.txt
python -m pip install -e .
git clone https://github.com/westlake-repl/Denovo-Pinal.git external/Pinal
git -C external/Pinal checkout d788376
git -C external/Pinal apply ../../patches/pinal.patch
hf download westlake-repl/Pinal --include 'T2struct-1.2B/*' 'SaProt-T/*' \
  --local-dir external/models/Pinal
ln -s T2struct-1.2B external/models/Pinal/T2struc-1.2B
ln -s SaProt-T external/models/Pinal/SaProtT

python scripts/run_pinal_k50.py --pinal-repository external/Pinal \
  --weights external/models/Pinal --input external/data/mol-instructions/test.json \
  --prompt-mode mol_instructions_numbered --seed 42 \
  --output outputs/mol_instructions/pinal.jsonl
python scripts/run_pinal_k50.py --pinal-repository external/Pinal \
  --weights external/models/Pinal --input external/data/CAMEO/test_keywords.json \
  --prompt-mode verbatim --seed 42 --output outputs/cameo/pinal.jsonl
```

The two links match today's weight-directory names to those expected by this
upstream version. Its repository includes the tokenizer files under `checkpoints/`.
The runner generates 50 raw candidates, removes duplicate sequences and selects
by joint structure/sequence log probability per residue. Output contains
`best_sequence` and the five highest-ranked candidates. Use `--limit 1` for a
small trial. The original Pinal sampling seed was not recorded; the commands
above set 42 for a repeatable new run.

## 6. ProDVa

ProDVa runs through its upstream `dvagen` command:

```bash
python3.12 -m venv .venv-prodva
source .venv-prodva/bin/activate
python -m pip install --upgrade pip
git clone https://github.com/sornkL/ProDVa.git external/ProDVa
git -C external/ProDVa checkout ff2d41c
git -C external/ProDVa apply ../../patches/prodva.patch
python -m pip install -r requirements/prodva.txt
python -m pip install -e external/ProDVa

hf download nwliu/ProDVa-Molinst-SwissProtCLAP --local-dir external/models/prodva-mol-instructions
hf download nwliu/ProDVa-CAMEO --local-dir external/models/ProDVa-CAMEO
hf download nwliu/Molinst-SwissProtCLAP --repo-type dataset --include phrases.json \
  --local-dir external/data/mol-instructions
hf download nwliu/CAMEO --repo-type dataset --include phrases.json --local-dir external/data/CAMEO

mkdir -p outputs
dvagen eval --config_path configs/prodva/mol_instructions.yaml
dvagen eval --config_path configs/prodva/cameo_keywords.yaml
```

These configs use the data and indexes prepared above. They save JSON arrays to
`outputs/prodva_mol_instructions.json` and `outputs/prodva_cameo_keywords.json`;
the generated sequence is in `prediction`. ProDVa reuses an existing output
file, so choose a new `save_results_path` in the config when rerunning generation.

## Score generated sequences

Top-1, Pinal and ProDVa save generated sequences. To calculate their ProTrek
scores, pair each sequence with its original test instruction. For Pinal:

First, prepare the pairs, matching predictions to test instructions by `sample_id`:

```bash
source .venv-baselines/bin/activate
python scripts/prepare_scoring_inputs.py --test external/data/mol-instructions/test.json \
  --predictions outputs/mol_instructions/pinal.jsonl --sequence-key best_sequence \
  --output outputs/mol_instructions/pinal.pairs.jsonl
```

`pinal.pairs.jsonl` contains `sample_id`, `instruction` and `sequence`.
This step prepares the input; it does not run a model.

Then calculate the scores:

```bash
python scripts/score_protrek.py --input outputs/mol_instructions/pinal.pairs.jsonl \
  --protrek-repository external/ProTrek --checkpoint-dir external/models/ProTrek_650M \
  --output outputs/mol_instructions/pinal.protrek.jsonl
```

`pinal.protrek.jsonl` contains one `protrek` score per `sample_id`.
ESM-2 and MCMC already calculate ProTrek during generation: their selected scores
are saved as `selected_protrek_score` and `best_score`, respectively. These can
be used directly without rescoring.

For another method, change the predictions file, sequence key and output names:

| Method | Predictions file for Mol-Instructions | `--sequence-key` |
|---|---|---|
| Top-1 | `outputs/mol_instructions/top1.jsonl` | `top1_sequence` |
| ESM-2 | `outputs/mol_instructions/esm2.jsonl` | `selected_sequence` |
| MCMC | `outputs/mol_instructions/mcmc.jsonl` | `best_sequence` |
| Pinal | `outputs/mol_instructions/pinal.jsonl` | `best_sequence` |
| ProDVa | `outputs/prodva_mol_instructions.json` | `prediction` |

For references, omit `--predictions` and `--sequence-key`. For CAMEO, use
`external/data/CAMEO/test_keywords.json` and the corresponding CAMEO outputs.
The preparation script joins by `sample_id`, or by test order for ProDVa,
and checks that any recorded instruction/reference agrees with the test input.

## Optional pLDDT evaluation

To evaluate the same Pinal outputs with ESMFold2-Fast, prepare reference/generated
sequence groups and run folding:

```bash
.venv-baselines/bin/python scripts/prepare_scoring_inputs.py \
  --test external/data/mol-instructions/test.json \
  --predictions outputs/mol_instructions/pinal.jsonl --sequence-key best_sequence \
  --output outputs/mol_instructions/pinal.pairs.jsonl \
  --folding-output outputs/mol_instructions/pinal.folding-input.jsonl
python3.12 -m venv .venv-fold
.venv-fold/bin/python -m pip install -e '.[fold]'
.venv-fold/bin/python scripts/score_esmfold2.py \
  --input outputs/mol_instructions/pinal.folding-input.jsonl \
  --output outputs/mol_instructions/pinal.plddt.jsonl --execute
```

Settings are in `configs/esmfold2.json`; weights download on first use.
Add `--structures-dir outputs/structures` to save mmCIF files.

## Released results

The checked-in `results/` directories contain the paper's frozen per-sample
outputs and summaries. The commands above write new runs under `outputs/`.

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

Each benchmark has `per_sample.jsonl` with sequences, ProTrek and pLDDT, plus
`summary.csv`. Missing measurements are `null`. Recalculate these saved summaries:

```bash
python scripts/summarize_results.py mol_instructions
python scripts/summarize_results.py cameo
python scripts/summarize_paper_tables.py
```

The last command summarizes the bundled inputs for
[Table 3](results/mol_instructions/training_overlap/),
[Table 4](results/mol_instructions/retrieval_quartiles/) and
[Table 5](results/mol_instructions/uniref50/).

`configs/methods.json` records the experiment settings; scripts use their own
constants and command-line arguments. `configs/esmfold2.json` is loaded by the
folding script. The ProDVa YAML files are loaded by upstream `dvagen`.