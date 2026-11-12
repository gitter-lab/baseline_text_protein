# Third-party components

Model weights and full datasets are downloaded separately from the sources below.

| Component | Source | Role | License |
|---|---|---|---|
| Pinal | [Code license](https://github.com/westlake-repl/Denovo-Pinal/blob/main/LICENSE); [weights](https://huggingface.co/westlake-repl/Pinal/blob/main/README.md) | Generation with T2struct-1.2B and SaProt-T | Apache-2.0 (code)[^1] |
| ProDVa | [Code license](https://github.com/sornkL/ProDVa/blob/master/LICENSE); [Molinst-SwissProtCLAP weights](https://huggingface.co/nwliu/ProDVa-Molinst-SwissProtCLAP); [CAMEO weights](https://huggingface.co/nwliu/ProDVa-CAMEO) | Generation | MIT |
| ProTrek | [Code license](https://github.com/westlake-repl/ProTrek/blob/main/LICENSE); [650M weights](https://huggingface.co/westlake-repl/ProTrek_650M) | Text-sequence scoring | MIT |
| ESM-2 | [Code license](https://github.com/facebookresearch/esm/blob/main/LICENSE); [650M weights](https://huggingface.co/facebook/esm2_t33_650M_UR50D) | Sequence infilling | MIT |
| PubMedBERT embeddings | [Model card](https://huggingface.co/NeuML/pubmedbert-base-embeddings); [license](https://www.apache.org/licenses/LICENSE-2.0) | Retrieval | Apache-2.0 |
| Molinst-SwissProtCLAP | [Dataset](https://huggingface.co/datasets/nwliu/Molinst-SwissProtCLAP) | Training corpus and benchmark splits | MIT |
| Mol-Instructions | [Dataset card](https://huggingface.co/datasets/zjunlp/Mol-Instructions); [DATA_LICENSE](https://github.com/zjunlp/Mol-Instructions/blob/main/DATA_LICENSE) | Source dataset for Molinst-SwissProtCLAP | CC BY 4.0[^2] |
| CAMEO | [Dataset](https://huggingface.co/datasets/nwliu/CAMEO) | CAMEO benchmark | MIT |
| ESMFold2-Fast | [Code license](https://github.com/Biohub/esm/blob/main/LICENSE.md); [model license](https://biohub.ai/esm/protein/get-started); [weights](https://huggingface.co/biohub/ESMFold2-Fast); [usage policy](https://biohub.org/acceptable-use-policy/) | Structure prediction | MIT |

[^1]: Pinal's model card does not specify a license for its weights.
[^2]: Mol-Instructions' dataset card also states research-only, non-commercial use, which conflicts with the [CC BY 4.0 terms](https://creativecommons.org/licenses/by/4.0/).
