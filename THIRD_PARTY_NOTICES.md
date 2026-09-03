# Third-party components

This repository does not redistribute model weights or benchmark corpora.
The code expects separately downloaded upstream assets:

| Component | Upstream links | Role |
|---|---|---|
| Pinal | [GitHub code](https://github.com/westlake-repl/Denovo-Pinal); [Hugging Face weights](https://huggingface.co/westlake-repl/Pinal) | Generative baseline |
| ProDVa | [GitHub code](https://github.com/sornkL/ProDVa); [Mol-Instructions weights](https://huggingface.co/nwliu/ProDVa-Molinst-SwissProtCLAP); [CAMEO weights](https://huggingface.co/nwliu/ProDVa-CAMEO) | Generative baseline |
| ProTrek | [GitHub code](https://github.com/westlake-repl/ProTrek); [Hugging Face 650M weights](https://huggingface.co/westlake-repl/ProTrek_650M) | Text-sequence scoring |
| ESM-2 | [GitHub code](https://github.com/facebookresearch/esm); [Hugging Face 650M model](https://huggingface.co/facebook/esm2_t33_650M_UR50D) | Sequence infilling |
| PubMedBERT embeddings | [Hugging Face model](https://huggingface.co/NeuML/pubmedbert-base-embeddings) | Retrieval |
| Mol-Instructions | [Hugging Face dataset (`zjunlp/Mol-Instructions`)](https://huggingface.co/datasets/zjunlp/Mol-Instructions); [GitHub repository](https://github.com/zjunlp/Mol-Instructions) | Mol-Instructions benchmark |
| CAMEO | [Hugging Face dataset (`nwliu/CAMEO`)](https://huggingface.co/datasets/nwliu/CAMEO) | CAMEO benchmark |

Upstream code, weights, and datasets remain governed by their original
licenses and model or dataset terms. Review those terms before redistribution.
