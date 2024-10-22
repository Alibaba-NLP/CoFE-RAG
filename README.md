# CoFE-RAG: A Comprehensive Full-chain Evaluation Framework for Retrieval-Augmented Generation with Enhanced Data Diversity
This is the repo for paper: [CoFE-RAG: A Comprehensive Full-chain Evaluation Framework for Retrieval-Augmented Generation with Enhanced Data Diversity](https://arxiv.org/abs/2410.12248).



## Quick Start
### Environment
```bash
conda create -n CoFE python=3.11
conda activate CoFE
pip install -r requirements.txt
```

### Document parsing and chunking
```python
python run_pipeline.py config/parse_and_chunk.json
```

### Retrieval
```python
python run_pipeline.py config/search_and_eval_search.json`
```

### Generation 
```python
python run_pipeline.py config/gen_response_and_eval_response.json
```

## Automatic Data Generation
If you want to construct evaluation data using custom documents, please refer to the following process.
### Query Generation
```python
python ./data_generation/query_generation.py
```

### Multi-granularity Keyword Generation
```python
python ./data_generation/keyword_generation.py
```

## Citation

If you found this work useful, consider citing our paper as followed:

```bigquery
@article{liu2024cofe,
  title={CoFE-RAG: A Comprehensive Full-chain Evaluation Framework for Retrieval-Augmented Generation with Enhanced Data Diversity},
  author={Liu, Jintao and Ding, Ruixue and Zhang, Linhao and Xie, Pengjun and Huang, Fie},
  journal={arXiv preprint arXiv:2410.12248},
  year={2024}
}
```
