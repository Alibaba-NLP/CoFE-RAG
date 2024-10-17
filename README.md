# CoFE-RAG: A Comprehensive Full-chain Evaluation Framework for Retrieval-Augmented Generation with Enhanced Data Diversity
This is the repo for paper: CoFE-RAG: A Comprehensive Full-chain Evaluation Framework for Retrieval-Augmented Generation with Enhanced Data Diversity.



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
### Query Generation
```python
python ./data_generation/query_generation.py
```

### Multi-granularity Keyword Generation
```python
python ./data_generation/keyword_generation.py
```
