# Hybrid NLP-DKT for Burmese STEM Curricula

Official PyTorch implementation of the proof-of-concept paper:  
**"Integrating Natural Language Processing with Deep Knowledge Tracing in Intelligent Tutoring Systems for Low-Resource Languages: A Case Study on Burmese STEM Curricula"**

## Architecture Overview
The system combines:
1. **Semantic Text Parsing Module**: Tokenization-free character-level TF-IDF (2-4 n-grams) with a calibrated Linear SVM to convert unsegmented Burmese explanations into a continuous correctness scalar ($c_t$).
2. **Deep Knowledge Tracing Engine**: An LSTM recurrent core integrating skill embeddings $e(s_t)$, dense text features $v_{text}$, and $c_t$.
3. **Pedagogical Prediction Layer**: Multi-skill mastery profile prediction for subsequent learning time-steps.

## Quickstart

### Installation
```bash
git clone [https://github.com/your-username/burmese-nlp-dkt.git](https://github.com/your-username/burmese-nlp-dkt.git)
cd burmese-nlp-dkt
pip install -r requirements.txt
```

### Run Training Demo
```bash
python train.py
```