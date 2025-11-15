# Urdu Transformer Models - Export Package

**Export Date:** 2025-11-15 09:40:27

## Package Contents

This package contains two trained transformer models for Urdu language processing:

### 1. T5-Style Span Corruption Model
- **File:** `urdu_transformer_t5.pt`
- **Training Method:** T5-style span corruption (denoising)
- **Best Validation Loss:** -0.7817
- **Epochs Trained:** 15

### 2. Causal Language Model
- **File:** `urdu_transformer_causal.pt`
- **Training Method:** Causal language modeling (GPT-style)
- **Best Validation Loss:** -0.4269
- **Epochs Trained:** 50
- **Pre-trained From:** T5 model

## Model Architecture

- **Layers (N):** 2 encoder + 2 decoder layers
- **Model Dimension (d_model):** 512
- **Feed-forward Dimension (d_ff):** 2048
- **Attention Heads (h):** 8
- **Dropout:** 0.1
- **Vocabulary Size:** 10000
- **Total Parameters:** ~30,084,880

## Tokenizer

- **Type:** SentencePiece BPE
- **Vocabulary Size:** 10000
- **Files:** 
  - `urdu_tokenizer.model` (main model)
  - `urdu_tokenizer.vocab` (vocabulary)

## Dataset Information

- **Total Sentences:** 20,044
- **Training Split:** 16,035 samples (80%)
- **Validation Split:** 2,004 samples (10%)
- **Test Split:** 2,005 samples (10%)

## Evaluation Metrics

### T5 Model
- **Perplexity:** 2477.9127
- **Custom Score:** 0.74%

### Causal Model
- **Perplexity:** 1657.2749
- **Custom Score:** 42.80%

## Usage

### Loading the Models

```python
import torch
import sentencepiece as spm

# Load tokenizer
sp = spm.SentencePieceProcessor()
sp.load('urdu_tokenizer.model')

# Load T5 model
t5_checkpoint = torch.load('urdu_transformer_t5.pt', map_location='cpu')
# ... create model with config and load state_dict

# Load Causal model
causal_checkpoint = torch.load('urdu_transformer_causal.pt', map_location='cpu')
# ... create model with config and load state_dict
```

### Model Selection Guide

- **Use T5 Model for:**
  - Text denoising and correction
  - Filling in masked/corrupted text
  - Sequence-to-sequence tasks

- **Use Causal Model for:**
  - Text generation and continuation
  - Autocomplete functionality
  - Creative writing assistance

## Files Included

1. `urdu_transformer_t5.pt` - T5 model bundle
2. `urdu_transformer_causal.pt` - Causal LM model bundle
3. `urdu_tokenizer.model` - SentencePiece tokenizer
4. `urdu_tokenizer.vocab` - Tokenizer vocabulary
5. `model_config.json` - Complete configuration and metrics
6. `training_history.json` - Training progress history
7. `README.md` - This file

## Requirements

- Python 3.10+
- PyTorch 2.0+
- SentencePiece
- NumPy

## Authors

- Muhammad Abu Huraira (22F-3853)
- Shayan Zawar (22F-3410)

## License

See LICENSE file for details.

---

Generated on 2025-11-15 09:40:27
