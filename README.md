# Neural Machine Translation with Sequence Models

This project implements and compares several neural machine translation (NMT) models using PyTorch and TorchText. The models are trained to translate sentences from English to German using the Multi30k dataset. Architectures include standard LSTM-based encoder-decoder models (with and without attention) as well as a Transformer model. BLEU scores are used to evaluate translation quality.

---

## Project Structure

```text
nmt_seq_models/
├── NMT with Various Sequence Models.ipynb    # Main notebook for training and evaluation
├── LSTMSeq2Seq_best.ckpt                     # Best checkpoint for LSTM Seq2Seq model
├── LSTMAttnSeq2Seq_best.ckpt                 # Best checkpoint for LSTM with Attention model
├── Transformer_best.ckpt                     # Best checkpoint for Transformer model
├── README.md                                 # Project documentation (you're here)
```

---

## Architectures

The notebook contains modular implementations of the following sequence models:

- **LSTMSeq2Seq**: Basic encoder-decoder using LSTMs, no attention  
- **LSTMAttnSeq2Seq**: Encoder-decoder with additive attention mechanism  
- **Transformer**: Full attention-based model using positional encodings and multi-head self-attention  

---

## Dataset & Training

- **Dataset**: Multi30k (English ↔ German)  
- **Tokenization**: Handled using `spaCy` and `torchtext`  
- **Optimization**:
  - LSTM Models: Adam optimizer with learning rate `0.001`
  - Transformer: Adam with `Noam`-style learning rate schedule
  - Batch size: 32  
  - Epochs: 20  
  - Dropout: 0.2 (for LSTMs), configurable for Transformer  
- **Evaluation**: BLEU score on validation set  
- **Device**: CUDA supported (if available)  

---

## Results

| Model              | BLEU Score |
|-------------------|------------|
| `LSTMSeq2Seq`      | 0.243      |
| `LSTMAttnSeq2Seq`  | 0.335      |
| `Transformer`      | 0.360      |

---

## How to Run

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/your-nmt-repo.git
   cd your-nmt-repo
   ```

2. **Install dependencies:**
   ```bash
    pip install torch torchtext==0.6.0 spacy tqdm
    python -m spacy download en
    python -m spacy download de
    ```

3. **If using Google Colab, mount your Google Drive:**
    ```bash
    from google.colab import drive
    drive.mount('/gdrive')
    ```

4. **Open the notebook:**
    ```bash
    NMT with Various Sequence Models.ipynb
    ```

5. **Run all cells to preprocess data, train models, and evaluate translations.**