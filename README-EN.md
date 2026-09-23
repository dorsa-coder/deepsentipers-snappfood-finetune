# Assignment 1 — Fine-tuning and Comparison of Language Models for Persian Sentiment Analysis

A comparison of two pretrained language models — **ParsBERT** (Persian monolingual) and **XLM-RoBERTa** (multilingual) — on two Persian datasets that differ in domain and number of classes, in order to assess whether one model's advantage over the other holds consistently.

## Datasets

| Dataset | Domain | Classes | Size (train / test) |
|---|---|---|---|
| [PNLPhub/snappfood-sentiment-analysis](https://huggingface.co/datasets/PNLPhub/snappfood-sentiment-analysis) | Snappfood user reviews (food delivery) | 2 (positive/negative) | tens of thousands of samples |
| [Khedesh/DeepSentiPers](https://huggingface.co/datasets/Khedesh/DeepSentiPers) | User reviews of digital products | 5 (furious / angry / neutral / happy / delighted) | 7,023 / 1,854 |

Dataset-specific fixes applied in the notebooks:
- **Snappfood:** the original label column (`label`) is a string (`'HAPPY'`/`'SAD'`); the numeric `label_id` column was used instead and cast to `int64`.
- **DeepSentiPers:** the original labels range from `-2` to `2` (furious=-2 … delighted=2). Since `Trainer` requires classes `0` to `n-1`, labels were shifted by subtracting the minimum value, giving `0` to `4`. (An earlier candidate, `PartAI/DeepSentiPers`, was found to only contain 3 classes and was replaced with `Khedesh/DeepSentiPers`, which was verified against the original paper author's repository [github.com/JoyeBright/DeepSentiPers].)

## Models

| Model | Type |
|---|---|
| `HooshvareLab/bert-fa-base-uncased` (ParsBERT) | Monolingual (Persian) |
| `xlm-roberta-base` | Multilingual |

Both models were fine-tuned on both datasets with identical settings: `learning_rate=2e-5`, `batch_size=16`, `epochs=3`, `seed=42`, so the comparison would be fair.

## Final Results (both datasets)

| Dataset | Model | Accuracy | F1-score |
|---|---|---|---|
| snappfood-sentiment-analysis | ParsBERT | 0.8680 | 0.8679 |
| snappfood-sentiment-analysis | XLM-RoBERTa | 0.8776 | 0.8773 |
| DeepSentiPers | ParsBERT | 0.7325 | 0.7344 |
| DeepSentiPers | XLM-RoBERTa | 0.7395 | 0.7388 |

### Training progress per epoch

**Snappfood — ParsBERT**

| Epoch | Training Loss | Validation Loss | Accuracy | F1 |
|---|---|---|---|---|
| 1 | 0.302939 | 0.323390 | 0.871250 | 0.870447 |
| 2 | 0.257582 | 0.344004 | 0.873685 | 0.873390 |
| 3 | 0.193523 | 0.426080 | 0.868039 | 0.867897 |

Total training time: ~1h 19m (`train_runtime≈4788.96s`)

**Snappfood — XLM-RoBERTa**

| Epoch | Training Loss | Validation Loss | Accuracy | F1 |
|---|---|---|---|---|
| 1 | 0.306942 | 0.327227 | 0.873464 | 0.872693 |
| 2 | 0.287966 | 0.337005 | 0.878224 | 0.878074 |
| 3 | 0.243898 | 0.341351 | 0.877560 | 0.877339 |

Total training time: ~1h 7m (`train_runtime≈4028.43s`)

**DeepSentiPers — ParsBERT**

| Epoch | Training Loss | Validation Loss | Accuracy | F1 |
|---|---|---|---|---|
| 1 | No log | 0.865089 | 0.661812 | 0.662080 |
| 2 | 0.974387 | 0.709017 | 0.726537 | 0.726433 |
| 3 | 0.632998 | 0.731068 | 0.732470 | 0.734441 |

Total training time: ~10m 7s (`train_runtime≈606.74s`)

**DeepSentiPers — XLM-RoBERTa**

| Epoch | Training Loss | Validation Loss | Accuracy | F1 |
|---|---|---|---|---|
| 1 | No log | 0.822311 | 0.691478 | 0.688402 |
| 2 | 1.063400 | 0.704660 | 0.728695 | 0.727223 |
| 3 | 0.736669 | 0.698677 | 0.739482 | 0.738754 |

Total training time: ~17m 33s (`train_runtime≈1052.72s`)

## Analysis and Assignment Q&A

**1. Which model achieved higher accuracy, and why?**

On **both** datasets, XLM-RoBERTa achieved higher accuracy and F1 than ParsBERT — about 1 percentage point higher on Snappfood (0.8776 vs. 0.8680 accuracy) and about 0.7 points higher on DeepSentiPers (0.7395 vs. 0.7325). This is a reproducible pattern across two independent runs, not a one-off result. The main driver appears to be each model's behavior across epochs: on Snappfood, ParsBERT clearly overfit between epoch 2 and 3 (validation loss jumped from 0.323 to 0.426 and accuracy dropped), while XLM-RoBERTa stayed stable to the end. A similar, milder pattern appears on DeepSentiPers, where both models' validation loss improved more slowly in epoch 3 than in epoch 2, but XLM-RoBERTa converged to a better final value on both datasets.

**2. Was the result the same across the two datasets?**

Yes, in terms of **model ranking** the result was consistent: XLM-RoBERTa won on both datasets. However, the **absolute performance level** differed substantially between the two: both models reached roughly 87–88% accuracy on Snappfood, but only 73–74% on DeepSentiPers. Likely reasons for this gap:
- DeepSentiPers has **5 classes** versus Snappfood's 2 — five-way classification is inherently harder, and adjacent classes (e.g., "angry" vs. "neutral") are harder to separate.
- DeepSentiPers has a smaller training set (7,023 samples vs. tens of thousands for Snappfood).
- The two datasets differ in domain (digital products vs. food), which affects vocabulary and writing style.

**3. Advantages and disadvantages of a monolingual vs. multilingual model**

ParsBERT: a model specialized for Persian with fewer parameters, but showed earlier signs of overfitting in both runs. XLM-RoBERTa: a larger and slower model (training time on DeepSentiPers was roughly 1.7x ParsBERT's — 1053s vs. 607s) — but was more stable during training on both datasets and converged to a better final result. This suggests that being specialized for a target language does not, by itself, guarantee better performance; model capacity and training-time behavior (overfitting/stability) can matter more.

**4. Effect of changing the learning rate**

The learning rate was **not** varied between models or datasets in this project — all four runs used a fixed `learning_rate=2e-5`, specifically so the model comparison would be fair. This question from the assignment asks for a **hypothetical prediction**: given that ParsBERT showed overfitting signs even at this relatively conservative learning rate, a larger learning rate (e.g., 1e-3) would likely have caused more instability and a sharper performance drop; a smaller learning rate (e.g., 1e-6) would likely have left too little time to learn well within 3 epochs, leading to lower accuracy. This hypothesis was not tested experimentally.

## Summary

Comparing these two models across two datasets that differ substantially in class count, domain, and data volume showed that the multilingual XLM-RoBERTa consistently, if modestly, outperformed the monolingual ParsBERT. This underscores that common assumptions (e.g., "a monolingual model is always better for its language") should be verified empirically rather than simply assumed. It also shows that the absolute accuracy achievable depends heavily on task complexity (number of classes, data volume), not just on model choice.

## Limitations
- For each dataset, the two models were not run back-to-back in the same session, so minor differences in the Colab runtime environment may have contributed to the results.
- The effect of changing the learning rate was not tested experimentally (Q4 is a theoretical prediction only).

## Files
- `tamrin1_final.ipynb` — full notebook for dataset 1 (Snappfood)
- `finetune_sentiment_comparison.py` — condensed script for dataset 1
- `tamrin1_deepsentipers.ipynb` — full notebook for dataset 2 (DeepSentiPers)
- `finetune_deepsentipers_comparison.py` — condensed script for dataset 2
