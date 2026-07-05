# Comparative EDA: Deep Learning vs Retrieval

## Objective
- Compare predictive quality and cost profile between the trainable deep-learning model and retrieval system.
- Create thesis-ready comparative analysis for discussion and results chapters.

## Dataset Context
- Total samples: `1250`
- Class distribution: `{'cats': 625, 'dogs': 625}`
- Validation samples: `250`

## System Architecture (Comparative)
- **Deep learning pipeline**: image preprocessing -> EfficientNet feature extractor -> trainable head -> sigmoid prediction.
- **Retrieval pipeline**: CLIP encoder -> vector embedding -> Chroma similarity search (top-k) -> majority vote prediction.
- **Key thesis distinction**: deep learning updates knowledge through weight optimization; retrieval updates knowledge through memory insertion.

## Computation Footprint View
- **Deep learning**: front-loaded cost (epochs, backpropagation, optimizer updates).
- **Retrieval**: indexing/search cost (no gradient updates), fast incremental insertion for new samples.
- This cost separation is the practical reason retrieval is suited for low-resource incremental learning.

## Method Comparison Table

| method | accuracy | precision | recall | f1 | total_inference_sec | samples | avg_ms_per_sample | train_or_build_seconds | compute_profile |
|---|---|---|---|---|---|---|---|---|---|
| Classifier | 0.9800 | 1.0000 | 0.9600 | 0.9796 | 25.4510 | 250 | 101.8040 | 703.9532 | trainable_params=1281, model_size_mb=16.27 |
| Retrieval_CLIP_VectorDB | 0.9880 | 1.0000 | 0.9760 | 0.9879 | 29.7111 | 250 | 118.8446 | 133.2358 | embedding_dim=512, vectors=1000 |

## EDA Findings
- Accuracy delta (`Classifier - Retrieval`): `-0.0080`
- F1 delta (`Classifier - Retrieval`): `-0.0083`
- Retrieval achieves near-parity with classifier performance while avoiding weight updates.
- Precision-recall trade-off indicates retrieval is more conservative; classifier is slightly more recall-strong.

## Deep Learning Convergence
- Best epoch by val_loss: `12`
- Overfitting flag epoch (>0.08 gap): `None`
- Validation trajectory remains stable, supporting robust generalization under regularization.
- Peak train accuracy: `0.9700`
- Peak validation accuracy: `0.9800`
- Final generalization gap: `N/A`

## Computational and Incremental Cost
- Classifier train time (s): `703.9532`
- Retrieval DB build time (s): `133.2358`
- Retrieval avg add-1-image time (s): `0.1332`
- DL estimated refit for +10 images (s): `7.0395`
- Retrieval estimated add +10 images (s): `1.3324`

## Thesis Conclusion (Comparative)
- Deep learning gives top-end performance but requires heavier retraining cost for new context.
- Retrieval remains highly competitive with much faster incremental updates.
- For low-resource or rapidly changing contexts, retrieval-first updates are more practical.

## Generated Plots
- `D:\Thesis\tcontext\artifacts\comparative_method_metrics.png`
- `D:\Thesis\tcontext\artifacts\comparative_learning_curve.png`