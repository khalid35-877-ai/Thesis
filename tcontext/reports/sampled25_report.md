# 25 Percent Sample Experiment Report

## Setup
- Run started at: 2026-04-14 20:58:29
- Seed used: 777
- Dataset: Cats vs Dogs (clean local subset)
- Sampling strategy: random stratified sampling, 25% per class
- Total samples: 1250
- Train/Test split: 1000 / 250
- Backbone: EfficientNetB0 (ImageNet pretrained, frozen)
- Classification head: GlobalAveragePooling2D + Dropout(0.45) + Dense(1, sigmoid, L2)
- Image size: 224x224
- Batch size: 32
- Optimizer/Loss: Adam + Binary Crossentropy
- Data augmentation: random flip + rotation + zoom (training only)
- Loss regularization: label smoothing=0.05
- Early stopping: monitor=val_loss, patience=3, restore_best_weights=True
- Learning-rate scheduling: ReduceLROnPlateau(factor=0.35, patience=1)
- Retrieval: CLIP image encoder + Chroma vector DB (cosine, top-5 voting)

## Model Architecture Summary
```text
Model: "functional_1"
┌─────────────────────────────────┬────────────────────────┬───────────────┐
│ Layer (type)                    │ Output Shape           │       Param # │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ input_layer_2 (InputLayer)      │ (None, 224, 224, 3)    │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ efficientnetb0 (Functional)     │ (None, 7, 7, 1280)     │     4,049,571 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ global_average_pooling2d        │ (None, 1280)           │             0 │
│ (GlobalAveragePooling2D)        │                        │               │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dropout (Dropout)               │ (None, 1280)           │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense (Dense)                   │ (None, 1)              │         1,281 │
└─────────────────────────────────┴────────────────────────┴───────────────┘
 Total params: 4,050,852 (15.45 MB)
 Trainable params: 1,281 (5.00 KB)
 Non-trainable params: 4,049,571 (15.45 MB)
```

## EDA
- Class balance: {'cats': 625, 'dogs': 625}
- Image size stats (H mean/std): 364.2 / 93.7
- Image size stats (W mean/std): 410.5 / 106.9
- Class-wise image statistics:
  - cats: n=625, H(mean/std)=359.9/95.0, W(mean/std)=413.4/107.2
  - dogs: n=625, H(mean/std)=368.4/92.2, W(mean/std)=407.7/106.6

## Model Performance
- Epochs trained: 12
- Training time (s): 703.95
- Accuracy: 0.9800
- Precision: 1.0000
- Recall: 0.9600
- F1: 0.9796
- Inference time total (s): 25.4510
- ROC-AUC: 0.9996
- Best learning epoch (val_loss minimum): 12
- Potential overfitting epoch (gap > 0.08): None

## Epoch Logs (Training History)

| epoch | loss | accuracy | auc | val_loss | val_accuracy | val_auc |
|---|---|---|---|---|---|---|
| 1 | 0.5933 | 0.7180 | 0.7883 | 0.4790 | 0.8640 | 0.9540 |
| 2 | 0.4395 | 0.8650 | 0.9558 | 0.3650 | 0.9200 | 0.9917 |
| 3 | 0.3591 | 0.9290 | 0.9774 | 0.2985 | 0.9440 | 0.9965 |
| 4 | 0.2934 | 0.9530 | 0.9912 | 0.2583 | 0.9640 | 0.9980 |
| 5 | 0.2725 | 0.9530 | 0.9901 | 0.2331 | 0.9680 | 0.9985 |
| 6 | 0.2523 | 0.9620 | 0.9917 | 0.2159 | 0.9680 | 0.9989 |
| 7 | 0.2301 | 0.9660 | 0.9938 | 0.2031 | 0.9680 | 0.9991 |
| 8 | 0.2198 | 0.9700 | 0.9939 | 0.1937 | 0.9720 | 0.9994 |
| 9 | 0.2145 | 0.9680 | 0.9946 | 0.1869 | 0.9760 | 0.9995 |
| 10 | 0.2067 | 0.9680 | 0.9961 | 0.1811 | 0.9800 | 0.9995 |
| 11 | 0.2034 | 0.9680 | 0.9962 | 0.1760 | 0.9800 | 0.9996 |
| 12 | 0.2005 | 0.9700 | 0.9952 | 0.1719 | 0.9800 | 0.9996 |

## Training/Inference Time Breakdown
- Total training time (s): 703.9532
- Classifier inference total (s): 25.4510
- Retrieval inference total (s): 29.7111
- Retrieval DB build time (s): 133.2358
- Retrieval average add-one-image time (s): 0.1332

## Computational Cost Profile
- Classifier total parameters: 4050852
- Classifier trainable parameters: 1281
- Saved model size (MB): 16.27
- Retrieval embedding dimension: 512
- Retrieval indexed vectors: 1000
- TensorFlow GPU available: False, PyTorch CUDA available: False, CLIP device used: cpu

## Incremental Update Cost Estimate
- Deep learning refit estimate (5 images): 3.5198 s
- Deep learning refit estimate (10 images): 7.0395 s
- Retrieval add estimate (5 images): 0.6662 s
- Retrieval add estimate (10 images): 1.3324 s

## Retrieval Performance
- Accuracy: 0.9880
- Precision: 1.0000
- Recall: 0.9760
- F1: 0.9879
- Inference time total (s): 29.7111
- Vector DB build time (s): 133.2358
- Encoder: openai/clip-vit-base-patch32
- Vector DB path: tcontext\vector_db\clip_chroma_db_seed_777

## Comparative Table

| method | accuracy | precision | recall | f1 | total_inference_sec | samples | avg_ms_per_sample |
|---|---|---|---|---|---|---|---|
| Classifier | 0.9800 | 1.0000 | 0.9600 | 0.9796 | 25.4510 | 250 | 101.8040 |
| Retrieval_CLIP_VectorDB | 0.9880 | 1.0000 | 0.9760 | 0.9879 | 29.7111 | 250 | 118.8446 |

## Output Files
- `tcontext\artifacts\sampled25_summary.json`
- `tcontext\artifacts\sampled25_comparison_metrics.csv`
- `tcontext\artifacts\sampled25_training_history.csv`
- `tcontext\artifacts\eda_class_distribution.png`
- `tcontext\artifacts\eda_size_distribution.png`
- `tcontext\artifacts\eda_samples.png`
- `tcontext\artifacts\training_curves.png`
- `tcontext\artifacts\classifier_roc_curve.png`
- `tcontext\artifacts\comparison_confusion_matrices.png`
- `tcontext\reports\sampled25_report.md`