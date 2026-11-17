# Brain Tumor MRI Prediction — AI FATE 2026, Phase A

Students: Liel (207411604), Yohad (211624317)

System name: Brain Tumor MRI Prediction

This repository is the code link for the Phase A report. Every required artifact has its own file link below. Hebrew setup notes are in [README_HE.md](README_HE.md).

## Code and results

| What the report asks for | Link |
|---|---|
| Prediction code | [brain_tumor.py](brain_tumor.py) |
| Dependencies | [requirements.txt](requirements.txt) |
| Trained model | [results/best_model.pt](results/best_model.pt) |
| Run settings and class counts | [results/run_config.json](results/run_config.json) |
| Metrics (accuracy, F1, per class, tumor detection) | [results/metrics.json](results/metrics.json) |
| Readable evaluation summary | [results/evaluation_summary.txt](results/evaluation_summary.txt) |
| Confusion matrix figure | [results/confusion_matrix.png](results/confusion_matrix.png) |
| Learning curve | [results/learning_curve.png](results/learning_curve.png) |
| Training history | [results/history.csv](results/history.csv) |
| Appendix A: 100 predictions (GitHub shows this CSV as a table) | [results/predictions_100.csv](results/predictions_100.csv) |
| All 1,475 test predictions | [results/predictions_all.csv](results/predictions_all.csv) |
| The 100 images chosen before training | [results/appendix_selection.csv](results/appendix_selection.csv) |
| Images used after the duplicate audit | [results/manifest_used.csv](results/manifest_used.csv) |
| Original file list | [results/manifest_original.csv](results/manifest_original.csv) |
| 104 excluded exact duplicates | [results/excluded_images.json](results/excluded_images.json) |
| What was checked before the reported run | [VALIDATION.md](VALIDATION.md) |

Dataset: [Brain Tumor MRI Deduplicated (clean version), M.Y. LEE_77](https://www.kaggle.com/datasets/mylee77/brain-tumor-mri-deduplicated-clean-version). Source images: [Masoud Nickparvar, Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset).

## The system

We chose medical brain-MRI image classification. Brain tumors are diagnosed from MRI slices, and separating glioma, meningioma, pituitary tumor, and no tumor is time-consuming for a radiologist. The course allows a medical or image prediction task, and this dataset is a labeled four-class Kaggle set with an official train, validation, and test split. The project is an educational classifier. It shows what a vision model can and cannot do as a second reader. It is not a clinical diagnostic device.

The system takes one axial brain MRI image and predicts one of four labels: glioma, meningioma, notumor, or pituitary. It also reports a binary tumor score, defined as one minus the notumor score, with a fixed threshold of 0.5. The purpose is to demonstrate a reproducible image classifier and to record held-out predictions for this phase. It is much more accurate than always predicting the majority class (92.9% versus 25.9% on the held-out test set), and specificity for no tumor is 99.3%. It does not mark tumor location. The scores are not calibrated clinical probabilities. Glioma recall is 82.5%. The same patient or near-duplicate slices may appear in more than one split. A missed tumor is the costly error: on the full test set the binary detector had 55 false negatives and 2 false positives.

## The dataset

- Four classes: glioma, meningioma, notumor, pituitary. Each file is one MRI slice, JPEG, mixed grayscale and RGB, commonly 512×512.
- The publisher started from about 7,200 images and removed perceptual-hash duplicates (pHash, Hamming distance 0). The extracted folder has 6,507 images: train 4,530, validation 502, test 1,475.
- Raw class counts. Train: glioma 1,258, meningioma 1,237, notumor 820, pituitary 1,215. Validation: 139, 137, 91, 135. Test: 382, 390, 303, 400.
- notumor is about one third smaller than the tumor classes. There is no patient id, age, sex, or scanner metadata, so demographic fairness cannot be measured.
- This run removed 104 additional exact pixel duplicates (95 from train, 9 from validation; test stayed 1,475). Images actually used: train 4,435 (glioma 1,254, meningioma 1,154, notumor 812, pituitary 1,215), validation 493 (139, 129, 90, 135), test 1,475. Counts are in [results/run_config.json](results/run_config.json). Removed files are listed in [results/excluded_images.json](results/excluded_images.json).
- The publisher warns that near-duplicates can remain, and that models may use image framing rather than the pathology, especially on notumor scans.

## The algorithm

Code: [brain_tumor.py](brain_tumor.py).

- Tooling: Python 3.12, PyTorch 2.7.1, torchvision, scikit-learn, Pillow. Model: EfficientNet-B0 pretrained on ImageNet, with a new 4-class linear head. The saved weights are [results/best_model.pt](results/best_model.pt).
- EfficientNet-B0 fits this task because the input is a single image, the label set is four classes, and about 4,400 training images are enough for transfer learning on a 6 GB laptop GPU.
- Preprocessing: convert to RGB, pad to a square with black borders, resize to 224×224, ImageNet normalization. Training augmentation is only a small rotation (±8°) and a 3% shift. Validation and test images are not augmented.
- The official split is kept. The model is never trained on validation or test. Class weights come from the training counts only, because notumor is smaller.
- Training has two stages. First the classifier head is trained for 3 epochs (AdamW, learning rate 0.001). Then the last three feature blocks are fine-tuned for up to 15 epochs (learning rate 0.0001, weight decay 0.0001, batch size 16, seed 42).
- The checkpoint is chosen by validation macro F1. The best checkpoint was fine-tune epoch 12 (validation accuracy 0.988, validation macro F1 0.988). Early stopping patience was 4 and did not fire before epoch 15. The test metrics below come from that checkpoint only. The epoch log is [results/history.csv](results/history.csv) and the curve is [results/learning_curve.png](results/learning_curve.png).
- Before training, a stratified sample of 100 test images was fixed (seed 42). The list is [results/appendix_selection.csv](results/appendix_selection.csv). Those 100 rows are Appendix A. They were not used to pick the model.

## Evaluation results

Held-out test set, 1,475 images. Full numbers: [results/metrics.json](results/metrics.json). Text summary: [results/evaluation_summary.txt](results/evaluation_summary.txt). Figure: [results/confusion_matrix.png](results/confusion_matrix.png).

| Metric | Result |
|---|---|
| Accuracy | 0.929 (1,370 / 1,475) |
| Balanced accuracy | 0.931 |
| Macro F1 | 0.927 |
| Majority-class baseline (always glioma) | 0.259 |

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| glioma | 0.997 | 0.825 | 0.903 | 382 |
| meningioma | 0.899 | 0.915 | 0.907 | 390 |
| notumor | 0.846 | 0.993 | 0.914 | 303 |
| pituitary | 0.978 | 0.993 | 0.985 | 400 |

Confusion matrix (rows are the true class, columns are glioma, meningioma, notumor, pituitary):

|  | glioma | meningioma | notumor | pituitary |
|---|---|---|---|---|
| glioma | 315 | 35 | 29 | 3 |
| meningioma | 1 | 357 | 26 | 6 |
| notumor | 0 | 2 | 301 | 0 |
| pituitary | 0 | 3 | 0 | 397 |

Binary tumor detection (tumor if the notumor score is below 0.5): sensitivity 0.953, specificity 0.993, ROC-AUC 0.998, 55 false negatives, 2 false positives.

The 100-image appendix was 92 / 100 correct. Its class mix is glioma 26, meningioma 26, notumor 21, pituitary 27. Eight errors: seven gliomas and one meningioma. Five of those eight were predicted as notumor.

- Accuracy alone is a weak summary because notumor is smaller, so balanced accuracy and macro F1 are reported too. Both are about 0.93, close to accuracy, so the result is not carried by one large class.
- The majority baseline is 0.259. The model’s 0.929 is far above that baseline.
- Pituitary is the easiest class (recall 0.993). Glioma is the failure mode (recall 0.825): 67 of 382 gliomas were missed, and 29 of those were called notumor.
- For a screening-style reading, the binary numbers matter more than the four-class accuracy. Specificity is 0.993, but 55 tumors were still called healthy.
- Many predicted scores are about 1.0. That is overconfidence. These scores are not calibrated probabilities and must not be read as clinical risk.
- The publisher’s own best test accuracy on this cleaned set was about 0.95 with a larger VGG19 model. Our 0.929 is in the same range with a smaller network and an extra exact-duplicate filter.
- Hyperparameters were not changed after seeing the test set.

Every test-image prediction, including the 100, is also in [results/predictions_all.csv](results/predictions_all.csv).

## Appendix A — 100 predictions

Open this file in GitHub. It renders as a table with image name, true label, prediction, correctness, and class scores:

[results/predictions_100.csv](results/predictions_100.csv)

92 of these 100 predictions are correct.
