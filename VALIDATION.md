# Validation performed

- Python syntax compilation passed.
- Real downloaded version 2 dataset folder discovery, all 6,507 image decodes, split/class validation and exact decoded-pixel audit passed.
- Audit excluded 104 images (duplicate/conflict groups): remaining train 4,435; validation 493; test 1,475. Source files are never modified.
- End-to-end synthetic smoke test passed using actual EfficientNetB0 with random initialization: head epoch, fine-tuning epoch, validation selection, checkpoint saving/loading, evaluation, exactly 100 appendix records, plots, and single-image prediction.
- Smoke test used random synthetic images; its scores are meaningless and are not included as project results.
- Full training was run locally on the real MRI set with pretrained EfficientNet-B0 weights on an NVIDIA RTX 1000 Ada (torch 2.7.1+cu126). Reportable numbers are in results/metrics.json, results/evaluation_summary.txt, and results/predictions_100.csv.
- The checkpoint used for those numbers is the validation-best model (fine-tune epoch 12, validation macro F1 0.988). Held-out test accuracy was 0.929 and test macro F1 was 0.927. These test numbers were not used to choose the checkpoint.
