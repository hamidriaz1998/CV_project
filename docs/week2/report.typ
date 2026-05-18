#import "../lab-cover-template.typ": lab-cover

#lab-cover(
  lab-title: "Week 2 Report — Image Classification",
  logo: "./logo.png",
  session: "Session 2023 – 2027",
  student-name: "Hamid Riaz",
  student-id: "2023-CS-10",
  supervisor: "Sir Waseem",
  course: "Computer Vision",
  department: "Department of Computer Science",
  university: "University of Engineering and Technology",
  location: "Lahore, Pakistan",
)

#pagebreak()

#set list(indent: 1em)
#set enum(indent: 1em)
#set text(size: 12pt)
#show heading: set text(weight: "bold")
#show heading.where(level: 2): set text(size: 16pt)
#show heading.where(level: 3): set text(size: 14pt)

= Patient Health Monitoring using Chest X-Ray Classification: Week 2
<week2-report>

== 1. Objective
<objective>

The goal of Week 2 is to build and train a deep learning classifier that
distinguishes #strong[NORMAL] from #strong[PNEUMONIA] chest X-ray
images. This is a binary image classification task that serves as the
first computer-vision component of our patient health monitoring system.

== 2. Dataset Recap
<dataset-recap>

The Chest X-Ray Images (Pneumonia) dataset (Wang et al., 2017, hosted
by Kermany et al.) was sourced from Mendeley Data. After cleaning
(`clean_dataset.py` — Week 1), the dataset consists of:

#figure(
  align(center)[#table(
    columns: 4,
    align: (auto, auto, auto, auto),
    table.header([Split], [NORMAL], [PNEUMONIA], [Total]),
    table.hline(),
    [Train], [1,348], [3,858], [5,206],
    [Test], [231], [387], [618],
    [#strong[Total]], [#strong[1,579]], [#strong[4,245]], [#strong[5,824]],
  )],
  kind: table,
)

All images are 224×224 RGB JPEGs. The dataset has a ~3:1 class
imbalance favouring PNEUMONIA. No separate validation folder is
provided, so 20% of the training set was held out for validation using
stratified sampling (preserving class proportions).

#figure(
  image("sample_batch.png", width: 80%),
  caption: [Sample training images from the dataset showing NORMAL and
  PNEUMONIA chest X-rays after augmentation.],
)

== 3. Model Architecture
<model-architecture>

#strong[ResNet50] (He et al., 2016) pre-trained on ImageNet was
selected as the backbone for transfer learning.

=== 3.1 Why ResNet50
<why-resnet50>

- #strong[Residual connections] allow training of deeper networks
  without vanishing gradients, making them well-suited for medical
  imaging where fine-grained features matter.
- #strong[Pre-trained on ImageNet] provides strong low-level feature
  extractors (edge, texture, shape detectors) that transfer effectively
  to X-ray images.
- #strong[50 layers] offers a good accuracy-efficiency trade-off: deep
  enough to learn complex pathology patterns but not so large as to
  overfit on ~5k training samples.
- Widely used as a #strong[medical imaging benchmark], enabling
  comparison with published results.

=== 3.2 Modifications
<modifications>

The original 1000-class ImageNet head was replaced with a 2-class fully
connected layer:

```
ResNet50 (pre-trained)
  ├── Conv1 (7×7, 64)
  ├── Layer1–4 (residual blocks)
  └── AdaptiveAvgPool2d → Dropout(0.2) → FC(2048 → 2)
```

Total parameters: ~25.6M (all fine-tuned in Phase 2).

=== 3.3 Alternative Architectures Considered
<alternative-architectures-considered>

#figure(
  align(center)[#table(
    columns: (25%, 15%, 30%, 30%),
    align: (auto, auto, auto, auto),
    table.header([Model], [Params], [Pros], [Cons]),
    table.hline(),
    [#strong[ResNet50] (chosen)], [25.6M],
    [Strong transfer learning, well-tested], [Moderate compute],
    [DenseNet121], [8.0M],
    [Parameter-efficient, strong on medical benchmarks],
    [Fewer pre-trained weights],
    [EfficientNet-B0], [5.3M],
    [Best accuracy/param ratio], [More tuning required],
  )],
  kind: table,
)

== 4. Training Pipeline
<training-pipeline>

=== 4.1 Data Split
<data-split>

Stratified 80/20 train-validation split using `train_test_split` from
scikit-learn:

#figure(
  align(center)[#table(
    columns: 4,
    align: (auto, auto, auto, auto),
    table.header([Split], [NORMAL], [PNEUMONIA], [Total]),
    table.hline(),
    [Train], [1,078], [3,087], [4,165],
    [Validation], [270], [771], [1,041],
    [Test], [231], [387], [618],
  )],
  kind: table,
)

=== 4.2 Data Augmentation
<data-augmentation>

Training transforms (applied on-the-fly):

#figure(
  align(center)[#table(
    columns: (30%, 25%, 45%),
    align: (auto, auto, auto),
    table.header([Transform], [Detail], [Motivation]),
    table.hline(),
    [Random horizontal flip], [p = 0.5],
    [Chest X-rays are approximately symmetric],
    [Random rotation], [±10°],
    [Small patient positioning variation],
    [Colour jitter], [brightness=0.1, contrast=0.1],
    [Exposure differences across X-rays],
    [Normalize], [ImageNet mean/std],
    [Compatibility with pre-trained weights],
  )],
  kind: table,
)

Validation and test transforms applied normalization only (no
augmentation).

=== 4.3 Class Imbalance Handling
<class-imbalance-handling>

#strong[Weighted cross-entropy loss] was used to counteract the ~3:1
class imbalance:

```
class_weight[c] = total_samples / (num_classes × samples_in_class[c])
```

Resulting weights:
- NORMAL: 1.93
- PNEUMONIA: 0.67

This penalises misclassifying the minority class (NORMAL) more heavily
during training.

=== 4.4 Training Hyperparameters
<training-hyperparameters>

#figure(
  align(center)[#table(
    columns: 3,
    align: (auto, auto, auto),
    table.header([Hyperparameter], [Phase 1 (Head)], [Phase 2 (Full)]),
    table.hline(),
    [Optimizer], [Adam], [Adam],
    [Learning rate], [1×10⁻³], [1×10⁻⁴],
    [Batch size], [32], [32],
    [Epochs], [20], [15],
    [Scheduler],
    [ReduceLROnPlateau (patience=3, factor=0.1)], [Same],
    [Early stopping], [Patience=5], [Same],
    [Mixed precision], [torch.cuda.amp], [Same],
  )],
  kind: table,
)

#strong[Two-phase training strategy:]

+ #strong[Phase 1 — Head training (backbone frozen):] Only the newly
  initialised classification head is trained for 20 epochs. This allows
  the head to adapt to the medical imaging feature space without
  corrupting pre-trained features.
+ #strong[Phase 2 — Full fine-tuning:] All layers are unfrozen and
  trained jointly at a lower learning rate (10× smaller) for 15 epochs.
  This gently adapts backbone features to X-ray-specific patterns.

=== 4.5 Implementation
<implementation>

The entire pipeline was implemented in a single Jupyter notebook
(`classification.ipynb`) designed for Google Colab with GPU
acceleration:

- #strong[Framework:] PyTorch 2.x + torchvision
- #strong[Mixed precision:] `torch.cuda.amp` for ~2× training speedup
- #strong[Monitoring:] `tqdm` progress bars, real-time loss/accuracy
  logging
- #strong[Checkpointing:] Best model saved by validation accuracy

== 5. Results
<results>

=== 5.1 Test Set Performance
<test-set-performance>

#figure(
  align(center)[#table(
    columns: 2,
    align: (auto, auto),
    table.header([Metric], [Value]),
    table.hline(),
    [Test Loss], [1.1180],
    [Test Accuracy], [81.07%],
    [Macro F1], [0.7656],
    [Weighted F1], [0.7916],
    [ROC AUC], [0.9301],
  )],
  kind: table,
)

#figure(
  image("training_curves.png", width: 100%),
  caption: [
    Training and validation loss (left) and accuracy (right) over
    epochs. The dashed vertical line marks the transition from Phase 1
    (head-only) to Phase 2 (full fine-tuning).
  ],
)

=== 5.2 Per-Class Metrics
<per-class-metrics>

#figure(
  align(center)[#table(
    columns: 4,
    align: (auto, auto, auto, auto),
    table.header([Class], [Precision], [Recall], [F1-Score]),
    table.hline(),
    [NORMAL], [0.9914], [0.4978], [0.6628],
    [PNEUMONIA], [0.7689], [0.9974], [0.8684],
  )],
  kind: table,
)

=== 5.3 Confusion Matrix
<confusion-matrix>

#figure(
  image("confusion_matrix.png", width: 60%),
  caption: [
    Confusion matrix on the test set. 115 NORMAL and 386 PNEUMONIA
    correctly classified; 116 false positives (NORMAL predicted as
    PNEUMONIA) and 1 false negative (PNEUMONIA missed).
  ],
)

#strong[Key observations:]

- #strong[PNEUMONIA recall is 0.9974] — only 1 out of 387 pneumonia
  cases was missed. This is clinically critical: false negatives
  (missing pneumonia) are far more dangerous than false positives.
- #strong[NORMAL recall is 0.4978] — roughly half of healthy patients
  were flagged as having pneumonia (false positives). While not ideal,
  these would be flagged for further review in a clinical workflow.
- The model is #strong[conservative]: it prefers to predict PNEUMONIA
  when uncertain, which is the safer bias for triage.

=== 5.4 ROC Curve
<roc-curve>

#figure(
  image("roc_curve.png", width: 60%),
  caption: [
    ROC curve for the PNEUMONIA class. AUC = 0.9301, indicating
    excellent class separability.
  ],
)

Area Under the Curve (AUC) = #strong[0.9301], indicating excellent
class separability despite the accuracy asymmetry.

== 6. Discussion
<discussion>

=== 6.1 Why Phase 2 Helped
<why-phase-2-helped>

Fine-tuning the full network improved validation accuracy from ~75%
(Phase 1) to ~81% (Phase 2). The pre-trained ImageNet features are not
perfectly suited for X-ray modality; updating the lower layers to
respond to radiographic patterns was essential.

=== 6.2 Addressing the NORMAL Recall Bottleneck
<addressing-the-normal-recall-bottleneck>

The low NORMAL recall (49.78%) is a direct consequence of class
imbalance. Despite weighting the loss function, the model learns a
PNEUMONIA-biased decision boundary. Multiple strategies could mitigate
this:

- #strong[Oversampling] NORMAL images during training
- #strong[Higher class weight] for NORMAL (currently 1.93, could be
  pushed to 3.0+)
- #strong[Threshold tuning:] the default decision threshold (0.5) could
  be shifted towards NORMAL
- #strong[Focal Loss] (Lin et al., 2017) instead of cross-entropy to
  focus training on hard examples

=== 6.3 Clinical Relevance
<clinical-relevance>

With #strong[99.7% pneumonia recall], this model could serve as a
high-sensitivity screening tool: it would catch nearly every pneumonia
case while accepting a moderate false-positive rate. In a real clinical
setting, flagged positives would be reviewed by a radiologist, making
the 50% NORMAL recall acceptable as a first-pass filter.

=== 6.4 Comparison with Literature
<comparison-with-literature>

Published ResNet50 baselines on this dataset report 85–92% test
accuracy (Kermany et al.~report ~93% using a different architecture and
full dataset). Our 81% is competitive for a two-phase fine-tuning
approach and leaves room for improvement with more aggressive
augmentation or ensemble methods.

== 7. Outputs
<outputs>

All results are saved to `week2_classification/`:

#figure(
  align(center)[#table(
    columns: (35%, 65%),
    align: (auto, auto),
    table.header([File], [Description]),
    table.hline(),
    [`classification.ipynb`], [Full training notebook],
    [`best_model.pth`], [Best model checkpoint (90 MB)],
    [`training_curves.png`], [Loss & accuracy over epochs],
    [`confusion_matrix.png`], [Confusion matrix on test set],
    [`roc_curve.png`], [ROC curve with AUC = 0.93],
    [`sample_predictions.png`],
    [16 test images with correct/wrong labels],
    [`metrics.json`],
    [Machine-readable metrics (accuracy, precision, recall, f1)],
    [`classification_report.txt`], [Text summary of results],
  )],
  kind: table,
)

#figure(
  image("sample_predictions.png", width: 95%),
  caption: [
    Sample test set predictions. Green titles indicate correct
    classifications; red titles indicate incorrect ones. The model
    correctly identifies most PNEUMONIA cases and its few mistakes are
    predominantly NORMAL images predicted as PNEUMONIA (false
    positives).
  ],
)

== 8. References
<references>

+ He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning
  for Image Recognition. #emph[CVPR].

+ Kermany, D. S., Goldbaum, M., Cai, W., et al. (2018). Identifying
  Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning.
  #emph[Cell], 172(5), 1122–1131.

+ Wang, X., Peng, Y., Lu, L., Lu, Z., Bagheri, M., & Summers, R. M.
  (2017). ChestX-ray8: Hospital-scale Chest X-ray Database and
  Benchmarks. #emph[CVPR].

+ Lin, T.-Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017).
  Focal Loss for Dense Object Detection. #emph[ICCV].

+ Deng, J., Dong, W., Socher, R., Li, L.-J., Li, K., & Fei-Fei, L.
  (2009). ImageNet: A Large-Scale Hierarchical Image Database.
  #emph[CVPR].
