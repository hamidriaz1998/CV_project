#import "../lab-cover-template.typ": lab-cover

#lab-cover(
  lab-title: "Dataset Proposal",
  logo: "./logo.png",
  session: "Session 2023 – 2027",
  student-name: "Hamid Riaz",
  student-id: "2023-CS-10",
  supervisor: "Sir Waseem",
  course: "Copmuter Vision",
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

= Dataset Proposal
<dataset-proposal>
== 1. Project Title
<project-title>
Patient Health Monitoring using Chest X-Ray Classification

== 2. Selected Dataset
<selected-dataset>
#strong[Chest X-Ray Images (Pneumonia)] --- Wang et al.~2017, hosted by
Kermany et al.

#strong[Source:]
#link("https://data.mendeley.com/datasets/rscbjbr9sj/2")[Mendeley Data]
--- `ChestXRay2017.zip` (1.15 GB)

#strong[Original paper:] Wang, X. et al.~"ChestX-ray8: Hospital-scale
Chest X-ray Database and Benchmarks." arXiv:1705.02315, 2017.

#strong[Kermany et al.~using the dataset:] Kermany, D.S. et
al.~"Identifying Medical Diagnoses and Treatable Diseases by Image-Based
Deep Learning." #emph[Cell], 172(5), 2018. DOI:
#link("https://doi.org/10.1016/j.cell.2018.02.010")[10.1016/j.cell.2018.02.010]
--- the dataset is re-hosted on Mendeley by Kermany et al.~via their
reproducibility study.

== 3. Dataset Description
<dataset-description>
The dataset consists of #strong[5,856 chest X-ray JPEG images]
classified into two classes --- #strong[NORMAL] and #strong[PNEUMONIA]
--- provided pre-split into `train/` and `test/` directories.

All images are anterior-posterior (AP) chest radiographs from
#strong[pediatric patients aged 1--5], collected retrospectively from
Guangzhou Women and Children's Medical Center. Image quality was
validated and grades assigned by two expert physicians, with a third
expert adjudicating the evaluation set.

== 4. Why This Dataset
<why-this-dataset>
- #strong[Real clinical data] with expert-verified labels, making it
  directly applicable to patient health monitoring.
- Contains sufficient samples to support deep learning training without
  requiring synthetic data augmentation as a prerequisite.
- #strong[Widely used benchmark] in the medical imaging ML community,
  enabling verifiable comparison against published results.

== 5. Preprocessing Steps
<preprocessing-steps>
`clean_dataset.py` applies a three-phase pipeline to every image across
all splits and classes:

#figure(
  align(center)[#table(
    columns: (22%, 36%, 48%),
    align: (auto, auto, auto),
    table.header([Phase], [Method], [Result]),
    table.hline(),
    [#strong[Corruption detection]],
    [Open each file with Pillow, call
      `.verify()`],
    [Corrupt/unreadable files silently deleted],
    [#strong[Deduplication]],
    [Compute MD5 hash of raw bytes; first
      occurrence kept],
    [Duplicate images removed],
    [#strong[Standardization]],
    [Resize to 224×224 px using LANCZOS
      resampling; convert grayscale to RGB],
    [All images normalized to
      `(224, 224, 3)` format compatible with standard CNN architectures],
  )],
  kind: table,
)

Surviving images are saved back in place, overwriting originals.

== 6. Annotation Plan
<annotation-plan>
A custom pygame-based bounding-box annotator (`annotator.py`) is used to
delineate the lung field region in chest X-rays:

- Images are loaded one at a time from `chest_xray/train/`, shuffled
  randomly.
- The user draws a single bounding box per image via click-and-drag.
- After drawing, the label is assigned by pressing #strong[N] (NORMAL)
  or #strong[P] (PNEUMONIA).
- #strong[S] saves the annotation; #strong[R] redoes the box; #strong[Q]
  quits and persists progress.
- Annotations are saved to `annotations.csv` with columns
  `filename, label, x1, y1, x2, y2`.
- Already-annotated images are skipped on resume.

#strong[Target for Week 1:] 20 annotated sample images, to be expanded
in subsequent weeks.

== 7. Class Distribution
<class-distribution>
Actual counts from `clean_dataset.py` after preprocessing:

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
