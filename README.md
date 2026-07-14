# RAW-Mix: Region-Aware Mixing for Unsupervised Domain Adaptive Object Detection

<p align="center">
  <img src="RawMix%20Pipeline%20for%20GSA.png" alt="RAW-Mix Framework" width="900"/>
</p>

<p align="center">
<b>Official implementation of RAW-Mix: Region-Aware Mixing for Unsupervised Domain Adaptive Object Detection.</b>
</p>

---

## Overview

Deep neural networks achieve impressive performance in object detection when trained on large annotated datasets. However, their performance often deteriorates when deployed in new environments because of **domain shift**, where the source and target domains follow different data distributions.

**RAW-Mix** is a novel **Unsupervised Domain Adaptation (UDA)** framework that improves object detection under domain shift by jointly aligning both **marginal** and **conditional** feature distributions.

Unlike conventional UDA methods that primarily focus on global feature alignment, RAW-Mix introduces a **region-aware mixing strategy** that transfers highly discriminative regions from unlabeled target images into labeled source images, allowing the detector to learn more robust and domain-invariant representations.

The framework combines:

* **Marginal Domain Alignment (MDA)** using adversarial learning.
* **Discriminative Feature Alignment (DFA)** using self-attention-guided region extraction.
* **Region-Aware Mixing** for conditional distribution alignment.
* **Self-supervised consistency learning** on mixed samples.

---

## Framework

RAW-Mix consists of two complementary modules:

### Marginal Domain Alignment (MDA)

MDA aligns global feature distributions between source and target domains using adversarial learning with gradient reversal. This encourages the feature extractor to produce domain-invariant representations that confuse the domain classifier.

### Discriminative Feature Alignment (DFA)

DFA extracts highly informative target regions using:

* YOLOv5 backbone features
* Self-attention maps
* Kernel Density Estimation (KDE)
* Adaptive thresholding

The extracted target regions are blended with labeled source images to generate augmented samples that improve conditional distribution alignment through self-supervised consistency learning.

---

## Region-Aware Mixing

<p align="center">
  <img src="RAWMix_Augment%20(2).png" alt="RAW-Mix Augmentation" width="850"/>
</p>

The proposed Region-Aware Mixing strategy identifies discriminative target-domain regions and integrates them into source-domain images. This allows the detector to observe target-specific visual characteristics while preserving reliable source annotations, significantly improving domain adaptation performance.

---

## Key Features

* Region-aware image mixing for domain adaptation
* Adversarial marginal feature alignment
* Self-supervised consistency learning
* Conditional feature distribution alignment
* Easy integration with existing object detectors
* End-to-end training framework

---

## Main Contributions

* Introduces **RAW-Mix**, a unified UDA framework for object detection using adversarial marginal alignment.
* Enhances class-discriminative representations through self-supervised consistency learning.
* Proposes a novel **Region-Aware Mixing** strategy that blends discriminative target patches with labeled source images.
* Achieves state-of-the-art performance across multiple challenging benchmark datasets, including weather adaptation, synthetic-to-real transfer, and cross-camera adaptation.

---

## Repository Structure

```text
RAW-Mix/
│
├── augmentation_utils/        # Region-aware mixing utilities
├── yolov5/                    # Modified YOLOv5 implementation
├── datasets/                  # Dataset loaders
├── models/                    # Network modules
├── utils/                     # Utility functions
├── configs/                   # Configuration files
├── scripts/                   # Training/testing scripts
├── requirements.txt
└── README.md
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/andualemw1/RAW-Mix-Region-Aware-Mixing.git

cd RAW-Mix-Region-Aware-Mixing
```

Create a Python environment:

```bash
conda create -n rawmix python=3.9

conda activate rawmix
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Dataset Preparation

The framework supports several standard UDA benchmarks:

* Cityscapes
* Foggy Cityscapes
* Sim10K
* KITTI

Organize the datasets according to the directory structure specified in the configuration files.

---

## Training

Example:

```bash
python train.py
```

or

```bash
python yolov5/train.py
```

depending on the project structure.

---

## Evaluation

```bash
python test.py
```

---

## Experimental Results

RAW-Mix consistently improves object detection performance under significant domain shifts by jointly optimizing:

* Marginal feature alignment
* Conditional feature alignment
* Region-aware augmentation
* Self-supervised consistency learning

Extensive experiments demonstrate strong performance on:

* Cityscapes → Foggy Cityscapes
* Sim10K → KITTI

---

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{tulu2025rawmix,
  title={RAW-Mix: Region-Aware Mixing for Unsupervised Domain Adaptive Object Detection},
  author={Tulu, Andualem Welabo and Conci, Nicola},
  booktitle={International Conference on Image Analysis and Processing (ICIAP)},
  year={2025}
}
```

---

## Acknowledgements

This work was conducted at the **University of Trento** in collaboration with the **Multimedia and Human Understanding Group**, with support from the **LIVIA Laboratory (ÉTS Montréal)**.

---

## License

This project is released under the MIT License.
