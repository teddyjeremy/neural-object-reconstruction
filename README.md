# Neural Object Reconstruction & Relighting Lab

> An end-to-end computer vision and neural rendering research project for reconstructing real-world objects from short phone videos and producing interactive, relightable 3D representations.

![Status](https://img.shields.io/badge/status-in%20development-orange)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-deep%20learning-red)
![CUDA](https://img.shields.io/badge/CUDA-GPU-green)
![Three.js](https://img.shields.io/badge/Three.js-WebGPU-black)

## Overview

The **Neural Object Reconstruction & Relighting Lab** explores an end-to-end pipeline that takes a 20–60 second handheld phone video of an object and turns it into an interactive 3D representation.

The long-term system combines foreground segmentation, camera estimation, monocular depth, multi-view reconstruction, 3D Gaussian Splatting, neural material decomposition, uncertainty estimation, and real-time WebGPU visualization.

The central research goal is to move beyond reproducing an object's observed RGB appearance and instead estimate its underlying material and lighting components so that the reconstructed object can be rendered under new lighting conditions.

## System Architecture

```text
20–60 s PHONE VIDEO
        │
        ▼
 Video preprocessing
        │
        ▼
 SAM 2 + BiRefNet
 Foreground segmentation
        │
        ▼
 Camera / pose estimation
        │
        ▼
 Depth estimation
        │
        ▼
 Multi-view reconstruction
        │
        ▼
 3D Gaussian Splatting
        │
        ▼
 Neural material decomposition
        │
        ├── Albedo
        ├── Roughness
        ├── Specular
        ├── Normal
        ├── Illumination
        └── Uncertainty
        │
        ▼
 Relightable neural object
        │
        ▼
 Three.js / WebGPU
        │
        ▼
 Interactive 3D viewer
```

## Core Research Idea

Most reconstruction pipelines primarily learn to reproduce the appearance observed in the input images. This project investigates whether a neural representation can instead separate intrinsic object properties from the lighting conditions in which the object was captured.

Conceptually:

```text
Input image
    │
    ▼
Neural material network
    │
    ├── Albedo
    ├── Roughness
    ├── Specular
    ├── Normal
    ├── Illumination
    └── Uncertainty
```

The predicted components can then be used to synthesize the object's appearance under lighting that was not present in the original capture.

## Uncertainty-Aware Material Decomposition

The main project-specific deep-learning feature is **neural material decomposition with uncertainty**.

Instead of predicting only a material value, the network is designed to estimate both a prediction and its uncertainty. This is useful for regions that are poorly constrained because of reflections, transparency, shadows, occlusions, textureless surfaces, motion blur, or insufficient viewpoints.

The uncertainty representation can be used for:

- uncertainty-aware training losses
- identifying unreliable reconstruction regions
- material refinement
- reconstruction diagnostics
- visualizing confidence in the final viewer

## Segmentation: SAM 2 + BiRefNet

The project intentionally combines **SAM 2** and **BiRefNet** rather than treating them as interchangeable models.

### SAM 2

SAM 2 provides object-level segmentation and temporal mask propagation across video frames. It is used as the primary mechanism for maintaining consistent object masks through the input sequence.

### BiRefNet

BiRefNet is used as a complementary refinement stage for high-quality object boundaries and fine structures.

```text
Video
  │
  ▼
SAM 2
  │
  ├── object mask
  └── temporal propagation
  │
  ▼
BiRefNet
  │
  ▼
refined foreground mask
```

The combination is intended to improve difficult boundaries such as thin structures, handles, cables, and fine object details.

## Dataset Strategy

The first dataset milestone is a curated collection of approximately **200 3D objects** from the Objaverse ecosystem.

The 200 objects are not treated as only 200 training examples. They are the source assets from which a much larger procedural synthetic dataset will be generated in Blender.

The initial target is:

```text
10 object categories
×
20 objects per category
=
200 curated objects
```

Example categories include:

- Furniture
- Kitchen / dining
- Electronics
- Tools
- Lighting
- Clothing / accessories
- Toys / games
- Musical instruments
- Sports / outdoor
- Decorative / household objects

The final category distribution may change during quality-control and licensing review.

## Why 200 Objects?

Each 3D object can generate many supervised observations through controlled rendering.

For example:

```text
200 objects
   ×
80 camera viewpoints
   ×
8 lighting configurations
   =
128,000 rendered observations
```

Additional material variations, camera variation, augmentation, and scene randomization can expand the training corpus further.

The important evaluation rule is that the final test set is split by **object identity**, not randomly by rendered image, to avoid leakage between training and testing.

## Dataset Pipeline

```text
Objaverse
   │
   ▼
Metadata filtering
   ├── license filtering
   ├── category filtering
   ├── geometry filtering
   ├── duplicate prevention
   └── quality filtering
   │
   ▼
200 curated objects
   │
   ▼
Blender procedural rendering
   ├── camera randomization
   ├── lighting randomization
   ├── material randomization
   ├── object rotation
   └── background variation
   │
   ▼
Synthetic training dataset
```

## Blender Synthetic Data Generation

Blender will be automated rather than requiring every object to be configured manually.

The rendering pipeline will generate variations of:

- camera pose
- camera focal length
- object rotation
- light direction
- light intensity
- light color
- background
- material parameters

Because Blender knows the underlying scene, it can also produce supervised ground truth.

A generated sample can contain:

```text
rgb.png
mask.png
depth.exr
normal.exr
albedo.png
roughness.png
specular.png
illumination.exr
camera.json
```

The exact representation and file formats will be finalized during the synthetic-data stage.

## Dataset Provenance

Every selected object receives a permanent internal dataset identifier while retaining its original source UID and metadata.

Example:

```text
Dataset ID:       OBJ_0042
Source UID:       <Objaverse UID>
Category:         electronics
Source:           Objaverse
License:          <source license>
```

This makes generated samples traceable back to their original source asset.

Third-party asset licenses are tracked at the asset level. The license of this repository does **not** automatically apply to third-party 3D assets.

## Dataset Storage

Large binary assets are intentionally separated from source code.

```text
GitHub
  └── source code, notebooks, configs, documentation

Hugging Face
  └── 3D assets and generated datasets

Colab / GPU worker
  └── temporary processing and training
```

The raw dataset repository is planned as:

`ndeda/neural-object-reconstruction-raw`

Large `.glb`, rendered images, caches, and model artifacts should not be committed to this Git repository.

## Model Strategy

The project is designed around pretrained foundation models and targeted fine-tuning rather than training every component from scratch.

Planned pretrained components include:

| Component | Planned approach |
|---|---|
| Foreground segmentation | SAM 2 |
| Boundary refinement | BiRefNet |
| Depth | Pretrained depth foundation model |
| Neural scene representation | 3D Gaussian Splatting |
| Material decomposition | Project-specific PyTorch model |
| Relighting | Learned / differentiable rendering pipeline |

The material decomposition and uncertainty components are the primary areas for project-specific model development.

## Training Objective

The material network will be developed as a multi-task model. A conceptual total loss is:

```text
L_total =
    λa L_albedo
  + λr L_roughness
  + λs L_specular
  + λn L_normal
  + λl L_illumination
  + λrender L_render
  + λu L_uncertainty
```

The exact architecture, losses, and weighting will be determined experimentally.

## Unseen-Object Evaluation

The dataset will be split by object identity rather than image identity.

An initial split can be:

```text
160 objects → training
20 objects  → validation
20 objects  → testing
```

All views and lighting configurations of an object must remain within the same split.

This makes the test set measure generalization to objects the model has never seen during training.

## Planned Evaluation

### Segmentation

- IoU
- Dice score
- Boundary F-score

### Depth

- Absolute Relative Error
- RMSE
- threshold accuracy

### Novel-view synthesis

- PSNR
- SSIM
- LPIPS

### Material decomposition

- albedo error
- roughness error
- specular error
- normal angular error

### Relighting

Predicted relit images will be compared against ground-truth Blender renders under lighting configurations not provided during inference.

## Interactive Viewer

The final system will provide a browser-based viewer using Three.js and WebGPU.

Users will be able to manipulate:

### Camera

- orbit
- zoom
- perspective

### Object

- rotation
- position
- scale

### Lighting

- direction
- intensity
- color

### Material

- albedo
- roughness
- specular

### Diagnostics

- uncertainty visualization
- reconstruction confidence
- potentially depth / mask overlays

## Backend Architecture

The planned inference backend uses asynchronous GPU jobs:

```text
FastAPI
   │
   ▼
Redis
   │
   ▼
Celery
   │
   ▼
GPU Worker
   ├── segmentation
   ├── depth estimation
   ├── reconstruction
   └── material decomposition
```

This prevents a long-running reconstruction job from blocking the API request.

## Repository Structure

The repository is being developed toward the following structure:

```text
neural-object-reconstruction/
│
├── notebooks/
│   ├── 01_objaverse_selection.ipynb
│   ├── 02_download_and_verify.ipynb
│   ├── 03_blender_dataset_generation.ipynb
│   ├── 04_dataset_quality_control.ipynb
│   ├── 05_segmentation.ipynb
│   ├── 06_depth_estimation.ipynb
│   ├── 07_material_decomposition.ipynb
│   └── 08_training.ipynb
│
├── scripts/
│   ├── objaverse_selector.py
│   ├── objaverse_downloader.py
│   ├── hf_uploader.py
│   ├── blender_renderer.py
│   ├── dataset_validator.py
│   └── training.py
│
├── configs/
│   ├── dataset.yaml
│   ├── rendering.yaml
│   └── training.yaml
│
├── models/
│   ├── segmentation/
│   ├── depth/
│   ├── reconstruction/
│   └── materials/
│
├── web/
│   └── viewer/
│
├── docs/
│   ├── architecture.md
│   ├── dataset.md
│   └── experiments.md
│
├── requirements.txt
├── .gitignore
└── README.md
```

## Development Roadmap

### Phase 1 — Dataset Foundation

- [x] Define project architecture
- [x] Define initial dataset strategy
- [x] Define object categories
- [x] Build Objaverse metadata discovery pipeline
- [x] Implement candidate filtering
- [ ] Select final 200 objects
- [ ] Download and verify raw assets
- [ ] Record complete provenance metadata
- [ ] Upload assets to Hugging Face
- [ ] Validate dataset

### Phase 2 — Blender Dataset Generation

- [ ] Automate Blender import
- [ ] Normalize object scale
- [ ] Generate camera trajectories
- [ ] Generate lighting conditions
- [ ] Generate material variations
- [ ] Render RGB
- [ ] Render masks
- [ ] Render depth
- [ ] Render normals
- [ ] Render albedo
- [ ] Render roughness
- [ ] Render specular
- [ ] Export camera parameters

### Phase 3 — Segmentation

- [ ] Integrate SAM 2
- [ ] Integrate BiRefNet
- [ ] Compare segmentation outputs
- [ ] Build mask refinement pipeline
- [ ] Evaluate segmentation quality

### Phase 4 — Depth and Camera

- [ ] Integrate depth foundation model
- [ ] Evaluate depth quality
- [ ] Estimate camera poses
- [ ] Align depth and camera coordinates

### Phase 5 — 3D Reconstruction

- [ ] Integrate 3D Gaussian Splatting
- [ ] Train Gaussian representations
- [ ] Evaluate novel-view synthesis
- [ ] Evaluate reconstruction quality

### Phase 6 — Neural Material Decomposition

- [ ] Design material network
- [ ] Predict albedo
- [ ] Predict roughness
- [ ] Predict specular
- [ ] Predict normals
- [ ] Predict illumination
- [ ] Predict uncertainty
- [ ] Implement multi-task loss
- [ ] Train on synthetic data
- [ ] Evaluate on unseen objects

### Phase 7 — Relighting

- [ ] Separate material from illumination
- [ ] Implement differentiable rendering
- [ ] Train relighting model
- [ ] Evaluate unseen lighting environments

### Phase 8 — WebGPU Viewer

- [ ] Build Three.js viewer
- [ ] Integrate WebGPU
- [ ] Load reconstructed representation
- [ ] Add camera controls
- [ ] Add object controls
- [ ] Add dynamic lighting
- [ ] Add material controls
- [ ] Visualize uncertainty

### Phase 9 — End-to-End Platform

- [ ] Integrate all components
- [ ] Build FastAPI backend
- [ ] Add Redis/Celery job queue
- [ ] Add GPU workers
- [ ] Build upload interface
- [ ] Add processing status
- [ ] Optimize inference
- [ ] Benchmark complete pipeline

## Technology Stack

### Deep Learning

- Python
- PyTorch
- CUDA
- TorchVision

### Computer Vision

- OpenCV
- SAM 2
- BiRefNet
- Depth foundation models

### 3D / Graphics

- Blender
- 3D Gaussian Splatting
- Three.js
- WebGPU

### Backend

- FastAPI
- Redis
- Celery

### Data

- Objaverse
- Hugging Face Hub

### Experimentation

- Google Colab
- GPU compute

## Current Status

**Current stage: Dataset preparation.**

The immediate milestone is to produce a clean, reproducible 200-object source dataset with complete provenance and licensing metadata.

The next major stage is procedural Blender rendering, which will turn those source assets into a substantially larger supervised synthetic dataset for model development.

The neural reconstruction, material decomposition, uncertainty, relighting, and WebGPU components are planned stages of the project and are not represented as completed until they are actually implemented and evaluated.

## Research Direction

The project ultimately aims to demonstrate a complete pipeline:

```text
Phone video
    ↓
Robust foreground extraction
    ↓
Camera and depth estimation
    ↓
3D neural reconstruction
    ↓
Material decomposition
    ↓
Uncertainty estimation
    ↓
Relightable neural representation
    ↓
Interactive WebGPU rendering
```

The key research question is whether neural reconstruction can produce a representation that is not merely view-dependent RGB, but useful for **material-aware rendering and relighting under novel conditions**.

## Reproducibility

Dataset selection records will preserve:

- source UID
- internal dataset ID
- category
- license metadata
- source URL
- geometry statistics
- selection configuration
- random seed where applicable

Generated datasets will be versioned separately from the source code so that experiments can be reproduced without committing large binary assets to Git.

## License and Third-Party Data

The source code license for this repository will be specified separately.

Third-party datasets and 3D assets retain their own licenses and attribution requirements. In particular, an object's presence in the project does not imply that the asset is freely redistributable for every purpose.

Asset-level provenance and license metadata will therefore be preserved throughout the pipeline.

## Author

**Jeremy Teddy Ndeda**

Deep Learning • Computer Vision • Neural Rendering • 3D Graphics

## Project Status

🚧 **Active research and development**

Current milestone:

```text
Objaverse metadata
       ↓
Candidate filtering
       ↓
200 curated objects
       ↓
Hugging Face dataset
       ↓
Blender procedural rendering
       ↓
Synthetic training dataset
```
