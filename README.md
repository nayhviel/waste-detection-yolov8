# Recyclable Waste Detection Using YOLOv8s

An object detection project for recognizing recyclable waste using YOLOv8s and the WaRP (Waste Recycling Plant) dataset. This project investigates the effect of lighting augmentation on object detection performance.

## Research Objective

To evaluate whether lighting augmentation can improve the detection of recyclable waste under varying visual conditions.

## Target Classes

The model detects five selected categories:

- Canister
- Cardboard
- Bottle
- Detergent
- Cans

## Models

- **Baseline:** YOLOv8s trained using the baseline configuration.
- **Lighting Augmentation:** YOLOv8s trained with additional lighting augmentation.

## Dataset

The project uses the WaRP Waste Recycling Plant dataset.

Dataset split:
- Training: 2,452 images
- Validation: 491 images
- Testing: 522 images

## Experimental Results

| Metric | Baseline | Lighting Augmentation |
|---|---:|---:|
| Precision | 0.689 | 0.638 |
| Recall | 0.501 | 0.610 |
| mAP@50 | 0.587 | 0.644 |
| mAP@50–95 | 0.460 | 0.506 |

Lighting augmentation improved recall by 10.9 percentage points, mAP@50 by 5.7 percentage points, and mAP@50–95 by 4.6 percentage points. Precision decreased by 5.1 percentage points, indicating a trade-off between detecting more objects and limiting false positives.

*Note: Confirm that the reported metrics correspond to the stated evaluation split before publication.*

## Tech Stack

- Python
- Ultralytics YOLOv8
- PyTorch
- OpenCV
- Streamlit (if used by the demo application)

## Project Files

- `app.py` — application entry point
- `requirements.txt` — Python dependencies
- `baseline_best.pt` — baseline model weights
- `augmented_best.pt` — lighting-augmented model weights
- `FIX FIX hasil riset.ipynb` — research and experimental notebook

## Getting Started

1. Clone this repository.
2. Create and activate a Python virtual environment.
3. Install dependencies from `requirements.txt`.
4. Ensure the model weight files are available at the paths expected by the application.
5. Run the application using the appropriate command for the framework used in `app.py`.

## Limitations

The model was evaluated on a specific dataset and a selected set of five classes. Further testing on unseen environments and different lighting conditions is needed to assess generalization.

## Research Reference

Research title: *Optimasi Kinerja YOLOv8 melalui Augmentasi Pencahayaan untuk Pengenalan Sampah Daur Ulang*.
