# GradTrack: Detecting Noisy Labels via Temporal Trajectories of Class-wise Gradient Misalignment

GradTrack: Detecting Noisy Labels via Temporal Trajectories of Class-wise Gradient Misalignment, NeurIPS 2026

Created by Townim Faisal Chowdhury and Zhibin Liao

## Introduction

Existing noisy-label detection methods rely on scalar signals such as loss, confidence, or gradient magnitude, which capture the scale of prediction errors but ignore the direction and temporal dynamics of gradients. We propose **GradTrack**, which estimates the class-wise gradient direction that clean labels would induce and measures how far each sample's gradient deviates from it. In each epoch, samples are ranked by this *class-wise gradient misalignment score*; the ranks across epochs form a *temporal rank trajectory*, and a Gaussian mixture model separates noisy samples from clean ones. GradTrack performs strongly on synthetic and real-world noise, especially instance-dependent noise, and works as a plug-in sample selection module for learning with noisy labels.

## Setup

The code was tested on a single Nvidia RTX A6000 GPU with Python 3.10.10, running Ubuntu 22.04.2 LTS. To install the required packages, use the following command:
```
pip install -r requirements.txt
```

CIFAR-100 is downloaded to `./data` on first use, and the noisy labels for each setting are generated and cached there.

## Directories

- [data.py](data.py): CIFAR-100 with symmetric, asymmetric, and instance-dependent label noise.
- [model.py](model.py): frozen CLIP ViT-B/32 image encoder and the MLP classification head.
- [train.py](train.py): trains the classification head on noisy labels and saves a checkpoint after every epoch to `./checkpoints`.
- [gradtrack.py](gradtrack.py): computes per-sample gradients of the final linear layer at each checkpoint, builds the temporal rank trajectories, and detects noisy-label samples. The F1 score is saved to `./results`.

## Running the models

We provide the [run.sh](run.sh) script to reproduce the CIFAR-100 results in Table 1 of the paper:
```
bash run.sh
```

To run a single setting:
```
python train.py --noise_type instance --noise_rate 0.4
python gradtrack.py --noise_type instance --noise_rate 0.4
```

## Source

CLIP: https://github.com/openai/CLIP

## Citation

```bibtex
@inproceedings{gradtrack,
	title        = {{GradTrack}: Detecting Noisy Labels via Temporal Trajectories of Class-wise Gradient Misalignment},
	author       = {Faisal Chowdhury, Townim and Duc Huy, Ta and Minh Hieu Phan, Vu and Hengel, Anton van den and W. Verjans, Johan and Carneiro, Gustavo and Liao, Zhibin},
	year         = 2026,
	booktitle    = {Advances in Neural Information Processing Systems (NeurIPS)}
}
```
