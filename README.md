# Computer-Vision-Examples

---

# Supervised Contrastive Learning on Fashion MNIST using TensorFlow/Keras

This project demonstrates how to implement supervised contrastive learning on the Fashion MNIST dataset using only built-in modules from TensorFlow, Keras, and scikit-learn. It showcases a two-stage training pipeline: contrastive pretraining followed by classifier fine-tuning.

## Overview

Contrastive learning is a technique that learns embeddings by bringing similar samples closer and pushing dissimilar ones apart. In supervised contrastive learning, label information is used to guide this process. This implementation follows these steps:

1. Normalize and flatten Fashion MNIST images.
2. Pretrain an encoder and projector using a supervised NT-Xent loss.
3. Visualize learned representations using PCA.
4. Freeze the encoder and train a classifier on top of it.
5. Evaluate the model using classification accuracy and a confusion matrix.

## Components

- **Encoder**: A two-layer dense neural network with L2-normalized outputs.
- **Projector**: A projection head that maps encoder outputs to a contrastive embedding space.
- **Classifier**: A single-layer dense network trained on frozen encoder outputs.
- **Supervised NT-Xent Loss**: A contrastive loss that uses label information to form positive pairs.

## Performance

After training for 20 epochs on the contrastive objective and 15 epochs for classifier fine-tuning:

- Train Accuracy: 95.31%
- Test Accuracy: 89.89%

The embeddings learned by contrastive learning are well-separated in PCA space, and the classifier performs competitively without unfreezing the encoder.


## Requirements

dependencies

- TensorFlow
- NumPy
- Matplotlib
- Seaborn
- scikit-learn
- pandas
  
---
# Transfer Learning on Various Modalities with TensorFlow

This project demonstrates how to apply transfer learning across four different data modalities: image, audio, video, and natural language. The implementation uses only TensorFlow and TensorFlow Hub, along with core Python and data science libraries.

## Overview

Transfer learning enables leveraging pretrained models for downstream tasks with minimal data and compute. This notebook explores transfer learning in the following domains:

1. **Image Classification**  
   - Dataset: CIFAR-10  
   - Model: MobileNetV2  
   - Approaches: Feature extraction, Fine-tuning  
   - Techniques: Data augmentation, Early stopping, LR scheduling

2. **Audio Classification**  
   - Embeddings: YAMNet (TF Hub) or simulated 1024-dim features  
   - Approaches: Feature extraction, Simulated fine-tuning  
   - Model: Dense feedforward networks  
   - Evaluation: Confusion matrix, classification report

3. **Video Classification**  
   - Simulated video data with synthetic motion patterns  
   - Backbone: Custom I3D-like Conv3D model  
   - Approaches: Feature extraction, Fine-tuning  
   - Metrics: Accuracy, Confusion matrix

4. **NLP Classification**  
   - Task: Sentiment analysis (synthetic data generation)  
   - Models: Universal Sentence Encoder, BERT (Feature + Fine-Tuning)  
   - Preprocessing: BERT tokenizer, Sentence embeddings  
   - Visualization: Training history, model comparison

## Key Features

- Modular class-based implementations for audio, video, and NLP pipelines
- Use of TensorFlow Hub pretrained models (YAMNet, I3D, USE, BERT)
- Synthetic data generators for fast prototyping in non-GPU environments
- Performance tracking using validation accuracy and training time
- Matplotlib and Seaborn-based visualizations for confusion matrices and learning curves



## Dependencies

- TensorFlow ≥ 2.9
- TensorFlow Hub
- NumPy, Pandas
- Matplotlib, Seaborn
- scikit-learn
- TensorFlow Text (for BERT preprocessing)

## Results Snapshot

| Modality | Model                     | Test Accuracy |
|----------|---------------------------|---------------|
| Image    | MobileNetV2 Fine-Tuning   | ~91–94%       |
| Audio    | YAMNet (Simulated)        | 100%          |
| Video    | I3D-like (Fine-Tuned)     | ~56%          |
| NLP      | BERT Fine-Tuning          | ~78%          |

*Note: Actual results may vary depending on runtime environment and synthetic data randomization.*


## Acknowledgements

- TensorFlow Hub for pretrained model access  
- CIFAR-10, YAMNet, and BERT for open datasets and models  
- DeepMind for I3D reference model  
- Google for Universal Sentence Encoder and BERT models

---

# Vision Classifiers: Transfer Learning Comparison on Multiple Datasets

This project demonstrates image classification using state-of-the-art vision models like EfficientNet, BiT, MLP-Mixer, and ConvNeXt across diverse datasets. Implementations are provided using both TensorFlow/Keras and PyTorch frameworks.

## Overview

The notebook evaluates modern convolutional and transformer-based models on the following datasets:

- MNIST (digit classification)
- Fashion MNIST (clothing classification)
- CIFAR-10 (object classification)
- EuroSAT (satellite land use classification)

Multiple architectures and training configurations are tested, including:

- EfficientNet (Keras and PyTorch)
- BiT (Big Transfer) via TF Hub
- MLP-Mixer (Hugging Face Transformers)
- ConvNeXt (Hugging Face Transformers)

## Key Features

- Image preprocessing pipelines for grayscale and RGB images
- TensorFlow models using EfficientNetB3 and EfficientNetV2
- PyTorch implementation of EfficientNet-B7 using the `efficientnet_pytorch` library
- Fine-tuning and feature extraction variants
- Integration with Hugging Face `datasets`, `transformers`, and `pipeline` APIs
- Training history visualization and test-time inference

## Experiments

| Dataset     | Framework | Model           | Accuracy   |
|-------------|-----------|------------------|------------|
| MNIST       | Keras     | EfficientNetB3   | ~99.5%     |
| FashionMNIST| PyTorch   | EfficientNet-B7  | ~91.1%     |
| CIFAR-10    | Keras     | EfficientNetV2B0 | ~97.1%     |
| EuroSAT     | PyTorch   | ConvNeXt-Tiny    | ~99.9%     |

> Results may vary based on runtime, hardware, and augmentation settings.

## Requirements

- TensorFlow ≥ 2.8
- PyTorch ≥ 1.10
- Hugging Face `datasets` and `transformers`
- `efficientnet_pytorch`, `albumentations`, `opencv-python`
- Python 3.7+
- 
## Highlights

- Comparison between pretraining and fine-tuning approaches
- Model portability across Keras and PyTorch ecosystems
- Scalable to new datasets with minimal code changes
- Clean data augmentation using `albumentations` and `ImageDataGenerator`



## Acknowledgments

- TensorFlow, PyTorch, and Hugging Face communities
- Qubvel's `efficientnet-pytorch`
- Datasets: MNIST, FashionMNIST, CIFAR-10, EuroSAT

---
# Pneumonia Detection from Chest X-rays and 3D CT Scans

This notebook presents a comprehensive workflow for detecting pneumonia using both 2D Chest X-rays and 3D CT scan volumes. It demonstrates image preprocessing, model construction, class balancing, training strategies, and evaluation using TensorFlow/Keras.

## Contents

- Part 1: Chest X-ray Classification (2D CNN)
- Part 2: CT Scan Classification (3D CNN)
- Data handling with TFRecords and NIfTI volumes
- Advanced data augmentation and visualization
- Training with TPU strategy (fallback to GPU/CPU)
- Evaluation metrics: accuracy, precision, recall



## Datasets Used

- **ChestXRay2017 (2D X-ray)**  
  Format: TFRecord  
  Classes: `NORMAL`, `PNEUMONIA`  

- **MosMedData (3D CT Scans)**  
  Format: NIfTI (`.nii.gz`)  
  Classes: `normal`, `abnormal`  

## Key Features

### Chest X-ray Pneumonia Classifier (2D CNN)

- Uses `SeparableConv2D` blocks with batch normalization and dropout.
- Pretrained-aware initialization using log class priors.
- Class balancing via computed `class_weight`.
- Data pipeline with TFRecord parsing, image decoding, and shuffling.
- Precision/Recall-aware evaluation.

**Validation Accuracy**: ~86%  
**Precision**: ~93%  
**Recall**: ~88%

---

###  CT Scan Pneumonia Classifier (3D CNN)

- Uses `Conv3D` blocks for volume-level feature extraction.
- Preprocessing includes NIfTI loading, Hounsfield normalization, and 3D resizing.
- 3D Data augmentation using SciPy-based random rotation.
- Volume visualization and slice grid plotting for insights.
- Model achieves strong separation between normal and pneumonia cases.

**Validation Accuracy**: ~80%  
**Best Performance Achieved On Epoch**: 44  
**Predicted Confidence Example**:  
> This model is 90.48% confident the CT scan is abnormal.


## Environment Requirements

- Python ≥ 3.7
- TensorFlow ≥ 2.9
- NumPy, Pandas, Matplotlib
- SciPy, NiBabel (for CT scan preprocessing)

## Acknowledgments

- TensorFlow team for ChestXRay2017 dataset
- Hasib Zunair and contributors for MosMedData tutorial
- NiBabel and SciPy developers for medical image processing tools

---

# Zero-Shot Image Classification using OpenAI CLIP on CIFAR-10

This notebook demonstrates zero-shot image classification using the OpenAI CLIP model on the CIFAR-10 dataset. Without any gradient-based training, the model predicts class labels based purely on natural language descriptions (text prompts).


## Overview

CLIP (Contrastive Language–Image Pretraining) is a powerful vision-language model trained to associate images and textual descriptions. This notebook leverages CLIP's ability to perform classification **without any fine-tuning** on target datasets like CIFAR-10.


## Key Features

- Zero-shot classification using **ViT-B/32** CLIP model
- Tested on **CIFAR-10 test set** with 10 object classes
- Text prompts like `"a photo of a dog"` used as classification anchors
- Real-time inference and visualization of predictions
- No training, backpropagation, or custom classifier layers required


## Dependencies

- `torch`
- `clip` (OpenAI's CLIP wrapper)
- `ftfy`, `regex`, `tqdm`
- `transformers`
- `torchvision`, `matplotlib`, `PIL`

---

### Youtube: [Computer-Vision-Examples](https://www.youtube.com/playlist?list=PLCGwaUpxPWO3p_he-GEpY75VQvEsO4FuU)
