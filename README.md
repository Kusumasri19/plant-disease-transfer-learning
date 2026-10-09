# Plant Disease Classification using Transfer Learning

An image-classification system that identifies **38 classes of healthy and diseased plant leaves** from a photo. A **Basic CNN** trained from scratch is compared against a pretrained **MobileNetV2** (Transfer Learning), and the best model is deployed as an interactive **Gradio** web demo.

**Tech stack:** Python, TensorFlow / Keras, scikit-learn, Matplotlib, Seaborn, Gradio, Google Colab (T4 GPU)

---

## Demo

The Gradio app accepts a leaf image and returns the top-3 predicted classes with confidence scores.

![Gradio demo: Apple Black rot predicted with 72% confidence](Screenshot%20%2810%29.png)

---

## Dataset

- **PlantVillage** (colour images), about 54,000 leaf images in **38 classes** across 14 crop species (Apple, Tomato, Potato, Grape, Corn, etc.).
- The classes are imbalanced, so at most **300 images per class** were used to keep training practical on a free Colab GPU and to reduce imbalance.
- Source: [PlantVillage dataset on Kaggle](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)

## Methodology

| Step | Details |
|------|---------|
| Split | Stratified **70% train / 15% validation / 15% test** (1,685 test images) |
| Preprocessing | Resize to 128 x 128, batch size 32, pixel scaling inside each model |
| Augmentation | Random flip, rotation and zoom (training only) |
| Model 1 | **Basic CNN**: 3 convolution blocks (32, 64, 128 filters), global average pooling, dense layer with dropout, softmax output |
| Model 2 | **MobileNetV2** pretrained on ImageNet: Phase 1 feature extraction (frozen base), Phase 2 fine-tuning (last 30 layers, learning rate 1e-5) |
| Training | Adam optimizer, sparse categorical cross-entropy, early stopping on validation accuracy |

## Results (Test Set)

| Model | Accuracy | Precision | Recall | F1-score | Training time |
|-------|:--------:|:---------:|:------:|:--------:|:-------------:|
| Basic CNN | 74.07% | 75.02% | 74.07% | 73.68% | 1.5 min |
| **MobileNetV2 (Transfer Learning)** | **86.71%** | **88.00%** | **86.71%** | **86.30%** | 1.8 min |

Precision, recall and F1-score are weighted averages over the 38 classes.

**Best model: MobileNetV2 (Transfer Learning)**, about **12.6 percentage points** higher accuracy than the Basic CNN on the same unseen test images.

### Key observations

- **Faster learning:** MobileNetV2 reached 79.6% validation accuracy after the first epoch, while the Basic CNN reached 13% after its first epoch. Total training times were similar, so the main advantage of transfer learning here is accuracy and fast convergence rather than shorter total training time.
- **Per-class performance:** MobileNetV2 reached an F1-score of 0.95 to 0.99 on many classes (for example Apple Black rot, Grape healthy, Raspberry healthy, Squash Powdery mildew).
- **Fine-tuning effect:** Phase 2 fine-tuning gave no clear gain over Phase 1 on this small subset (validation accuracy stayed around 87 to 88%).

### Error analysis

The most frequent mistakes occurred between diseases with visually similar symptoms:

- Corn Northern Leaf Blight predicted as Corn Gray Leaf Spot (10 images)
- Potato Early blight predicted as Potato Late blight (9 images)
- Tomato Target Spot predicted as Tomato healthy (8 images)

Tomato classes were the weakest overall (for example Tomato Early blight recall of 0.24), because several Tomato diseases produce small, similar-looking spots.

## Project Structure

```
plant-disease-transfer-learning/
├── Plant_Disease_Transfer_Learning.ipynb   # full notebook with code, outputs and analysis
├── Screenshot (10).png                     # Gradio demo screenshot
└── README.md
```

## How to Run

1. Open `Plant_Disease_Transfer_Learning.ipynb` in [Google Colab](https://colab.research.google.com/).
2. Set the runtime to GPU: `Runtime -> Change runtime type -> T4 GPU`.
3. Run all cells: `Runtime -> Run all`. The dataset is downloaded automatically with `kagglehub`.
4. The last section launches the Gradio demo and prints a public link.

## Limitations and Future Work

- Only up to 300 images per class and 128 x 128 resolution were used. Using the full dataset and 224 x 224 images should improve accuracy.
- PlantVillage images have plain backgrounds, so performance on real field photos with complex backgrounds needs to be tested.
- Compare other pretrained models (ResNet50, EfficientNet, VGG16).
- Handle class imbalance with class weights or stronger augmentation.
- Convert the model to TensorFlow Lite and deploy it in a mobile app for farmers.
