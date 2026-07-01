# 🌾 Rice Classification Using PyTorch

<p align="center">

<img src="https://img.shields.io/badge/Task-Tabular%20Classification-1f77b4?style=for-the-badge"/>
<img src="https://img.shields.io/badge/Architecture-MLP%20Neural%20Network-6f42c1?style=for-the-badge"/>
<img src="https://img.shields.io/badge/Framework-PyTorch-ee4c2c?style=for-the-badge"/>
<img src="https://img.shields.io/badge/Language-Python-3776AB?style=for-the-badge"/>

<br>

<img src="https://img.shields.io/badge/Dataset-Rice%20Type%20Classification-3CB043?style=for-the-badge"/>
<img src="https://img.shields.io/badge/Classes-5-F28C28?style=for-the-badge"/>
<img src="https://img.shields.io/badge/Accuracy-98.39%25-38B000?style=for-the-badge"/>
<img src="https://img.shields.io/badge/Status-Completed-38B000?style=for-the-badge"/>

</p>

## 🔗 Project Resources

<p align="center">

<a href="https://www.kaggle.com/code/rushikeshraghatate/rice-classification-using-pytorch">
<img src="https://img.shields.io/badge/📒%20VIEW-KAGGLE%20NOTEBOOK-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white"/>
</a>
<a href="https://www.kaggle.com/datasets/mssmartypants/rice-type-classification">
<img src="https://img.shields.io/badge/📊%20VIEW-KAGGLE%20DATASET-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white"/>
</a>
<a href="https://www.kaggle.com/datasets/mssmartypants/rice-type-classification">
<img src="https://img.shields.io/badge/🙏%20DATASET%20BY-MSSMARTYPANTS-00599C?style=for-the-badge"/>
</a>

</p>

<p align="center">
<b>Click any badge above to open the corresponding Kaggle resource.</b>
</p>

---

# 📖 Project Overview

Rice Classification Using PyTorch is a **deep learning-based tabular classification project** that predicts the type of rice grain from its physical and geometric characteristics.

The project demonstrates how a **Multi-Layer Perceptron (MLP)** built with **PyTorch** can effectively learn complex relationships between numerical features and achieve a **98.39% test accuracy**.

The workflow includes data preprocessing, feature normalization, model training, evaluation, visualization, and inference on unseen rice samples.

---

# 🎯 Objectives

- Build a deep learning classifier using PyTorch
- Preprocess and normalize numerical features
- Train an MLP neural network for rice type classification
- Evaluate model performance using accuracy and loss metrics
- Visualize learning curves
- Perform inference on custom rice grain measurements

---

# 🚀 Key Features

- ✅ End-to-end PyTorch implementation
- ✅ Tabular Deep Learning
- ✅ Feature Normalization
- ✅ Model Training & Validation
- ✅ Accuracy and Loss Visualization
- ✅ Real-time Prediction Support
- ✅ Simple and Reproducible Notebook

---

# 🛠 Tech Stack

| Category | Technology |
|-----------|------------|
| Language | Python |
| Deep Learning | PyTorch |
| Data Processing | Pandas |
| Machine Learning | Scikit-learn |
| Visualization | Matplotlib |
| Notebook | Jupyter Notebook |

---

# 📂 Dataset

The model is trained on the **Rice Type Classification Dataset** available on Kaggle.

### Dataset Highlights

- Numerical tabular dataset
- Physical and geometric characteristics of rice grains
- Multiple rice classes
- Suitable for supervised classification tasks

### Input Features

- Area
- Major Axis Length
- Minor Axis Length
- Eccentricity
- Convex Area
- EquivDiameter
- Extent
- Perimeter
- Roundness
- Aspect Ratio

---

# 🙏 Dataset Acknowledgement

Special thanks to **mssmartypants** for making the Rice Type Classification dataset publicly available on Kaggle.

Your contribution enables students, researchers, and developers to explore practical machine learning applications.

---

# 🚀 Getting Started

## Clone the Repository

```bash
git clone https://github.com/rushikeshraghatate90/Rice-Classification-Using-PyTorch.ipynb.git
```

```bash
cd rice-classification-pytorch
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Launch Jupyter Notebook

```bash
jupyter notebook
```

Open

```
Rice Classification Using PyTorch.ipynb
```

and execute all cells sequentially.

---

# 🧠 Model Architecture

The classifier is implemented as a **Fully Connected Neural Network (MLP)** using PyTorch.

Training configuration:

- Neural Network (MLP)
- Cross Entropy Loss
- Adam Optimizer
- Feature Normalization
- GPU Support (CUDA)

---

# 📊 Model Performance

### Test Accuracy

> **98.39%**

### Training Strategy

- Feature normalization
- Supervised learning
- Cross Entropy Loss
- Adam Optimizer

---

# 📈 Training Visualization

The notebook visualizes:

- Training Loss
- Validation Loss
- Training Accuracy
- Validation Accuracy

```python
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(15,5))
```

These plots help monitor convergence and identify overfitting during training.

---

# 🔍 Model Inference

After training, the model accepts numerical measurements of a rice grain and predicts its corresponding class.

Example workflow:

1. Enter rice grain measurements.
2. Normalize the input features.
3. Pass the tensor through the trained PyTorch model.
4. Obtain the predicted rice class.

---

# 📋 Sample Prediction

```text
Area: 6431.279

Major Axis Length: 145.21338

Minor Axis Length: 56.902

Eccentricity: 0.919981821

Convex Area: 6518.93759999

EquivDiameter: 90.483541

Extent: 0.8506668

Perimeter: 329.972

Roundness: 0.742255516

AspectRation: 2.551696

Prediction

Class: 1
```

---

# 📌 Future Improvements

- Hyperparameter Optimization
- Advanced Neural Network Architectures
- Automated Model Selection
- Flask/FastAPI Deployment
- Docker Support
- Model Quantization
- ONNX Export
- Web-based Prediction Interface

---

# 🤝 Contributing

Contributions are welcome.

If you'd like to improve this project:

- Fork the repository
- Create a feature branch
- Commit your changes
- Open a Pull Request

---

# ⭐ Support

If you found this project helpful, please consider giving it a ⭐ on GitHub.

It helps support future open-source AI and Deep Learning projects.

---
