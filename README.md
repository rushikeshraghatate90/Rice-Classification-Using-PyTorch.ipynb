# 🌾 Rice Classification Using PyTorch  

## 📖 Project Overview  
This project focuses on **tabular data classification** to distinguish between different types of rice grains. Using **PyTorch**, we train a deep learning model to classify rice based on its **physical and geometric attributes**. The model achieves a high accuracy of **98.39%** on the test dataset! 🚀  

## 🎯 Objectives  
✔️ Preprocess and normalize tabular data  
✔️ Train a **PyTorch-based neural network** for classification  
✔️ Evaluate the model using accuracy and loss metrics  
✔️ Perform inference to classify new rice samples  
✔️ Visualize training performance with loss and accuracy plots  

---

## 🛠 Technologies Used  
| Technology | Purpose |
|------------|---------|
| **Python** 🐍 | Core programming language |
| **PyTorch** 🔥 | Deep learning framework |
| **Pandas** 📊 | Data handling and preprocessing |
| **Scikit-learn** 🤖 | Machine learning utilities |
| **Matplotlib & Seaborn** 📈 | Data visualization |

---

## 📂 Dataset Description  
The dataset consists of multiple numerical features representing **physical characteristics of rice grains**. These include:  
- 🌱 **Area**  
- 📏 **Major Axis Length**  
- 📐 **Minor Axis Length**  
- 🔺 **Eccentricity**  
- 📦 **Convex Area**  
- 🔵 **EquivDiameter**  
- 🔳 **Extent**  
- 🌀 **Perimeter**  
- ⚫ **Roundness**  
- 📊 **Aspect Ratio**  

The target variable indicates the **class of rice** based on these features.  

---

## 🚀 How to Run the Project  

### 1️⃣ Clone the Repository  
```bash
git clone https://github.com/your-username/rice-classification-pytorch.git
cd rice-classification-pytorch
```

### 2️⃣ Install Dependencies  
```bash
pip install -r requirements.txt
```

### 3️⃣ Run the Jupyter Notebook  
```bash
jupyter notebook
```
Open **Rice Classification Using PyTorch.ipynb** and execute the cells sequentially.  

---

## 📊 Model Training & Evaluation  

- The dataset is **normalized** for better model performance.  
- A **fully connected neural network** (MLP) is implemented using PyTorch.  
- The model is trained using **cross-entropy loss** and optimized using **Adam**.  
- **Accuracy on the test dataset: 98.39%** 🎯  

### 📉 Performance Visualization  
The following plots show how the model performed during training:  

🔹 **Loss Curve**: Training vs. Validation Loss  
🔹 **Accuracy Curve**: Training vs. Validation Accuracy  

```python
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

axs[0].plot(total_loss_train_plot, label='Training Loss')
axs[0].plot(total_loss_validation_plot, label='Validation Loss')
axs[0].set_title('Training and Validation Loss over Epochs')
axs[0].set_xlabel('Epochs')
axs[0].set_ylabel('Loss')
axs[0].set_ylim([0, 2])
axs[0].legend()

axs[1].plot(total_acc_train_plot, label='Training Accuracy')
axs[1].plot(total_acc_validation_plot, label='Validation Accuracy')
axs[1].set_title('Training and Validation Accuracy over Epochs')
axs[1].set_xlabel('Epochs')
axs[1].set_ylabel('Accuracy')
axs[1].set_ylim([0, 100])
axs[1].legend()

plt.tight_layout()
plt.show()
```

---

## 🏆 Model Inference  
Once the model is trained, it can be used for **real-time predictions**. Here’s an example where the model classifies a new rice grain sample:

```python
area = float(input("Area: "))/original_df['Area'].abs().max()
MajorAxisLength = float(input("Major Axis Length: "))/original_df['MajorAxisLength'].abs().max()
MinorAxisLength = float(input("Minor Axis Length: "))/original_df['MinorAxisLength'].abs().max()
Eccentricity = float(input("Eccentricity: "))/original_df['Eccentricity'].abs().max()
ConvexArea = float(input("Convex Area: "))/original_df['ConvexArea'].abs().max()
EquivDiameter = float(input("EquivDiameter: "))/original_df['EquivDiameter'].abs().max()
Extent = float(input("Extent: "))/original_df['Extent'].abs().max()
Perimeter = float(input("Perimeter: "))/original_df['Perimeter'].abs().max()
Roundness = float(input("Roundness: "))/original_df['Roundness'].abs().max()
AspectRation = float(input("AspectRation: "))/original_df['AspectRation'].abs().max()

my_inputs = [area, MajorAxisLength, MinorAxisLength, Eccentricity, ConvexArea, EquivDiameter, Extent, Perimeter, Roundness, AspectRation]

model_inputs = torch.Tensor(my_inputs).to(device)
prediction = model(model_inputs)
print("Predicted Class:", round(prediction.item()))
```

### 📝 Sample Output  
```bash
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
====================
tensor([0.8427], device='cuda:0', grad_fn=<SigmoidBackward0>)
Class is: 1
```

---

## 🔮 Future Improvements  
🚀 **Hyperparameter tuning** to improve accuracy  
📈 **Exploring different deep learning architectures**  
🎯 **Deploying the model as an API using Flask**  

---

## 🤝 Contributing  
If you find this project useful, feel free to:  
✔️ **Fork the repo**  
✔️ **Report issues**  
✔️ **Suggest improvements**  

---
