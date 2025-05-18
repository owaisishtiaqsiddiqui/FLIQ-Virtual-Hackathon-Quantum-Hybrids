Python version: 3.9.21

### 1. **Library Installation and Imports**
- Installs required libraries: `torch`, `openpyxl`, and checks `qiskit-machine-learning`.
- Imports standard Python, pandas, numpy, matplotlib, seaborn, scikit-learn, PyTorch, and Qiskit modules.
- Imports a wide range of Qiskit transpiler passes and quantum machine learning modules, including `QSVC`.

---

### 2. **Data Loading**
- Loads the **Drug-Induced Autoimmunity Prediction** dataset:
  - Training set: `DIA_trainingset_RDKit_descriptors.csv`
  - Test set: `DIA_testset_RDKit_descriptors.csv`
  - Chemical descriptor info: `RDKit_ChemDes.xlsx`

---

### 3. **Data Exploration**
- Checks for missing data in train and test sets.
- Displays descriptive statistics for all datasets.
- Prints data types and previews the first few rows.

---

### 4. **Data Visualization**
- Plots the distribution of the target variable (`Label`) for both train and test sets.
- Plots a heatmap of feature correlations.
- Plots the top 10 features with the highest variance (in two different color schemes).

---

### 5. **Data Preprocessing**
- Removes non-numeric columns (`SMILES`, `ID`, `Name`) from the datasets.
- Separates features (`X_train_set`) and labels (`y_train_set`).
- Splits the training data into train and validation sets (80/20 split).
- Applies standard scaling to features.

---

### 6. **Dimensionality Reduction**
- Uses PCA to reduce the feature space to 2 principal components for quantum input:
  - `X_train_pca` and `X_valid_pca` are created.

---

### 7. **Quantum Machine Learning Model (QSVC)**
- Defines a quantum feature map using `ZZFeatureMap`.
- Sets up a quantum kernel using `FidelityQuantumKernel`.
- Trains a `QSVC` (Quantum Support Vector Classifier) on the PCA-reduced training data.
- Makes predictions on the validation set.

---

### 8. **Model Evaluation**
- Calculates and prints:
  - Accuracy, Precision, Recall, F1-score, ROC AUC.
  - Confusion matrix and classification report.
- Plots:
  - Precision-Recall curve for class 0.
  - Confusion matrix for both classes.

---

**Summary:**  
This notebook builds a full pipeline for quantum machine learning on the drug-induced autoimmunity dataset:  
- Loads and explores the data,  
- Preprocesses and reduces dimensionality,  
- Trains a quantum kernel SVM (QSVC) using Qiskit Machine Learning,  
- Evaluates and visualizes the results.

All code is tailored for the specific dataset and uses quantum-enhanced methods for classification.
