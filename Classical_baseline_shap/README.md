Python version: 3.9.21

### 1. **Environment & Library Setup**

- Installs required libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `torch`, `torchvision`, `torchaudio`, `shap`, `ucimlrepo`.
- Imports standard Python, pandas, numpy, matplotlib, seaborn, scikit-learn, PyTorch, and SHAP modules.
- Sets a global random `SEED` (42) for reproducibility across numpy, torch, and os.

---

### 2. **Data Loading & Rigorous Preprocessing**

- **Fetch Dataset:** Loads the **Wisconsin Diagnostic Breast Cancer** dataset using `ucimlrepo` (ID: 17).
  - `X_raw` (features) and `y_raw` (targets) are extracted.
  - Initial inspection of shapes, head, and target value counts is performed.
- **Encode Target Variable:**
  - Encodes the `Diagnosis` column ('M'/'B') to numeric labels (0/1).
  - Ensures 'M' (Malignant) is 1 (positive class) and 'B' (Benign) is 0 (negative class).
- **Data Splitting (Stratified):**
  - Splits data into **train (70%)**, **validation (15%)**, and **test (15%)** sets using `StratifiedShuffleSplit` to maintain class proportions.
  - Prints shapes and target distributions for each set.
- **Feature Scaling:**
  - Applies `StandardScaler` to normalize features (fit on train, transform on val/test).
- **Dimensionality Reduction (PCA):**
  - **Analysis:** Fits PCA on scaled training data and plots cumulative explained variance to determine the optimal number of components.
  - **Transformation:** Applies PCA with **8 components** (based on analysis aiming for ~90-95% variance) to train, validation, and test feature sets.

---

### 3. **Classical Baseline Models & Explainability**

- **Evaluation Helper Function:** Defines `evaluate_model` to calculate and print:
  - Accuracy, Precision, Recall, F1-score, Matthews Correlation Coefficient (MCC).
  - AUC-ROC, AUC-PR (Average Precision).
  - Displays a Confusion Matrix.
- **Logistic Regression:**
  - Trains a `LogisticRegression` model (solver='liblinear', class_weight='balanced') on PCA-transformed training data.
  - Evaluates the model on the PCA-transformed test set using the helper function.
- **PyTorch MLP (Multi-Layer Perceptron):**
  - **Datasets & DataLoaders:** Converts PCA-transformed data into PyTorch `TensorDataset` and `DataLoader` for train, validation, and test sets (Batch size: 32).
  - **MLP Model Definition:** Defines `SimpleMLP` class with one hidden layer (input: 8 PCA components, hidden: 16 units, output: 1 logit). Uses ReLU activation.
  - **Loss Function & Optimizer:**
    - Calculates `pos_weight` for `BCEWithLogitsLoss` to handle class imbalance.
    - Uses `Adam` optimizer (lr=0.001).
  - **MLP Training Loop:**
    - Trains for `N_EPOCHS` (e.g., 100) with **early stopping** based on validation F1-score (patience: 10 epochs).
    - Tracks and prints train/validation loss and F1-score per epoch.
    - Saves the best model state (`best_classical_mlp_model.pth`).
  - **Training History Plotting:** Plots training & validation loss and F1-score over epochs.
  - **MLP Evaluation on Test Set:** Evaluates the best trained MLP on the PCA-transformed test set using the helper function.
- **Explainability (SHAP for MLP):**
  - Applies `shap.KernelExplainer` to the trained PyTorch MLP using a sample of PCA-transformed training data as background.
  - Calculates SHAP values for a sample of the PCA-transformed test data.
  - Plots SHAP **summary plot** and **dependence plot** for the most important PCA component.

---

**Summary:**
This notebook establishes a classical machine learning baseline for the Wisconsin Diagnostic Breast Cancer dataset:

- Loads, explores, and preprocesses the data including PCA for dimensionality reduction.
- Trains and evaluates Logistic Regression and a PyTorch MLP.
- Implements class imbalance handling and early stopping for the MLP.
- Provides model explainability for the MLP using SHAP values on PCA components.
- Plots training performance and feature importance.
