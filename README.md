# 🚗 Automobile Analytics — Unsupervised & Supervised Learning (Combined Project)

## 📘 Overview

This project demonstrates the application of **Machine Learning (ML)** techniques on **automobile datasets**, covering both **unsupervised** and **supervised learning** approaches in a **single notebook**.

The notebook includes:

* **Part A — K-Means Clustering** on car attributes dataset
* **Part B — PCA + SVM Classification** on vehicle silhouette dataset

These combined projects showcase end-to-end ML capabilities — from **data cleaning**, **EDA**, **feature engineering**, and **modeling** to **evaluation** and **optimization**.

---

## 📂 Domain

**Automobile Industry** — focusing on:

* Fuel consumption analysis and car segmentation
* Vehicle type classification using geometric and performance-based features

---

## 🎯 Objectives

### **Part A — K-Means Clustering (Unsupervised Learning)**

To segment cars into distinct groups based on performance and design characteristics using **K-Means Clustering**.

### **Part B — PCA + SVM Classification (Supervised Learning)**

To apply **Principal Component Analysis (PCA)** for dimensionality reduction and train an **SVM model** to classify vehicle silhouettes, followed by **hyperparameter tuning** for performance improvement.

---

## 🧠 Project Workflow

### 🔹 Part A: K-Means Clustering on Car Dataset

1. **Data Understanding & Exploration**

   * Read and merged datasets (`Car name.csv` and `Car-Attributes.json`)
   * Computed descriptive statistics for numerical features

   **Key Observations:**

   * MPG ranges from **9 to 46.6**, with an average of **23.5**
   * Cars have **3 to 8 cylinders**, most around **5**
   * **Displacement:** 68 to 455; **Weight:** 1613 to 5140
   * **Acceleration:** 8 to 24.8; **Year:** 1970 to 1982
   * Majority of cars originate from region code **2**

2. **Data Preparation & Analysis**

   * Verified absence of null and duplicate values
   * Identified and imputed missing values in `hp` column (6 nulls) with median
   * Converted data types and cleaned anomalies (`'?'` values)
   * Visualized relationships using `pairplot` and scatter plots

     * **Positive correlation:** Weight ↔ Displacement
     * **Negative correlation:** MPG ↔ Weight / Horsepower / Displacement

3. **K-Means Clustering**

   * Trained K-Means for clusters **k = 2 to 10**
   * Used **Elbow Method** and **Silhouette Score** for optimal cluster identification
   * Optimal cluster: **k = 3** (highest silhouette = 0.586)
   * Added cluster labels and visualized results
   * Predicted cluster for a **new data point** using trained model

   **Key Insight:**

   * Vehicles naturally cluster into three main groups — typically representing **compact**, **mid-size**, and **heavy vehicles**.

---

### 🔹 Part B: PCA + SVM Classification on Vehicle Dataset

1. **Data Understanding & Cleaning**

   * Loaded `vehicle.csv`
   * Imputed missing values with median
   * Verified no duplicate rows
   * Visualized class distribution:

     * **Car:** 50.7%
     * **Bus:** 25.8%
     * **Van:** 23.5%

2. **Data Preparation**

   * Split dataset into `X` (features) and `y` (class labels)
   * Standardized data using **Z-score normalization**

3. **Model Building**

   * Trained baseline **SVM model**

     * Accuracy: **98%** on full data
   * Applied **PCA (10 components)** and visualized cumulative variance

     * Found **4 components** explain **~90% variance**
   * Retrained SVM on PCA-reduced data

     * Accuracy dropped to **80%** (information loss due to dimensionality reduction)

4. **Model Optimization (Grid Search + Hyperparameter Tuning)**

   * Tuned SVM using **GridSearchCV** with parameters:

     * C = [0.01, 0.1, 1, 10]
     * Kernel = ['linear', 'rbf']
     * Gamma = [0.1, 1, 5]
   * Best parameters:

     * **Kernel:** rbf
     * **C:** 10
     * **Gamma:** 0.1

---

## 📊 Model Performance & Observations

### **After Hyperparameter Tuning (SVM with PCA)**

| Metric              | Before Tuning | After Tuning |
| ------------------- | ------------- | ------------ |
| **Accuracy**        | 80%           | **86%**      |
| **Precision (Bus)** | 85%           | **88%**      |
| **Precision (Car)** | 83%           | **88%**      |
| **Precision (Van)** | 67%           | **78%**      |
| **Recall (Bus)**    | 66%           | **81%**      |
| **Recall (Car)**    | 90%           | **92%**      |
| **Recall (Van)**    | 73%           | **77%**      |
| **F1-Score (Bus)**  | 74%           | **84%**      |
| **F1-Score (Car)**  | 86%           | **90%**      |
| **F1-Score (Van)**  | 70%           | **77%**      |

### **Key Insights**

* Model’s **overall accuracy improved from 80% → 86%** after tuning.
* Precision, recall, and F1-score increased across all classes.
* Indicates better generalization and separation of vehicle classes.
* PCA successfully reduced dimensionality with minimal performance loss after tuning.

---

## 🧰 Tools & Technologies

* **Language:** Python 🐍
* **Libraries:** NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn, SciPy
* **Algorithms:**

  * K-Means Clustering
  * Principal Component Analysis (PCA)
  * Support Vector Machine (SVM)
  * GridSearchCV (Hyperparameter Optimization)
* **Environment:** Jupyter Notebook / VSCode

---

## 🧩 Notebook Structure

```
Automobile_ML_Combined_Project.ipynb
│
├── Part A: K-Means Clustering (Unsupervised Learning)
│   ├── Data Understanding & Cleaning
│   ├── Exploratory Data Analysis (EDA)
│   ├── K-Means Clustering & Evaluation
│   └── Cluster Visualization & Prediction
│
└── Part B: PCA + SVM Classification (Supervised Learning)
    ├── Data Cleaning & Preparation
    ├── PCA Analysis & Visualization
    ├── Model Building (SVM)
    ├── Hyperparameter Tuning (GridSearchCV)
    └── Final Evaluation & Insights
```

---

## 🧠 Skills Demonstrated

* Data Preprocessing & Cleaning
* Exploratory Data Analysis (EDA)
* Unsupervised Learning (K-Means)
* Dimensionality Reduction (PCA)
* Supervised Learning (SVM)
* Model Evaluation & Tuning
* Feature Scaling & Visualization
* Insight Generation & Reporting

---

## 🚀 Future Enhancements

* Extend K-Means results using hierarchical clustering or DBSCAN
* Apply **SMOTE** for class imbalance (if applicable)
* Deploy classification model via **Streamlit / Flask**
* Compare PCA+SVM performance with **Random Forest** and **XGBoost**
* Experiment with **Autoencoders** for unsupervised feature learning

---

## 👤 Author

**G. Siva Kumar**
💼 [LinkedIn](https://linkedin.com/in/g-siva)| 
🧑‍💻 [GitHub](https://github.com/gesivak21)
