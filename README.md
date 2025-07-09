# 🧠 Stacking Ensemble for Text Classification

This project demonstrates a **multi-layer stacking ensemble** for a classification task on a high-dimensional, sparse dataset. It leverages several classical machine learning algorithms and builds a deep stacking pipeline with performance tuning and cross-validation to achieve high accuracy.

---

## 📊 Dataset

* **Shape**: `(50000, 73392)`
* **Type**: Sparse matrix (99.87% sparse)
* **Features**: Preprocessed vectorized inputs (likely from text data such as TF-IDF or bag-of-words)
* **Labels**: Multi-class or binary classification

---

## 🛡️ Stacking Architecture

```
Layer 0 (Base Models):
    ✔️ SGDClassifier
    ✔️ LogisticRegression
    ✔️ LinearSVC
    ✔️ MultinomialNB

↓ Outputs

Layer 1 (Meta Model):
    ✔️ LogisticRegression
```

---

## ✅ Final Accuracy Results

| Model                     | Accuracy (%) |
| ------------------------- | ------------ |
| MultinomialNB             | 85.41        |
| BernoulliNB               | 85.27        |
| LogisticRegression        | 88.64        |
| LinearSVC                 | 88.54        |
| SGDClassifier             | 88.65        |
| RidgeClassifier           | 86.09        |
| PassiveAggressive         | 81.02        |
| **1-Layer Stack**         | **89.15** ✅  |
| **Deep Stack (2 layers)** | 89.09        |

> ✅ **Best performance achieved using 1-layer stacking with Logistic Regression as the meta-model.**

---

## 🧪 Techniques Used

* `train_test_split` from `sklearn.model_selection`
* Grid Search (`GridSearchCV`) for hyperparameter tuning:

  * `C` for `LogisticRegression`, `LinearSVC`
  * `alpha` for `MultinomialNB`, `SGDClassifier`
* `StackingClassifier` for layered architecture
* `accuracy_score` for evaluation
* `mode()` for hard voting in final layer

---

## 🧠 Requirements

Install necessary packages:

```bash
pip install numpy scikit-learn scipy
```

---

## 🧠 Learning Highlights

* Using **classical ML models** in a stacked architecture
* Understanding how and when additional layers in stacking help (or don’t)
* Measuring accuracy improvements layer-by-layer using validation and voting

---

## 📊 Model Progression & Insights

This section outlines the iterative improvements made during model experimentation:

### 🔹 Baseline Models (Naive Bayes Family)

| Model         | Accuracy (%) |
| ------------- | ------------ |
| GaussianNB    | 82.00        |
| BernoulliNB   | 83.00        |
| MultinomialNB | 83.50        |

---

### 🔹 After Hyperparameter Tuning (Grid Search)

| Model                 | Tuned Accuracy (%) |
| --------------------- | ------------------ |
| MultinomialNB (alpha) | 85.40              |

---

### 🔹 Ensemble of Classical ML Models

Tried:

* LogisticRegression
* SGDClassifier
* RidgeClassifier
* PassiveAggressiveClassifier
* LinearSVC

Among them, 4 models performed best, achieving around 87% accuracy.

---

### 🔹 Grid Search on Top 4 Models

After tuning `C`, `alpha`, `penalty`, `loss`, the best accuracy improved to **88.6%**.

---

### 🔹 Stacking

Performed model stacking with:

* Base Models: MultinomialNB, SGD, LogisticRegression, LinearSVC
* Meta Model: LogisticRegression (solver='lbfgs')
* Cross-Validation: 5-fold

**🌟 Final Accuracy: 89.12%**
**🌟 Cross-Validation Score: 89.15%**

---

### 🔹 Multi-layer Stacking Attempt

Tried adding a second stacking layer with Ridge & SGD followed by hard voting.

* **Result**: Accuracy dropped to **89.09%**
* **Conclusion**: Layer 1 was already optimal. Extra depth didn’t help.

---
