# SVM Kernel Comparison for Job Classification

## Overview

This project explores how different kernel functions in Support Vector Machines (SVM) affect classification performance. The goal is to understand when a simple model is sufficient and when more complex kernels are useful.

The dataset used consists of job postings, where each job is represented by a set of required skills. The task is to classify each job into predefined categories based on these features.

---

## What this project covers

* Implementation of SVM for classification
* Comparison of three kernels:

  * Linear
  * Polynomial
  * RBF (Radial Basis Function)
* Evaluation using accuracy, classification reports, and confusion matrices
* Analysis of how model complexity impacts performance

---

## Dataset

The dataset contains job-related information, including skill indicators such as Python, SQL, Machine Learning, and others. These are represented as binary features (0 or 1).

Target variable:

* `Class` → represents job category

Source:

* Kaggle LinkedIn Jobs Dataset

---

## Key Observations

* The Linear SVM achieved perfect accuracy, suggesting that the dataset is largely linearly separable.
* The Polynomial kernel performed poorly, showing sensitivity to class imbalance and unnecessary complexity.
* The RBF kernel achieved high accuracy but did not significantly outperform the linear model.

This highlights an important point: increasing model complexity does not always improve performance.

---


## How to Run

1. Clone the repository:

```
git clone https://github.com/poluananya13/LinkedIn_Data_Set.git
cd LinkedIn_Data_Set
```

2. Install required libraries:

```
pip install -r requirements.txt
```

3. Run the code:

```
python src/main.py
```

---

## Accessibility Notes

* Plots use clear labels and readable layouts
* Code is structured and easy to follow
* Outputs are explained in text for clarity

---

## Final Thoughts

This project shows that understanding the data is more important than choosing complex models. In this case, a simple linear SVM was enough to achieve excellent results due to the structure of the dataset.

---

## References

* Cortes, C. and Vapnik, V., 1995. Support-vector networks. Machine learning, 20(3), pp.273-297. https://doi.org/10.1007/BF00994018
* Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V. and Vanderplas, J., 2011. Scikit-learn: Machine learning in Python. the Journal of machine Learning research, 12, pp.2825-2830.