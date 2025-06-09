# LSH-DynED

The proposed method is located inside the [Model](Model) folder.

# LSH-DynED: A Dynamic Ensemble Framework with LSH-Based Undersampling for Evolving Multi-Class Imbalanced Classification

This repository contains the implementation of the LSH-DynED model, a novel, robust, and resilient approach for classifying imbalanced and non-stationary data streams with multiple classes. This work is authored by Soheil Abadifard and Fazli Can.

**Authors:**
* Soheil Abadifard, Kansas State University (abadifard@ksu.edu)
* Fazli Can, Bilkent University (canf@cs.bilkent.edu.tr)

**Publication:**
Soheil Abadifard and Fazli Can. 2025. LSH-DynED: A Dynamic Ensemble Framework with LSH-Based Undersampling for Evolving Multi-Class Imbalanced Classification. 1, 1 (June 2025), 31 pages. https://doi.org/10.1145/nnnnnnn.nnnnnnn

---

## Overview

The classification of imbalanced data streams, where class distributions are unequal and change over time, is a significant challenge in machine learning, especially in multi-class scenarios. LSH-DynED addresses this challenge by integrating Locality Sensitive Hashing with Random Hyperplane Projections (LSH-RHP) into the Dynamic Ensemble Diversification (DynED) framework. This marks the first application of LSH-RHP for undersampling in the context of imbalanced non-stationary data streams.

LSH-DynED undersamples the majority classes using LSH-RHP to create a balanced training set, which in turn improves the prediction accuracy of the ensemble. Our comprehensive experiments on 23 real-world and ten semi-synthetic datasets demonstrate that LSH-DynED outperforms 15 state-of-the-art methods in terms of both Kappa and mG-Mean effectiveness. The model excels in handling large-scale, high-dimensional datasets with significant class imbalances and shows strong adaptation and robustness in real-world scenarios.

### Key Features:

* **Novel Undersampling Technique:** First application of Locality Sensitive Hashing with Random Hyperplane Projections (LSH-RHP) for undersampling in multi-class imbalanced non-stationary data streams.
* **Dynamic Ensemble Framework:** Extends and modifies the DynED framework to handle dynamic imbalance ratios in multi-class imbalanced data stream tasks.
* **State-of-the-Art Performance:** Outperforms other methods in both Kappa and mG-Mean effectiveness measures on a wide range of datasets.
* **Robust and Resilient:** Effectively handles concept drift and dynamic changes in class distributions.
* **Open Source:** The implementation is publicly available to encourage further research and improvements.

---

<details>
<summary><b>How it Works</b></summary>

LSH-DynED operates in three main stages:

1.  **Prediction and Training:** A subset of the ensemble, the "selected components," predicts the label of incoming data instances via majority voting. These components are then trained on the new data instance.
2.  **Drift Detection and Adaptation:** The ADWIN drift detector monitors the system's performance. If drift is detected, a new component is trained on recent data from a balanced dataset created by our novel undersampling method and added to a pool of "reserved components".
3.  **Component Selection:** This stage updates the ensemble's components to maintain a balance between diversity and accuracy. Components are selected from the combined pool of "selected" and "reserved" components based on their accuracy and a modified Maximal Marginal Relevance (MMR) algorithm.

A key innovation in LSH-DynED is the use of **multi-sliding windows**, with each class allocating its own sliding window to buffer the most recent data samples of a fixed size. This ensures that minority classes remain consistently available for training, addressing the challenge of data availability in imbalanced streams.

The undersampling of majority classes is guided by the **Weighted Average of Ratios ($WAR_t$)**, which dynamically identifies majority classes based on their representation in the stream.
</details>

<br>

<details>
<summary><b>Implementation Details</b></summary>

The proposed method is implemented in **Python 3.11.7** and utilizes the following libraries:
* **River 0.21.1**
* **Faiss 1.7.4**

The base classifier used is a **Hoeffding Tree**.

### Dependencies:

* A list of all required packages can be found in the `requirements.txt` file.

### Reproducibility:

For the reproducibility of our results, our implementation is available on GitHub. We have provided all experimental details to make our approach open to new improvements. The baseline methods used for comparison are from the MOA framework, and the BELS implementation is also publicly available.
</details>

<br>

<details>
<summary><b>Usage</b></summary>

To run the LSH-DynED model, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/user/LSH-DynED.git](https://github.com/user/LSH-DynED.git)
    cd LSH-DynED
    ```
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Run the model:**
    ```python
    # Example script to run LSH-DynED on a dataset
    # (Please provide a more detailed script or instructions on how to run your code)
    ```

### Hyperparameters:

The default hyperparameter values used in our experiments are detailed in the paper and are set for broad applicability without tuning to any specific dataset. The optimal values we determined are as follows:
* **Active Components ($S_{slc}$):** 10
* **Training Samples ($n_{train}$):** 20
* **Test Samples ($n_{test}$):** 50
* **Hyperplanes ($n_v$):** 5

</details>

<br>

<details>
<summary><b>Experimental Evaluation</b></summary>

We conducted a thorough experimental evaluation on 33 imbalanced datasets, which include 23 real datasets and ten semi-synthetic data streams. The results show that LSH-DynED demonstrates superior performance, especially on datasets with dynamic imbalance ratios.

For a detailed analysis of our results, including performance on specific datasets and comparisons with 15 other methods, please refer to the full paper.
</details>

<br>

---

## Future Work

We envision several avenues for future research:
* Adapting the approach for binary imbalanced data streams.
* Studying data streams with a high dimensionality of classes or newly emerging class labels.
* Introducing novel datasets representing diverse data streams from various fields.
* Extending the approach to multi-label streams, where each instance may belong to more than one class.

---

<!-- ## Citation

If you use LSH-DynED in your research, please cite our paper:

 ```bibtex
@article{Abadifard2025LSHDynED,
  author = {Abadifard, Soheil and Can, Fazli},
  title = {LSH-DynED: A Dynamic Ensemble Framework with LSH-Based Undersampling for Evolving Multi-Class Imbalanced Classification},
  year = {2025},
  issue_date = {June 2025},
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
  volume = {1},
  number = {1},
  issn = {},
  url = {[https://doi.org/10.1145/nnnnnnn.nnnnnnn](https://doi.org/10.1145/nnnnnnn.nnnnnnn)},
  doi = {10.1145/nnnnnnn.nnnnnnn},
  journal = {},
  month = {jun},
  pages = {31},
  numpages = {31}
}-->
