# LSH-DynED

The proposed method is located inside the [Model](Model) folder.

# LSH-DynED: A Dynamic Ensemble Framework with LSH-Based Undersampling for Evolving Multi-Class Imbalanced Classification

This repository contains the implementation of the LSH-DynED model, a novel, robust, and resilient approach for classifying imbalanced and non-stationary data streams with multiple classes. This work has been published in [insert journal/conference name here] and is authored by Soheil Abadifard and Fazli Can.

**Authors:**
* Soheil Abadifard, Kansas State University (abadifard@ksu.edu)
* Fazli Can, Bilkent University (canf@cs.bilkent.edu.tr)

**Publication:**
Soheil Abadifard and Fazli Can. 2025. LSH-DynED: A Dynamic Ensemble Framework with LSH-Based Undersampling for Evolving Multi-Class Imbalanced Classification. [cite_start]1, 1 (June 2025), 31 pages. https://doi.org/10.1145/nnnnnnn.nnnnnnn 

---

## Overview

[cite_start]The classification of imbalanced data streams, where class distributions are unequal and change over time, is a significant challenge in machine learning, especially in multi-class scenarios.  [cite_start]LSH-DynED addresses this challenge by integrating Locality Sensitive Hashing with Random Hyperplane Projections (LSH-RHP) into the Dynamic Ensemble Diversification (DynED) framework.  [cite_start]This marks the first application of LSH-RHP for undersampling in the context of imbalanced non-stationary data streams. 

[cite_start]LSH-DynED undersamples the majority classes using LSH-RHP to create a balanced training set, which in turn improves the prediction accuracy of the ensemble.  [cite_start]Our comprehensive experiments on 23 real-world and ten semi-synthetic datasets demonstrate that LSH-DynED outperforms 15 state-of-the-art methods in terms of Kappa and mG-Mean effectiveness.  [cite_start]The model excels in handling large-scale, high-dimensional datasets with significant class imbalances and shows strong adaptation and robustness in real-world scenarios. 

### Key Features:

* [cite_start]**Novel Undersampling Technique:** First application of Locality Sensitive Hashing with Random Hyperplane Projections (LSH-RHP) for undersampling in multi-class imbalanced non-stationary data streams. 
* [cite_start]**Dynamic Ensemble Framework:** Extends the DynED framework to handle dynamic imbalance ratios in multi-class imbalanced data stream tasks. 
* [cite_start]**State-of-the-Art Performance:** Outperforms other methods in both Kappa and mG-Mean effectiveness measures on a wide range of datasets. 
* [cite_start]**Robust and Resilient:** Effectively handles concept drift and dynamic changes in class distributions. 
* [cite_start]**Open Source:** The implementation is publicly available to encourage further research and improvements. 

---

## How it Works

LSH-DynED operates in three main stages:

1.  **Prediction and Training:** A subset of the ensemble, the "selected components," predicts the label of incoming data instances via majority voting. [cite_start]These components are then trained on the new data. 
2.  [cite_start]**Drift Detection and Adaptation:** The ADWIN drift detector monitors the system's performance.  [cite_start]If drift is detected, a new component is trained on recent data from a balanced dataset created by our novel undersampling method and added to a pool of "reserved components." 
3.  [cite_start]**Component Selection:** The ensemble's components are updated to maintain a balance between accuracy and diversity.  [cite_start]Components are selected from the combined pool of "selected" and "reserved" components based on their accuracy and diversity, measured by the kappa statistic. 

[cite_start]A key innovation in LSH-DynED is the use of **multi-sliding windows**, with each class having its own window to buffer recent data samples.  [cite_start]This ensures that minority classes remain consistently available for training, addressing the challenge of data availability in imbalanced streams. 

[cite_start]The undersampling of majority classes is guided by the **Weighted Average of Ratios (WARt)**, which dynamically identifies majority classes based on their representation in the stream. 

---

## Implementation Details

The proposed method is implemented in **Python 3.11.7** and utilizes the following libraries:
* [cite_start]**River 0.21.1** 
* [cite_start]**Faiss 1.7.4** 

[cite_start]The base classifier used is a **Hoeffding Tree**. 

### Dependencies:

* A list of all required packages can be found in the `requirements.txt` file.

### Reproducibility:

[cite_start]For the reproducibility of our results, our implementation is available on GitHub.  [cite_start]We have provided all experimental details to make our approach open to new improvements.  [cite_start]The baseline methods used for comparison are from the MOA framework and the BELS implementation is also publicly available. 

---

## Usage

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

[cite_start]The default hyperparameter values used in our experiments are detailed in the paper and are set for broad applicability without tuning to any specific dataset.  The optimal values we determined are:
* [cite_start]**Active Components:** 10 
* [cite_start]**Training Samples:** 20 
* [cite_start]**Test Samples:** 50 
* [cite_start]**Hyperplanes:** 5 

---

## Experimental Evaluation

[cite_start]We conducted a thorough experimental evaluation on 33 imbalanced datasets, including 23 real-world and 10 semi-synthetic datasets.  [cite_start]The results show that LSH-DynED demonstrates superior performance, especially on datasets with dynamic imbalance ratios. 

For a detailed analysis of our results, please refer to the full paper.

---

## Future Work

We envision several avenues for future research:
* [cite_start]Adapting the approach for binary imbalanced data streams. 
* [cite_start]Studying data streams with high dimensionality of classes or newly emerging classes. 
* [cite_start]Introducing novel datasets from various fields. 
* [cite_start]Extending the approach to multi-label streams. 

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
