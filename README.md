# LSH-DynED: A Dynamic Ensemble Framework with LSH-Based Undersampling for Evolving Multi-Class Imbalanced Classification

This repository contains the implementation of the LSH-DynED model, a novel, robust, and resilient approach for classifying imbalanced and non-stationary data streams with multiple classes.

**Authors:**
* Soheil Abadifard, Kansas State University (abadifard@k-state.edu)
* Fazli Can, Bilkent University (canf@cs.bilkent.edu.tr)
  
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

</details>

<br>

<details>
<summary><b>Implementation Details</b></summary>

The proposed method is implemented in **Python 3.11.7** and utilizes the following libraries:
* **River 0.21.1**
* **Faiss 1.7.4**

The base classifier used is a **Hoeffding Tree**.

### Reproducibility:

For the reproducibility of our results, our implementation is available on GitHub. We have provided all experimental details to make our approach open to new improvements. The baseline methods used for comparison are from the MOA framework, and other implementations are also publicly available.

#### Baselines

| Method | Implementation Link |
| :--- | :--- |
| **General-Purpose Methods (GPM)** | |
| OzaBagAdwin (OBA) | [MOA Framework](https://github.com/Waikato/moa) |
| Leveraging Bagging (LB) | [MOA Framework](https://github.com/Waikato/moa) |
| ARF | [MOA Framework](https://github.com/Waikato/moa) |
| SRP | [MOA Framework](https://github.com/Waikato/moa) |
| KUE | [MOA Framework](https://github.com/canoalberto/Kappa-Updated-Ensemble) |
| BELS | [GitHub Repository](https://github.com/sepehrbakhshi/BELS) | 
| DynED | [GitHub Repository](https://github.com/soheilabadifard/DynED) |
| **Imbalance-Specific Methods (ISM)** | |
| HD-VFDT | [MOA Framework](https://github.com/Waikato/moa) |
| GH-VFDT | [MOA Framework](https://github.com/Waikato/moa) |
| MUOB | [MOA Framework](https://github.com/Waikato/moa) |
| MOOB | [MOA Framework](https://github.com/Waikato/moa) |
| ARFR | [MOA Framework](https://github.com/Waikato/moa) |
| CSARF | [MOA Framework](https://github.com/Waikato/moa) |
| CALMID | [MOA Framework](https://github.com/Waikato/moa) |
| ROSE | [GitHub Repository](https://github.com/canoalberto/ROSE) |
| MicFoal | [MOA Framework](https://github.com/Waikato/moa) |

</details>

<br>

<details>
<summary><b>Usage</b></summary>

To run the LSH-DynED model, follow these steps:

1.  **Prepare Your Datasets:**
    * Create a directory (e.g., `datasets/`).
    * Place all your dataset files (e.g., in `.arff` format) inside this directory. The script will iterate through and process every file in this folder.

2.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/user/LSH-DynED.git](https://github.com/user/LSH-DynED.git)
    cd LSH-DynED
    ```
3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Configure the Script:**
    * Open the `main.py` file.
    * Go to the last line of the script:
        ```python
        if __name__ == "__main__":
            main('Path to dataset Directory')
        ```
    * Modify the path inside the `main()` function to point to the directory you created in step 1. For example, if your folder is named `datasets`, the line should look like this:
        ```python
        if __name__ == "__main__":
            main('datasets/')
        ```
5.  **Run the Model:**
    * Execute the script from your terminal:
        ```bash
        python main.py
        ```
    * The script will now run the LSH-DynED model on each dataset in the specified folder.

#### Output

For each dataset processed (e.g., `my_data.arff`), the script will generate two new CSV files in the same directory:
* `my_data.arff_mgmean.csv`: Contains the G-Mean scores calculated every 500 instances.
* `my_data.arff_kappa.csv`: Contains the prequential Kappa scores.

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

## CitationAdd

If you use LSH-DynED in your research, please cite our paper:

 ```bibtex
@article{Abadifard2025LSHDynED,
  title={LSH-DynED: A Dynamic Ensemble Framework with LSH-Based Undersampling for Evolving Multi-Class Imbalanced Classification}, 
      author={Soheil Abadifard and Fazli Can},
      year={2025},
      eprint={2506.20041},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2506.20041},
}
