# LSH-DynED: A Dynamic Ensemble Framework with LSH-Based Undersampling for Evolving Multi-Class Imbalanced Classification

This repository contains the implementation of the LSH-DynED model, a novel, robust, and resilient approach for classifying imbalanced and non-stationary data streams with multiple classes.

**Authors:**
* Soheil Abadifard, Kansas State University (abadifard@k-state.edu)
* Fazli Can, Bilkent University (canf@cs.bilkent.edu.tr)

---

## Overview

The classification of imbalanced data streams, where class distributions are unequal and change over time, is a significant challenge in machine learning, especially in multi-class scenarios. LSH-DynED addresses this challenge by integrating Locality Sensitive Hashing with Random Hyperplane Projections (LSH-RHP) into the Dynamic Ensemble Diversification (DynED) framework. This marks the first application of LSH-RHP for undersampling in the context of imbalanced non-stationary data streams.

LSH-DynED undersamples the majority classes using LSH-RHP to create balanced training batches, which in turn improves the minority-class performance of the ensemble. Our experiments on 33 real-world and semi-synthetic datasets against 15 state-of-the-art methods show improvements in Kappa and mG-Mean, with the best average rank on both metrics.

### Key Features:

* **Novel Undersampling Technique:** First application of Locality Sensitive Hashing with Random Hyperplane Projections (LSH-RHP) for undersampling in multi-class imbalanced non-stationary data streams.
* **Dynamic Ensemble Framework:** Extends and modifies the DynED framework to handle dynamic imbalance ratios in multi-class imbalanced data stream tasks.
* **State-of-the-Art Performance:** Achieves the best average rank in both Kappa and mG-Mean effectiveness measures on a wide range of datasets.
* **Robust and Resilient:** Effectively handles concept drift and dynamic changes in class distributions.
* **Open Source:** The implementation is publicly available to encourage further research and improvements.

---

<details>
<summary><b>How it Works</b></summary>

LSH-DynED operates in three main stages:

1.  **Prediction and Training:** A subset of the ensemble, the "selected components," predicts the label of incoming data instances by averaging the components' class probabilities (soft voting). These components are then trained on the new data instance.
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

A portable conda environment with a pinned package set (Python 3.10) is provided in `environment.yml`.

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
<summary><b>Datasets</b></summary>

The paper evaluates LSH-DynED on 33 imbalanced multi-class streams: 23 real-world datasets and ten semi-synthetic streams with dynamic imbalance ratios. The full list and their characteristics are given in the paper's dataset table.

* **Benchmark streams (real and semi-synthetic):** distributed with the reproducible experimental framework of Aguiar, Sousa, and Cano's survey on learning from imbalanced data streams ([imbalanced-streams](https://github.com/canoalberto/imbalanced-streams)) — the same framework used to run the MOA baselines. The semi-synthetic `*-D1` streams follow the construction described by Korycki and Krawczyk. Note that some streams in this collection are extended versions of small datasets: `zoo` is distributed as a 1,000,000-instance stream recirculated from the 101-instance UCI Zoo dataset, with the original schema including the `animal` attribute.
* **KEEL datasets** (`*_cleaned.arff`): multi-class imbalanced datasets from the [KEEL repository](https://sci2s.ugr.es/keel/datasets.php), converted from KEEL `.dat` format to ARFF.
* The remaining real-world datasets are publicly accessible from the [UCI repository](https://archive.ics.uci.edu) and Kaggle, as detailed in the paper.

Place the `.arff` files under `Imbalance Datasets/` following the folder layout described in Usage.

</details>

<br>

<details>
<summary><b>Usage</b></summary>

To run the LSH-DynED model, follow these steps:

1.  **Clone the repository and create the environment:**
    ```bash
    git clone https://github.com/soheilabadifard/LSH-DynED.git
    cd LSH-DynED
    conda env create -f environment.yml
    conda activate dyned-imb
    ```
2.  **Prepare your datasets:**
    * The entry point reads `.arff` streams from `Imbalance Datasets/<group>/<source>/` under the repository root.
    * For the multi-class real datasets, create `Imbalance Datasets/arff-datasets-multiclass/Alberto_data/` and place your `.arff` files there.
    * Other groups (for example `KEEL_imbalanced_multiclass` or `arff-multi-class-semi-synthetic/Alberto_data_new`) are selected through the `mode` and `src` arguments of `main()` at the bottom of `Model/Main-newreaddata.py`.
3.  **Run the model:**
    ```bash
    PYTHONPATH=. python Model/Main-newreaddata.py
    ```
    Results are written under `new_res/`, mirroring the input folder structure. Datasets that already have outputs are skipped, so interrupted runs can be resumed.

#### Output

For each dataset processed, the run writes two CSV files:

* `<dataset>.csv`: one row per 500-instance evaluation window, with the columns
  * `counter`: the number of instances processed at the end of the window,
  * `kappa`: Cohen's kappa over the window,
  * `avg_geo`, `recall`, `precision`, `f1`: window-level convenience metrics from imblearn's `classification_report_imbalanced`,
  * `CM[i][j]`: the window's full confusion matrix, where row `i` is the true class and column `j` is the predicted class.
* `<dataset>_prequential.csv`: the same metrics computed over a 500-instance sliding window.

CSVs produced by runs before September 2026 name the `avg_geo` column `g_mean`.

### Hyperparameters:

The default hyperparameter values used in our experiments are detailed in the paper and are set for broad applicability without tuning to any specific dataset. The optimal values we determined are as follows:
* **Active Components ($S_{slc}$):** 10
* **Training Samples ($n_{train}$):** 20
* **Test Samples ($n_{test}$):** 50
* **Hyperplanes ($n_v$):** 5

</details>

<br>

<details>
<summary><b>How the Paper's Metrics Are Computed</b></summary>

The paper reports two effectiveness metrics per method and dataset. Both are computed per 500-instance window and then averaged over the dataset's windows.

* **Kappa** is the `kappa` column.
* **mG-Mean** is the geometric mean of per-class recalls, recomputed from the `CM[i][j]` columns of each window: rebuild the window's confusion matrix, take each class's recall (the diagonal cell divided by its row sum), multiply the recalls of the classes present in the window, and take the c-th root, where c is the number of classes in the dataset. A class with no true instance in a window has undefined recall and contributes a neutral factor; the exponent stays 1/c. A class that is present but never correctly predicted makes the window's mG-Mean zero. The same convention applies to every method and every table in the paper.

**The `avg_geo` column is not mG-Mean.** `avg_geo` is imblearn's support-weighted average of per-class `sqrt(sensitivity * specificity)`, a different statistic that remains nonzero when a present class has zero recall. No value reported in the paper uses this column; it exists only as a run-time convenience. Because every window's full confusion matrix is stored in the CSVs, all reported values can be recomputed directly from the `CM[i][j]` columns.

</details>

<br>

<details>
<summary><b>Experimental Evaluation</b></summary>

We conducted a thorough experimental evaluation on 33 imbalanced datasets, covering real-world datasets and semi-synthetic data streams. The results show that LSH-DynED demonstrates superior performance, especially on datasets with dynamic imbalance ratios.

For a detailed analysis of our results, including performance on specific datasets and comparisons with 15 other methods, please refer to the full paper.

</details>

<br>

---

## Citation

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
  DOI={10.48550/ARXIV.2506.20041}
}
```
