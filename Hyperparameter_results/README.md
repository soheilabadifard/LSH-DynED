In our evaluation of the proposed method, we focus on four key parameters to assess how they impact overall performance using the Kappa metric. To investigate different values for each parameter, we conduct this analysis over four semi-synthetic datasets that have a sufficient amount of samples and a variety of features: 'ACTIVITY-D1', 'DJ30-D1', 'GAS-D1', and 'TAGS-D1'. The parameters we examine are:

- The number of active components for classification, three values: 5, 10, and 15.
- The number of samples used for training, three values: 20, 50, and 100.
- The number of samples used for testing purposes, three values: 50, 100, and 200.
- The number of hyperplanes used in the undersampling process, five values: 2, 3, 4, 5, and 6.

By adjusting these parameters, we aim to understand their influence on the method's performance and identify optimal configurations. These values produced 135 possible combinations of parameters for each dataset, totaling 540 results across four datasets.
