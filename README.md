# FairSD

FairSD is a package that implement subgroup discovery algorithms for finding the most discriminated subgroups in a dataset or in a machine learning model. The meaning of “discriminated subgroups” refers to the Fairness-aware Machine Learning topic.<br/>
This package has been designed to offer the user the possibility to use different discrimination measures as quality measures and also offers integration with the [Fairlearn library]( https://fairlearn.github.io/). This integration makes possible to use all the [Fairlearn metrics](https://fairlearn.github.io/v0.6.0/api_reference/fairlearn.metrics.html) as quality measures. The user can also define custom quality measures by extending the QualityFunction class present in the [fairsd.qualitymeasures](https://github.com/MaurizioPulizzi/fairsd/blob/main/fairsd/qualitymeasures.py) module.

