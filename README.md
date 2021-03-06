# FairSD

FairSD is a package that implements top-k subgroup discovery algorithms for identifying subgroups that may be treated unfairly by a machine learning model.<br/>

The package has been designed to offer the user the possibility to use different notions of fairness as quality measures. Integration with the [Fairlearn]( https://fairlearn.github.io/) package allows the user to use all the [fairlearn metrics](https://fairlearn.github.io/v0.6.0/api_reference/fairlearn.metrics.html) as  quality measures. The user can also define custom quality measures, by extending the QualityFunction class present in the [fairsd.qualitymeasures](https://github.com/MaurizioPulizzi/fairsd/blob/main/fairsd/qualitymeasures.py) module.


## Usage
For common usage refer to the [Jupyter notebooks](https://github.com/MaurizioPulizzi/fairsd/tree/main/notebooks). In particular:
* [Quick start - FairSD usage](https://github.com/MaurizioPulizzi/fairsd/blob/main/notebooks/fairsd_usage.ipynb).
* [FairSD settings](https://github.com/MaurizioPulizzi/fairsd/blob/main/notebooks/fairsd_settings.ipynb), for a detailed explanation of how inizialize the SugbgroupDiscoveryTask object and use the implemented subgroup discovery algorithms.


## Contributors
* [Maurizio Pulizzi](https://github.com/MaurizioPulizzi)
* [Hilde Weerts](https://github.com/hildeweerts)


## Acknowledgements
Some parts of the code are an adaptation of the [pysubgroup package](https://github.com/flemmerich/pysubgroup). These parts are indicated in the code.
