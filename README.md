# FairSD

FairSD is a package that implements top-k subgroup discovery algorithms for identifying subgroups that may be treated unfairly by a machine learning model.<br/>

The package has been designed to offer the user the possibility to use different notions of fairness as quality measures. Integration with the [Fairlearn]( https://fairlearn.github.io/) package allows the user to use all the [fairlearn metrics](https://fairlearn.github.io/v0.6.0/api_reference/fairlearn.metrics.html) as  quality measures. The user can also define custom quality measures, by extending the QualityFunction class present in `qualitymeasures.py` module.

## Acknowledgements
This package is a fork of the original fairsd repository by [Maurizio Pulizzi](https://github.com/MaurizioPulizzi). The original repository can be found [here](https://github.com/MaurizioPulizzi/fairsd). However, since the original repository is no longer maintained, my fork is used to add convenience features and bug fixes and thus will be released in this package.

### Acknowledgements to the original repository
Some parts of the code are an adaptation of the [pysubgroup package](https://github.com/flemmerich/pysubgroup). These parts are indicated in the code.

### Contributors to the original repository
* [Maurizio Pulizzi](https://github.com/MaurizioPulizzi)
* [Hilde Weerts](https://github.com/hildeweerts)
