from sklearn.datasets import fetch_openml
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
#from fairlearn.metrics import MetricFrame
#import numpy as np
'''author: Maurizio'''

import fairsd.DiscriminatedSubgroupsDiscovery as dsd



#Import dataset, training the classifier, producing y_pred
d = fetch_openml(data_id=1590, as_frame=True)
dataset = d.data
d_train=pd.get_dummies(dataset)
y_true = (d.target == '>50K') * 1
classifier = DecisionTreeClassifier(min_samples_leaf=10, max_depth=4)
classifier.fit(d_train, y_true)
y_pred = classifier.predict(d_train)
print("ACCURACY OF THE CLASSIFIER:")
print(accuracy_score(y_true,y_pred))
print("-------------------------------------------"+"\n")

dataset['y_true'] = y_true
dataset['y_pred'] = y_pred
#dataset=dataset.head(10)

target= dsd.BinaryTarget(target_attribute='y_true',dataset=dataset, predicted_target_attr='y_pred', target_value=True)

search_space=dsd.SearchSpace(dataset, ignore=['y_true', 'y_pred', 'fnlwgt', 'education'])

discretizer=dsd.Discretizer(discretization_type='mdlp', target='y_true')

task=dsd.SubgroupDiscoveryTask(
        dataset,
        target,
        search_space,
        discretizer =discretizer,
        qf=dsd.EqualOpportunity(),
        min_support=250,
        result_set_size=10)



result_set=dsd.DSSD(beam_width=10).execute(task)

print("Result:")
print(" QUALITY              DESCRIPTION")
for s in result_set:
       print(s)
