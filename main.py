from sklearn.datasets import fetch_openml
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from fairlearn.metrics import MetricFrame
import numpy as np
'''author: Maurizio'''

import DiscriminatedSubgroupsDiscovery as dsd

class EqualOpportunity(dsd.QualityFunction):
    def evaluate(self, description, task):
        data = task.data
        target_attr = task.target.target_attribute
        target_val = task.target.target_value
        pred_attr = task.target.predicted_target_attr

        p_subset = data[description.to_boolean_array(data) & (data[target_attr] == target_val)]
        p_sg = p_subset.shape[0]
        tp_sg = p_subset[(p_subset[pred_attr] == target_val)].shape[0]

        p_complement = data[description.complement_to_boolean_array(data) & (data[target_attr] == target_val)]
        p_c = p_complement.shape[0]
        tp_c = p_complement[(p_complement[pred_attr] == target_val)].shape[0]

        if p_sg == 0: return 0
        if p_c == 0: return 0
        return (tp_c / p_c) - (tp_sg/p_sg)

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

target= dsd.BinaryTarget('y_true',dataset, 'y_pred', target_value=True)

search_space=dsd.SearchSpace(dataset, ignore=['y_true', 'y_pred', 'fnlwgt', 'education'])

discretizer=dsd.Discretizer(discretization_type='mdlp', target='y_true')

task=dsd.SubgroupDiscoveryTask(
        dataset,
        target,
        search_space,
        discretizer =discretizer,
        qf=EqualOpportunity(),
        min_support=250,
        result_set_size=10)



result_set=dsd.BeamSearch(beam_width=10).execute(task)

print("Result:")
print(" QUALITY              DESCRIPTION")
for s in result_set:
       print(s)
