from sklearn.datasets import fetch_openml
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
#from fairlearn.metrics import MetricFrame
#import numpy as np
'''author: Maurizio'''

import fairsd.subgroupdiscovery as dsd



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


#dataset=dataset.head(10)


task=dsd.SubgroupDiscoveryTask(dataset, y_true, y_pred)
result_set=dsd.BeamSearch(beam_width=10).execute(task)



print("Result:")
print(" QUALITY              DESCRIPTION")
for s in result_set:
       print(s)
