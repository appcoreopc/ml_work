
Bootstrap - sample used multipel times for a single tree

Boostrap aggragating or bagging - training individual learner woth boostrap data and then averaging the prediction is known as bagging


from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import numpy as np

model = RandomForestClassifier(n_estimators=10)

model.fit(iris.data, iris.target)

There are type 0,1,2 (3 differemt class)

iris.data.tolist()
iris.target.tolist()

sample = np.array([[5.1,3.5,1.4,0.2]])
model.predict_proba(sample) ### (should returns 1,0,0 - which means type 0)

sample2 = np.array([[5.9, 3.0, 5.1, 1.8]])
model.predict_proba(sample2) ### (should returns 0,0.1,0.9 - which means type)
which means 0.1 chance of type 1 
and 0.9 chances of type 2
