# Phase3
Project Phase 3 - Classification

Classification. Employ a classification algorithm (e.g. decision
tree, support vector machine, lazy learner, ensemble, etc.) to explore the
data.

*Construct a model to determine the characteristics of accidents that
occurred when road surfaces were not dry. That is, consider only the
instances when the road surfaces were wet, had loose or packed snow,
were icy or covered with slush*

i) Data Processing
Data reduction - removed rows with null values results in reduction from 14843 to 7453 rows; Class imbalance - our processed data contains twice as many dry cases as there are "not dry" cases (i.e. wet, had loose or packed snow, were icy or covered with slush). To deal with the class imbalance, we used 5-fold cross validation.

ii) Model Construction:
We tested multiple classifiers including KNN, Adaboost, Random Forest, Decision Trees, and Logistic Regression. All classifiers, on average, seemed to perform reasonably well within the 70-85% range. We chose logistic regression with class_weight='balanced' (to accomodate for class imbalance) because it was a fast and simple algorithm intended for binary classification that achieved the same results as more advanced classifiers.

iii) Model Evaluation:
5-fold cross validation on recall, accuracy, precision, and f1 score.
Our model also performed ~40% better when compared with a dummy classifier.

iv) Insights and Lessons Learned:
Discovered that Environment, Date, and Light were the three most important features in determining whether or not an accident
occured on wet or dry roads. This makes sense as wet accidents can only occur during or after a rain/snow storm. 