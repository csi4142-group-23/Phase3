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
Preprocessing: Removed rows with null values results in reduction from 14843 to 7453 rows
Class imbalance: Our processed data contains twice as many dry cases as there are "not dry" cases (i.e. wet, had loose or packed snow, were icy or covered with slush). To deal with the class imbalance, we used 5-fold cross validation.

ii) Model Construction:
Logistic Regression

iii) Model Evaluation:
5-fold cross validation on recall, accuracy, precision, and f1 score using 33% of the data for testing