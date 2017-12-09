import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier


train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")


#MLPClassifier

#Used the below commented out section to get the idea about the prediction accuracy of the MLPClassifier
#I divided the train data set into two sections to do so.

x_train = train[:400000]
xx = x_train.drop(['id','target'], axis = 1)
yy = x_train['target']
#Remove 'target' from the test data set
Y_test = train[400000:]
Y_test = Y_test.drop(['target'], axis = 1)

clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1)
clf.fit(xx, yy)


samplePredictions = clf.predict(Y_test.drop(['id'], axis = 1))
#Compare with the actual target in the test data
Y_test = train[400000:]
y_true = Y_test['target']
print(accuracy_score(y_true, samplePredictions))

#0.96348072864373091 -> Very Bad



#Actual analysis
X = train.drop(['id','target'], axis = 1)
Y = train['target']

clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1)
clf.fit(X, Y)


DNNPredictions = clf.predict(test.drop(['id'], axis = 1))

#save to submission file as required
test['target'] = DNNPredictions
submission = test.loc[:,['id', 'target']]
submission.columns = ['id', 'target']
submission.to_csv('DNNPredictions.csv', index=False)