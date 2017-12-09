import pandas as pd
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier



train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

all_ids = test['id']
main_test = test.drop(['id'], axis = 1)

#Validation Dataset
x_train = train[:400000]
xx = x_train.drop(['id','target'], axis = 1)
yy = x_train['target']
#Remove 'target' from the test data set
Y_test = train[400000:]
Y_test = Y_test.drop(['target'], axis = 1)


clf = XGBClassifier()
clf.fit(xx, yy)

samplePredictions = clf.predict(Y_test.drop(['id'], axis = 1))
predictions1 = [round(value) for value in samplePredictions]

#Compare with the actual target in the test data
Y_test = train[400000:]
y_true = Y_test['target']
print("The Accuracy of xgBoost: ", accuracy_score(y_true, predictions1))


#The Code below is the actual predictions on the test data provided.
X = train.drop(['id','target'], axis = 1)
Y = train['target']

clf = XGBClassifier()
clf.fit(X, Y)


xgBPredictions = clf.predict(test.drop(['id'], axis = 1))
predictions = [round(value) for value in xgBPredictions]

#save to submission file as required
test['target'] = predictions
submission = test.loc[:,['id', 'target']]
submission.columns = ['id', 'target']
submission.to_csv('xgBPredictions.csv', index=False)
