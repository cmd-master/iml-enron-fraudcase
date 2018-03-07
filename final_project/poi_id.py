#!/usr/bin/python
import sys
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pprint
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from sklearn.cross_validation import train_test_split

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi', 'salary', 'bonus', 'deferral_payments', 'total_payments', 
                 'loan_advances', 'restricted_stock_deferred', 
                 'deferred_income', 'total_stock_value', 'expenses', 
                 'exercised_stock_options', 'other', 'long_term_incentive', 
                 'restricted_stock', 'director_fees', 'to_messages', 
                 'from_poi_to_this_person', 'from_messages', 
                 'from_this_person_to_poi', 'shared_receipt_with_poi']

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Store to my_dataset for easy export below.
my_dataset = data_dict

### Counts to describe the dataset
print "Count of People: ", len(my_dataset.keys())
print "Count of Features: ", len(my_dataset["METTS MARK"].keys())
print "Count of POI: ", len(dict([(k,r) for k,r in my_dataset.iteritems() if r['poi'] == 1]))

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys=True, remove_all_zeroes=False)

### Task 2: Remove outliers
### Scatterplot function that marks the person of interest in red.
def plotScatter(dat, title, labels=features_list):
    for point in dat:
        poi = point[0]
        if poi == 0:
            plt.scatter(point[1], point[2], color = "blue")
        else:
            plt.scatter(point[1], point[2], color = "red")
    plt.ylabel(labels[1])
    plt.xlabel(labels[2])
    plt.title(title)
    plt.show()
    
plotScatter(data, "Before Outlier Removal")
data_dict.pop("THE TRAVEL AGENCY IN THE PARK", 0)
data_dict.pop("TOTAL", 0)
data = featureFormat(my_dataset, features_list, sort_keys=True, remove_all_zeroes=False)
plotScatter(data, "After Outlier Removal")

### Task 3: Create new feature(s)
from sklearn.preprocessing import MinMaxScaler

labels, features = targetFeatureSplit(data)

def doScale(dat):
    scaler = MinMaxScaler()
    rescaled = scaler.fit_transform(dat)
    return rescaled

def addNewFeature(dat, num, name):
    new = {}   
    keys = []
    if name not in features_list:
        features_list.append(name)

    for key in dat.keys():
        keys.append(key)
    keys = sorted(keys)
    for i, f in enumerate(keys):
        elem = {}
        elem[name] = num[i][0]
        for key, val in dat[f].items():
            elem[key] = val
        new[f] = elem
    return new

def selectFeatures(dat, var):
    new = []
    feat = []
    for i, u in enumerate(var):
        feat.append(features_list.index(u) - 1)
        
    for v in dat:
        elem = []
        for f in feat:
            elem.append(v[f])
        new.append(elem)
    return np.array(new)

def compute(dat, how):
    new = []
    if how == "sum":
        for i in dat:
            elem = []
            total = sum(i)
            elem.append(total)            
            new.append(elem)
    elif how == "divide":
        for i in dat:
            elem = []
            if (i[0] <= 0) | (i[1] <= 0):
                elem.append(0.)
            else:
                elem.append(i[0] / i[1])
            new.append(elem)
    return np.array(new)

def plotComparison(labels, feat1, feat2):
    for xx, jj, ii in zip(labels, feat1, feat2):
        if xx == 0:
            plt.scatter(jj, ii, color="b")
        else:
            plt.scatter(jj, ii, color="r")
        
    plt.ylabel("Bonus and Salary")
    plt.xlabel("Email Sent to POI")
    plt.show()

feat_re = doScale(features)
bonus_re = selectFeatures(feat_re, ["bonus", "salary"])
bonus_sm = compute(bonus_re, "sum")

email_from = selectFeatures(features, ["from_this_person_to_poi", "from_messages"])
emailto_poi = compute(email_from, "divide")

plotComparison(labels, emailto_poi, bonus_sm)

my_dataset = addNewFeature(my_dataset, bonus_sm, "bonus_salary")
my_dataset = addNewFeature(my_dataset, emailto_poi, "emailto_poi")

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.naive_bayes import GaussianNB as GNB

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

data = featureFormat(my_dataset, features_list, sort_keys=True, remove_all_zeroes=False)
labels, features = targetFeatureSplit(data)

custom_features = selectFeatures(features, ["bonus_salary", "emailto_poi"])

features_train, features_test, labels_train, labels_test = \
    train_test_split(custom_features, labels, test_size=0.3, random_state=42)

clf = SVC()
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
print "SVM accuracy score: ", round(accuracy_score(labels_test, pred), 4)
print "SVM precision score: ", round(precision_score(labels_test, pred), 4)
print "SVM recall score: ", round(recall_score(labels_test, pred), 4)
print

clf = DTC()
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
print "Decision Tree accuracy score: ", round(accuracy_score(labels_test, pred), 4)
print "Decision Tree precision score: ", round(precision_score(labels_test, pred), 4)
print "Decision Tree recall score: ", round(recall_score(labels_test, pred), 4)
print

clf = GNB()
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
print "Gaussian accuracy score: ", round(accuracy_score(labels_test, pred), 4)
print "Gaussian precision score: ", round(precision_score(labels_test, pred), 4)
print "Gaussian recall score: ", round(recall_score(labels_test, pred), 4)
print

from sklearn.feature_selection import SelectKBest

data = featureFormat(my_dataset, features_list,
                     sort_keys=True, remove_all_zeroes=False)
labels, features = targetFeatureSplit(data)
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

clf = SelectKBest(k = 6)
clf.fit_transform(features_train, labels_train)
features_train_tr = clf.transform(features_train)
features_train_ts = clf.transform(features_test)

def findScores(score, names):
    new = {}
    board = []
    player = []
    for i, value in enumerate(score):
        player.append(names[i+1])
        board.append(round(value, 2))
    new["feature"] = player
    new["score"] = board
    return pd.DataFrame(new).sort_values(by=["score"], 
                                     ascending=False).set_index('feature', drop=True)

feature_sc = findScores(clf.scores_, features_list)
print feature_sc

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.

from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report as classReport

def doPipes(steps):
    pipe = Pipeline(steps)
    pipe.fit(features_train, labels_train).predict(features_test)
    return pipe

def getAccuracy(pipe):
    pred = pipe.fit(features_train, labels_train).predict(features_test)
    return round(accuracy_score(labels_test, pred), 2)

def getClassReport(pipe):
    pipe.fit(features_train, labels_train)
    pred = pipe.predict(features_test)
    return classReport(labels_test, pred)

scalepcasvc = doPipes([
    ("scale", MinMaxScaler()),
    ("pca", PCA()),
    ("svc", SVC())
])

selectKrfc = doPipes([
    ("selectK", SelectKBest()),
    ("rfc", RFC())
])

selectKsteps = doPipes([
    ("selectK", SelectKBest()),
    ("gnb", GNB())
])

scalepcadtc = doPipes([
    ("scale", MinMaxScaler()),
    ("pca", PCA()),
    ("dtc", DTC())
])

print "SelectKBest, RandomForest"
print getAccuracy(selectKrfc), ": Accuracy Score"
print getClassReport(selectKrfc)
print "Scale, PCA, SVC"
print getAccuracy(scalepcasvc), ": Accuracy Score"
print getClassReport(scalepcasvc)
print "SelectKBest, GaussianNB"
print getAccuracy(selectKsteps), ": Accuracy Score"
print getClassReport(selectKsteps)
print "Scale, PCA, DecisionTree"
print getAccuracy(scalepcadtc), ": Accuracy Score"
print getClassReport(scalepcadtc)

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

from sklearn.grid_search import GridSearchCV as GridSVC

param = dict(
    pca__n_components = [1, 2, 3, 4, 5, 6],
    dtc__min_samples_split = [2, 4]
)

clf = scalepcadtc # My Best Algorithm

cv = GridSVC(scalepcadtc, param_grid=param)

from sklearn.model_selection import StratifiedShuffleSplit

sss = StratifiedShuffleSplit(n_splits=2, test_size=0.3, random_state=0)
for train, test in sss.split(features, labels):
    features_train = [features[ii] for ii in train]
    features_test = [features[ii] for ii in test]
    labels_train = [labels[ii] for ii in train]
    labels_test = [labels[ii] for ii in test]

print "Best Algorithm"
print getAccuracy(cv), ": Accuracy Score"
print getClassReport(cv)
print "Best Parameters: ", cv.best_params_
print "PCA explained variance: ", cv.best_estimator_.named_steps["pca"].explained_variance_
print
print "Alternative Algorithm"
print getAccuracy(scalepcasvc), ": Accuracy Score"
print getClassReport(scalepcasvc)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)