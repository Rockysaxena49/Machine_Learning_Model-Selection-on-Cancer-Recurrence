import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import sklearn.model_selection as model_selection
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
import numpy as np
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns
# pycharm code below to print out how many features to display

pd.set_option('display.width', 400)
pd.set_option('display.max_columns', 36)

#====================================================data pre-proccessing===============================================
# load the dataset
dataset = pd.read_csv("breast_cancer.csv",sep='\s*,\s*', engine='python',encoding='utf-8-sig') # sep used to ignore spaces in column name when calling them

print(dataset.head())
print(dataset.shape)# 198 by 35

# replace R with 1 and N with 0
dataset.replace(to_replace = ['R','N'], value=[1,0], inplace=True)
print(dataset)

# visualize the data
sns.set_style("whitegrid")
sns.FacetGrid(dataset, hue="outcome", height=4) \
   .map(plt.scatter, "time", "tumor size") \
   .add_legend()
plt.show()

print(dataset['outcome'].value_counts()) # count how many numbers are in each class of the target variable
#The dataset is really unbalanced at class 0: 151 and class 1: 47

sns.countplot(x = 'outcome', data=dataset) # counts how many different class and how many of each there are
plt.show()

print(dataset.groupby('outcome').mean()) # displays the average value of each column that gives that particular class

#########################################Data-processing ###############################################################
# this is to check if there are any missing values in our dataset in which case there are 4 in lymph node
print('Nan before impute\n ',dataset.isna().sum()) # count the number of NaN values (missing feature values)

# replace Nan values with average value of the feature
dataset.fillna(dataset.mean(), inplace=True)

# check if nan have been changed with average value
print('Nan before impute\n ',dataset.isna().sum()) # count the number of NaN values (missing feature values)

# drop the ID column because it is not useful
dataset.drop('ID', axis=1, inplace=True)
print(dataset)

# Separate the features and target variables
X = dataset.iloc[:,0:33]
y = dataset.iloc[:,-1]
print(X.head())
# The next step is to standardize our features we will use the z score which gives mean 0 and variance 1
sc = StandardScaler() # create object of class
xScaled = sc.fit_transform(X)


#split the scaled data into train (70%), test (20%), and validation (10%)
#This is done by running train test split twice
X_train, X_test, y_train, y_test = model_selection.train_test_split(xScaled, y, test_size=0.2, random_state=1)
X_train, X_val, y_train, y_val = model_selection.train_test_split(X_train, y_train,test_size=0.125, random_state=1)

print('X_train: ', X_train.shape)
print('X_test: ', X_test.shape)
print('X_val shape: ',X_val.shape)
# Apply PCA for dimensionality reduction
pca = PCA(.95) # 95% variance retained
pca.fit(X_train) # fit only on the training set
print('95% of variance amounts to ', pca.n_components_,' components')
# transform on everything
X_train = pca.transform(X_train)
X_val = pca.transform(X_val)
X_test = pca.transform(X_test)
# we want to over sample the minority class to help balance the dataset and generate "new" samples of patients whose breast cancer reoccured
smote = SMOTE(random_state=0)
xSmote,ySmote=smote.fit_sample(X_train, y_train)

# compare class counts before and after
print(y_train.value_counts()) # before 0: 107, 1: 31
print(ySmote.value_counts()) # after 0: 107, 1:107
sns.countplot(x = ySmote, data= ySmote) # counts how many different class and how many of each there are
plt.show()

def naives():
    THRESHOLD = 0.031
    classifier = GaussianNB()
    classifier.fit(X_train, y_train)
    fpr, tpr, threshold = roc_curve(y_test, classifier.predict_proba(X_test)[:,1])

    # Parameters 2
    clf = GaussianNB()
    clf.fit(xSmote,ySmote)
    fpr2, tpr2, threshold2 = roc_curve(y_test, clf.predict_proba(X_test)[:, 1])
    naive_roc_auc2 = auc(fpr2,tpr2)

    naive_roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, label='NO SMOTE = %0.2f' % naive_roc_auc)
    plt.plot(fpr2, tpr2, label='SMOTE = %0.2f' % naive_roc_auc2)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')

    # # create the axis of thresholds (scores)
    # ax2 = plt.gca().twinx()
    # ax2.plot(fpr, threshold, markeredgecolor='r', linestyle='dashed', color='r')
    # ax2.set_ylabel('Threshold', color='r')
    # ax2.set_ylim([threshold[-1], threshold[0]])
    # ax2.set_xlim([fpr[0], fpr[-1]])
    plt.show()

    # Predicting the model
    preds_threshold = np.where(classifier.predict_proba(X_test)[:, 1] > THRESHOLD, 1, 0)
    tn, fp, fn, tp = confusion_matrix(y_test, preds_threshold).ravel()
    print('tn: {} fp: {} fn: {} tp: {}' .format(tn, fp, fn, tp))
    print("Accuracy score: {}".format(accuracy_score(y_test, preds_threshold)*100))
    data = {'FPR':fpr, 'TPR':tpr}
    table = pd.DataFrame(data, index = threshold)
    print(table)
    return fpr, tpr, threshold


def logRegres():
    THRESHOLD = 0.202807
    classifier = LogisticRegression(penalty='l1',solver='liblinear', max_iter=100000)  # create object of the class LogisticRegression
    classifier.fit(X_train, y_train)
    fpr, tpr, threshold = roc_curve(y_test, classifier.predict_proba(X_test)[:,1])

    # Parameters 2
    clf2 = LogisticRegression(penalty='l2', solver='lbfgs', max_iter=100000)
    clf2.fit(X_train, y_train)
    fpr2, tpr2, threshold2 = roc_curve(y_test, clf2.predict_proba(X_test)[:, 1])
    logs_roc_auc2 = auc(fpr2, tpr2)

    #parameter 3
    clf3 = LogisticRegression(penalty='none', solver='newton-cg', max_iter=100000)
    clf3.fit(X_train, y_train)
    fpr3, tpr3, threshold3 = roc_curve(y_test, clf3.predict_proba(X_test)[:, 1])
    logs_roc_auc3 = auc(fpr3, tpr3)

    logs_roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, label='Parameter 1 L1 AUC = %0.2f' % logs_roc_auc)
    plt.plot(fpr2, tpr2, label='Parameter 2 L2 AUC = %0.2f' % logs_roc_auc2)
    plt.plot(fpr3, tpr3, label='Parameter 3 None AUC = %0.2f' % logs_roc_auc3)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    #
    # # create the axis of thresholds (scores)
    # ax2 = plt.gca().twinx()
    # ax2.plot(fpr, threshold, markeredgecolor='r', linestyle='dashed', color='r')
    # ax2.set_ylabel('Threshold', color='r')
    # ax2.set_ylim([threshold[-1], threshold[0]])
    # ax2.set_xlim([fpr[0], fpr[-1]])
    plt.show()

    preds_threshold = np.where(classifier.predict_proba(X_test)[:, 1] > THRESHOLD, 1, 0)
    tn, fp, fn, tp = confusion_matrix(y_test, preds_threshold).ravel()
    print('tn: {} fp: {} fn: {} tp: {}'.format(tn, fp, fn, tp))
    print("Accuracy score: {}".format(accuracy_score(y_test, preds_threshold) * 100))
    data = {'FPR': fpr, 'TPR': tpr}
    table = pd.DataFrame(data, index=threshold)
    print(table)
    return fpr, tpr, threshold


def svm():
    THRESHOLD = 0.150286
    classifier = SVC(kernel= 'rbf', random_state=0, probability=True, tol=1e-5,max_iter=100000,C= 50)
    classifier.fit(X_train, y_train)
    fpr, tpr, threshold = roc_curve(y_test, classifier.predict_proba(X_test)[:,1])

    # Parameters 2
    clf2 = SVC(kernel='sigmoid', C=2, probability=True, max_iter=100000, random_state=0)
    clf2.fit(X_train, y_train)
    fpr2, tpr2, threshold2 = roc_curve(y_test, clf2.predict_proba(X_test)[:, 1])
    svm_roc_auc2 = auc(fpr2, tpr2)

    # parameter 3
    clf3 = SVC(kernel='linear', C= 0.0001, probability=True, max_iter=100000, random_state=0)
    clf3.fit(X_train, y_train)
    fpr3, tpr3, threshold3 = roc_curve(y_test, clf3.predict_proba(X_test)[:, 1])
    svm_roc_auc3 = auc(fpr3, tpr3)

    svm_roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, label='kernel:rbf, reg:50 AUC = %0.2f' % svm_roc_auc)
    plt.plot(fpr2, tpr2, label='kernel:sigmoid, reg:2 AUC = %0.2f' % svm_roc_auc2)
    plt.plot(fpr3, tpr3, label='kernel:rbf, reg:0.0001 AUC = %0.2f' % svm_roc_auc3)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    #
    # # create the axis of thresholds (scores)
    # ax2 = plt.gca().twinx()
    # ax2.plot(fpr, threshold, markeredgecolor='r', linestyle='dashed', color='r')
    # ax2.set_ylabel('Threshold', color='r')
    # ax2.set_ylim([threshold[-1], threshold[0]])
    # ax2.set_xlim([fpr[0], fpr[-1]])
    plt.show()

    preds_threshold = np.where(classifier.predict_proba(X_test)[:, 1] > THRESHOLD, 1, 0)
    tn, fp, fn, tp = confusion_matrix(y_test, preds_threshold).ravel()
    print('tn: {} fp: {} fn: {} tp: {}'.format(tn, fp, fn, tp))
    print("Accuracy score: {}".format(accuracy_score(y_test, preds_threshold) * 100))
    data = {'FPR': fpr, 'TPR': tpr}
    table = pd.DataFrame(data, index=threshold)
    print(table)
    return fpr, tpr, threshold


def mlp():
    THRESHOLD = 0.5
    classifier = MLPClassifier(random_state=0,learning_rate_init=0.01, activation='relu', alpha=1.04,
                        hidden_layer_sizes=100, max_iter=100000, batch_size= 4)
    classifier.fit(X_train, y_train)
    fpr, tpr, threshold = roc_curve(y_test, classifier.predict_proba(X_test)[:,1])

    # Parameters 2
    clf2 = MLPClassifier(random_state=0, learning_rate_init=0.0001, activation='tanh',alpha= 2,
                         hidden_layer_sizes= 6, batch_size= 6, max_iter=100000  )
    clf2.fit(X_train, y_train)
    fpr2, tpr2, threshold2 = roc_curve(y_test, clf2.predict_proba(X_test)[:, 1])
    mlp_roc_auc2 = auc(fpr2, tpr2)

    # parameter 3
    clf3 = MLPClassifier(random_state=0, learning_rate_init=0.05, activation='logistic',alpha= 20,
                         hidden_layer_sizes=25, batch_size=1, max_iter=100000)
    clf3.fit(X_train, y_train)
    fpr3, tpr3, threshold3 = roc_curve(y_test, clf3.predict_proba(X_test)[:, 1])
    mlp_roc_auc3 = auc(fpr3, tpr3)

    mlp_roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, label='relu, learning: 0.01, L2: 1.04, AUC = %0.2f' % mlp_roc_auc)
    plt.plot(fpr2, tpr2, label='adam, learning: 1, L2: 2, AUC = %0.2f' % mlp_roc_auc2)
    plt.plot(fpr3, tpr3, label='logistic, learning: 0.05, L2: 20, AUC = %0.2f' % mlp_roc_auc3)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    #
    # # create the axis of thresholds (scores)
    # ax2 = plt.gca().twinx()
    # ax2.plot(fpr, threshold, markeredgecolor='r', linestyle='dashed', color='r')
    # ax2.set_ylabel('Threshold', color='r')
    # ax2.set_ylim([threshold[-1], threshold[0]])
    # ax2.set_xlim([fpr[0], fpr[-1]])
    plt.show()

    preds_threshold = np.where(classifier.predict_proba(X_test)[:, 1] > THRESHOLD, 1, 0)
    tn, fp, fn, tp = confusion_matrix(y_test, preds_threshold).ravel()
    print('tn: {} fp: {} fn: {} tp: {}'.format(tn, fp, fn, tp))
    print("Accuracy score: {}".format(accuracy_score(y_test, preds_threshold) * 100))
    data = {'FPR': fpr, 'TPR': tpr}
    table = pd.DataFrame(data, index=threshold)
    print(table)
    return fpr, tpr, threshold


def KNN():
    THRESHOLD = 0.227311
    classifier = KNeighborsClassifier(n_neighbors=5, algorithm='auto', weights= 'distance', metric='minkowski')
    classifier.fit(X_train, y_train)
    fpr, tpr, threshold = roc_curve(y_test, classifier.predict_proba(X_test)[:, 1])

    # Parameters 2
    clf2 = KNeighborsClassifier(n_neighbors=7, metric='minkowski', weights='uniform')
    clf2.fit(X_train, y_train)
    fpr2, tpr2, threshold2 = roc_curve(y_test, clf2.predict_proba(X_test)[:, 1])
    knn_roc_auc2 = auc(fpr2, tpr2)

    # parameter 3
    clf3 = KNeighborsClassifier(n_neighbors=13, metric='euclidean', weights='uniform')
    clf3.fit(X_train, y_train)
    fpr3, tpr3, threshold3 = roc_curve(y_test, clf3.predict_proba(X_test)[:, 1])
    knn_roc_auc3 = auc(fpr3, tpr3)
    KNN_roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, label='neighbors:5, weights: distance,\n distance: minkowski AUC = %0.2f' % KNN_roc_auc)
    plt.plot(fpr2, tpr2, label='neighbors:7, weights: uniform,\n distance: euclidean AUC = %0.2f' % knn_roc_auc2)
    plt.plot(fpr3, tpr3, label='neighbors:13, weights: uniform,\n distance: euclidean AUC = %0.2f' % knn_roc_auc3)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    #
    # # create the axis of thresholds (scores)
    # ax2 = plt.gca().twinx()
    # ax2.plot(fpr, threshold, markeredgecolor='r', linestyle='dashed', color='r')
    # ax2.set_ylabel('Threshold', color='r')
    # ax2.set_ylim([threshold[-1], threshold[0]])
    # ax2.set_xlim([fpr[0], fpr[-1]])
    plt.show()

    preds_threshold = np.where(classifier.predict_proba(X_test)[:, 1] > THRESHOLD, 1, 0)
    tn, fp, fn, tp = confusion_matrix(y_test, preds_threshold).ravel()
    print('tn: {} fp: {} fn: {} tp: {}'.format(tn, fp, fn, tp))
    print("Accuracy score: {}".format(accuracy_score(y_test, preds_threshold) * 100))
    data = {'FPR': fpr, 'TPR': tpr}
    table = pd.DataFrame(data, index=threshold)
    print(table)
    return fpr, tpr, threshold

def roc_auc():
    iter = 1000
    naivefpr, naivetpr, naivethreshold = naives()
    logsfpr, logstpr, logsthreshol = logRegres()
    svmfpr, svmtpr, svmthreshold = svm()
    mlpfpr, mlptpr, mlpthreshold = mlp()
    adafpr, adatpr, adathreshold = KNN()

    naive_roc_auc = auc(naivefpr, naivetpr)
    logs_roc_auc = auc(logsfpr, logstpr)
    svm_roc_auc = auc(svmfpr, svmtpr)
    mlp_roc_auc = auc(mlpfpr, mlptpr)
    knn_roc_auc = auc(adafpr, adatpr)

    # image drawing
    plt.figure()
    plt.title('Receiver Operating Characteristic')
    plt.plot(naivefpr, naivetpr, label='Naive AUC = %0.2f' % naive_roc_auc)
    plt.plot(logsfpr, logstpr, label='Logistic AUC = %0.2f' % logs_roc_auc)
    plt.plot(svmfpr, svmtpr, label='SVM AUC = %0.2f' % svm_roc_auc)
    plt.plot(mlpfpr, mlptpr, label='MLP AUC = %0.2f' % mlp_roc_auc)
    plt.plot(adafpr, adatpr, label='KNN AUC = %0.2f' % knn_roc_auc)


    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

def classifierSelection(argument):
    switcher = {
        1: naives,
        2: logRegres,
        3: svm,
        4: mlp,
        5: KNN,
        6: roc_auc

    }
    # Get the function from switcher dictionary
    func = switcher.get(argument, lambda: "Invalid Classifier")
    # Execute the function
    print(func())



# Create a menu system for the user to choose which classifier to run
def menu():
    print("############################   Menu   #############################################")

    print(' Displaying a list of classifier to use\n')

    clf = int(input("""
                      1: Naive Bayes  
                      2: Logistic Regression 
                      3: Primal Support Vector Machine  
                      4: Multi-Layer Perceptron 
                      5: K Nearest Neighbors   
                      6: roc_auc
                      7: Quit/Log Out


                      Please enter your choice: """))

    if clf == 1 or clf == 2 or clf == 3 or clf == 4 or clf == 5 or clf == 6:
        classifierSelection(clf)
        menu()
    elif clf == 7:
        print('Quiting program ')
        return None
    else:
        print("Invalid choice, choose again")
        menu()


menu()
