#;;*****************************************************************;;;
#;;*****************************************************************;;;
#;;;****************************************************************;;;
#;;;***  FIRMA          : PARADOX                                ***;;;
#;;;***  Autor          : Alexander Wagner                       ***;;;
#;;;***  STUDIEN-NAME   : ASNI-MED                               ***;;;
#;;;***  STUDIEN-NUMMER :                                        ***;;;
#;;;***  SPONSOR        :                                        ***;;;
#;;;***  ARBEITSBEGIN   : 01.11.2023                             ***;;;
#;;;****************************************************************;;;
#;;;*--------------------------------------------------------------*;;;
#;;;*---  PROGRAMM      : MLReports2025.ipynb                   ---*;;;
#;;;*---  Parent        : MLReports2025_Test.ipynb, 19.05.2025  ---*;;;
#;;;*---  BESCHREIBUNG  : System                                ---*;;;
#;;;*---                :                                       ---*;;;
#;;;*---                :                                       ---*;;;
#;;;*---  VERSION   VOM : 20.05.2025                            ---*;;;
#;;;*--   KORREKTUR VOM :                                       ---*;;;
#;;;*--                 :                                       ---*;;;
#;;;*---  INPUT         :.INI                                   ---*;;;
#;;;*---  OUTPUT        : CSV, Image (PNG, etc.)                ---*;;;
#;;;*--------------------------------------------------------------*;;;
#;;;************************ Änderung ******************************;;;
#;;;****************************************************************;;;
#;;;  Wann              :               Was                        *;;;
#;;;*--------------------------------------------------------------*;;;
#;;;* 20.05.2025        : Neu Version, Bereinigung                 *;;;
#;;;****************************************************************;;;

import os, sys, inspect, time, datetime
from time import time, strftime, localtime
import datetime as dt
import time
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import metrics
import matplotlib.pyplot as plt
import math
import random
import seaborn as sns
import warnings

# preprocessing
import sklearn
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, log_loss 
from sklearn.metrics import confusion_matrix, classification_report
from sklearn import metrics
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, learning_curve, ShuffleSplit
from sklearn.model_selection import cross_val_predict as cvp
from sklearn.calibration import CalibratedClassifierCV

# models
from sklearn.linear_model import LogisticRegression, LogisticRegression, Perceptron, RidgeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC, LinearSVC, SVR, NuSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
from xgboost import XGBClassifier
import lightgbm as lgb
from lightgbm import LGBMClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

# ensemble
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier 
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, VotingClassifier

from sklearn.metrics import roc_auc_score
from scikitplot.metrics import plot_roc_curve
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.metrics import RocCurveDisplay, roc_curve
from sklearn.metrics import PrecisionRecallDisplay, precision_recall_curve
from sklearn import metrics
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, classification_report,f1_score,confusion_matrix,precision_score,recall_score,balanced_accuracy_score
import io 
from PIL import Image 

cwd=os.getcwd()
print("cwd: ",cwd)

def RunEda():
    import time
    exec(open(r"C:\AW75\AsniMed\ImportBib.py").read(), globals())
    time.sleep(2.0)
    
    exec(open(r"C:\AW75\AsniMed\ConfigINI2025.py").read(), globals())
    import time
    time.sleep(2.0)
              
    exec(open(r"C:\AW75\AsniMed\AsniDef.py").read(), globals())
    import time
    time.sleep(2.0)
    
    exec(open(r"C:\AW75\AsniMed\AsNiDefFa2.py").read(), globals())
    import time
    time.sleep(2.0)
    
RunEda()

timestart = datetime.datetime.now()
date_time = timestart.strftime("%d.%m.%Y %H:%M:%S")

warnings.filterwarnings('ignore')
sns.set()
stop=18

def fig2imgNo(fig): 
	buf = io.BytesIO() 
	fig.savefig(buf) 
	buf.seek(0) 
	img = Image.open(buf) 
	return img 

import io 
from PIL import Image 
import matplotlib.pyplot as plt

def fig2img(fig):
    import io
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return img

#x = np.arange(-3,3)
#plt.plot(x)
#fig = plt.gcf()
#img = fig2img(fig)
#img.save(os.path.join(cwd, 'ImageNow/TestPlot.png'))

df = pd.read_csv(os.path.join(cwd, "data\heard.csv")) 
df.rename({'Y': 'target'}, axis=1, inplace=True)
df = df.fillna(0)

train, test = train_test_split(df, test_size = 0.4)
X_train = train[train.columns.difference(['target'])]
y_train = train['target']

X_test = test[test.columns.difference(['target'])]
y_test = test['target']

cv_n_split = 2
random_state = 0
test_train_split_part = 0.2

train0, test0 = X_train, X_test
target0 = y_train
train, test, target, target_test = train_test_split(train0, target0, test_size=test_train_split_part, random_state=random_state)

#st.write("3. Modellenauswahl")
cv_train = ShuffleSplit(n_splits=cv_n_split, test_size=test_train_split_part, random_state=random_state)

metrics_all = {1 : 'r2_score', 2: 'acc', 3 : 'rmse', 4 : 're'}
metrics_now = [1, 2, 3, 4] 

# list of accuracy of all model - amount of metrics_now * 2 (train & test datasets)
num_models = 18
acc_train = []
acc_test = []
ShuffleSplit(n_splits=2, random_state=0, test_size=0.2, train_size=None)

# GradientBoostingClassifier
mGBC=GradientBoostingClassifier()
GBC=GradientBoostingClassifier()
    
# KNeighborsClassifier
mKNC=KNeighborsClassifier()
KNC=KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)

# SVC
mSVC = SVC()
SVC = SVC(kernel="rbf", C=0.025, probability=True)

# DecisionTreeClassifier
mDTC=DecisionTreeClassifier()
DTC=DecisionTreeClassifier(criterion = 'entropy', random_state = 51)

# RandomForestClassifier
mRF=RandomForestClassifier()
RF=RandomForestClassifier(n_estimators = 20, criterion = 'entropy', random_state = 51)

# XGBClassifier
mXGBC=XGBClassifier()
XGBC=XGBClassifier()

# AdaBoostClassifier
mAdaBoost=AdaBoostClassifier()
AdaBoost=AdaBoostClassifier(DecisionTreeClassifier(criterion = 'entropy', random_state = 200),
                              n_estimators=2000,
                              learning_rate=0.1,
                              algorithm='SAMME.R',
                              random_state=1,)    


# LogisticRegression LR
LogReg=LogisticRegression()
LR=LogisticRegression(random_state = 51)

############### Neue Modelle ###############
# Linear Regression
param_grid = {'C': np.linspace(.1, 1.5, 15)}
linreg = LogisticRegression()
linreg_CV = GridSearchCV(linreg, param_grid=param_grid, cv=cv_train, verbose=False)

# Linear SVR
linear_svc = LinearSVC()
linear_svc_CV = GridSearchCV(linear_svc, param_grid={}, cv=cv_train, verbose=False)

mlp = MLPClassifier()
param_grid = {'hidden_layer_sizes': [i for i in range(2,5)],
              'solver': ['sgd'],
              'learning_rate': ['adaptive'],
              'max_iter': [1000]
              }
mlp_GS = GridSearchCV(mlp, param_grid=param_grid, cv=cv_train, verbose=False)

# Decision Tree Classifier
decision_tree = DecisionTreeClassifier()
decision_tree_CV = GridSearchCV(decision_tree, param_grid={}, cv=cv_train, verbose=False)

# Stochastic Gradient Descent
sgd = SGDClassifier()
sgd_CV = GridSearchCV(sgd, param_grid={}, cv=cv_train, verbose=False)

# Ridge Classifier
ridge = RidgeClassifier()
ridge_CV = GridSearchCV(estimator=ridge, param_grid={'alpha': np.linspace(.1, 1.5, 15)}, cv=cv_train, verbose=False)

# Bagging Classifier
bagging = BaggingClassifier()
bagging_CV = GridSearchCV(estimator=bagging, param_grid={}, cv=cv_train, verbose=False)

# AdaBoost Classifier
Ada_Boost = AdaBoostClassifier()
Ada_Boost_CV = GridSearchCV(estimator=Ada_Boost, param_grid={'learning_rate' : [.01, .1, .5, 1]}, cv=cv_train, verbose=False)

# LogisticRegression
logreg = LogisticRegression()
logreg_CV = GridSearchCV(estimator=logreg, param_grid={'C' : [.1, .3, .5, .7, 1]}, cv=cv_train, verbose=False)

# Gaussian Naive Bayes
gaussian = GaussianNB()
gaussian_CV = GridSearchCV(estimator=gaussian, param_grid={}, cv=cv_train, verbose=False)

# Perceptron
perceptron = Perceptron()
perceptron_CV = GridSearchCV(estimator=perceptron, param_grid={}, cv=cv_train, verbose=False)

classifiers = [
    linreg_CV,
    logreg_CV,
    perceptron_CV,
    linear_svc_CV,
    mlp_GS,
    decision_tree_CV,
    sgd_CV,
    ridge_CV,
    bagging_CV,
    Ada_Boost_CV,
    GBC,
    KNC,
    DTC,
    RF,
    XGBC,
    AdaBoost,
    gaussian_CV,
    SVC
  ] 

models = pd.DataFrame({'Model': [
    'Linear Regression', 
    'Logistic Regression',
    'Perceptron',
    'Linear SVC', 
    'MLPClassifier', 
    'Decision Tree Classifier 1', 
    'Stochastic Gradient Decent', 
    'RidgeClassifier', 
    'BaggingClassifier', 
    'AdaBoostClassifier 1', 
    'GradientBoostingClassifie',
    'KNeighborsClassifier',
    'DecisionTreeClassifier 2',
    'RandomForestClassifier',
    'XGBClassifier',
    'AdaBoostClassifier 2',
    'Naive Bayes',    
    'SVC' ]})

i=-1
result_table = pd.DataFrame(columns=['classifiers', 
                                     'fpr',
                                     'tpr',
                                     'accurate',
                                     'auc'])

for classifier in classifiers:
    i=i+1
    name_model = models.iloc[i]['Model']

    if i < stop:
        pipe = Pipeline(steps=[('classifier', classifier)])
        model = pipe.fit(X_train, y_train) 
        acc_metrics_calc(i,model,train,test,target,target_test)

        title = "Model: " + name_model
        figure, axes = plt.subplots(2, 1, figsize=(20, 10))
    
        if axes is None:
            _, axes = plt.subplots(1, 2, figsize=(20, 5))

        axes[0].set_title(title)
        axes[0].set_xlabel("Training examples")
        axes[0].set_ylabel("Score")

        cv_train = ShuffleSplit(n_splits=cv_n_split, test_size=test_train_split_part, random_state=random_state)
        train_sizes, train_scores, test_scores, fit_times, _ = \
            learning_curve(estimator=model, X=train, y=target, cv=cv_train,
                           train_sizes=np.linspace(.1, 1.0, 5),
                           return_times=True)
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        fit_times_mean = np.mean(fit_times, axis=1)
        fit_times_std = np.std(fit_times, axis=1)

        axes[0].grid()
        axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                             train_scores_mean + train_scores_std, alpha=0.1,
                             color="r")
        axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std,
                             test_scores_mean + test_scores_std, alpha=0.1,
                             color="g")
        axes[0].plot(train_sizes, train_scores_mean, 'o-', color="r",
                     label="Training score")
        axes[0].plot(train_sizes, test_scores_mean, 'o-', color="g",
                     label="Cross-validation score")
        axes[0].legend(loc="best")

        # Plot n_samples vs fit_times
        axes[1].grid()
        axes[1].plot(train_sizes, fit_times_mean, 'o-')
        axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std,
                             fit_times_mean + fit_times_std, alpha=0.1)
        axes[1].set_xlabel("Training examples")
        axes[1].set_ylabel("fit_times")
        axes[1].set_title("Scalability of the model")

        figure.savefig(os.path.join(cwd,'ImageNow/plot_learning_curve' + str(i) + '.png'))
        ypred   = model.predict(X_test)
         
        if (i==2): 
            clf = CalibratedClassifierCV(perceptron_CV) 
            clf.fit(X_train, y_train)
            y_pred = clf.predict_proba(X_test)[::,1]
        elif (i==3):
            #svm = LinearSVC()
            clf = CalibratedClassifierCV(linear_svc_CV) #svm) 
            clf.fit(X_train, y_train)
            y_pred = clf.predict_proba(X_test)[::,1]
        elif (i==6):
            clf = CalibratedClassifierCV(sgd_CV)
            clf.fit(X_train, y_train)
            y_pred = clf.predict_proba(X_test)[::,1]            
        elif (i==7):
            clf = CalibratedClassifierCV(ridge_CV)
            clf.fit(X_train, y_train)
            y_pred = clf.predict_proba(X_test)[::,1]            
               
        else:
            y_pred = model.predict_proba(X_test)[::,1]
  
            
        fpr, tpr, _ = roc_curve(y_test, y_pred) 
        auc         = roc_auc_score(y_test, y_pred).round(3) 
        accuracy    = accuracy_score(ypred, y_test).round(2)
        F1_Score = f1_score(y_test, ypred).round(2)
        
        Precision_Score = precision_score(y_test, ypred).round(2)
        Recall_Score = recall_score(y_test, ypred).round(2)
        Balanced_Accuracy_Score = balanced_accuracy_score(y_test, ypred).round(2)
        target_names = ['class 0', 'class 1']
        report = classification_report(y_test, ypred, target_names=target_names, output_dict=True, digits=4)
        
        print(report)
        CR = pd.DataFrame(report).transpose()
        
        print(CR)
        CR.to_csv(os.path.join(cwd,'data/CLSB_' + str(i) +  '.csv'))
                   
        Accurate_train=pipe.score(X_train, y_train).round(2)   
        Accurate_test =pipe.score(X_test,  y_test).round(2)  
        cm = confusion_matrix(y_test, ypred)
        fig=plt.figure(figsize=(8,5))
        plt.title('Heatmap of Confusion Matrix', fontsize = 15)
        sns.heatmap(cm, annot = True)
        fig.savefig(os.path.join(cwd, 'ImageNow/Heatmap' + str(i) + '.png'))  
        
        Roc_Auc_Score = roc_auc_score(y_test, y_pred).round(2)
             
        fig2=plt.figure(figsize=(8,5))            
        plt.style.use('seaborn-v0_8-darkgrid')
        plt.plot(fpr, tpr, label='%s ROC (area = %0.2f)' % (name_model, auc))
        plt.plot([0, 1], [0, 1],'r--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('1-Specificity(False Positive Rate)')
        plt.ylabel('Sensitivity(True Positive Rate)')

        plt.legend(loc="lower right")
        #fig = plt.gcf()
        #img = fig2img(fig)
        fig2.savefig(os.path.join(cwd, 'ImageNow/PlotROC' + str(i) + '.png')) 
        #img.savefig(os.path.join(cwd, 'ImageNow/PlotROC' + str(i) + '.png')) 

i=-1
plt.style.use('seaborn-v0_8-darkgrid')
figC = plt.figure(figsize=(16,10))


for classifier in classifiers:
    i=i+1
    name_model = models.iloc[i]['Model']
  
    if i < stop:
        pipe = Pipeline(steps=[('classifier', classifier)])
        model = pipe.fit(X_train, y_train) 
        ypred   = model.predict(X_test)
        if (i==2): 
            clf = CalibratedClassifierCV(perceptron_CV) 
            clf.fit(X_train, y_train)
            y_pred = clf.predict_proba(X_test)[::,1]
        elif (i==3):
            #svm = LinearSVC()
            clf = CalibratedClassifierCV(linear_svc_CV) #svm) 
            clf.fit(X_train, y_train)
            y_pred = clf.predict_proba(X_test)[::,1]
        elif (i==6):
            clf = CalibratedClassifierCV(sgd_CV)
            clf.fit(X_train, y_train)
            y_pred = clf.predict_proba(X_test)[::,1]            
        elif (i==7):
            clf = CalibratedClassifierCV(ridge_CV)
            clf.fit(X_train, y_train)
            y_pred = clf.predict_proba(X_test)[::,1]            
               
        else:
            y_pred = model.predict_proba(X_test)[::,1]
             
        fpr, tpr, _ = roc_curve(y_test, y_pred) 
        auc         = roc_auc_score(y_test, y_pred).round(3) 
        accuracy    = accuracy_score(ypred, y_test).round(2)
        F1_Score = f1_score(y_test, ypred).round(2)
        
        Precision_Score = precision_score(y_test, ypred).round(2)
        Recall_Score = recall_score(y_test, ypred).round(2)
        Balanced_Accuracy_Score = balanced_accuracy_score(y_test, ypred).round(2)
        Classification_Report = classification_report(y_test, ypred)
        Accurate_train=pipe.score(X_train, y_train).round(2)   
        Accurate_test =pipe.score(X_test,  y_test).round(2)        

        #fig3=plt.figure(figsize=(8,5))  
        plt.plot(fpr, tpr, label='%s ROC (area = %0.2f)' % (name_model, auc))
        plt.plot([0, 1], [0, 1],'r--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('1-Specificity(False Positive Rate)')
        plt.ylabel('Sensitivity(True Positive Rate)')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")

figC = plt.gcf()
img = fig2img(figC) 
img.save(os.path.join(cwd, 'ImageNow/PlotROC.png'))

############## 6. Speicher der Ergebnisse #################
metricsnow = pd.DataFrame(metrics_now)
#metricsnow.to_csv(os.path.join(cwd, 'data\Nostalgi2023_metrics_now.csv'), index=False) 
 
accall= pd.DataFrame(acc_all)
#accall.to_csv(os.path.join(cwd, 'data\Nostalgi2023_acc_all.csv'), index=False)   

models = pd.DataFrame({'Model': [
    'Linear Regression', 
    'Logistic Regression',
    'Perceptron',
    'Linear SVC', 
    'MLPClassifier', 
    'Decision Tree Classifier 1', 
    'Stochastic Gradient Decent', 
    'RidgeClassifier', 
    'BaggingClassifier', 
    'AdaBoostClassifier 1', 
    'GradientBoostingClassifie',
    'KNeighborsClassifier',
    'DecisionTreeClassifier 2',
    'RandomForestClassifier',
    'XGBClassifier',
    'AdaBoostClassifier 2',
    'Naive Bayes',    
    'SVC' ]})

metrics_all = {1 : 'r2_score', 2: 'acc', 3 : 'rmse', 4 : 're'}
metrics_now = [1, 2, 3, 4]

models=models[0:stop] 
for x in metrics_now:
    xs = metrics_all[x]
    models[xs + '_Train'] = acc_all[(x-1)*2]
    models[xs + '_Test']  = acc_all[(x-1)*2+1]
    if xs == "acc":
        models[xs + '_Diff'] = models[xs + '_Train'] - models[xs + '_Test']

ms = metrics_all[metrics_now[1]] # the accuracy
models.sort_values(by=[(ms + '_Test'), (ms + '_Train')], ascending=False)
pd.options.display.float_format = '{:,.2f}'.format

# Plots
plt.style.use('seaborn-v0_8-darkgrid')

i=0
for x in metrics_now: 
    i=i+1
    xs = metrics_all[x]
    xs_train = metrics_all[x] + '_Train'
    xs_test = metrics_all[x] + '_Test'
    figD = plt.figure(figsize=[25,6])
    xx = models['Model']
    plt.tick_params(labelsize=14)
    plt.plot(xx, models[xs_train], label = xs_train)
    plt.plot(xx, models[xs_test], label = xs_test)
    plt.legend()
    plt.title(str(xs) + ' criterion for ' + str(num_models) + ' popular models for train and test datasets')
    plt.xlabel('Models')
    plt.ylabel(xs + ', %')
    plt.xticks(xx, rotation='vertical')
    
    img = fig2img(figD) 
    #img.save("assets/PlotE" + str(i) + ".png")
    img.save(os.path.join(cwd, 'ImageNow/PlotE' + str(i) + '.png'))

metrics_main = 2
xs = metrics_all[metrics_main]
xs_train = metrics_all[metrics_main] + '_Train'
xs_test = metrics_all[metrics_main] + '_Test'
direct_sort = False if (metrics_main >= 2) else True

models_sort = models.sort_values(by=[xs_test, xs_train], ascending=direct_sort)
models_best = models_sort[(models_sort.acc_Diff < 5) & (models_sort.acc_Train > 90)]
models_best[['Model', ms + '_Train', ms + '_Test', 'acc_Diff']].sort_values(by=['acc_Test'], ascending=False)
models_best.to_csv(os.path.join(cwd,'data\RT1.csv'), index=False) 

metrics_all = {1 : 'r2_score', 2: 'acc', 3 : 'rmse', 4 : 're'}
metrics_now = [1, 2, 3, 4] 

# list of accuracy of all model - amount of metrics_now * 2 (train & test datasets)
num_models = 18
acc_train = []
acc_test = []
acc_all = np.empty((len(metrics_now)*2, 0)).tolist()

models = pd.DataFrame({'Model': [
    'Linear Regression', 
    'Logistic Regression',
    'Perceptron',
    'Linear SVC', 
    'MLPClassifier', 
    'Decision Tree Classifier 1', 
    'Stochastic Gradient Decent', 
    'RidgeClassifier', 
    'BaggingClassifier', 
    'AdaBoostClassifier 1', 
    'GradientBoostingClassifie',
    'KNeighborsClassifier',
    'DecisionTreeClassifier 2',
    'RandomForestClassifier',
    'XGBClassifier',
    'AdaBoostClassifier 2',
    'SVC'
 ]})

models_pred = pd.DataFrame(models.Model, columns = ['Model'])
N_best_models = len(models_pred.Model)

from sklearn.svm import SVC
i=0
for i in range(N_best_models):
    if i < N_best_models+1:
        name_model = models_pred.iloc[i]['Model']
        ii=i+1
        model = model_fit(name_model,train0,target0)
        acc_metrics_calc_pred(i,model,name_model, train0, test0, target0)

models_pred=models_pred[0:N_best_models]

i=0
for x in metrics_now:
    xs = metrics_all[x]
    #Auswahl N_best_models(17 im Moment) Spalten aus der acc_all_pred!  
    acc_all_pred2=acc_all_pred[x-1][0:N_best_models] 
    models_pred[xs + '_train'] = acc_all_pred2 

sort_pred=models_pred[['Model', 'acc_train']].sort_values(by=['acc_train'], ascending=False)
sort_pred.to_csv(os.path.join(cwd,'data\RT2.csv'), index=False) 

models_pred=models_pred[0:N_best_models]
i=0
for x in metrics_now:
    xs = metrics_all[x]
    #Auswahl N_best_models(17 im Moment) Spalten aus der acc_all_pred!  
    acc_all_pred2=acc_all_pred[x-1][0:N_best_models] 
    models_pred[xs + '_train'] = acc_all_pred2 

sort_pred=models_pred[['Model', 'r2_score_train','acc_train','rmse_train', 're_train']].sort_values(by=['acc_train'], ascending=False)
sort_pred.to_csv(os.path.join(cwd,'data\RT3.csv'), index=False) 

#import pandas as pd
#RT4=pd.read_csv(os.path.join(cwd, "data\allpred2023.csv"))

print('models_best RT1', models_best)
print('------------------------------------------------------------------------- ')
print(' ')
print('sort_pred RT2', sort_pred)
print('------------------------------------------------------------------------- ')
print(' ')
print('models_pred RT3', models_pred)
print('------------------------------------------------------------------------- ')
print(' ')
#print('models_pred RT4', RT4)

#print("Programm Start: ", date_time)
timeend = datetime.datetime.now()
date_time = timeend.strftime("%d.%m.%Y %H:%M:%S")
#print("Programm Finish:",date_time)
timedelta = round((timeend-timestart).total_seconds(), 2) 

r=(timeend-timestart) 
t=int(timedelta/60)
if timedelta-t*60 < 10:
    t2=":0" + str(int(timedelta-t*60))
else:
    t2=":" + str(int(timedelta-t*60))
txt="Общее время работы программы MLReports2025.ipynb составляет: 00:" + str(t) + t2 
print(txt)
