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
#;;;*---  PROGRAMM      : MLReports2025V01.ipynb                ---*;;;
#;;;*---  Parent        : MLReports2025.ipynb, 20.05.2025       ---*;;;
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
from AsniDef import *
from AsNiDefFa2 import *

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

timestart = datetime.datetime.now()
date_time = timestart.strftime("%d.%m.%Y %H:%M:%S")

warnings.filterwarnings('ignore')
sns.set()
stop=18
def fig2img(fig):
    import io
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return img

x = np.arange(-3,3)
plt.plot(x)
fig = plt.gcf()
img = fig2img(fig)
img.save(os.path.join(cwd, 'Image/TestPlot.png'))

df = pd.read_csv(os.path.join(cwd, "data\heard.csv")) 
df.rename({'Y': 'target'}, axis=1, inplace=True)
df = df.fillna(0)
print(df)
print(' ')

timeend = datetime.datetime.now()
date_time = timeend.strftime("%d.%m.%Y %H:%M:%S")
timedelta = round((timeend-timestart).total_seconds(), 2) 

r=(timeend-timestart) 
t=int(timedelta/60)
if timedelta-t*60 < 10:
    t2=":0" + str(int(timedelta-t*60))
else:
    t2=":" + str(int(timedelta-t*60))
txt="Общее время работы программы MLReports2025.ipynb составляет: 00:" + str(t) + t2 
print(txt)
