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
#;;;*---  PROGRAMM      : ASNIMED-EDA-Voll2025.ipynb            ---*;;;
#;;;*---  Parent        : AsNiVollPlus.ipynb, 09.06.2024        ---*;;;
#;;;*---                : AsNiVollEnd2025.ipynb, 15.05.2025     ---*;;;
#;;;*---  BESCHREIBUNG  : System                                ---*;;;
#;;;*---                :                                       ---*;;;
#;;;*---                :                                       ---*;;;
#;;;*---  VERSION   VOM : 18.05.2025                            ---*;;;
#;;;*--   KORREKTUR VOM : 19.05.2025                            ---*;;;
#;;;*--                 :                                       ---*;;;
#;;;*---  INPUT         :.INI                                   ---*;;;
#;;;*---  OUTPUT        : Image (PNG, etc.)                     ---*;;;
#;;;*--------------------------------------------------------------*;;;
#;;;************************ Änderung ******************************;;;
#;;;****************************************************************;;;
#;;;  Wann              :               Was                        *;;;
#;;;*--------------------------------------------------------------*;;;
#;;;* 18.05.2025        : Neu Version                              *;;;
#;;;* 19.05.2025        : Bereinigung                              *;;;
#;;;****************************************************************;;;

import os, sys, inspect, time, datetime
import subprocess
import numpy as np 
import pandas as pd 
import json
from time import time, strftime, localtime
from datetime import timedelta
import shutil
from IPython.core.display import HTML 
from collections import Counter 
from colorama import Fore, Style 
import matplotlib 
from matplotlib import * 
from matplotlib import pyplot as plt 
from matplotlib.colors import ListedColormap 
import plotly as pl 
import plotly as pplt 
from plotly import tools 
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot 
from plotly.subplots import make_subplots 
import plotly.express as px 
import plotly.figure_factory as ff 
import plotly.graph_objects as go 
import plotly.io as pio 
import plotly.offline 
import plotly.offline as po 
import random 
import scipy.stats as stats 
import seaborn as sns 
import warnings 
from collections import Counter 
import colorama 
import cufflinks as cf 
import patchworklib as pw 

pio.renderers
warnings.filterwarnings("ignore") 
pd.set_option("display.max_rows",None) 

print("Programm ASNIMED-EDA-Voll2025.ipynb Start")
now=datetime.datetime.now()
timestart = now.replace(microsecond=0)
print("Programm Start: ", timestart)
today = datetime.date.today()
year = today.year
cwd = os.getcwd()

path=cwd 
pathIm=os.path.join(cwd, "image")
print("path: ", path)
print("pathIm: ", pathIm)
pd.set_option("display.max_rows",None)

try:
    raw_df = pd.read_csv(path+'/data/heart.csv')
except:
    raw_df =pd.read_csv(path+'/data/heart.csv')

df=raw_df
des0=raw_df[raw_df['HeartDisease']==0].describe().T.applymap('{:,.2f}'.format)
des1=raw_df[raw_df['HeartDisease']==1].describe().T.applymap('{:,.2f}'.format)

cat = ['Sex', 'ChestPainType','FastingBS','RestingECG','ExerciseAngina',  'ST_Slope','HeartDisease']
num = ['Age','RestingBP','Cholesterol','MaxHR','Oldpeak']

categorical_columns = list(raw_df.loc[:,['Sex', 'ChestPainType', 'FastingBS', 'RestingECG', 'ExerciseAngina', 'ST_Slope']])
numerical_columns = list(raw_df.loc[:,['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak', 'HeartDisease']])
numerical=numerical_columns

index = 0
plt.figure(figsize=(20,20))
for feature in numerical:
    if feature != "HeartDisease":
        index += 1
        plt.subplot(2, 3, index)
        sns.boxplot(x='HeartDisease', y=feature, data=df)
        
plt.savefig(pathIm + '/EDA1.png')  

#################  Programm-Ende #########################
timeend = datetime.datetime.now()
date_time = timeend.strftime("%d.%m.%Y %H:%M:%S")
timedelta = round((timeend-timestart).total_seconds(), 2) 

r=(timeend-timestart) 
t=int(timedelta/60)
if timedelta-t*60 < 10:
    t2=":0" + str(int(timedelta-t*60))
else:
    t2=":" + str(int(timedelta-t*60))
txt="Programm Dauer: 00:" + str(t) + t2 
print(txt)
print("Programm ASNIMED-EDA-Voll2025.ipynb Ende")
