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

import os, sys, inspect, time, datetime
from time import time, strftime, localtime
import datetime as dt
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io 
from PIL import Image 

cwd=os.getcwd()
print("cwd: ",cwd)

timestart = datetime.datetime.now()
date_time = timestart.strftime("%d.%m.%Y %H:%M:%S")

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
