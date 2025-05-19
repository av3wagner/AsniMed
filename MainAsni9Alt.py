#https://docs.streamlit.io/deploy/streamlit-community-cloud/share-your-app/embed-your-app
print("#;;*****************************************************************;;;")
print("#;;*****************************************************************;;;")
print("#;;;****************************************************************;;;")
print("#;;;***  FIRMA          : PARADOX                                ***;;;")
print("#;;;***  Autor          : Alexander Wagner                       ***;;;")
print("#;;;***  STUDIEN-NAME   : AsNiFen                                ***;;;")
print("#;;;***  STUDIEN-NUMMER :                                        ***;;;")
print("#;;;***  SPONSOR        :                                        ***;;;")
print("#;;;***  ARBEITSBEGIN   : 01.11.2023                             ***;;;")
print("#;;;****************************************************************;;;")
print("#;;;*--------------------------------------------------------------*;;;")
print("#;;;*---  PROGRAMM      : Asni2025-V09.ipynb                    ---*;;;")
print("#;;;*---  Parent        : Asni2025-V08.ipynb                    ---*;;;")
print("#;;;*---  BESCHREIBUNG  : System                                ---*;;;")
print("#;;;*---                :                                       ---*;;;")
print("#;;;*---                :                                       ---*;;;")
print("#;;;*---  VERSION   VOM : 31.05.2025                            ---*;;;")
print("#;;;*--   KORREKTUR VOM : 04.04.2025                            ---*;;;")
print("#;;;*--                 :                                       ---*;;;")
print("#;;;*---  INPUT         :.INI, .Json, .CSV                      ---*;;;")
print("#;;;*---  OUTPUT        :                                       ---*;;;")
print("#;;;*--------------------------------------------------------------*;;;")
print("#;;;************************ Änderung ******************************;;;")
print("#;;;****************************************************************;;;")
print("#;;;  Wann              :               Was                        *;;;")
print("#;;;*--------------------------------------------------------------*;;;")
print("#;;;* 23.03.2025        : Старт Модулей                            *;;;")
print("#;;;* 28.03.2025        : PDF Seite                                *;;;")
print("#;;;* 31.03.2025        : Чистка                                   *;;;")
print("#;;;* 04.04.2025        : Neue PDF-File                            *;;;")
print("#;;;* 04.05.2025        : Revision                                 *;;;")
print("#;;;* 09.05.2025        : Revision, Korrektur                      *;;;")
print("#;;;****************************************************************;;;")

def close_app(app_name):
    running_apps=psutil.process_iter(['pid','name']) #returns names of running processes
    found=False
    for app in running_apps:
        sys_app=app.info.get('name').split('.')[0].lower()

        if sys_app in app_name.split() or app_name in sys_app:
            pid=app.info.get('pid') #returns PID of the given app if found running
            
            try: #deleting the app if asked app is running.(It raises error for some windows apps)
                app_pid = psutil.Process(pid)
                app_pid.terminate()
                found=True
            except: pass
            
        else: pass
    if not found:
        print(app_name+" not found running")
    else:
        print(app_name+'('+sys_app+')'+' closed')

def kill_processes_by_port(port):
    killed_any = False

    for proc in psutil.process_iter(['pid', 'name', 'connections']):
        for conn in proc.info['connections']:
            if conn.laddr.port == port:
                try:
                    print(f"Found process with PID {proc.pid} and name {proc.info['name']}")

                    if proc.info['name'].startswith("docker"):
                        print("Found Docker. You might need to stop the container manually")

                    kill_process_and_children(proc)
                    killed_any = True

                except (PermissionError, psutil.AccessDenied) as e:
                    print(f"Unable to kill process {proc.pid}. The process might be running as another user or root. Try again with sudo")
                    print(str(e))

                except Exception as e:
                    print(f"Error killing process {proc.pid}: {str(e)}")

    return killed_any

def kill_process_and_children(proc):
    children = proc.children(recursive=True)
    for child in children:
        kill_process(child)
    kill_process(proc)

def kill_process(proc):
    print(f"Killing process with PID {proc.pid}")
    proc.kill()

def Rmain(path_to_main):
    executable = sys.executable
    proc = Popen(
        [
            executable,
            "-m",
            "streamlit",
            "run",
            path_to_main,
            "--server.headless=true",
            "--global.developmentMode=false",
        ],
        stdin=PIPE,
        stdout=PIPE,
        stderr=STDOUT,
        text=True,
    )
    proc.stdin.close()
  
    time.sleep(3)
    webbrowser.open("http://localhost:8501")
    while True:
        s = proc.stdout.read()
        if not s:
            break
        print(s, end="")
    proc.wait()

def execute_python_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            python_code = file.read()
            exec(python_code, globals())
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' does not exist.")
        
import dash
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import sqlite3
from dash import dash_table
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
from dash import Dash, dcc, html, Input, Output, State, callback
import plotly.express as px
import plotly.graph_objects as go
import chart_studio.plotly as py 
from jupyter_dash import JupyterDash
import flask
import json
import requests
from urllib.request import urlopen
from prophet import Prophet
from pandas_datareader import data, wb
import base64
import os, sys, inspect, time, datetime
import subprocess
import json
from time import time, strftime, localtime
from datetime import timedelta
import shutil
from subprocess import Popen, PIPE, STDOUT
import sys
import webbrowser
import pandas as pd
from configparser import ConfigParser
import streamlit as st
import matplotlib.pyplot as plt
from IPython.display import IFrame
from dash import Dash, dcc, html, callback, Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px
import dash_ag_grid as dag
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time
import psutil
import dash_pdf
from dash import Dash, html, dcc, Input, Output, State

path=os.getcwd()
os.chdir(path)
print("MainAsni9.py: ", path)

header_height, footer_height = "7rem", "10rem"
sidebar_width, adbar_width = "12rem", "12rem"

HEADER_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "right": 0,
    "height": header_height,
    "padding": "2rem 1rem",
    "background-color": "white",
}

SIDEBAR_STYLE = {
    "position": "fixed",
    "top": header_height,
    "left": 0,
    "bottom": footer_height,
    "width": sidebar_width,
    "padding": "2rem 1rem",
    "background-color": "lightgreen",
}

SIDEBAR_STYLE2 = {
    "position": "fixed",
    "top": header_height,
    "left": 0,
    "bottom": 0,
    "width": "15rem",
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
}

ADBAR_STYLE = {
    "position": "fixed",
    "top": header_height,
    "right": 0,
    "bottom": footer_height,
    "width": adbar_width,
    "padding": "1rem 1rem",
    "background-color": "lightblue",
}

FOOTER_STYLE = {
    "position": "fixed",
    "bottom": 0,
    "left": 0,
    "right": 0,
    "height": footer_height,
    "padding": "1rem 1rem",
    "background-color": "gray",
}

CONTENT_STYLE2 = {
    "margin-top": header_height,
    "margin-left": sidebar_width,
    "margin-right": adbar_width,
    "margin-bottom": footer_height,
    "padding": "1rem 1rem",
}

CONTENT_STYLE = {
    "margin-left": "8rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
}

DocuList= ['Кардиология', 'Диабетология','Геникология'] 

header = html.Div([
    html.H4("Республика Казахстан АСНИ-МЕД")], style=HEADER_STYLE
)

PLOTLY_LOGO = "https://images.plot.ly/logo/new-branding/plotly-logomark.png"
def b64_image(image_filename):
    with open(image_filename, 'rb') as f:
        image = f.read()
    return 'data:image/png;base64,' + base64.b64encode(image).decode('utf-8')

File1="assets/ArtikelList.md"
File2="assets/README05.md"
File3="assets/+Resume04.md"
MdFile="assets/AWresumeF.md"
MdAW="assets/AWresume2025.md"
Front="assets/FrontSeite.md" 
Asni="assets/ASNIKonzept.md"
Literatur="assets/ArtikelList.md"

def demo_explanation(File):
    with open(File, "r", encoding="utf-8") as file:    
        demo_md = file.read()
    return html.Div(
        html.Div([dcc.Markdown(demo_md, className="markdown")]),
        style={"margin": "20px"},
    )

app = JupyterDash(external_stylesheets=[dbc.themes.SLATE])
server = app.server
sidebar = html.Div(
    [
        html.Div(
            [
                html.Img(src=PLOTLY_LOGO, style={"width": "3rem"}),
                html.H2("Sidebar"),
            ],
            className="sidebar-header",
        ),
        html.Hr(),        
     dbc.Nav(
            [
                dbc.NavLink("Главная страница", href="/", active="exact"),
                dbc.NavLink("Ввведение в ASNI-MED",   href="/page-1", active="exact"),
                dbc.NavLink("Разработчики системы", href="/page-5", active="exact"),
                dbc.NavLink("Литература", href="/page-6", active="exact"),
                dbc.NavLink("Просмотр PDF-отчетов", href="/page-7", active="exact"),
                dbc.NavLink("Старт программ избранных проектов", href="/page-8", active="exact"),
                #dbc.NavLink("Окончание работы", href="/page-12", active="exact"),
            ],
            vertical=True,
            pills=True,
        ),
    ],
    style=SIDEBAR_STYLE,
)

dash._dash_renderer._set_react_version("18.2.0")
content = html.Div(id="page-content", style=CONTENT_STYLE)
app.title = "RK Asni-Med"
app.layout = html.Div([dcc.Location(id="url"), sidebar, content])

@app.callback(Output("page-content", "children"), [Input("url", "pathname")])
def render_page_content(pathname):
    if pathname == "/":    
       return html.Div([  
                  html.Div(
                    html.Iframe(
                      sandbox='',
                      srcDoc='''
                      <!DOCTYPE html>
                      <html lang="en"> 
                       <html>
                        <head>
                            <style>
                                .myDiv {border: 5 outset red; background-color: black; text-align: center;}
                            </style>
                        </head>
                        
                        <body>
                        <div class="myDiv">
                          <iframe src="/assets/Front7.jpg" height="873" width="1350" marginLeft=270 scrolling="yes"></iframe>
                        </div>
                        </body>
                       </html> 
                    ''',
                     style={'width':'99.9%',"height": '900px','display':'inline-block',
                            'overflow-y':'auto', 'color': 'yellow', "font-size": "1.4rem",
                            'background-color': 'black', 'marginLeft':50, 'vertical-align':'middle'},
                           className="four columns instruction",      
                         
                   ))
                ])  
      
    elif pathname == "/page-1":
        return html.Div([  
                html.Div(   
                  html.Iframe(
                      sandbox='',
                      srcDoc='''
                      <!DOCTYPE html>
                      <html lang="en"> 
                       <html>
                        <head>
                            <style>
                                .myDiv {border: 5 outset red; background-color: lightblue; text-align: center;}
                            </style>
                        </head>
                        
                        <body>
                        <div class="myDiv">
                          <h1> Введение в Автоматизированную систему научных исследований в медицине "АСНИ-МЕД" </h1>
                        </div>
                        </body>
                       </html> 
                    ''',
                     style={'width':'95.0%',"height": '70px','display':'inline-block',
                           'overflow-y':'auto', 'color': 'yellow', "font-size": "1.4rem",
                           'marginLeft':50, 'vertical-align':'middle'},
                           className="four columns instruction",      
           )),
            html.Div(
             [html.Div(id="demo-explanation", children=[demo_explanation(Asni)])],
               style={'width':'95.0%',"height": '1100px','display':'inline-block',
                      'overflow-y':'auto', 'color': 'yellow', "font-size": "1.4rem",
                      'marginLeft':50, 'vertical-align':'middle'},
               className="four columns instruction",         
          )
        ])  
        
    elif pathname == "/page-5":
        return html.Div([  
                html.Div(   
                  html.Iframe(
                      sandbox='',
                      srcDoc='''
                      <!DOCTYPE html>
                      <html lang="en"> 
                       <html>
                        <head>
                            <style>
                                .myDiv {border: 5 outset red; background-color: lightblue; text-align: center;}
                            </style>
                        </head>
                        
                        <body>
                        <div class="myDiv">
                          <h1> Информация о разработчиках "АСНИ-МЕД" </h1>
                        </div>
                        </body>
                       </html> 
                      ''',
                     style={'width':'95.0%',"height": '70px','display':'inline-block',
                           'overflow-y':'auto', 'color': 'yellow', "font-size": "1.4rem",
                           'marginLeft':50, 'vertical-align':'middle'},
                           className="four columns instruction",      
           )), 

         html.Iframe(
                    id="my-output",
                    src="assets/Maksut.html",
                    style={'width':'99.5%',"height": '450px','display':'inline-block',
                    'backgroundColor': 'white',       
                    'overflow-y':'auto', 'color': 'yellow', "font-size": "1.4rem",
                    'marginLeft':50, 'marginRight':1, 'vertical-align':'middle'},
                    className="four columns instruction",  
                ),    
            
        html.Iframe(
                    id="my-output2",
                    src="assets/WagnerCV.html",
                    style={'width':'99.5%',"height": '550px','display':'inline-block',
                    'backgroundColor': 'white',       
                    'overflow-y':'auto', 'color': 'yellow', "font-size": "1.4rem",
                    'marginLeft':50, 'marginRight':1, 'vertical-align':'middle'},
                    className="four columns instruction",  
                ),   
       ]) 
        
    elif pathname == "/page-6":
         return html.Div([  
                html.Div(   
                  html.Iframe(
                      sandbox='',
                      srcDoc='''
                      <!DOCTYPE html>
                      <html lang="en"> 
                       <html>
                        <head>
                            <style>
                                .myDiv {border: 5 outset red; background-color: lightblue; text-align: center;}
                            </style>
                        </head>
                        
                        <body>
                        <div class="myDiv">
                          <h1> Список литературных источников "АСНИ-МЕД" </h1>
                        </div>
                        </body>
                       </html> 
                    ''',
                     style={'width':'95.0%',"height": '70px','display':'inline-block',
                           'overflow-y':'auto', 'color': 'yellow', "font-size": "1.4rem",
                           'marginLeft':50, 'vertical-align':'middle'},
                           className="four columns instruction",      
           )), 
             
          html.Div(
             [html.Div(id="demo-explanation", children=[demo_explanation(Asni)])],
               style={'width':'95.0%',"height": '1100px','display':'inline-block',
                      'overflow-y':'auto', 'color': 'yellow', "font-size": "1.4rem",
                      'marginLeft':50, 'vertical-align':'middle'},
               className="four columns instruction",         
          )
        ]) 

    elif pathname == "/page-7":
        return html.Div([  
                html.Div(   
                  html.Iframe(
                      sandbox='',
                      srcDoc='''
                      <!DOCTYPE html>
                      <html lang="en"> 
                       <html>
                        <head>
                            <style>
                                .myDiv {border: 5 outset red; background-color: lightblue; text-align: center;}
                            </style>
                        </head>
                        
                        <body>
                        <div class="myDiv">
                          <h1> Выбор и просмотр PDF-файлов "АСНИ-МЕД" </h1>
                        </div>
                        </body>
                       </html> 
                    ''',
                     style={'width':'95.0%',"height": '70px','display':'inline-block',
                           'overflow-y':'auto', 'color': 'yellow', "font-size": "1.4rem",
                           'marginLeft':50, 'vertical-align':'middle'},
                           className="four columns instruction",      
           )), 
            
          html.Div([
                html.Label(['Пожалуйста, выберете PDF-файл'], style={'color': 'yellow', 'marginLeft':55}),  
                dcc.Dropdown(
                    id="input2",
                    options=[
                        {"label": "Введение",  "value": "Einleitung.pdf"},
                        {"label": "Концепт",     "value": "GesamtNNRZ.pdf"},
                        {"label": "История",  "value": "GeschichteNNRZ.pdf"},  
                        {"label": "EDA Report",  "value": "ASNI_Result2025.pdf"}, 
                        {"label": "IPYNB Report",  "value": "EDA_ReportFinal20240323.pdf"}, 
                        {"label": "Фонд страхования РК",  "value": "FomsAI.pdf"}, 
                    ], 
                    style={'width':'99.5%',"height": '40px',
                    'overflow-y':'auto', 'color': 'black', "font-size": "1.0rem",
                    'marginLeft':30, 'marginRight':1, 'vertical-align':'middle'},
                ),   
                
         html.ObjectEl(
            id="my-output2", 
            data=" ",
            type="application/pdf",
            style={'width':'98.5%',"height": '920px', 'display':'inline-block',
                           'overflow-y':'auto', 'color': 'yellow', "font-size": "1.4rem",
                           'backgroundColor': 'black', 
                           'marginLeft':60, 'marginRight':40, 'vertical-align':'middle'},
                           className="four columns instruction",  
             ), 
           ]), 
         ])   

    elif pathname == "/page-8":
        return html.Div([  
                html.Div(   
                  html.Iframe(
                      sandbox='',
                      srcDoc='''
                      <!DOCTYPE html>
                      <html lang="en"> 
                       <html>
                        <head>
                            <style>
                                .myDiv {border: 5 outset red; background-color: lightblue; text-align: center;}
                            </style>
                        </head>
                        
                        <body>                        <div class="myDiv">
                          <h1> Запуск программ "АСНИ-МЕД" и визуализация результатов работы</h1>
                        </div>
                        </body>
                       </html> 
                    ''',
                     style={'width':'95.0%',"height": '65px','display':'inline-block',
                           'overflow-y':'auto', 'color': 'yellow', "font-size": "1.4rem",
                           'marginLeft':50, 'vertical-align':'middle'},
                           className="four columns instruction",      
               )), 
               html.Div([
                html.Label(['Старт программы и визуализация отчёта'], style={'color': 'yellow', 'marginLeft':55}),   
                dcc.Dropdown(
                    id="input",
                    options=[
                        {"label": "ML-Report",  "value": "MLReportsMod2025.py"},
                        {"label": "EDA-Report", "value": "EDAReportsTestV01.py"},
                        #{"label": "End-Report", "value": "ASNI-Reports2025.py"},
                    ], 
                    style={'width':'99.5%',"height": '40px',
                    'overflow-y':'auto', 'color': 'black', "font-size": "1.0rem",
                    'marginLeft':30, 'marginRight':1, 'vertical-align':'middle'},
                ),
                html.Iframe(
                    id="my-output",
                    src="https://asnimednew-86i7ukkxxsjq22umxwp79q.streamlit.app/",
                    style={'width':'99.5%',"height": '875px','display':'inline-block',
                    'backgroundColor': 'white',       
                    'overflow-y':'auto', 'color': 'yellow', "font-size": "1.4rem",
                    'marginLeft':50, 'marginRight':1, 'vertical-align':'middle'},
                    className="four columns instruction",  
                ),
             ])            
         ]),                  
    
    
       
       #elif pathname == "/page-12":
       #   return html.Div([  
       #     close_app("firefox") 
       #   ]), 
        
    return dbc.Jumbotron(
        [
            html.H1("404: Not found", className="text-danger"),
            html.Hr(),
            html.P(f"The pathname {pathname} was not recognised..."),
        ]
    )
    
#Seite 7
@app.callback(
    Output("my-output2", "data"), 
    Input("input2", "value") 
)
def update_output_div(input_value):
    return f"assets/{input_value}"

#Seite 8
@app.callback(
    Output("my-output", "src"), 
    Input("input", "value"), prevent_initial_call=True)


def update_output_div(input_value):
    print(input_value)
    return f"https://asnimednew-86i7ukkxxsjq22umxwp79q.streamlit.app/" 
            
#Seite 9
       
if __name__ == "__main__":
    app.run_server(debug=False, port=8083)   
