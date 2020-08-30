#############################################
#  Projekt Mood Optimizer
#  Gruppe 7: Charlotta, Leonie und Lena
#  Techlabs Sommersemester 2020
#############################################

import pandas as pd
from joblib import dump, load

# import PySimpleGUI as sg
import seaborn as sbn

##Idee: Eingabefeld bauen, in das man seine Variablen einträgt und dann eine Vorhersage herausbekommt

RF = load("./data/random_forest.joblib")

predictors = ["sleep_duration_h", 'sleep_quality',
              'glasses_of_fluid', 'steps', 'very_active_minutes', 'sedentary_minutes']

#Use Case: User benennt unbhängige Variablen und bekommt vorhergesagte abhängige Variable (mood)

print("This is a basic command-line interface for determining your mood.\n")

Questions = ["How many hours have you slept last night? (as float)",
             "How well did you sleep on a scale from 1 (very bad) to 5 (great)? (as integer)",
             "How many glasses of fluid did you drink today? (as float)",
             "How many steps have you taken today? (as integer)",
             "How many active minutes did you have today? (as integer)",
             "How many inactive minutes did you have today? (as integer)"]
df = pd.DataFrame(columns=predictors)

for i, item in enumerate(predictors):
    user_in = input(Questions[i])
    df[item] = [user_in]

pred = RF.predict(df)

print(f"Our AI estimated your mood on a scale from 1 (bad) to 5 (great) as: {pred[0]:.2f}")
