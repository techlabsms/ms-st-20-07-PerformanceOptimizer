#############################################
#  Projekt Mood Optimizer
#  Gruppe 7: Charlotta, Leonie und Lena
#  Techlabs Sommersemester 2020
#############################################

#Pakete laden
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import copy as cp
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score,mean_squared_error

from joblib import dump, load
import seaborn as sbn


#### Schritt 3:Deskriptive Analysen
# Erstellen von Dictionary für die Excel mit den einzelnen Sheets pro Teilnehmer:
dictdfs = pd.read_excel("./data/daten_277.xlsx", sheet_name=None)

# - Zur Erinnerung: Ansprechen von Variablen -
# dictdfs["merged_v01"]["mood"]
# erst in eckigen Klammern, welches Sheet man ansprechen will, dann die Variable davon

# Deskriptive Daten von den einzelnen Teilnehmern erstellen:
#als Schleife für alle Teilnehmer:
dictdes ={} #leeres Dictionary für die deskriptiven Daten erstellt
for item in dictdfs.keys():
    dictdes[item] = dictdfs[item].describe()

#Ausgeben lassen der deskriptiven Daten für die 15 Teilnehmer:
for item in dictdes.keys():
    print(dictdes[item])
    #für eine bessere Übersicht rechts unten in der Leiste auf "View as DataFrame" gehen

## Korrelationsmatrix
for item in dictdfs.keys():
    x_train = dictdfs[item]["sleep_quality"].dropna().values
    y_train= dictdfs[item]["mood"].dropna().values


matsch = pd.DataFrame()
person_id = []
for i, item in enumerate(dictdfs.keys()):
    person_id.extend([i for entry in range(len(dictdfs[item]))])
    matsch = matsch.append(dictdfs[item])
matsch["person_id"] = person_id
matsch = matsch.reset_index()


corrMatrix = matsch.corr()
sbn.heatmap(corrMatrix, annot=True, cmap="RdBu", center=0, vmin=-0.8,vmax=0.8)
plt.tight_layout()
plt.show()

matschna = matsch.dropna()
len(matschna)

#### Schritt 4: Random Forest Model
predictors = ["sleep_duration_h", 'sleep_quality',
              'glasses_of_fluid','steps', 'very_active_minutes' ,'sedentary_minutes']

x = matschna[predictors]
y = matschna.mood

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=42)

Rf = RandomForestRegressor()
Rf.fit(x_train, y_train)

y_pred = Rf.predict(x_test)
y_pred

mse=mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(rmse)

# Save trained model to file (alternative: pickle)
dump(Rf, "./data/random_forest.joblib")


