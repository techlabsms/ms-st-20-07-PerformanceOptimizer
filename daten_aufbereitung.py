#############################################
#  Projekt Mood Optimizer
#  Gruppe 7: Charlotta, Leonie und Lena
#  Techlabs Sommersemester 2020
#############################################

# !/usr/bin/env python
# coding: utf-8

#### Schritt 1: Wichtige Pakete laden
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import copy as cp
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

from joblib import dump, load

from sklearn import metrics
import pandas as pd

#### Schritt 2: Daten einlesen
folder = "data"
sub_folders = [f"p{i + 1:02}" for i in range(15)]
table_name = [f"merged_v{i + 1:02}" for i in range(15)]
list_files = []
our_files = ["googledocs/reporting.csv", "pmsys/wellness.csv", "pmsys/srpe.csv", "fitbit/calories.json",
                "fitbit/sedentary_minutes.json", "fitbit/steps.json", r"fitbit/very_active_minutes.json"]
for i, person in enumerate(sub_folders):
    if i == 14:
        continue

    files = []
    for file in stupid_files:
        files.append("./" + folder + "/" + person + "/" + file)
    files.append(table_name[i])
    list_files.append(files)

results = {}
# for a (reporting), b (wellness), c (sport), d (calories), e (sedentary minutes),
# f (steps), g (very active minutes, h (name of excel sheet)  in list_files
for a, b, c, d, e, f, g, h in list_files:
    # REPORTING dataset
    reporting = pd.read_csv(a, usecols=['timestamp', 'glasses_of_fluid', 'alcohol_consumed'])
    reporting['timestamp'] = pd.to_datetime(reporting['timestamp'], yearfirst=True,
                                            format="%d/%m/%Y %H:%M:%S")
    reporting['timestamp'] = reporting['timestamp'].dt.strftime('%Y-%m-%d')
    reporting.set_index('timestamp', inplace=True)
    reporting.index.name = 'date_reporting'

    # change variable 'alcohol_consumed', so it cant be used with groupby function
    # as a result value=1 indicate "alcohol consumed", value=2 indicate "no alcohol consumed or missing value"
    reporting['alcohol_consumed'] = reporting['alcohol_consumed'].map({'Yes': 1, 'No': 2})

    # WELLNESS dataset
    wellness = pd.read_csv(b, usecols=['effective_time_frame', 'mood',
                                       'sleep_duration_h', 'sleep_quality', 'stress'])
    wellness['effective_time_frame'] = pd.to_datetime(wellness['effective_time_frame'],
                                                      yearfirst=True, format="%Y-%m-%dT%H:%M:%S.%fZ")
    wellness['effective_time_frame'] = wellness['effective_time_frame'].dt.strftime('%Y-%m-%d')
    wellness.set_index('effective_time_frame', inplace=True)
    wellness.index.name = 'date_wellness'

    # SPORT dataset
    sport = pd.read_csv(c, parse_dates=True, usecols=['duration_min', 'end_date_time'])
    sport['end_date_time'] = pd.to_datetime(sport['end_date_time'], yearfirst=True, format="%Y-%m-%dT%H:%M:%S.%fZ")
    sport['end_date_time'] = sport['end_date_time'].dt.strftime('%Y-%m-%d')
    sport.set_index('end_date_time', inplace=True)
    sport.index.name = 'date_sport'

    # CALORIES dataset
    calories = pd.read_json(d)
    calories['dateTime'] = pd.to_datetime(calories['dateTime'], yearfirst=True, format="%Y-%m-%d %H:%M:%S")
    calories['dateTime'] = calories['dateTime'].dt.strftime('%Y-%m-%d')
    calories.set_index('dateTime', inplace=True)
    calories = calories.groupby(calories.index).sum()
    calories.rename(columns={'value': 'calories_per_day'}, inplace=True)

    # SEDENTARY MINUTES dataset
    sedentary_minutes = pd.read_json(e)
    sedentary_minutes['dateTime'] = pd.to_datetime(sedentary_minutes['dateTime'], yearfirst=True, format="%Y-%m-%d")
    sedentary_minutes.set_index('dateTime', inplace=True)
    sedentary_minutes.rename(columns={'value': 'sedentary_minutes'}, inplace=True)

    # STEPS dataset
    steps = pd.read_json(f)
    steps['dateTime'] = pd.to_datetime(steps['dateTime'], yearfirst=True, format="%Y-%m-%d")
    steps['dateTime'] = steps['dateTime'].dt.strftime('%Y-%m-%d')
    steps.set_index('dateTime', inplace=True)
    steps = steps.groupby(steps.index).sum()
    steps.rename(columns={'value': 'steps'}, inplace=True)

    # VERY ACTIVE MINUTES dataset
    very_active_minutes = pd.read_json(g)
    very_active_minutes['dateTime'] = pd.to_datetime(very_active_minutes['dateTime'], yearfirst=True, format="%Y-%m-%d")
    very_active_minutes.set_index('dateTime', inplace=True)
    very_active_minutes.rename(columns={'value': 'very_active_minutes'}, inplace=True)

    ###### Merging ######
    # merging of subsets reporting, wellness, and sport
    csv_merge_1 = reporting.merge(wellness, left_index=True, right_index=True, how='outer', validate='m:m')
    csv_merge_2 = csv_merge_1.merge(sport, left_index=True, right_index=True, how='outer', validate='m:m')

    # merging of json subets with calories per day, sedentary minutes, steps, and very active minutes
    json_merge_1 = calories.merge(sedentary_minutes, left_index=True, right_index=True)
    json_merge_2 = json_merge_1.merge(steps, left_index=True, right_index=True, how="outer", validate='m:m')
    json_merge_3 = json_merge_2.merge(very_active_minutes, left_index=True, right_index=True, how="outer",
                                      validate='m:m')

    # merge csv_merge_2 and json_merge_3
    merged = csv_merge_2.merge(json_merge_3, left_index=True, right_index=True, how='outer', validate='m:m')
    print(merged.head(3))

    # set index in merged dataset and combine redundant rows
    merged.index = pd.to_datetime(merged.index)
    merged.index = merged.index.to_period('d')
    merged.index.name = 'date'
    merged = merged.groupby(merged.index)[
        ['glasses_of_fluid', 'alcohol_consumed', 'mood', 'sleep_duration_h', 'sleep_quality',
         'stress', 'duration_min', 'calories_per_day', 'sedentary_minutes', 'steps', 'very_active_minutes']].mean()

    # save one dataset per person with title
    results[h] = merged

# create excel file for each participant
output_file = "./" + folder + '/daten_277.xlsx'
writer = pd.ExcelWriter(output_file, engine='xlsxwriter', datetime_format='YYYY-mm-dd')

for key, value in results.items():
    value.to_excel(writer, sheet_name=key)
writer.save()
