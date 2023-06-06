# -*- coding: utf-8 -*-
"""
Created on Sat Jun  3 22:46:34 2023

@author: dusan
"""

import pandas as pd
from tensorflow.keras import layers
from tensorflow.keras import models
import numpy as np

#Ucitavamo podatke
ds1 = pd.read_csv(r"Datasets\Season1.csv")
ds2 = pd.read_csv(r"Datasets\Season2.csv")
ds3 = pd.read_csv(r"Datasets\Season3.csv")
dataset = pd.concat([ds1, ds2, ds3])

#Filtriramo dataset
selected_columns = ['Date', 'Time', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'HTHG', 'HTAG', 'HS', 'AS', 'HST', 'AST', 'HC', 'AC', 'HY', 'AY', 'HR', 'AR']
dataset = dataset[selected_columns]

#Dodajemo dodatne potrebne kolone koje cemo koristiti za NM
dataset["Date"] = pd.to_datetime(dataset["Date"], format="%d/%m/%Y")
dataset["Hour"] = dataset["Time"].str.replace(":.+", "", regex=True).astype("int")
dataset["Day_Code"] = dataset["Date"].dt.dayofweek
dataset = dataset.drop(["Time"], axis=1)



#Mapiramo Timove na indekse, da bi mogli da ih koristimo u NM
categorical_columns = ["HomeTeam", "AwayTeam"]
teams = pd.unique(dataset[['HomeTeam']].values.ravel())
numerical_labels = pd.RangeIndex(len(teams))
team_mapping = dict(zip(teams, numerical_labels))
dataset['HomeTeam'] = dataset['HomeTeam'].map(team_mapping).astype('int')
dataset['AwayTeam'] = dataset['AwayTeam'].map(team_mapping).astype('int')

#Delimo dataset na training dataset i testni dataset
training_data = dataset[dataset['Date'] < '08/01/2021']
test_data = dataset[dataset['Date'] > '08/01/2021']

#Delimo ulazne kolone na dva dela, jer cemo koristiti Sijamsku NM, gde ce nam jedna podmreza biti za domacu ekipu, a druga za gostujucu
home_columns=['Hour','Day_Code', 'HomeTeam', 'HTHG', 'HS', 'HST', 'HC', 'HY', 'HR']
away_columns=['Hour','Day_Code', 'AwayTeam', 'HTAG', 'AS', 'AST', 'AC', 'AY', 'AR']
training_input_home = training_data[home_columns]
training_input_away = training_data[away_columns]
training_output_home = training_data['FTHG']
training_output_away = training_data['FTAG']

#Definisemo velicinu ulaza u mrezu i broj neurona u sloju
input_shape = (len(home_columns),)
hidden_units = 64

#Definisemo podmrezu sa jednim slojem za domaci tim
input_home = layers.Input(shape=input_shape)
hidden_home1 = layers.Dense(hidden_units, activation='relu')(input_home)
hidden_home2 = layers.Dense(hidden_units, activation='relu')(hidden_home1)

#Definisemo podmrezu sa jednim slojem za gostujuci tim
input_away = layers.Input(shape=input_shape)
hidden_away1 = layers.Dense(hidden_units, activation='relu')(input_away)
hidden_away2 = layers.Dense(hidden_units, activation='relu')(hidden_away1)

#Koristimo sloj Concatenate da bi spojili izlaze iz prethodne dve podmreze
concatenated = layers.Concatenate()([hidden_home1, hidden_away2])

#Kreiramo dva izlazna sloja
output_home = layers.Dense(1, activation='linear', name='output_home')(concatenated)
output_away = layers.Dense(1, activation='linear', name='output_away')(concatenated)

#Kreiramo model za Sijamsku NM
sm = models.Model(inputs=[input_home, input_away], outputs=[output_home, output_away])


sm.compile(optimizer='adam', loss='mse')


sm.fit([training_input_home, training_input_away], [training_output_home, training_output_away], epochs=20, batch_size=32, validation_split=0.2)


test_input_home = test_data[home_columns]
test_input_away = test_data[away_columns]
test_output_home = test_data['FTHG'].values.ravel()
test_output_away = test_data['FTAG'].values.ravel()



predicted_home_goals, predicted_away_goals = sm.predict([test_input_home, test_input_away])

predicted_home_goals = predicted_home_goals.astype(int)
predicted_away_goals = predicted_away_goals.astype(int)

combined = pd.DataFrame(dict(actual=test_output_home, prediction=predicted_home_goals.flatten()))
print(pd.crosstab(index = combined["actual"], columns=combined["prediction"]))


accuracy = (predicted_home_goals.flatten() == test_output_home).mean()

print("Precision Accuracy for Home Team:", accuracy)

