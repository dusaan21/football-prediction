# -*- coding: utf-8 -*-
"""
Created on Sat Jun  3 22:46:34 2023

@author: dusan
"""

import pandas as pd
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping

import numpy as np

def train_and_predict_snn(home_columns, away_columns, training_data, test_data, output_columns):
    #Definisemo velicinu ulaza u mrezu i broj neurona u sloju
    input_shape = (len(home_columns),)
    hidden_units = 64
    l2_strength = 0.015

    #Definisemo velicinu ulaza u mrezu i broj neurona u sloju
    training_input_home = training_data[home_columns]
    training_input_away = training_data[away_columns]
    training_output_home = training_data[output_columns[0]]
    training_output_away = training_data[output_columns[1]]

    #Definisemo podmrezu sa 5 slojeva za domaci tim
    input_home = layers.Input(shape=input_shape)
    hidden_home1 = layers.Dense(hidden_units, activation='LeakyReLU', kernel_regularizer=regularizers.l2(l2_strength))(input_home)
    hidden_home2 = layers.Dense(hidden_units/2, activation='LeakyReLU', kernel_regularizer=regularizers.l2(l2_strength))(hidden_home1)
    hidden_home3 = layers.Dense(hidden_units/4, activation='LeakyReLU', kernel_regularizer=regularizers.l2(l2_strength))(hidden_home2)
    hidden_home4 = layers.Dense(hidden_units/8, activation='LeakyReLU', kernel_regularizer=regularizers.l2(l2_strength))(hidden_home3)
    hidden_home5 = layers.Dense(1, activation='linear', kernel_regularizer=regularizers.l2(l2_strength))(hidden_home4)

    #Definisemo podmrezu sa 5 slojeva za gostujuci tim
    input_away = layers.Input(shape=input_shape)
    hidden_away1 = layers.Dense(hidden_units, activation='LeakyReLU', kernel_regularizer=regularizers.l2(l2_strength))(input_away)
    hidden_away2 = layers.Dense(hidden_units/2, activation='LeakyReLU', kernel_regularizer=regularizers.l2(l2_strength))(hidden_away1)
    hidden_away3 = layers.Dense(hidden_units/4, activation='LeakyReLU', kernel_regularizer=regularizers.l2(l2_strength))(hidden_away2)
    hidden_away4 = layers.Dense(hidden_units/8, activation='LeakyReLU', kernel_regularizer=regularizers.l2(l2_strength))(hidden_away3)
    hidden_away5 = layers.Dense(1, activation='linear', kernel_regularizer=regularizers.l2(l2_strength))(hidden_away4)

    #Koristimo sloj Concatenate da bi spojili izlaze iz prethodne dve podmreze
    concatenated = layers.Concatenate()([hidden_home5, hidden_away5])

    #Kreiramo dva izlazna sloja
    output_home = layers.Dense(1, activation='linear', name='output_home')(concatenated)
    output_away = layers.Dense(1, activation='linear', name='output_away')(concatenated)

    #Kreiramo model za SNN
    sm = models.Model(inputs=[input_home, input_away], outputs=[output_home, output_away])
    
    #Postavljamo optimizer i loss funkciju da bi nam mreza bila spremna za treniranje
    sm.compile(optimizer='adam', loss='mae')

    #Na ovaj nacin, nasa mreza ce automatski prestati da se trenira kada nema vise znacajnih promena u preciznosti iz epohe u epohu
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    #Treniramo model
    sm.fit([training_input_home, training_input_away], [training_output_home, training_output_away], epochs=500, batch_size=32, validation_split=0.2, callbacks=[early_stop])

    #Definisemo ulaze i izlaze za testni dataset
    test_input_home = test_data[home_columns]
    test_input_away = test_data[away_columns]

    #Dobijamo predikcije golova za nas test set
    predicted_home, predicted_away = sm.predict([test_input_home, test_input_away])

    #Zaokruzujemo na ceo broj nase predikcije
    predicted_home = predicted_home.round().astype(int)
    predicted_away = predicted_away.round().astype(int)

    #Stavljamo negativne vrednosti na nulu
    predicted_home = np.maximum(predicted_home, 0)
    predicted_away = np.maximum(predicted_away, 0)

    return predicted_home.flatten(), predicted_away.flatten()

#Ucitavamo podatke
ds1 = pd.read_csv(r"Datasets\Season1.csv")
ds2 = pd.read_csv(r"Datasets\Season2.csv")
ds3 = pd.read_csv(r"Datasets\Season3.csv")
dataset = pd.concat([ds1, ds2, ds3])

#Filtriramo dataset
selected_columns = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'HTHG', 'HTAG', 'HF', 'AF', 'HS', 'AS', 'HST', 'AST', 'HC', 'AC', 'HY', 'AY', 'HR', 'AR']
dataset = dataset[selected_columns]

#Dodajemo dodatne potrebne kolone koje cemo koristiti za NM
dataset["Date"] = pd.to_datetime(dataset["Date"], format="%d/%m/%Y")

#formiramo dodatne kolone koje ce nam dodatno optimizovati mrezu
dataset['Home_SOT_Perc'] = dataset['HST']/dataset['HS']
dataset['Away_SOT_Perc'] = dataset['AST']/dataset['AS']


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
home_columns=['HomeTeam', 'HTHG', 'HS', 'HST', 'HC', 'Home_SOT_Perc', 'HY', 'HR']
away_columns=['AwayTeam', 'HTAG', 'AS', 'AST', 'AC', 'Away_SOT_Perc', 'AY', 'AR']

predicted_home_goals, predicted_away_goals = train_and_predict_snn(home_columns, away_columns, training_data, test_data, ['FTHG', 'FTAG'])
predicted_home_fouls, predicted_away_fouls = train_and_predict_snn(home_columns, away_columns, training_data, test_data, ['HF', 'AF'])


test_output_home_goals = test_data['FTHG'].values.ravel()
test_output_away_goals = test_data['FTAG'].values.ravel()
test_output_home_fouls = test_data['HF'].values.ravel()
test_output_away_fouls = test_data['AF'].values.ravel()

#Kreiramo crosstab koji ce nam u tabeli oznaciti gde smo bili tacni, a gde smo imali odstupanja
combined = pd.DataFrame(dict(actual=test_output_home_goals, prediction=predicted_home_goals.flatten()))
print("Crosstab for Home Team Goals: ")
print(pd.crosstab(index = combined["actual"], columns=combined["prediction"]))
accuracy = (predicted_home_goals.flatten() == test_output_home_goals).mean()
print("Precision Accuracy for Home Team Goals:", accuracy)

combined = pd.DataFrame(dict(actual=test_output_away_goals, prediction=predicted_away_goals.flatten()))
print("Crosstab for Away Team Goals: ")
print(pd.crosstab(index = combined["actual"], columns=combined["prediction"]))
accuracy = (predicted_away_goals.flatten() == test_output_away_goals).mean()
print("Precision Accuracy for Away Team Goals:", accuracy)

combined = pd.DataFrame(dict(actual=test_output_home_fouls, prediction=predicted_home_fouls.flatten()))
print("Crosstab for Home Team Fouls: ")
print(pd.crosstab(index = combined["actual"], columns=combined["prediction"]))
accuracy = (abs(predicted_home_fouls.flatten() - test_output_home_fouls) <= 2).mean()
print("Precision Accuracy for Home Team Fouls:", accuracy)

combined = pd.DataFrame(dict(actual=test_output_away_fouls, prediction=predicted_away_fouls.flatten()))
print("Crosstab for Away Team Fouls: ")
print(pd.crosstab(index = combined["actual"], columns=combined["prediction"]))
accuracy = (abs(predicted_away_fouls.flatten() - test_output_away_fouls) <= 2).mean()
print("Precision Accuracy for Away Team Fouls:", accuracy)