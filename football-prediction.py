# -*- coding: utf-8 -*-
"""
Created on Sat Jun  3 22:46:34 2023

@author: dusan
"""

import pandas as pd

training_data1 = pd.read_csv(r"D:\faks\3\2. Semestar\MITNOP\Projekat\football-prediction\Datasets\Season1.csv")
training_data2 = pd.read_csv(r"D:\faks\3\2. Semestar\MITNOP\Projekat\football-prediction\Datasets\Season2.csv")
training_data = pd.concat([training_data1, training_data2])
test_data = pd.read_csv(r"D:\faks\3\2. Semestar\MITNOP\Projekat\football-prediction\Datasets\Season3.csv")

selected_columns = ['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'HTHG', 'HTAG', 'HS', 'AS', 'HST', 'AST', 'HC', 'AC', 'HF', 'AF', 'HY', 'AY', 'HR', 'AR']
training_data = training_data[selected_columns]
test_data = test_data[selected_columns]

