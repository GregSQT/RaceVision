#!/usr/bin/env python
# coding: utf-8

# # RaceVision - Data Preparation
# Data are from Ergast DB : http://ergast.com/mrd/db/  
# 
# This model uses an LSTM to predict the positions, laptimes and pitstops for 20 drivers. 
# Features to add : 
# - Weather
# - Tyres and track types
# - (if rules chage, fuel weight impact on lap time and pit stop)

# ## Initialization

# In[1]:


#%pip install matplotlib
import matplotlib.pyplot as plt
#%pip install pandas
import pandas as pd
#%pip install numpy
import numpy as np
#%pip install tqdm
from tqdm.auto import tqdm
#%pip install os
import os
#%pip install torch
import torch, torch.nn as nn, torch.optim as optim, torch.nn.functional as F
#%pip install math
import math
#%pip install pathlib
from pathlib import Path


# In[2]:


db_dir = Path(r"E:\Dropbox\Informatique\Holberton\F1_Project\db")


# In[3]:


# The time format in the Ergast database is MM:SS.ms
# For a better work, we split aroud the ":" and express it in seconds (float)
def time_to_int(t):
  # If input is already float type, return as-is
  if (t == float):
    return t
  t2 = str(t)
  ts = t2.rsplit(':')
  # Handle missing values
  if ('\\N' in t2):
    return None # missing data
  if (not '.' in t2):
    return None # unexpected format
  # Convert minutes and seconds to total seconds
  if (len(ts) > 1):
    return int(ts[0]) * 60 + float(ts[1])
  else:
    return float(ts[0])


# ## Data preparation
# 
# Each csv file contains information of one race, and each row contains information of a lap.

# In[4]:


# -----------------------------------------------------------------------------
# Build Per-Race, Per-Lap Data for Modeling/Analysis
# -----------------------------------------------------------------------------
# This script transforms raw F1 CSVs into per-race CSV files where each row
# represents a lap and each column block represents one driver on the grid.
# It captures pre-race (lap 0) qualifying snapshots, live positions, lap times,
# pit-stop flags, and driver/constructor standings at race start.
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# 1) Load Master Tables
# -----------------------------------------------------------------------------
races = pd.read_csv(db_dir / 'races.csv')                       # - races: race metadata (year, round, circuit, etc.)
d_standings = pd.read_csv(db_dir / 'driver_standings.csv')      # - Driver_standings: standings after each round
c_standings = pd.read_csv(db_dir / 'constructor_standings.csv') # - Constructor_standings: standings after each round
quali = pd.read_csv(db_dir / 'qualifying.csv')                  # - Qualifying: Q1/Q2/Q3 lap times
pit_stops = pd.read_csv(db_dir / 'pit_stops.csv')               # - Pit_stops: pit stop events (from 2012 onward)
lap_times = pd.read_csv(db_dir / 'lap_times.csv')               # - Lap_times: every driver’s lap-by-lap time
results = pd.read_csv(db_dir / 'results.csv')                   # - Results: final finishing order and status per driver per race

# -----------------------------------------------------------------------------
# 2) Select Relevant Races
# -----------------------------------------------------------------------------
races_selection = races.query('year  >= 2018')
rids = races_selection['raceId']  # Getting the IDs of all the races since 2018


# -----------------------------------------------------------------------------
# 3) Iterate over each race
# -----------------------------------------------------------------------------
# Loop over each race ID with a progress bar
for i in tqdm(rids):
  # Getting the raceID, and the year of this race
  race_info = races.query(f'raceId == {i}')
  year = race_info['year'].item()

  # Skip the 2021 season (we save it for tests)
  if (year == 2021):
    continue

  # Get the circuit where the race was held
  circuit = race_info['circuitId'].item()

  # Prepare output path: db_dir/races/<year>
  if not os.path.exists(db_dir / f'races/{year}'):
      os.makedirs(db_dir / f'races/{year}')

  # Skip if we've already built the CSV for this race
  if os.path.exists(db_dir / f'races/{year}/race{i}.csv'):
    continue

  # Determine which standings to use (prior to this race)
  if (race_info['round'].item() > 1):
    # For rounds >1, use previous raceId = i-1
    d_standing_pre_race = d_standings.query(f'raceId == {i-1}')
    c_standing_pre_race = c_standings.query(f'raceId == {i-1}')
  else:
    # If this is the first race, roll back to the last race of the previous season
    prev_s = races.query(f'year == {year - 1}')
    prev_s = prev_s.sort_values(by=['round'])
    prev_s = prev_s.reset_index()
    prev_last_race = prev_s['raceId'].iloc[-1]
    d_standing_pre_race  = d_standings.query(f'raceId == {prev_last_race}')
    c_standing_pre_race  = c_standings.query(f'raceId == {prev_last_race}')
  
  # Load all per-race sub-tables
  quali_info = quali.query(f'raceId == {i}')          # qualifying data
  r_laptimes = lap_times.query(f'raceId == {i}')      # lap times
  r_pitstops = pit_stops.query(f'raceId == {i}')      # pit stops
  r_results = results.query(f'raceId == {i}')         # final results

  # Sort results to find number of laps (winner’s laps) and to get grid order
  r_results_sorted = r_results.sort_values('position').reset_index(drop=True)
  num_of_laps = r_results_sorted['laps'].iloc[0]

  # Get the starting grid order (pole position to P20)
  r_results_sorted_grid = r_results.sort_values('grid').reset_index(drop=True)
  
  # -----------------------------------------------------------------------------
  # 4) Build blank DataFrame schema for per-lap records
  # -----------------------------------------------------------------------------
  # Prepare empty DataFrame: one row per lap, columns for circuit + 20 drivers * various features

  columns = ['circuitId']
  for k in range(20):
    columns.extend([
      f'driverId{k+1}',
      f'driverStanding{k+1}',
      f'constructorStanding{k+1}',
      f'position{k+1}',
      f'inPit{k+1}',
      f'status{k+1}',
      f'laptime{k+1}'
    ])
  df = pd.DataFrame(columns=columns)

  # Instead of concatenating inside the loop, collect rows here
  all_rows = []

  # -----------------------------------------------------------------------------
  # 5) Populate each lap row (lap 0 = qualifying baseline)
  # -----------------------------------------------------------------------------
  # Each lap is a row, each race is a dataframe
  for lap_num in range(0, num_of_laps + 1):
   
     # Lists to collect per-driver data for this lap
    driver_ids = []
    d_s = [] # Driver's standings
    c_s = [] # Constructor's standings
    pos = [] # Driver's position
    pit = [] # Pit stop (1) or not (0)
    statuses = [] # Status of the driver
    lps = [] # Laptimes (in seconds)

    # Build each lap row: lap 0 is pre-race (qualifying), laps 1..N are race laps
    # Loop drivers in starting grid order (pole → P20)
    for id in r_results_sorted_grid['driverId']:
      driver_ids.append(id)
      constructorId = r_results.query(f'driverId == {id}')['constructorId'].item()
      d_s_p = d_standing_pre_race.query(f'driverId == {id}')['position']
      if (not d_s_p.empty):
        d_s.append(d_s_p.item())
      else:
        d_s.append(20)
      c_s_p = c_standing_pre_race.query(f'constructorId == {constructorId}')['position']
      if (not c_s_p.empty):
        c_s.append(c_s_p.item())
      else:
        c_s.append(10)
      if (lap_num == 0):
        p = quali_info.query(f'driverId == {id}')['position']
        if (not p.empty):
          p = p.item()
        else:
          p = 21
      else:
        p = r_laptimes.query(f'driverId == {id} & lap == {lap_num}')['position']
        if (not p.empty): # position could be null
          p = p.item()
        else:
          p = 21 # 21 means retired
      pos.append(p)
      inp = r_pitstops.query(f'driverId == {id} & lap == {lap_num}')
      if (not inp.empty):
        pit.append(1)
      else:
        pit.append(0)

      # Zeroth lap laptime is quali laptime
      if (lap_num == 0):
        q3_s = quali_info.query(f'driverId == {id}')['q3']
        if (not q3_s.empty):
          q3 = time_to_int(q3_s.item())
        else:
          q3 = None
        q2_s = quali_info.query(f'driverId == {id}')['q2']
        if (not q2_s.empty):
          q2 = time_to_int(q2_s.item())
        else:
          q2 = None
        q1_s = quali_info.query(f'driverId == {id}')['q1']
        if (not q1_s.empty):
          q1 = time_to_int(q1_s.item())
        else:
          q1 = None
        if (q3):
          lps.append(q3)
        elif (q2):
          lps.append(q2)
        elif (q1):
          lps.append(q1)
        else:
          lps.append(0)
        statuses.append(0) # 0 when in race or before race start
      elif (r_results.query(f'driverId == {id}')['laps'].item() <= lap_num): # check if driver has retired
        statuses.append(r_results.query(f'driverId == {id}')['statusId'].item())
        lps.append(0)
      else:
        statuses.append(0)
        t = r_laptimes.query(f'driverId == {id} & lap == {lap_num}')['time']
        if (not t.empty): 
          lps.append(time_to_int(t.item()))
        else: # if somehow we cant find a laptime
          lps.append(0)
    row = {}
    row['circuitId'] = circuit
    for j in range(len(driver_ids)):
        row[f'driverId{j+1}'] = driver_ids[j]
        row[f'driverStanding{j+1}'] = d_s[j]
        row[f'constructorStanding{j+1}'] = c_s[j]
        row[f'position{j+1}'] = pos[j]
        row[f'inPit{j+1}'] = pit[j]
        row[f'status{j+1}'] = statuses[j]
        row[f'laptime{j+1}'] = lps[j]
    all_rows.append(row)
  df = pd.DataFrame(all_rows, columns=columns)
  output_path = db_dir / 'races' / str(year) / f'race{i}.csv'
  df.to_csv(output_path, index=False)
    


# ## Data preparation
# 
# Each csv file contains information of one race, and each row contains information of a lap.

# In[5]:


# Open relevant files
races = pd.read_csv(db_dir / 'races.csv')
results = pd.read_csv(db_dir / 'results.csv')

# Collect all driverIds across selected races
all_drivers = []
for race_id in tqdm(rids, desc="Gathering driver IDs"):
    r_results = results.query(f'raceId == {race_id}')
    all_drivers.extend(r_results['driverId'].tolist())

# Create DataFrame of unique drivers, sorted
dddf = pd.DataFrame({'driverId': sorted(set(all_drivers))})

# Write to CSV without an unwanted index column
# This line was modified to drop the default index
dddf.to_csv(db_dir / 'drivers_short.csv', index=False)


# In[7]:


# Create a DataFrame with unique driver IDs
#ddf = pd.DataFrame({'driverId':df['driverId'].unique()})


# In[8]:


#
dddf = dddf.sort_values(by=['driverId']).reset_index()


# In[9]:


dddf = dddf.drop(columns=['index'])
dddf


# In[10]:


dddf.to_csv(db_dir / f'drivers_short.csv')


# In[11]:


years = range(2018, 2020)   #range(2001, 2020)
for year in years:
  if not os.path.exists(db_dir / f'races/{year}'):
      os.makedirs(db_dir / f'races/{year}')

  cur_year = os.listdir(db_dir / f'races/{year}/')
  for r in cur_year:
    #if os.path.exists(db_dir / f'races/{year}/{r}'):
    #  continue
    
    cur_race = pd.read_csv(db_dir / f'races/{year}/{r}')
    for j in range(20):
      col = f'inPit{j+1}'
      for i in range(len(cur_race)-1):
        if cur_race.at[i+1, col] == 1:
            cur_race.loc[i,   col] = 1
            cur_race.loc[i+1, col] = 0
      cur_race.rename(columns={col: f'inPit{j+1}'}, inplace=True)

    cur_race.to_csv(db_dir / f'races/{year}/{r}', index=False)
  print(year)
    


# In[12]:


years = range(2018, 2020)   #range(2001, 2020)
numlaps = []
for y in years:
  cur_year = os.listdir(db_dir / f'races/{y}/')
  for r in cur_year:
    cur_race = pd.read_csv(db_dir / f'races/{y}/{r}')
    numlaps.append(len(cur_race) - 1)
  print(y)


# In[13]:


numlaps.sort()
print(numlaps)
print(numlaps[-1])

