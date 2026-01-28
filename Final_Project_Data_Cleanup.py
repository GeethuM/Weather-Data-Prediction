#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 24 12:43:19 2022

@author: geethumuru
"""




from __future__ import print_function




from pandas import read_csv



pd_df = read_csv("/Users/geethumuru/Documents/Grad School/BU/CS777/Homeworks/Final Project/weatherAUS.csv")

# Remove columns that are unnecessary based on visual analysis of csv file
pd_df = pd_df.drop(columns = ["Sunshine", "Evaporation", "Location", 
                              "WindGustDir", "WindDir9am", "WindDir3pm", 
                              "WindSpeed9am","WindSpeed3pm"])
# Drop all rows containing NA values
pd_df = pd_df.dropna()


# create average for cloud, humidity, pressure, temp for each day. 
# adding up all the recorded values and divide by the number of values:
pd_df["cloud"]= (pd_df["Cloud9am"]+pd_df["Cloud3pm"])/2
pd_df["pressure"]= (pd_df["Pressure9am"]+pd_df["Pressure3pm"])/2
pd_df["humidity"]= (pd_df["Humidity9am"]+pd_df["Humidity3pm"])/2
pd_df["temp"]= (pd_df["Temp9am"]+pd_df["Temp3pm"]+pd_df["MinTemp"]+pd_df["MaxTemp"])/4

# Drop unnecessary columns
pd_df= pd_df.drop(columns = ["Cloud9am", "Cloud3pm", "Pressure9am",
                             "Pressure3pm", "Humidity9am", "Humidity3pm",
                             "Temp9am", "Temp3pm","MinTemp","MaxTemp"])


# converting rain columns

pd_df.loc[pd_df["RainToday"] == "Yes", "RainToday"] = 1
pd_df.loc[pd_df["RainTomorrow"] == "Yes", "RainTomorrow"] = 1

pd_df.loc[pd_df["RainToday"] == "No", "RainToday"] = 0
pd_df.loc[pd_df["RainTomorrow"] == "No", "RainTomorrow"] = 0




# Drop unnecessary columns
pd_df = pd_df.drop(columns = ["Date","Rainfall"])

# Rearranging the columns
pd_df = pd_df[['WindGustSpeed','cloud', 'pressure',
       'humidity', 'temp','RainToday', 'RainTomorrow']]

# Save Dataframe to CSV file
pd_df.to_csv("Aus_Weather_data_clean_final.csv")


pd_df.head()



























