# -*- coding: utf-8 -*-
"""
Created on Sun Mar  5 13:02:05 2023

@author: SrijaSwarna
"""
#Importing necessary Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



# Load the data into a DataFrame
df = pd.read_csv("World_Development.csv")
df['Country Code'] = df['Country Code'].astype(str)


# Columns
df.columns = ["Country Name", "Country Code", "Series Name", "Series Code", "2012 [YR2012]", "2013 [YR2013]", "2014 [YR2014]","2015 [YR2015]", "2016 [YR2016]", "2017 [YR2017]", "2018 [YR2018]", "2019 [YR2019]", "2020 [YR2020]", "2021 [YR2021]"]

# Line Plot
df.plot(x="Country Code", y=["2012 [YR2012]",  "2013 [YR2013]", "2014 [YR2014]","2015 [YR2015]", "2016 [YR2016]", "2017 [YR2017]", "2018 [YR2018]", "2019 [YR2019]", "2020 [YR2020]", "2021 [YR2021]"])
plt.title("Unemployment percentage change in Ind, UK and USA")
plt.xlabel("Countries")
plt.ylabel("Unemployment %")
plt.show()

# Bar Chart
df.plot.bar(x="Series Name", y=["2012 [YR2012]",  "2013 [YR2013]", "2014 [YR2014]","2015 [YR2015]", "2016 [YR2016]", "2017 [YR2017]", "2018 [YR2018]", "2019 [YR2019]", "2020 [YR2020]", "2021 [YR2021]"])
plt.title("Unemployment change in Ind, UK and USA")
plt.xlabel("India UK USA")
plt.ylabel("Unemployment %")
plt.show()

# Scatter Plot
df_long = df.melt(id_vars=["Country Code"], 
                  value_vars=["2012 [YR2012]",  "2013 [YR2013]", "2014 [YR2014]","2015 [YR2015]", 
                              "2016 [YR2016]", "2017 [YR2017]", "2018 [YR2018]", "2019 [YR2019]", 
                              "2020 [YR2020]", "2021 [YR2021]"],
                  var_name="Year",
                  value_name="Unemployment Rate")

sns.scatterplot(data=df_long, x="Country Code", y="Unemployment Rate", hue="Year" )
plt.title("Unemployment change in Ind, UK and USA")
plt.xlabel("Countries")
plt.ylabel("Unemployment %")
plt.show()
