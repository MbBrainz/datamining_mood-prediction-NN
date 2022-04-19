#%%
# imports
from curses import raw
from email import header
from os import stat
from pickle import TRUE
from pstats import Stats
from re import X
from sre_constants import GROUPREF
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
DATAFILE = "data/dataset_mood_smartphone.csv"
# %%
raw_df = pd.read_csv(DATAFILE, index_col=0)
nrows_dupl = raw_df.shape[0]
raw_df = raw_df.replace()

# convert time variable from str type to datetime object type
raw_df["time"] = pd.to_datetime(raw_df["time"], infer_datetime_format=True, exact="ms") #this doesn't influances the number of duplicates removed. (check by reordering this line)

raw_df = raw_df.drop_duplicates()
nrows = raw_df.shape[0]
ndupl = nrows_dupl-nrows
print(f"The dataframe has {nrows} rows. ({ndupl} duplicates removed)")
raw_df.head()

# %%
#converting the time ti date time 
raw_df["time"] = pd.to_datetime(raw_df["time"], infer_datetime_format=True, exact="ms")

#%%
#getting the names of all the individuals who participated in the study 
ids = raw_df.loc[:,["id"]]

individuals = ids.drop_duplicates(subset="id", keep='first', inplace=False)

arrayind = []

for (columnName, columnData) in individuals.iteritems():
    arrayind.append(columnData.values)

Individuals_list  = []

for i in arrayind:
  for j in i:
    Individuals_list.append(j)

#%%
#creating the list of variables

Variables = raw_df.loc[:,["variable"]]
Vars = Variables.drop_duplicates(subset="variable", keep='first', inplace=False)
Vars

Variable_A= []

for (columnName, columnData) in Vars.iteritems():
    Variable_A.append(columnData.values)

Vars_list  = []

for i in Variable_A:
  for j in i:
    Vars_list.append(j)

Vars_list

#%%
#creating a dataframe for the first entry

vars_val1 = raw_df.loc[:,["id","variable","value"]]

id1 = vars_val1[vars_val1["id"] == "AS14.01"]

group1 = id1.groupby(by = ['variable']).agg('mean')
print(group1)

#%%
#creating a dataframe for the average values
df = pd.DataFrame({'Variable': group1.reset_index().variable , 'AS14.01': group1.reset_index().value})
df

#%%
##############################
group = raw_df.groupby(by = ["id","variable"]).mean().reset_index()
groups = group[group["variable"] == "mood"]

df2 = pd.DataFrame({'User': Individuals_list, 'mood': groups.reset_index().value})
df2
#%%
for i in Vars_list:
  if i ==  "mood":
    print("skip mood")
  else:
    groupsloop= group[group["variable"] == i]
    idlist = []
    dlist = []
    for j in groupsloop.reset_index().id:
      idlist.append(j)

    for k in Individuals_list:
      if (k in idlist) == True:
        id_val = groupsloop[groupsloop["id"] == k]
        dlist.append(id_val.reset_index().value[0])
      else:
        dlist.append('')
    df2["{}".format(i)] = dlist

#%%
df2
#%%
fig, axs = plt.subplots(ncols=3)
sns.regplot(x='User', y='mood', data=df2, ax=axs[1])
sns.regplot(x='User', y='circumplex.arousal	', data=df2, ax=axs[2])
# sns.boxplot(x='education',y='wage', data=df_melt, ax=axs[2])

#%%
newd = df.set_index('Variable').transpose()
newd["mood"]
#%%
sea = sns.FacetGrid(newd, col = "mood")

sea.map(sns.regplot, "activity", "appCat.builtin	", color = ".3")

newd
plt.figure(figsize=(10,10))
i = sns.regplot(x="Users", y="activity", data=newd)
i.set_xticklabels(i.get_xticklabels(),rotation=85)
# fig, axs = plt.subplots(ncols=3)
# sns.regplot(x='value', y='wage', data=df_melt, ax=axs[0])
# sns.regplot(x='value', y='wage', data=df_melt, ax=axs[1])
# sns.boxplot(x='education',y='wage', data=df_melt, ax=axs[2])


# sns.pairplot(newd)
#%%
##############################################################

#%%
vars_val = raw_df.loc[:,["id", "variable","value"]]
#%%
#normalizing arousal data
g = vars_val[vars_val["variable"] == "circumplex.arousal"]
h = g['value'] + 2
g["addition"] = h

g["addition"]/h.sum()

g["nomralized"] = g["addition"]/h.sum()
g
#%%
#normalizing valence data 
i = vars_val[vars_val["variable"] == "circumplex.valence"]
i.min()
j = i["value"] + 2
i["addition"] = j
i["nomralized"] = i["addition"]/j.sum()
i
#%%

#appCAT
#builtin
builtin = vars_val[vars_val["variable"] == "appCat.builtin"]
#communications 
communication = vars_val[vars_val["variable"] == "appCat.communication"]
#entertainment 
entertainment = vars_val[vars_val["variable"] == "appCat.entertainment"]
#finance
finance = vars_val[vars_val["variable"] == "appCat.finance"]
#game
game = vars_val[vars_val["variable"] == "appCat.game"]
#office
office = vars_val[vars_val["variable"] == "appCat.office"]
#office
other = vars_val[vars_val["variable"] == "appCat.other"]
#social
social = vars_val[vars_val["variable"] == "appCat.social"]
#travel
travel = vars_val[vars_val["variable"] == "appCat.travel"]
#unknown
unknown = vars_val[vars_val["variable"] == "appCat.unknown"]
#utilities
utilities = vars_val[vars_val["variable"] == "appCat.utilities"]
# weather
weather = vars_val[vars_val["variable"] == "appCat.weather"]
#%%
# sns.set_theme(style="whitegrid")
# sns.lineplot(data=raw_df[raw_df["variable"]=="activity"], palette="tab10", linewidth=2.5)

#%%
################################################################
#Looking at the averages of the users per day 
raw_df['Day'] = raw_df['time'].dt.day
Days_list = list(range(1,32))

day_vals = raw_df.loc[:,["Day","id","variable","value"]]
day_vals.sort_values( by = "Day")

d1 = day_vals[day_vals["Day"] == 1]
d1

#%%

d11 = d1.groupby(by = ["id","variable"]).mean().reset_index()
d111 = d11[d11["variable"] == 'mood']

df_perday = pd.DataFrame({'User': Individuals_list, 'mood day1': d111.reset_index().value})

#%%

#%%

for i in Vars_list:
  for d in Days_list:
    day_num = day_vals[day_vals["Day"] == d]
    if d ==  1 and i == "mood":
      print("skip day 1 mood1 ")
    else:
      day_num_l= day_num[day_num["variable"] == i]
      idlist = []
      dlist = []
      for j in day_num_l.reset_index().id:
        idlist.append(j)

      for k in Individuals_list:
        if (k in idlist) == True:
          id_val = day_num_l[day_num_l["id"] == k]
          dlist.append(id_val.reset_index().value[0])
        else:
          dlist.append('')
      df_perday["{} Day {}".format(i, d)] = dlist

#%%
df_perday
#%%
#creating the data frame for the average usage of apps for all users

Dayscol = raw_df.loc[:,["Day"]]

Days = Dayscol.drop_duplicates(subset="Day", keep='first', inplace=False)

Days.sort_values(by = "Day")

Days_list = list(range(1,32))

groupday1 = d1.groupby(by = ['variable']).agg('mean')

DFUser1 = pd.DataFrame({'Variable': groupday1.reset_index().variable , 'Average for Day 1 ': groupday1.reset_index().value})

DFUser1

for i in Individuals_list:
  day = day_vals[day_vals["id"] == i]
  day_sorted = day.sort_values(by= "Day")

  for j in Days_list:
    d = day_sorted[day_sorted["Day"] == j]
    grouptoadd = d.groupby(by = ['variable']).agg('mean')
    DFUser1["Average for day {} User {}".format(j,i)] = grouptoadd.reset_index().value

DFUser1
#%%

list(df_perday.columns)
# %%
