#%%
# imports
from curses import raw
from email import header
from os import stat
from pstats import Stats
from re import X
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
#creating a dataframe for the first entry

vars_val1 = raw_df.loc[:,["id","variable","value"]]

id1 = vars_val1[vars_val1["id"] == "AS14.01"]

group1 = id1.groupby(by = ['variable']).agg('mean')
 
group1.reset_index().variable

df = pd.DataFrame({'Variable': group1.reset_index().variable , 'Average User AS14.01': group1.reset_index().value})


#%%

for i in Individuals_list:
  if i ==  "AS14.01":
    print("skip this bugger AS14.01")
  else:
    id = vars_val1[vars_val1["id"] == i ]
    group = id.groupby(by = ['variable']).agg('mean')
    df[i] = group.reset_index().value
  
#%%
import seaborn as sns
#%%
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



vars_val = raw_df.loc[:,["id", "variable","value"]]
sns.pairplot(vars_val)

# mood = id1[id1["variable"] == "mood"]

# mood
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

dfpairs = pd.DataFrame({'mood': mood.value })



# dfpairs['builtin'] = communication.value

com = communication.value.to_list()

dfpairs['builtin'] = communication.value.to_list()

dfpairs

#%%


for i in Vars_list:

  plt.figure(figsize=(10,5))
  i = sns.lineplot(x="id", y="value", data=raw_df[raw_df["variable"]==i])
  i.set_ylabel(i, fontsize = 15)
  i.set_xticklabels(i.get_xticklabels(),rotation=90)

#%%
ax = sns.boxplot(data=raw_df[raw_df["variable"]=="appCat.builtin"], x="id", y="value")
ax.set_xticklabels(ax.get_xticklabels(),rotation=80)

# Initialize the figure
f, ax1 = plt.subplots()

# Show each observation with a scatterplot
ax1 = sns.lineplot(x="id", y="value", data=raw_df[raw_df["variable"]=="appCat.builtin"])
ax1.set_xticklabels(ax1.get_xticklabels(),rotation=90)

#%%

df

# sns.set_theme(style="whitegrid")
# sns.lineplot(data=raw_df[raw_df["variable"]=="activity"], palette="tab10", linewidth=2.5)

#%%
transpose = df.T

transpose.plot.box()

#trying to create plots of the activites and the individuals 





# plt.scatter(x = Individuals_list, y = a)
# one = df.iloc[0]

# vals = []

# for i in range(1,len(one)):
#   vals.append(one[i])

# plt.scatter(x = Individuals_list, y = vals)
# plt.xticks(rotation = 75)

#%%
#Looking at the averages of the users per day 
raw_df['Day'] = raw_df['time'].dt.day

day_vals = raw_df.loc[:,["Day","id","variable","value"]]

day1 = day_vals[day_vals["id"] == "AS14.01"]

day1_sorted = day1.sort_values(by= "Day")

day1_sorted.head()

d1 = day1_sorted[day1_sorted["Day"] == 1]
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
#number of days used for the Users
Dayscol = raw_df.loc[:,["Day"]]

Days = Dayscol.drop_duplicates(subset="Day", keep='first', inplace=False)

Days.sort_values(by = "Day")

Days_list = list(range(1,32))

# day1_sorted = day1.sort_values(by= "Day")

# number_days = day1_sorted.Day

# num_days = number_days.drop_duplicates(subset="Day", keep='first', inplace=False)

# number_days

#%%

M = id1[id1["variable"] =='mood']
M['value'].mean()
# id1_var = id1.groupby(by=["id","variable"]).agg(["mean","count"])

# id1_var.index[1:2]
#%%

# ids_ind = vars_val1.groupby(by = ["id", "variable"])

# ids_ind.loc[:,1:10]

#%%

idsind = raw_df.groupby(by=[raw_df['id']])

id1 = idsind[idsind['id'] == "AS14.01"]

#%%

id1 = raw_df.id['AS14.01']



#%%

idsind1 = raw_df.groupby(by=["id","variable"]).agg(["mean"])

idsind1

id1_var1 = id1.groupby(by=["id","variable"]).agg(["mean"])



for i in range(len(id1_var)) :
  print(id1_var.iloc[i, 0])

#%%
vars_val = raw_df.loc[:,["variable","value"]]
mood = vars_val[vars_val["variable"] == "mood"]
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
