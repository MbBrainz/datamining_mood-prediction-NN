#%% [markdown] 
# Every thingh is going to be inside this 
# %% 
# imports
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import datetime
DATAFILE = "data/dataset_mood_smartphone.csv"
#%%

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
# %% [markdown]
# The dataframe is now without duplicates and the time column is recognised as a datetime object 
#  
# %% 
n_ids = raw_df.id.drop_duplicates().count()
n_vars = raw_df.variable.drop_duplicates().count()
print(f"The dataframe contains {n_ids} ids and {n_vars} variables.")
#%%
id_var_df = raw_df.groupby(by=["id", "variable"]).agg(["mean","count"]) # note the .count() here
display(id_var_df.head())
#%% [markdown]
# ## Completeness of the Data
# **The next table** computes the count, std, min, max and average(over ids) **number**(computed by .count() above) of datapoints per variable.
# in other words: the count column for example shows a number of times it counted a datapoint per id. if the dataset were complete, it would show ONLY 27 in this column.
# this is not the case however, which means not for all ids we have all the variable types available. 
# 
# It shows that there is a significant difference between the amount of data collected per id **AND**
# that some values are missing.
# To help understand: for example look at the first row: it counted 27 times that there are 
#%%
id_var_df.reset_index().drop(columns=['id']).groupby(by=["variable"]).agg(["count", "mean", "std", "min", "max"])["value"]

#%%
# %% 
# # Split the time into day and timestamp
# This might be USELESS
raw_df["date"] = raw_df["time"].apply(lambda x: x.date())
raw_df["year"] = raw_df["time"].apply(lambda x: x.year)
raw_df["month"] = raw_df["time"].apply(lambda x: x.month)
raw_df["hour"] = raw_df["time"].apply(lambda x: x.time().strftime('%H:%M:%S'))

# display(raw_df.head())

print(f" number of years: {raw_df.year.drop_duplicates()}") # All data is form 2014
print(f" number of months: \n {raw_df.month.drop_duplicates()}") # Only 5 months in dataset and highly skewed across months! #TODO: Discuss and eliminate data
sum_months = 1+8+61+204+3854  # this is not the total amount of data points... #TODO: Find out how this data is aggregated so we can verify the number
fract_june = 3854 / sum_months
print(f"{fract_june*100:.2f}% of the data is from june")

#%%
ax = sns.lineplot(data=raw_df[raw_df["variable"]=="mood"], x="hour", y="value", hue="id")
# sns.relplot(data=raw_df, x="hour", y="value", hue="id", col="variable", kind="line")
# ax.set_xticklabels(rotation = 30)
ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
labels = ax.get_xticklabels()
plt.setp(labels, rotation=45)
ax.grid(True)
plt.title("mood")
# %%
# Checking wether there are duplicate values per timeframe 
timecount_per_id_per_var = raw_df.groupby(by=["variable","id","time"]).count()
# timecount_per_id_per_var[timecount_per_id_per_var["datetime"] != 1]# %%
timecount_per_id_per_var.head()
# %%

circ_df = raw_df[(raw_df["variable"] == "circumplex.valence") | (raw_df["variable"] == "circumplex.arousal")]
# raw_df[raw_df["variable"] == "circumplex.valence"].groupby(by=["value"]).count()
# %%
# raw_df[raw_df["value"]=="mood"]
circ_df[(circ_df["value"] != -2) & (circ_df["value"] != -1) & (circ_df["value"] != 0) & (circ_df["value"] != 1) & (circ_df["value"] != 2) ]
# %%
circ_df[pd.isnull(circ_df["value"])] #TODO: Ask TA what to do with these and fix
# %%

# TODO: Boxx plot of all variables

# %% [markdown]
# # data division:
# - mood: final 
# - circumplex [arousal, valence]
# - countables [call, sms]
# - time based [ appCat.all ]
# - screen time 
#%%
# group by data types
# %%
# AppCat
appcat_df = raw_df.query('variable.str.startswith("appCat.")')
appcat_df["variable"] = appcat_df["variable"].apply(lambda x: x.removeprefix("appCat."))

apcat_list = appcat_df.variable.drop_duplicates().values
print(apcat_list)

# %%
appcat_avg_df = appcat_df.groupby(by=["id", "date", "variable"])["value"].sum().reset_index()
appcat_avg_df["day"] = appcat_avg_df["date"].apply(lambda x: pd.to_datetime(x).day_name())
appcat_avg_df.head()
#%%
#  ------------ PIEPLOT ---------------------
sorted_pie_df = pd.DataFrame(appcat_avg_df.groupby(by=["variable"])["value"].mean().reset_index())
# sorted_pie_df = sorted_pie_df.set_index('variable')
# ax = sorted_pie_df.plot.pie(y="value", autopct=lambda x: "yea")
values = (sorted_pie_df["value"].values/60).round(1)
total = sum(values)
explode=[0.05]* len(values)

plt.pie(sorted_pie_df["value"].values, labels=np.array((sorted_pie_df["value"].values/60).round(1)),radius=1.0, shadow=True,autopct='%1.1f%%', explode=explode)
plt.legend(sorted_pie_df["variable"].values,bbox_to_anchor=(1.04, 0.9), loc='upper left', borderaxespad=0)
plt.xlabel("usage in minutes")
plt.savefig("figures/stats_mean_app_usage.pdf", bbox_inches="tight")

# %%
# ax = sns.scatterplot(data=appcat_avg_df, x="variable", y="value", hue="id")
plot_id = "AS14.01"
ax = sns.relplot(data=appcat_avg_df[appcat_avg_df["id"]==plot_id].drop(columns=["date"]), x="variable", y="value", row="day")
# labels = ax.get_xticklabels()
# plt.setp(labels, rotation=45)
# ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
plt.title("time per app")

# %%
# COMPUTE the number of calls/msn per day per user 
callsms_df = raw_df[(raw_df["variable"] == "sms") |
                    (raw_df["variable"] == "call")]
callsms_m_day_df = callsms_df.groupby(by=["id", "date", "variable"])["value"].sum().reset_index()
# %%
