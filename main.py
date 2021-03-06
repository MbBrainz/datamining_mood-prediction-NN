#%% [markdown] 
# Every thingh is going to be inside this 
# %% 
# imports
from sklearn.preprocessing import MinMaxScaler
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


# %%
# Checking wether there are duplicate values per timeframe 
timecount_per_id_per_var = raw_df.groupby(by=["variable","id","time"]).count().sort_values("value", ascending=False)
print(timecount_per_id_per_var[timecount_per_id_per_var["value"] != 1].shape)# %%
timecount_per_id_per_var.head()
# %%
#TODO: Discuss the best way of aggregating the circumplex values. now just sums
circ_df = raw_df[(raw_df["variable"] == "circumplex.valence") | (raw_df["variable"] == "circumplex.arousal")]
circ_avg_df = circ_df.set_index("time") \
    .groupby(by=["id","variable"])["value"] \
    .resample("1D").sum().reset_index()
circ_avg_df= circ_avg_df.rename(columns={"time":"date"})
# raw_df[raw_df["variable"] == "circumplex.valence"].groupby(by=["value"]).count()
# %%
# raw_df[raw_df["value"]=="mood"]
circ_df[(circ_df["value"] != -2) & (circ_df["value"] != -1) & (circ_df["value"] != 0) & (circ_df["value"] != 1) & (circ_df["value"] != 2) ]

#%%
circ_df
# TODO: Make a choise what to do with circumplex data. same issue as with activity -> not every hour, lot of missing values
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

# appcat_list = raw_df[]

appcat_list = [x for x in raw_df["variable"].drop_duplicates().tolist() if x.startswith("appCat.")]
appcat_df = raw_df[raw_df["variable"].isin(appcat_list)]
# appcat_df = raw_df.query(variable.str.startswith("appCat."))
appcat_df["variable"] = appcat_df["variable"].apply(lambda x: x.removeprefix("appCat."))

print(appcat_list)

appcat_df
# %% [markdown]
# # Negative Outliers
# This appcat dataset has 4 neative outliers. those are shown and filtered from raw_df in the next cell
# As there are multiple possible reasons for these values to be negative (like integer overflow, device error, or anything) the solution 
#%%
display(appcat_df[appcat_df["value"] < 0])

appcat_df = appcat_df[appcat_df["value"] > 0]


#%%
appcat_avg_df = appcat_df.set_index("time") \
    .groupby(by=["id","variable"])["value"] \
    .resample("1D").sum().reset_index()
appcat_avg_df= appcat_avg_df.rename(columns={"time":"date"})

appcat_avg_df["day"] = appcat_avg_df["date"].apply(lambda x: pd.to_datetime(x).day_name())
appcat_avg_df[(appcat_avg_df["variable"] == "builtin") & (appcat_avg_df["value"] == 0)]

#%%
#  ------------ PIEPLOT ---------------------
sorted_pie_df = pd.DataFrame(appcat_avg_df.groupby(by=["variable"])["value"].mean().reset_index())
# sorted_pie_df = sorted_pie_df.set_index('variable')
# ax = sorted_pie_df.plot.pie(y="value", autopct=lambda x: "yea")
values = (sorted_pie_df["value"].values/60).round(1)
total = sum(values)
explode= [0.05]* len(values)

plt.pie(sorted_pie_df["value"].values, labels=np.array((sorted_pie_df["value"].values/60).round(1)),radius=1.0, shadow=True,autopct='%1.1f%%', explode=explode)
plt.legend(sorted_pie_df["variable"].values,bbox_to_anchor=(1.04, 0.9), loc='upper left', borderaxespad=0)
plt.xlabel("usage in minutes")
# plt.savefig("figures/stats_mean_app_usage.pdf", bbox_inches="tight")

# %%
# ax = sns.scatterplot(data=appcat_avg_df, x="variable", y="value", hue="id")
weekday_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
plot_id = "AS14.01"
# # Per specific user
# ax = sns.catplot(data=appcat_avg_df[appcat_avg_df["id"]==plot_id].drop(columns=["date"]), x="day", y="value", row="variable", kind="box")
sns.catplot(data=appcat_avg_df[["id","day", "variable","value"]], x="day", y="value", row="variable", kind="box", order=weekday_order)
# plt.savefig("figures/box-plot-app-per-weekday.pdf", bbox_inches="tight")
# ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
# plt.title("time per app")
#%%
# TODO: Check with TA if the log(x+1) solution is valid
sns.set_theme(style="ticks", palette="tab10")
appcat_avg_df["log(value)"] = np.log(appcat_avg_df["value"]+1)
appcat_log_df = appcat_avg_df
appcat_log_df["value"] = np.log(appcat_avg_df["value"]+1)

ax = sns.violinplot(data=appcat_avg_df.drop(columns=["date"]), x="variable", y="log(value)", showfliers = True )
sns.despine(offset=10, trim=True)
labels = ax.get_xticklabels()
plt.setp(labels, rotation=45)
plt.tight_layout()
plt.savefig("figures/violinplot_app_usage_log(x+1).pdf", bbox_inches="tight")


#%%


# %%
# COMPUTE  the number of calls/msn per day per user 
callsms_df = raw_df[(raw_df["variable"] == "sms") |
                    (raw_df["variable"] == "call")]

callsms_avg_df = callsms_df.set_index("time") \
    .groupby(by=["id","variable"])["value"] \
    .resample("1D").sum().reset_index()
    
callsms_avg_df = callsms_avg_df.rename(columns={"time":"date"})
callsms_avg_df
# callsms_avg_df.groupby(by=["id","variable"]).describe()
# %%
# TODO: Handle activity properly
activity_df = raw_df[raw_df["variable"] == "activity"].sort_index() #.groupby(by=["id","date"]).count()
activity_avg_df = activity_df.set_index("time") \
    .groupby(by=["id","variable"])["value"] \
    .resample("1D").sum().reset_index()
activity_avg_df = activity_avg_df.rename(columns={"time":"date"})


# %%
# TODO: handle screentime properly
raw_df[raw_df["variable"] == "screen"].sort_index() #.groupby(by=["id","date"]).count()
screen_df = raw_df[(raw_df["variable"] == "screen")]

screen_avg_df = screen_df.set_index("time") \
    .groupby(by=["id","variable"])["value"] \
    .resample("1D").sum().reset_index()
screen_avg_df= screen_avg_df.rename(columns={"time":"date"})
screen_avg_df


#%% 
mood_df = raw_df[raw_df["variable"] == "mood"]

mood_avg_df = pd.DataFrame(mood_df.set_index("time") \
    .groupby(by=["id","variable"])["value"] \
    .resample("1D").mean())

mood_avg_df = mood_avg_df.fillna(method='ffill', limit=2).reset_index()
mood_avg_df = mood_avg_df.rename(columns={"time":"date"})

# ---------[<user-id>, <index of first non-nan>, <remove rows>]
NANuser1 = ["AS14.01", 23, 23]
NANuser2 = ["AS14.12", 424, 12]
NANuser3 = ["AS14.17", 691, 17]
NANusers= [NANuser1, NANuser2, NANuser3]
remove_idxs = []
for u in NANusers:
    idxs = np.arange(u[1]-u[2], u[1])
    remove_idxs += list(idxs)

# remove_idxs
mood_avg_df = mood_avg_df.drop(remove_idxs)




#%%
#valence data processing
circumplex_df = raw_df[(raw_df["variable"] == "circumplex.valence") | (raw_df["variable"] == "circumplex.arousal")]
circumplex_df["value"] = circumplex_df["value"] + 2

circumplex_avg_df = circumplex_df.set_index("time")\
    .groupby(by=["id","variable"])["value"] \
    .resample("1D").mean().reset_index()

# first forward fill with limit of 2 for intermediate nan values, then set all nan values left to average(2)
circumplex_avg_df['value'] = circumplex_avg_df['value'].fillna(method='ffill', limit=2)
circumplex_avg_df['value'] = circumplex_avg_df['value'].replace(np.nan, 2)
circumplex_avg_df= circumplex_avg_df.rename(columns={"time":"date"})

# circumplex_avg_df

# %%
# concat. all the separately processed data
avg_day_df = pd.DataFrame(pd.concat([callsms_avg_df, appcat_log_df, screen_avg_df, mood_avg_df, circumplex_avg_df, activity_avg_df ])[["id","date","variable", "value"]])

# creating columns from all variables 
raw_train_df = avg_day_df.pivot(index=["id","date"],columns=["variable"], values=["value"])["value"]
raw_train_df
# IMPORTANT: after the pivot step here, there are some NAN values at mood. These values are all in the beginning or 
#               the end of a period of a user, therefore we can safely discard those values

# %%
# # checking if my assumption is correct:
# # all the values are in the beginning of the dataset and if the builtin is zero, then all the other appcat values are also zero.
# raw_train_df[pd.isnull(raw_train_df["builtin"])]

# removing rows which have builtin = NAN and/or mood NAN -> there is no phone data available for those days, only in the beginning
train_df = pd.DataFrame(raw_train_df[ (~pd.isnull(raw_train_df["builtin"])) ])
train_df = pd.DataFrame(train_df[ (~pd.isnull(train_df["mood"])) ])


# %%
# very fancy way to put the mood column last :p
train_df = train_df[[c for c in train_df if c not in ['mood']]+ ['mood']]
train_df = train_df.fillna(0)
display(train_df)

# %%


scaler = MinMaxScaler(feature_range=(0,1))

# This applies a scalar transform per id per column
df_scaled = train_df.groupby(level=0).apply(lambda x : pd.DataFrame(scaler.fit_transform(x), columns=x.columns, index=x.index).round(5))
# df_scaled = train_df.reset_index(drop=True).apply(lambda x : pd.DataFrame(scaler.fit_transform(x.reshape(1,-1)), columns=x.columns, index=x.index).round(5))
# df_scaled = train_df.groupby(level=0).apply(lambda x : pd.DataFrame(scaler.fit_transform(x), columns=x.columns, index=x.index).round(5))
df_scaled

#%%
df_scaled.to_csv("data/train_data_v1.csv")

