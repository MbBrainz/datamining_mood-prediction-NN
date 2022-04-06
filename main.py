#%% [markdown] 
# Every thingh is going to be inside this 
# %% 
# imports
import pandas as pd
import seaborn as sns
import numpy as np
DATAFILE = "data/dataset_mood_smartphone.csv"
#%%

raw_df = pd.read_csv(DATAFILE, index_col=0)
nrows_dupl = raw_df.shape[0]

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
print(f"The dataframe contains {n_ids} ids and {n_vars}.")
#%%
id_var_df = raw_df.groupby(by=["id", "variable"]).agg(["mean","count"]) # note the .count() here
display(id_var_df.head())
#%% [markdown]
# ## Completeness of the Data
# **The next table** computes the count, std, min, max and average(over ids) **number**(computed by .count() above) of datapoints per variable.
# in other words: the count column for example shows a number of times it counted a datapoint per id. if the dataset were complete, it would show ONLY 27 in this column.
# this is not the case however, which means not for all ids we have all the variable types available. 
# 
# #TODO: decide on which variable we should apply what to do with these "structural inconsistencies"
# #TODO: determine which of these values are redundant aka analyse single variables piece by piece
# #TODO: Check for nan/corrupt values
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

raw_df.year.drop_duplicates() # All data is form 2014
raw_df.month.drop_duplicates() # Only 5 months in dataset and highly skewed across months! #TODO: Discuss and eliminate data
sum_months = 1+8+61+204+3854  # this is not the total amount of data points... #TODO: Find out how this data is aggregated so we can verify the number
fract_june = 3854 / sum_months
print(f"{fract_june*100:.2f}% of the data is from june")

#%%
sns.relplot(data=raw_df[raw_df["variable"]=="mood"], x="hour", y="value", hue="id", kind="line")
# sns.relplot(data=raw_df, x="hour", y="value", hue="id", col="variable", kind="line")

# %%
# Checking wether there are duplicate values per timeframe 
timecount_per_id_per_var = raw_df.groupby(by=["variable","id","time"]).count()
timecount_per_id_per_var[timecount_per_id_per_var["datetime"] != 1]# %%

# %%

# %%
