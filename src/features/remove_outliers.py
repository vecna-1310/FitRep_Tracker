import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy
from sklearn.neighbors import LocalOutlierFactor  

# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------
df = pd.read_pickle("../../data/interim/01_data_processed.pkl")

outlier_columns=list(df.columns[:6])
# --------------------------------------------------------------
# Plotting outliers
# --------------------------------------------------------------
plt.style.use("fivethirtyeight")
plt.rcParams["figure.figsize"]=(20,5)
plt.rcParams["figure.dpi"]=100



df[["acc_y","label"]].boxplot(by="label",figsize=(20,10))
plt.show()

df[["gyr_y","label"]].boxplot(by="label",figsize=(20,10))
plt.show()

df[outlier_columns[:3] + ["label"]].boxplot(by="label", figsize=(20, 10),layout=(1,3))
plt.show()

df[outlier_columns[3:] + ["label"]].boxplot(by="label", figsize=(20, 10),layout=(1,3))
plt.show()


def plot_binary_outliers(dataset, col, outlier_col, reset_index):


    dataset = dataset.dropna(axis=0, subset=[col, outlier_col])
    dataset[outlier_col] = dataset[outlier_col].astype("bool")

    if reset_index:
        dataset = dataset.reset_index()

    fig, ax = plt.subplots()

    plt.xlabel("samples")
    plt.ylabel("value")

    # Plot non outliers in default color
    ax.plot(
        dataset.index[~dataset[outlier_col]],
        dataset[col][~dataset[outlier_col]],
        "+",
    )
    # Plot data points that are outliers in red
    ax.plot(
        dataset.index[dataset[outlier_col]],
        dataset[col][dataset[outlier_col]],
        "r+",
    )

    plt.legend(
        ["outlier " + col, "no outlier " + col],
        loc="upper center",
        ncol=2,
        fancybox=True,
        shadow=True,
    )
    plt.show()

# --------------------------------------------------------------
# Interquartile range (distribution based)
# --------------------------------------------------------------

# Insert IQR function

def mark_outliers_iqr(dataset, col):
    

    dataset = dataset.copy()

    Q1 = dataset[col].quantile(0.25)
    Q3 = dataset[col].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    dataset[col + "_outlier"] = (dataset[col] < lower_bound) | (
        dataset[col] > upper_bound
    )

    return dataset
# Plot a single column

col ="acc_x"
dataset = mark_outliers_iqr(df,col)
plot_binary_outliers(dataset = dataset, col = col,outlier_col =col+"_outlier",reset_index =True)
# Loop over all columns
for col in outlier_columns:
  dataset = mark_outliers_iqr(df,col)
  plot_binary_outliers(dataset = dataset, col = col,outlier_col =col+"_outlier",reset_index =True)
  

# --------------------------------------------------------------
# Chauvenets criteron (distribution based)
# --------------------------------------------------------------


# Check for normal distribution

df[outlier_columns[:3] + ["label"]].plot.hist(by="label", figsize=(20, 20),layout=(3,3))
plt.show()

df[outlier_columns[3:] + ["label"]].plot.hist(by="label", figsize=(20, 20),layout=(3,3))
plt.show()

# Insert Chauvenet's function

def mark_outliers_chauvenet(dataset, col, C=2):
    dataset = dataset.copy()
    # Compute the mean and standard deviation.
    mean = dataset[col].mean()
    std = dataset[col].std()
    N = len(dataset.index)
    criterion = 1.0 / (C * N)

    # Consider the deviation for the data points.
    deviation = abs(dataset[col] - mean) / std

    # Express the upper and lower bounds.
    low = -deviation / math.sqrt(C)
    high = deviation / math.sqrt(C)
    prob = []
    mask = []

    # Pass all rows in the dataset.
    for i in range(0, len(dataset.index)):
        # Determine the probability of observing the point
        prob.append(
            1.0 - 0.5 * (scipy.special.erf(high[i]) - scipy.special.erf(low[i]))
        )
        # And mark as an outlier when the probability is below our criterion.
        mask.append(prob[i] < criterion)
    dataset[col + "_outlier"] = mask
    return dataset
# Loop over all columns
for col in outlier_columns:
  dataset = mark_outliers_chauvenet(df,col)
  plot_binary_outliers(dataset = dataset, col = col,outlier_col =col+"_outlier",reset_index =True)

# --------------------------------------------------------------
# Local outlier factor (distance based)
# --------------------------------------------------------------

# Insert LOF function

def mark_outliers_lof(dataset, columns, n=20):
    
    dataset = dataset.copy()

    lof = LocalOutlierFactor(n_neighbors=n)
    data = dataset[columns]
    outliers = lof.fit_predict(data)
    X_scores = lof.negative_outlier_factor_

    dataset["outlier_lof"] = outliers == -1
    return dataset, outliers, X_scores
# Loop over all columns
 dataset,outliers, X_scores = mark_outliers_lof(df,outlier_columns)
for col in outlier_columns:
 
  plot_binary_outliers(dataset = dataset, col = col,outlier_col ="outlier_lof",reset_index =True)
# --------------------------------------------------------------
# Check outliers grouped by label
# --------------------------------------------------------------
label ="dead"
for col in outlier_columns:
  dataset = mark_outliers_iqr(df[df["label"]==label],col)
  plot_binary_outliers(dataset,col,col+"_outlier",reset_index =True)
  
for col in outlier_columns:
  dataset = mark_outliers_chauvenet(df[df["label"]==label],col)
  plot_binary_outliers(dataset,col,col+"_outlier",reset_index =True)
  

dataset,outliers, X_scores = mark_outliers_lof(df[df["label"]==label],outlier_columns)
for col in outlier_columns:
 
  plot_binary_outliers(dataset = dataset, col = col,outlier_col ="outlier_lof",reset_index =True)
# --------------------------------------------------------------
# Choose method and deal with outliers
# --------------------------------------------------------------



# Test on single column
col ="gyr_z"
dataset = mark_outliers_chauvenet(df,col=col)
dataset[dataset["gyr_z_outlier"]]

dataset.loc[dataset["gyr_z_outlier"],"gyr_z"]=np.nan
# Create a loop
outliers_remove_df = df.copy()
for col in outlier_columns:
  for label in df["label"].unique():
    dataset = mark_outliers_chauvenet(df[df["label"]==label],col)
    dataset.loc[dataset[col + "_outlier"],col]=np.nan
    #update columns in original data frames
    outliers_remove_df.loc[(outliers_remove_df["label"]==label),col]=dataset[col]
    n_outliers =len(dataset)-len(dataset[col].dropna())
    print(f"Removed {n_outliers} from {col} for {label} ")
    
  

outliers_remove_df.info()
# --------------------------------------------------------------
# Export new dataframe
# --------------------------------------------------------------
outliers_remove_df.to_pickle("../../data/interim/02_outliers_removed_chauvenets.pkl")
