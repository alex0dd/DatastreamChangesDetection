import pandas as pd

def load_cavitation_single_type(base_path, fnames_list, verbose=0):
    dfs_to_concat = []
    for fname in fnames_list:
        fpath = base_path + fname + ".csv"
        if verbose:
            print("Loading", fpath)
        df = pd.read_csv(fpath, sep='\t')
        if verbose:
            print("Loaded", fpath)
        dfs_to_concat.append(df)
    df_merged = pd.concat(dfs_to_concat, sort=False)
    # free memory
    del dfs_to_concat
    return df_merged
    

def load_cavitation_data(base_path, data_csvs_dict, verbose=0):
    """
    Given a dictionary indexed by status (OK, IN, STANDING), containing list of csv filenames.
    Returns a pandas dataframe containing all the cavitation data
    """
    df_okay = load_cavitation_single_type(base_path, data_csvs_dict["OK"], verbose=verbose)
    df_okay["status"] = 0
    df_in = load_cavitation_single_type(base_path, data_csvs_dict["IN"], verbose=verbose)
    df_in["status"] = 1
    df_standing = load_cavitation_single_type(base_path, data_csvs_dict["STANDING"], verbose=verbose)
    df_standing["status"] = 2
    df_merged = pd.concat([df_okay, df_in, df_standing], sort=False)
    del df_okay, df_in, df_standing
    # convert time string to datetime type
    df_merged["time"] = pd.to_datetime(df_merged["time"])
    df_merged = df_merged.drop(columns=["cycle"])
    return df_merged
    
def get_cavitation_features(df):
    """
    Given a dataframe containing cavitation data, performs a group by on "time" field
    to condense 75000 observations per second into a single feature 
    (e.g. mean, std, skew, kurtosis)
    """
    grouped_by_df = df.groupby("time")
    df_mean = grouped_by_df.mean()
    df_std = grouped_by_df.std()
    df_skewness = grouped_by_df.skew()
    df_kurtosis = grouped_by_df.apply(pd.DataFrame.kurt)
    del grouped_by_df
    return {
        "mean": df_mean,
        "std": df_std,
        "skewness": df_skewness,
        "kurtosis": df_kurtosis,
    }
    
def average_signal_columns(df):
    df["average_column"] = (df["P1_x"] + df["P1_y"] + df["P1_z"] + df["P2_x"] + df["P2_y"] + df["P2_z"])/6
    return df.drop(columns=["P1_x", "P1_y", "P1_z", "P2_x", "P2_y", "P2_z"])

    
