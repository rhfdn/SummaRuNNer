# Convert pandas dataframe to numpy array
def convert_pandas_df_to_array(df):
    arr = []
    for i in range(df.shape[0]):
        arr.append({col : df.iloc[i][col] for col in df.columns})
    return arr
