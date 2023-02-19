import pandastotensor as ptt
def json_reader(filepath):
    df=pd.json_csv(filepath)
    tensor=ptt.df_to_tensor(df)
    return tensor