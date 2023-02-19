import pandastotensor as ptt
def csv_reader(filepath):
    df=pd.read_csv(filepath)
    tensor=ptt.df_to_tensor(df)
    return tensor
