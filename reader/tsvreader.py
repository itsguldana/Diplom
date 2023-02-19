import pandastotensor as ptt
def tsv_reader(filepath):
    df=pd.read_tsv(filepath)
    tensor=ptt.df_to_tensor(df)
    return tensor
