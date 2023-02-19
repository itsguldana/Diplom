import pandastotensor as ptt
def xls_reader(filepath):
    df=pd.read_exel(filepath)
    tensor=ptt.df_to_tensor(df)
    return tensor
