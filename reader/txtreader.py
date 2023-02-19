import pandastotensor as ptt
def txt_reader(filepath):
    df = pd.DataFrame()
    with open(filepath,'r') as txcxtfile:
        for f in txcxtfile:
            rl=f.readline()
            df.append(rl.split())
    
    tensor=ptt.df_to_tensor(df)
    return tensor