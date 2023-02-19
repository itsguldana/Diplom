import pandastotensor as ptt
def xml_reader(filepath):
    df=pd.read_xml(filepath)
    tensor=ptt.df_to_tensor(df)
    return tensor
