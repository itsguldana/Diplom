#!/usr/bin/python3
import csvreader
import txtreader
import xmlreader
import xlsreader
def universal_reader(finame):

    fsplit=finame.split('.')
    print(fsplit[1])
    if fsplit[1]=='csv':
        tensor=csvreader.csv_reader(finame)
    elif fsplit[1]=='txt':
        tensor=txtreader.txt_reader(finame)
    elif fsplit[1]=='tsv':
        tensor=tsvreader.tsv_reader(finame)
    elif fsplit[1]=='xml':
        tensor=xmlreader.xml_reader(finame)
    elif fsplit[1]=='json':
        tensor=jsonreader.json_reader(finame)
    elif fsplit[1]=='xls':
        tensor=xlsreader.xls_reader(finame)
    return tensor
if __name__ == "__main__":
        finame=input("File path:")
        tensor=universal_reader(finame)
        print(tensor)
