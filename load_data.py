import json
import codecs

def load_CLINC150():
    readfile = codecs.open('/export/home/Dataset/CLINC150/data_full.json', 'r', 'utf-8')

    file2dict =  json.load(readfile)
    list_of_list = file2dict.get('oos_val')
    size = len(list_of_list)
    print(size)

if __name__ == "__main__":
    load_CLINC150()
