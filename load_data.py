import json
import codecs

def load_CLINC150():
    readfile = codecs.open('/export/home/Dataset/CLINC150/data_full.json', 'r', 'utf-8')

    file2dict =  json.load(readfile)
    for key, value in file2dict.items():
        print(key, len(value))

if __name__ == "__main__":
    load_CLINC150()
