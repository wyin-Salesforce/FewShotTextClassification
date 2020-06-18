import json
import codecs
from tabula import read_pdf

def load_CLINC150():
    readfile = codecs.open('/export/home/Dataset/CLINC150/data_full.json', 'r', 'utf-8')

    file2dict =  json.load(readfile)
    for key, value in file2dict.items():
        print(key, len(value))


def read_supplementary():
    content = read_pdf('/export/home/Dataset/CLINC150/supplementary.pdf', output_format='json', pages=2)
    print(content)

if __name__ == "__main__":
    # load_CLINC150()
    read_supplementary()
