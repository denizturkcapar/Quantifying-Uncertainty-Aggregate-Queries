
import platform
import csv

#defines an iterator over the google catalog
class Catalog():

    def __init__(self, filename):
      self.filename = filename
      f = open(self.filename, 'r', encoding = "ISO-8859-1")
      reader = csv.reader(f, delimiter=',', quotechar='"')
      self.length = 0
      for row in reader:
        self.length += 1

    def __iter__(self):
      f = open(self.filename, 'r', encoding = "ISO-8859-1")
      self.reader = csv.reader(f, delimiter=',', quotechar='"')
      next(self.reader)
      return self

    def __next__(self):
      row = next(self.reader)
      return {'id': row[0],
               'name': row[1],
               'description': row[2],
               'mfg': row[3],
               'price': row[4]
              }
def data_catalog(filename):
    return Catalog(filename)

def google_catalog():
    return Catalog('Matching/data/GoogleProducts.csv')

def amazon_catalog():
    return Catalog('Matching/data/Amazon.csv')


def eval_matching(matching):
    f = open('Matching/data/Amzon_GoogleProducts_perfectMapping.csv', 'r', encoding = "ISO-8859-1")
    reader = csv.reader(f, delimiter=',', quotechar='"')
    matches = set()
    proposed_matches = set()

    tp = set()
    fp = set()
    fn = set()
    tn = set()

    for row in reader:
        matches.add((row[0],row[1]))

    for m in matching:
        proposed_matches.add(m)

        if m in matches:
            tp.add(m)
        else:
            fp.add(m)

    for m in matches:
        if m not in proposed_matches:
            fn.add(m)
    print("TP: ", len(tp))
    print("FP: ", len(fp))

    try:
        prec = len(tp)/(len(tp) + len(fp))
    except ZeroDivisionError:
        prec = 0
    
    try:
        rec = len(tp)/(len(tp) + len(fn))
    except ZeroDivisionError:
        rec = 0
    
    try:
        accuracy = round(2*(prec*rec)/(prec+rec),2)
    except ZeroDivisionError:
        accuracy = 0

    print(prec)
    print(rec)

    return {'false positive': round(1-prec,2), 
            'false negative': round(1-rec,2),
            'accuracy': accuracy }
