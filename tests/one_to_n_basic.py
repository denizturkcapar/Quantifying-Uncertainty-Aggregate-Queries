import sys
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from src import one_to_n
#from src import helpers

import datetime
import textdistance
import editdistance
import pandas as pd
import networkx as nx

table_ex = one_to_n.convert_df("table_a.csv")
print(table_ex.head())
repeat = one_to_n.create_duplicates(table_ex, "aa", 3)