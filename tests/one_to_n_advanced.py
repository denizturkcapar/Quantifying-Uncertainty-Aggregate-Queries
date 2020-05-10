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

table_a = one_to_n.lat_convert_df("ACM.csv")

table_b = one_to_n.lat_convert_df("DBLP2.csv")


#repeat = one_to_n.create_duplicates(table_a, "id", 3)
#print("The duplicated table is as follows:", repeat, "\n")

now = datetime.datetime.now()
treshold_graph_maximal = one_to_n.valcomp_treshold_updated_maximal_construct_graph(table_a, table_b, 0.5)
timing_tresh = (datetime.datetime.now()-now).total_seconds()
print("---- Timing for Graph Construction with Treshold Constraint ----")
print(timing_tresh,"seconds")


"""
Outputs the matching that has the maximal weight for each edge in the bipartite graph
Input: A Bipartite Graph
Output: A set of matchings. Ex: {('journals/vldb/MedjahedBBNE03__2', '775457__1')}
"""
now = datetime.datetime.now()
matching_set = nx.algorithms.matching.max_weight_matching(treshold_graph_maximal)
timing_match = (datetime.datetime.now()-now).total_seconds()
print("---- Timing for Matching (Done on the graph constructed with the treshold constraint) ----")
print(timing_match,"seconds")
print("The number of edges in the graph is:", treshold_graph_maximal.number_of_edges(), "\n")


print("The Matching Set is:", matching_set, "\n")


out = one_to_n.collapse(matching_set)
print("The cleaned matching is:", out, "\n")

last = one_to_n.collapsed_dict(out)
print("The de-duplicated final result is:", last, "\n")


check = one_to_n.more_than_one(last)
print("The non 1:n matchings found in this dataset are:", check, "\n")
