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

table_a = one_to_n.convert_df("table_a.csv")
table_b = one_to_n.convert_df("table_b.csv")
repeat = one_to_n.create_duplicates(table_a, "aa", 3)
print("The duplicated table is as follows:\n", repeat, "\n")

"""
*******************************************************************
MAXIMAL MATCHING
*******************************************************************
"""
now = datetime.datetime.now()
treshold_graph_maximal = one_to_n.keycomp_treshold_updated_maximal_construct_graph(table_a, table_b, "aa", 0.5)
timing_tresh = (datetime.datetime.now()-now).total_seconds()
print("---- Timing for Graph Construction with Treshold Constraint ----")
print(timing_tresh,"seconds")


"""
Outputs the matching that has the maximal weight for each edge in the bipartite graph
Input: A Bipartite Graph
Output: A set of matchings. Ex: {('journals/vldb/MedjahedBBNE03__2', '775457__1')}
"""
print("\n\n MAXIMAL MATCHING RESULTS:")
now = datetime.datetime.now()
matching_set = nx.algorithms.matching.max_weight_matching(treshold_graph_maximal)
timing_match = (datetime.datetime.now()-now).total_seconds()
print("---- Timing for Matching (Done on the graph constructed with the treshold constraint) ----")
print(timing_match,"seconds")
print("The number of edges in the graph is:", treshold_graph_maximal.number_of_edges(), "\n")


print("The Maximal Matching Set is:", matching_set, "\n")


out = one_to_n.collapse(matching_set)
print("The Cleaned Maximal Matching is:", out, "\n")

max_final = one_to_n.collapsed_dict(out)
print("The De-duplicated Maximal Matching Final Result Is:", max_final, "\n")

"""
*******************************************************************
MINIMAL MATCHING
*******************************************************************
"""
now = datetime.datetime.now()
bipartite_graph_minimal = one_to_n.keycomp_treshold_updated_minimal_construct_graph(table_a, table_b, "aa", -0.5)
timing_tresh = (datetime.datetime.now()-now).total_seconds()
print("---- Timing for Graph Construction with Treshold Constraint ----")
print(timing_tresh,"seconds")


"""
Outputs the matching that has the minimal weight for each edge in the bipartite graph
Input: A Bipartite Graph
Output: A set of matchings. Ex: {('journals/vldb/MedjahedBBNE03__2', '775457__1')}
"""
print("\n\n MINIMAL MATCHING RESULTS:")
now = datetime.datetime.now()
matching_set2 = nx.algorithms.bipartite.matching.maximum_matching(bipartite_graph_minimal)
timing_match = (datetime.datetime.now()-now).total_seconds()
print("---- Timing for Matching (Done on the graph constructed with the treshold constraint) ----")
print(timing_match,"seconds")
print("The number of edges in the graph is:", bipartite_graph_minimal.number_of_edges(), "\n")
# print(bipartite_graph_minimal.edges.data())
print("The Minimal Matching Set is:", matching_set2, "\n")

out2 = one_to_n.collapse2(matching_set2)
print("The Cleaned Minimal Matching is:", out, "\n")

min_final = one_to_n.collapsed_dict(out2)
print("The De-duplicated Minimal Matching Final Result Is:", min_final, "\n")
print(nx.bipartite.is_bipartite(bipartite_graph_minimal))
# matching_set2 = nx.algorithms.bipartite.matching.minimum_weight_full_matching(bipartite_graph_minimal)
# print("Secondary Minimal Matching Set is:", matching_set2, "\n")

# out2 = one_to_n.collapse(matching_set2)
# print("(2) The Cleaned Minimal Matching is:", out2, "\n")

# last2 = one_to_n.collapsed_dict(out2)
# print("(2) The De-duplicated Minimal Matching Final Result Is:", last2, "\n")

"""
********************************************************************
UNCERTAINTY QUANTIFICATION
********************************************************************
"""
print("\n\n UNCERTAINTY QUANTIFICATION")

lookup = one_to_n.create_val_lookup(table_a, table_b, 2)
(min_sum, max_sum) = one_to_n.SUM_result_with_uncertainties(max_final, min_final, lookup)
final_uncertainty = one_to_n.form_formal_output(min_final, max_final, min_sum, max_sum)
for k, v in final_uncertainty.items():
	print (k, '-->', v)
