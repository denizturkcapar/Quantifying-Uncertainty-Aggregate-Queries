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
print("---- Timing for Graph Construction with Treshold Constraint for Maximum Bipartite Matching----")
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
bipartite_graph_minimal = one_to_n.keycomp_treshold_updated_maximal_construct_graph(table_a, table_b, "aa", 0.5)
timing_tresh = (datetime.datetime.now()-now).total_seconds()
print("---- Timing for Graph Construction with Treshold Constraint for Minimum Bipartite Matching----")
print(timing_tresh,"seconds")


"""
Outputs the matching that has the minimal weight for each edge in the bipartite graph
Input: A Bipartite Graph
Output: A set of matchings. Ex: {('journals/vldb/MedjahedBBNE03__2', '775457__1')}
"""
print("\n\n MINIMAL MATCHING RESULTS:")
now = datetime.datetime.now()
#matching_set2 = nx.algorithms.bipartite.matching.minimum_weight_full_matching(bipartite_graph_minimal)
timing_match = (datetime.datetime.now()-now).total_seconds()
print("---- Timing for Matching (Done on the graph constructed with the treshold constraint) ----")
print(timing_match,"seconds")
#print(bipartite_graph_minimal.edges())
#print(bipartite_graph_minimal.edges.data())
# print(nx.bipartite.is_bipartite(bipartite_graph_minimal))
# data_edge = bipartite_graph_minimal.edges()
# for i in data_edge:
# 	print(i)
print("Is graph bipartite? ", nx.bipartite.is_bipartite(bipartite_graph_minimal))
# G = nx.bipartite.gnmk_random_graph(3, 5, 10, seed=123)

# G = nx.complete_bipartite_graph(2,2)
# print("Is graph bipartite? ", nx.bipartite.is_bipartite(G))
# print(G.edges())

matching_set2 = nx.algorithms.bipartite.matching.minimum_weight_full_matching(bipartite_graph_minimal)

# top = nx.bipartite.sets(bipartite_graph_minimal)[0]
# print(nx.drawing.layout.bipartite_layout(bipartite_graph_minimal, top))

# matching_set2 = nx.algorithms.bipartite.matching.minimum_weight_full_matching(bipartite_graph_minimal)

print("The number of edges in the graph is:", bipartite_graph_minimal.number_of_edges(), "\n")
print("The Minimal Matching Set is:", matching_set2, "\n")

out2 = one_to_n.collapse2(matching_set2)
print("The Cleaned Minimal Matching is:", out2, "\n")

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
# print("\n\n UNCERTAINTY QUANTIFICATION")

# lookup = one_to_n.create_val_lookup(table_a, table_b, 2)
# (min_sum, max_sum) = one_to_n.SUM_result_with_uncertainties(max_final, min_final, lookup)
# final_uncertainty = one_to_n.form_formal_output(min_final, max_final, min_sum, max_sum)
# for k, v in final_uncertainty.items():
# 	print ("\n\n", k, '-->', v)



"""
Editing Edge Weight for SUM
"""
sum_weighted_graph = one_to_n.SUM_edit_edge_weight(treshold_graph_maximal)

print("\n\n 'SUM' MAXIMAL MATCHING RESULTS:")
now = datetime.datetime.now()
matching_set_maximal = nx.algorithms.matching.max_weight_matching(treshold_graph_maximal)
timing_match = (datetime.datetime.now()-now).total_seconds()
print("---- Timing for Matching (Done on the graph constructed with the treshold constraint) ----")
print(timing_match,"seconds")
print("The number of edges in the graph is:", sum_weighted_graph.number_of_edges(), "\n")


print("The Maximal Matching Set is:", matching_set_maximal, "\n")


out_maximal = one_to_n.collapse(matching_set_maximal)
print("The Cleaned Maximal Matching is:", out_maximal, "\n")

max_final_out = one_to_n.collapsed_dict(out_maximal)
print("The De-duplicated Maximal Matching Final Result Is:", max_final_out, "\n")



print("\n\n 'SUM' MINIMAL MATCHING RESULTS:")
matching_set_minimal = nx.algorithms.bipartite.matching.minimum_weight_full_matching(bipartite_graph_minimal)
print("The Minimal Matching Set is:", matching_set_minimal, "\n")

out_minimal = one_to_n.collapse2(matching_set_minimal)
print("The Cleaned Minimal Matching is:", out_minimal, "\n")

min_final_out = one_to_n.collapsed_dict(out_minimal)
print("The De-duplicated Minimal Matching Final Result Is:", min_final_out, "\n")
# print(nx.bipartite.is_bipartite(bipartite_graph_minimal))

"""
COUNT
"""

count_weighted_graph = one_to_n.COUNT_edit_edge_weight(treshold_graph_maximal)

print("\n\n 'COUNT' MAXIMAL MATCHING RESULTS:")
now = datetime.datetime.now()
matching_set_maximal_count = nx.algorithms.matching.max_weight_matching(treshold_graph_maximal)
timing_match = (datetime.datetime.now()-now).total_seconds()
print("---- Timing for Matching (Done on the graph constructed with the treshold constraint) ----")
print(timing_match,"seconds")
print("The number of edges in the graph is:", sum_weighted_graph.number_of_edges(), "\n")


print("The Maximal Matching Set is:", matching_set_maximal_count, "\n")


out_maximal_count = one_to_n.collapse(matching_set_maximal_count)
print("The Cleaned Maximal Matching is:", out_maximal_count, "\n")

max_final_out_count = one_to_n.collapsed_dict(out_maximal_count)
print("The De-duplicated Maximal Matching Final Result Is:", max_final_out_count, "\n")



print("\n\n 'COUNT' MINIMAL MATCHING RESULTS:")
matching_set_minimal_count = nx.algorithms.bipartite.matching.minimum_weight_full_matching(bipartite_graph_minimal)
print("The Minimal Matching Set is:", matching_set_minimal, "\n")

out_minimal_count = one_to_n.collapse2(matching_set_minimal_count)
print("The Cleaned Minimal Matching is:", out_minimal_count, "\n")

min_final_out_count = one_to_n.collapsed_dict(out_minimal_count)
print("The De-duplicated Minimal Matching Final Result Is:", min_final_out_count, "\n")
# print(nx.bipartite.is_bipartite(bipartite_graph_minimal))



