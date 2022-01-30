import sys
from os import path
import pandas as pd
import re
import csv
import datetime
import matplotlib.transforms as transforms
# import matplotlib.axes.Axes as ax
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import networkx as nx
import experiment_funcs
import one_to_n
from Matching import core
from Matching import analyze
from Matching import matcher
from Matching import core_scholar
import textdistance
import editdistance
import collections

def find_max_n_amazon(file1):
	data_table = one_to_n.lat_convert_df(file1)
	records1 = data_table.to_records(index=False)
	result_1 = list(records1)
	n_counter_map = {}
	for (col1,col2) in result_1:
		if col1 in n_counter_map:
			n_counter_map[col1] += 1
		else:
			n_counter_map[col1] = 1
	# print(n_counter_map)
	all_values = n_counter_map.values()
	max_value = max(all_values)
	print(max_value)
	return max_value

def find_max_n_google(file1):
	data_table = one_to_n.lat_convert_df(file1)
	records1 = data_table.to_records(index=False)
	result_1 = list(records1)
	n_counter_map = {}
	for (col1,col2) in result_1:
		if col2 in n_counter_map:
			n_counter_map[col2] += 1
		else:
			n_counter_map[col2] = 1
	# print(n_counter_map)
	all_values = n_counter_map.values()
	max_value = max(all_values)
	print(max_value)
	return max_value

def data_to_df(file1, file2):
	table_a = one_to_n.lat_convert_df(file1)
	table_b = one_to_n.lat_convert_df(file2)

	# Keep track of table 1 and table 2 items
	records1 = table_a.to_records(index=False)
	result_1 = list(records1)

	records2 = table_b.to_records(index=False)
	result_2 = list(records2)

	joined_list = result_1 + result_2
	tables_map = {}

	data1_map = {}
	data2_map = {}

	for (col1,col2,col3,col4,col5) in result_1:
		data1_map[col2] = col5

	for (col1,col2,col3,col4,col5) in result_2:
		data2_map[col2] = col5

	for (col1,col2,col3,col4,col5) in joined_list:
		tables_map[col2] = col5

	name_to_id_dict_1 = {}
	name_to_id_dict_2 = {}

	for (col1,col2,col3,col4,col5) in result_1:
		name_to_id_dict_1[col2] = col1

	for (col1,col2,col3,col4,col5) in result_2:
		name_to_id_dict_2[col2] = col1

	# print("TABLES MAP", tables_map, '\n\n')

	return table_a, table_b, tables_map, data1_map, data2_map, name_to_id_dict_1, name_to_id_dict_2

def SUM_edit_edge_weight(bip_graph, data1_map, data2_map):
	for u,v,d in bip_graph.edges(data=True):
		splitted_u = u.split("_")[0]
		splitted_v = v.split("_")[0]

		if u.split("_")[1].isdigit() == True:
			val_u = data1_map[splitted_u]
			val_v = data2_map[splitted_v]
		else:
			val_u = data2_map[splitted_u]
			val_v = data1_map[splitted_v]

		# val_u = lookup_table[splitted_u]
		# val_v = lookup_table[splitted_v]

		if isinstance(val_u, (np.floating, float, int)):
			val1 = float(val_u)
		else:
			val1 = float(val_u.split(" ")[0])
		
		if isinstance(val_v, (np.floating, float, int)):
			val2 = float(val_v)
		else:
			val2 = float(val_v.split(" ")[0])

		d['weight'] = val1 + val2
		# print("splitted_u: ", splitted_u, "\n")
		# print("splitted_v: ", splitted_v, "\n")
		# print("val_u: ", val_u, "\n")
		# print("val_v: ", val_v, "\n")
		# print("val1: ", val1, "\n")
		# print("val2: ", val2, "\n")
		# print("weight: ", d['weight'], "\n")

	return bip_graph

def create_perfect_mapping(perf_matching_file, file1_amazon, file2_google):
	perf_matching_df = one_to_n.lat_convert_df(perf_matching_file)
	perf_match_records_1 = perf_matching_df.to_records(index=False)
	result_perfmatch = list(perf_match_records_1)

	perf_match_dict = {}

	name_to_id_dict = collections.defaultdict(list)

	for val1,val2 in result_perfmatch:
		perf_match_dict[val1] = val2

	table_a = one_to_n.lat_convert_df(file1_amazon)
	table_b = one_to_n.lat_convert_df(file2_google)

	# Keep track of table 1 and table 2 items
	records1 = table_a.to_records(index=False)
	result_1 = list(records1)

	records2 = table_b.to_records(index=False)
	result_2 = list(records2)

	joined_list = result_1 + result_2

	joined_table = collections.defaultdict(list)

	for col1,col2,col3,col4,col5 in joined_list:
		joined_table[col1].append((col2,col3,col4,col5))

	for col1,col2,col3,col4,col5 in joined_list:
		name_to_id_dict[col2].append(col1)


	# print("PERF MATCHING DICT: ", perf_match_dict)
	# print("JOINED TABLE", joined_table)

	return result_perfmatch, joined_table, perf_match_dict, name_to_id_dict

def find_perfect_sum_result(perf_matching_file, file1_amazon, file2_google):
	res_sum = 0

	result_perfmatch, joined_table, perf_match_dict, name_to_id_dict = create_perfect_mapping(perf_matching_file, file1_amazon, file2_google)
	match_count = 0
	for i in result_perfmatch:
		if i[0] and i[1] in joined_table:
			# print(i[0], i[1])
			if isinstance(joined_table[i[0]][0][-1], (np.floating, float)):
				amazon_price = float(joined_table[i[0]][0][-1])
			else:
				amazon_price = float(joined_table[i[0]][0][-1].split(" ")[0])

			if isinstance(joined_table[i[1]][0][-1], (np.floating, float)):
				google_price = float(joined_table[i[1]][0][-1])
			else:
				google_price = float(joined_table[i[1]][0][-1].split(" ")[0])
			# print("GOOGLE PRICE", google_price, "AMAZON PRICE", amazon_price)
			res_sum += google_price + amazon_price
			# if i[0] or i[1] == 'b000fjs8sc':
			# 	print("\n", "HERE IS TYPE: ", type(joined_table[i[0]][0][1]))
			# 	print(str(joined_table[i[0]][0][1]))
			# 	print(str(joined_table[i[0]][0][2]))
			match_count += 1
			# print("Google Item name: ", joined_table[i[1]][0][0],"GOOGLE PRICE", google_price, "Amazon Item Name", joined_table[i[0]][0][0],"AMAZON PRICE", amazon_price, "SUM: ", res_sum)
	print("Perf Matching Count: ", match_count)
	return res_sum



def minimal_matching(sum_weighted_graph):
	new_graph = sum_weighted_graph.copy()
	# print(new_graph.edges(data=True))
	max_weight = max([d['weight'] for u,v,d in new_graph.edges(data=True)])
	for u,v,d in new_graph.edges(data=True):
		# print("max weight:", max_weight)
		# print("BEFORE:", d['weight'])
		d['weight'] = max_weight - d['weight']
		# print("AFTER", d['weight'])
	# print(new_graph.edges(data=True))
	matching_set_minimal = nx.algorithms.matching.max_weight_matching(new_graph)
	return matching_set_minimal

def fetch_sum(bip_graph, matching):
	output = []
	for u,v,d in bip_graph.edges(data=True):
		l = (u, v)
		k = (v, u)
		if l in matching:
			output.append([u,v, d['weight']])
		if k in matching:
			output.append([v,u, d['weight']])
	return output

def formatted_output(out_max, out_min):
	out_dict = {}
	for (val1,val2, weight) in out_min:
		splitted1 = val1.split("_")
		splitted2 = val2.split("_")
		if len(splitted1) == 4:
			if splitted1[0] in out_dict:
				out_dict[splitted1[0]].append((splitted2[0], "min", weight))
			else:
				out_dict[splitted1[0]] = [(splitted2[0], "min", weight)]

		if len(splitted2) == 4:
			if splitted2[0] in out_dict:
				out_dict[splitted2[0]].append((splitted1[0], "min", weight))
			else:
				out_dict[splitted2[0]] = [(splitted1[0], "min", weight)]

	for (val1,val2, weight) in out_max:
		splitted1 = val1.split("_")
		splitted2 = val2.split("_")
		if len(splitted1) == 4:
			if splitted1[0] in out_dict:
				out_dict[splitted1[0]].append((splitted2[0], "max", weight))
			else:
				out_dict[splitted1[0]] = [(splitted2[0], "max", weight)]

		if len(splitted2) == 4:
			if splitted2[0] in out_dict:
				out_dict[splitted2[0]].append((splitted1[0], "max", weight))
			else:
				out_dict[splitted2[0]] = [(splitted1[0], "max", weight)]
	return out_dict

def sum_total_weights(max_min_list):
	# print("MAX-MIN LIST ", max_min_list)
	if max_min_list == [] or max_min_list == None:
		print("ERROR: NO SIMILARITY FOUND IN BIPARTITE, NAIVE OR RANDOM SAMPLING APPROACH. Suggestion: Decrease Similarity Matching Threshold.")
		return None
	total = 0
	for i in max_min_list:
		total += i[-1]
	return total

def realdata_sum_bip_script(table_a_non_duplicated, table_b, column_name, similarity_threshold, n_matches, data1_map, data2_map, num_swaps=None):

	now = datetime.datetime.now()
	bipartite_graph_result = one_to_n.realdata_keycomp_treshold_updated_maximal_construct_graph(table_a_non_duplicated, table_b, column_name, similarity_threshold, n_matches)
	timing_tresh = (datetime.datetime.now()-now).total_seconds()
	# print("---- Timing for Graph Construction with Treshold Constraint ----")
	# print(timing_tresh,"seconds")
	# print(bipartite_graph_result.edges())

	if num_swaps != None:
		bipartite_graph_result = one_to_n.randomize_by_edge_swaps(bipartite_graph_result, num_swaps)
		sum_weighted_graph = SUM_edit_edge_weight(bipartite_graph_result, data1_map, data2_map)	
	else:
		sum_weighted_graph = SUM_edit_edge_weight(bipartite_graph_result, data1_map, data2_map)
	# print("BIPARTITE GRAPH RES: ", bipartite_graph_result.edges(data=True))
	# print("BIPARTITE GRAPH RES: ", bipartite_graph_result.number_of_edges())
	# print("\n\n 'SUM' MAXIMAL MATCHING:")
	now = datetime.datetime.now()
	matching_set_maximal = nx.algorithms.matching.max_weight_matching(sum_weighted_graph)
	timing_match_maximal = (datetime.datetime.now()-now).total_seconds()
	# print("MATCHED BIP RESULT: ", matching_set_maximal)
	now = datetime.datetime.now()
	min_bipartite_graph_result = one_to_n.realdata_keycomp_treshold_updated_maximal_construct_graph(table_a_non_duplicated, table_b, column_name, similarity_threshold, n_matches)
	min_timing_tresh = (datetime.datetime.now()-now).total_seconds()
	# print("---- Timing for Graph Construction with Treshold Constraint ----")
	# print(timing_tresh,"seconds")
    
	min_sum_weighted_graph = SUM_edit_edge_weight(min_bipartite_graph_result, data1_map, data2_map)
	# print(nx.bipartite.is_bipartite(min_sum_weighted_graph))
	# print("\n\n 'SUM' MINIMAL MATCHING RESULTS:")
	# print("MIN EDGES:", min_sum_weighted_graph.edges())
	# for u,v,d in min_sum_weighted_graph.edges(data=True):
	# 	print("table a: ", u, "table b: ", v, "distance: ", d)
	now = datetime.datetime.now()
	matching_set_minimal = minimal_matching(min_sum_weighted_graph)
	timing_match_minimal = (datetime.datetime.now()-now).total_seconds()
	# print("The Minimal Matching Set is:", matching_set_minimal, "\n")
	# print("---- Timing for Matching (Done on the graph constructed with the treshold constraint) ----")
	# print(timing_match_minimal,"seconds")
	# print("The number of edges in the graph is:", sum_weighted_graph.number_of_edges(), "\n")

	out_max = fetch_sum(sum_weighted_graph, matching_set_maximal)
	out_min = fetch_sum(min_sum_weighted_graph, matching_set_minimal)
	# print("OUT MIN: ", out_min)
	# form_output = formatted_output(out_max,out_min)
    
	total_max = sum_total_weights(out_max)
	print("BP Matching: Highest bound for maximum:", total_max)

	total_min = sum_total_weights(out_min)
	print("BP Matching: Lowest bound for minimum:", total_min)
	# print("BP MAX MATCHING OUTPUT WITH SUMS:", out_max)
	print("BP Max Match Count: ", len(out_max))
	# print("BP MIN MATCHING OUTPUT WITH SUMS:", out_min)
	print("BP Min Match Count: ", len(out_min))

	max_bp_match_count = len(out_max)
	min_bp_match_count = len(out_min)

	return total_max, total_min, timing_match_minimal+min_timing_tresh, timing_match_maximal+timing_tresh, out_max, out_min

def realdata_sum_naive_script(sim_threshold, filename1, filename2, table_a_non_duplicated, n_matches, filename1_dup, num_swaps=None):
    # print(sim_threshold, filename1, filename2)
    table_a_unprocessed = one_to_n.lat_convert_df(filename1)
    table_a_dup = one_to_n.create_duplicates(table_a_unprocessed, "name", n_matches)

    table_a_dup.to_csv(filename1_dup, index = False, header=True)
    cat_table1_dup = core.data_catalog(filename1_dup)
    cat_table1 = core.data_catalog(filename1)
    cat_table2 = core.data_catalog(filename2)
    # print('Loaded catalogs.')
    
    
    # NAIVE MAX MATCHING
    # print("NAIVE MAX MATCHING")
    # print('Performing compare all match (edit distance)...')
    now = datetime.datetime.now()
    # if num_swaps != None:
    # 	max_compare_all_edit_match = matcher.matching_with_random_swaps(num_swaps,n_matches, True, cat_table1,cat_table2,editdistance.eval, matcher.all, sim_threshold)
    # else:
    # 	max_compare_all_edit_match = matcher.matcher_updated(n_matches, True, cat_table1,cat_table2,editdistance.eval, matcher.all, sim_threshold)
    # naive_time_edit_max = (datetime.datetime.now()-now).total_seconds()
    # print("Naive Edit Distance Matching computation time taken: ", naive_time_edit_max, " seconds")
    #print('Compare All Matcher (Edit Distance) Performance: ' + str(core.eval_matching(compare_all_edit_match)))


    print('Performing compare all match (jaccard distance)...')
    now = datetime.datetime.now()
    if num_swaps != None:
    	max_compare_all_jaccard_match, res_for_eval_max = matcher.matching_with_random_swaps(num_swaps,n_matches, True, cat_table1,cat_table2,one_to_n.calc_jaccard, matcher.all, sim_threshold)
    else:
    	max_compare_all_jaccard_match, res_for_eval_max = matcher.realdata_matcher_updated(n_matches, True, cat_table1,cat_table2,one_to_n.calc_jaccard, matcher.all, sim_threshold)
    naive_time_jaccard_max = (datetime.datetime.now()-now).total_seconds()
    print("Naive Jaccard Matching computation time taken: ", naive_time_jaccard_max, " seconds", "\n")
    # print(max_compare_all_jaccard_match)
    # print('Compare All Matcher (Jaccard Distance) Performance: ' + str(core.eval_matching(res_for_eval_max)))

    # NAIVE MIN MATCHING
    # print("NAIVE MIN MATCHING")
    # print('Performing compare all match (edit distance)...')
    # now = datetime.datetime.now()
    # if num_swaps != None:
    # 	min_compare_all_edit_match = matcher.matching_with_random_swaps(num_swaps,1, True, cat_table1,cat_table2,editdistance.eval, matcher.random_sample, sim_threshold)
    # else:
    # 	min_compare_all_edit_match = matcher.matcher_updated(1, False, cat_table1,cat_table2,editdistance.eval, matcher.all, sim_threshold)
    # print(min_compare_all_edit_match)
    # naive_time_edit_min = (datetime.datetime.now()-now).total_seconds()
    # print("Naive Edit Distance Matching computation time taken: ", naive_time_edit_min, " seconds")
    #print('Compare All Matcher (Edit Distance) Performance: ' + str(core.eval_matching(compare_all_edit_match)))


    print('Performing compare all match (jaccard distance)...')
    now = datetime.datetime.now()
    if num_swaps != None:
    	min_compare_all_jaccard_match, res_for_eval_min = matcher.matching_with_random_swaps(num_swaps,1, True, cat_table1,cat_table2,one_to_n.calc_jaccard, matcher.random_sample, sim_threshold)
    else:
    	min_compare_all_jaccard_match, res_for_eval_min = matcher.realdata_matcher_updated(1, False, cat_table1,cat_table2,one_to_n.calc_jaccard, matcher.all, sim_threshold)
    naive_time_jaccard_min = (datetime.datetime.now()-now).total_seconds()
    print("Naive Jaccard Matching computation time taken: ", naive_time_jaccard_min, " seconds")
    # print('Compare All Matcher (Jaccard Distance) Performance: ' + str(core.eval_matching(res_for_eval_min)))


    naive_total_max = sum_total_weights(max_compare_all_jaccard_match)
    naive_total_min = sum_total_weights(min_compare_all_jaccard_match)
    print("NAIVE MAX Matching Bound: ", naive_total_max)
    print("NAIVE MIN Matching Bound: ", naive_total_min)
    # print("NAIVE MAX MATCHING WITH SUMS:", max_compare_all_jaccard_match)
    # print("NAIVE MIN MATCHING WITH SUMS:", min_compare_all_jaccard_match)
    print("Naive Max Match Count: ", len(max_compare_all_jaccard_match))
    print("Naive Min Match Count", len(min_compare_all_jaccard_match))
    return naive_total_max, naive_total_min, naive_time_jaccard_min, naive_time_jaccard_max, max_compare_all_jaccard_match, min_compare_all_jaccard_match, res_for_eval_max, res_for_eval_min


# def realdata_sum_random_sample_script(sim_threshold, sample_size, filename1, filename2, table_a_non_duplicated, n_matches, filename1_dup,num_swaps=None):

#     table_a_dup = one_to_n.create_duplicates(table_a_non_duplicated, "id", n_matches)
#     table_a_dup.to_csv(filename1_dup, index = False, header=True)
#     cat_table1_dup = core.data_catalog(filename1_dup)
#     cat_table1 = core.data_catalog(filename1)
#     cat_table2 = core.data_catalog(filename2)
#     # print('Loaded catalogs.')
    
#     # RANDOM SAMPLING MAX MATCHING
#     # print("RANDOM SAMPLE MAX MATCHING")
#     # print('Performing random sample match (edit distance)...')
#     # now = datetime.datetime.now()
#     # if num_swaps != None:
#     # 	max_compare_sampled_edit_match = matcher.matching_with_random_swaps(num_swaps,n_matches, True, cat_table1,cat_table2,editdistance.eval, matcher.random_sample, sim_threshold, sample_size)
#     # else:
#     # 	max_compare_sampled_edit_match = matcher.matcher_updated(n_matches, True, cat_table1,cat_table2,editdistance.eval, matcher.random_sample, sim_threshold, sample_size)
#     # sim_time_edit_max = (datetime.datetime.now()-now).total_seconds()
#     # print("Simulation-Based Edit Distance Matching computation time taken: ", sim_time_edit_max, " seconds")
#     #print('Random Sample Matcher (Edit Distance) Performance: ' + str(core.eval_matching(compare_all_edit_match)))

#     print('Performing random sample match (jaccard distance)...')
#     now = datetime.datetime.now()
#     if num_swaps != None:
#     	max_compare_sampled_jaccard_match, res_for_eval_max = matcher.matching_with_random_swaps(num_swaps,n_matches, True, cat_table1,cat_table2,one_to_n.calc_jaccard, matcher.random_sample, sim_threshold, sample_size)
#     else:
#     	max_compare_sampled_jaccard_match, res_for_eval_max = matcher.realdata_matcher_updated(n_matches, True, cat_table1,cat_table2,one_to_n.calc_jaccard, matcher.random_sample, sim_threshold, sample_size)
#     sim_time_jaccard_max = (datetime.datetime.now()-now).total_seconds()
#     print("Simulation-Based Jaccard Matching computation time taken: ", sim_time_jaccard_max, " seconds", "\n")
#     print('Random Sample Matcher (Jaccard Distance) Performance: ' + str(core.eval_matching(max_compare_sampled_jaccard_match)))


#     # RANDOM SAMPLING MIN MATCHING
#     print("RANDOM SAMPLE MIN MATCHING")
#     print('Performing random sample match (edit distance)...')
#     # now = datetime.datetime.now()
#     # if num_swaps != None:
#     # 	min_compare_sampled_edit_match = matcher.matching_with_random_swaps(num_swaps,1, True, cat_table1,cat_table2,editdistance.eval, matcher.random_sample, sim_threshold, sample_size)
#     # else:
#     # 	min_compare_sampled_edit_match = matcher.matcher_updated(1, False, cat_table1,cat_table2,editdistance.eval, matcher.random_sample, sim_threshold, sample_size)
#     # sim_time_edit_min = (datetime.datetime.now()-now).total_seconds()
#     # print("Simulation-Based Edit Distance Matching computation time taken: ", sim_time_edit_min, " seconds")
#     #print('Random Sample Matcher (Edit Distance) Performance: ' + str(core.eval_matching(compare_all_edit_match)))

#     print('Performing random sample match (jaccard distance)...')
#     now = datetime.datetime.now()
#     if num_swaps != None:
#     	min_compare_sampled_jaccard_match, res_for_eval_min = matcher.matching_with_random_swaps(num_swaps,1, True, cat_table1,cat_table2,one_to_n.calc_jaccard, matcher.random_sample, sim_threshold, sample_size)
#     else:
#     	min_compare_sampled_jaccard_match, res_for_eval_min = matcher.realdata_matcher_updated(1, False, cat_table1,cat_table2,one_to_n.calc_jaccard, matcher.random_sample, sim_threshold, sample_size)
#     sim_time_jaccard_min = (datetime.datetime.now()-now).total_seconds()
#     print("Simulation-Based Jaccard Matching computation time taken: ", sim_time_jaccard_min, " seconds")
#     print('Random Sample Matcher (Jaccard Distance) Performance: ' + str(core.eval_matching(min_compare_sampled_jaccard_match)))

#     sampled_total_max = sum_total_weights(max_compare_sampled_jaccard_match)
#     sampled_total_min = sum_total_weights(min_compare_sampled_jaccard_match)
#     # print("SAMPLED MAX Matching Bound: ", sampled_total_max, "\n")
#     # print("SAMPLED MIN Matching Bound: ", sampled_total_min)
#     return sampled_total_max, sampled_total_min, sim_time_jaccard_min, sim_time_jaccard_max, max_compare_sampled_jaccard_match, min_compare_sampled_jaccard_match, res_for_eval_max, res_for_eval_min


def accuracy_eval(formatted_proposed_matching, perfect_mapping, joined_table, name_to_id_dict_1, name_to_id_dict_2, perf_match_file):
	matches = set()
	proposed_matches = set()
	tp = set()
	fp = set()
	fn = set()
	tn = set()

	f = open(perf_match_file, 'r', encoding = "ISO-8859-1")
	reader = csv.reader(f, delimiter=',', quotechar='"')

	for row in reader:
		matches.add((row[0],row[1]))

	# print("PERFECT MAPPING: ", perfect_mapping)
	for key,vals in perfect_mapping.items():
		# print("KEY: ", key,"\n","vals: ", vals)
		matches.add((key,vals))

	for (m1,m2,w) in formatted_proposed_matching:
		if m1 in name_to_id_dict_1:
			id1 = name_to_id_dict_1[m1]
			id2 = name_to_id_dict_2[m2]
		else:
			id1 = name_to_id_dict_2[m1]
			id2 = name_to_id_dict_1[m2]
		# print("id1 :", id1)
		# print("id2: ", id2)
		proposed_matches.add((id1,id2))
		if (id1,id2) in matches:
			tp.add((id1,id2))
		else:
			fp.add((id1,id2))

	for m in matches:
		if m not in proposed_matches:
			fn.add((id1,id2))

	# print("fn", len(fn))
	# print("tp", len(tp))
	# print("fp", len(fp))

	# print("PERFECT MATCHES: ", matches)
	# print("BIP MATCH CANDIDATES: ", proposed_matches)

	try:
		prec = len(tp)/(len(tp) + len(fp))
	except ZeroDivisionError:
		prec = 0

	try:
		rec = len(tp)/(len(tp) + len(fn))
	except ZeroDivisionError:
		rec = 0


	# print("prec", prec)
	# print("rec", rec)
	false_pos = 1-prec
	false_neg = 1-rec

	try:
		accuracy = round(2*(prec*rec)/(prec+rec),2)
	except ZeroDivisionError:
		accuracy = 0

	# print(accuracy)
	return false_pos, false_neg, accuracy

def full_evaluation(bp_min,bp_max, naive_min,naive_max, perfect_mapping, perf_match_dict, joined_table, name_to_id_dict, name_to_id_dict_1, name_to_id_dict_2, perf_match_file):
	formatted_max_bp = experiment_funcs.fix_form_bp(bp_max)
	formatted_min_bp = experiment_funcs.fix_form_bp(bp_min)

	# formatted_max_naive = fix_form_naive(naive_max)
	# formatted_min_naive = fix_form_naive(naive_min)

	# formatted_max_sampled = fix_form_naive(sampled_max)
	# formatted_min_sampled = fix_form_naive(sampled_min)

	# print("PERFECT MAPPING", perfect_mapping)
	# print("NAIVE MIN", formatted_min_naive)

	records_tuple = []
	# print("naive min: ", naive_min)
	# print("naive max: ", naive_max)

	bp_min_fp, bp_min_fn, bp_min_acc = accuracy_eval(formatted_min_bp, perf_match_dict, joined_table, name_to_id_dict_1, name_to_id_dict_2, perf_match_file)
	bp_max_fp, bp_max_fn, bp_max_acc = accuracy_eval(formatted_max_bp, perf_match_dict, joined_table, name_to_id_dict_1, name_to_id_dict_2, perf_match_file)

	print("bp_min_fp: ", bp_min_fp, "\n", "bp_min_fn: ", bp_min_fn, "\n", "bp_min_acc: ", bp_min_acc)
	print("bp_max_fp: ", bp_max_fp, "\n", "bp_max_fn: ", bp_max_fn, "\n", "bp_max_acc: ", bp_max_acc)
	naive_min_fp, naive_min_fn, naive_min_acc = core.eval_matching(naive_min, perf_match_file)
	naive_max_fp, naive_max_fn, naive_max_acc = core.eval_matching(naive_max, perf_match_file)
	print("naive_min_fp: ", naive_min_fp, "\n", "naive_min_fn: ", naive_min_fn, "\n", "naive_min_acc: ", naive_min_acc)
	print("naive_max_fp: ", naive_max_fp, "\n", "naive_max_fn: ", naive_max_fn, "\n", "naive_max_acc: ", naive_max_acc)


	# sampled_min_fp, sampled_min_fn, sampled_min_acc = core.eval_matching(sampled_min)
	# sampled_max_fp, sampled_max_fn, sampled_max_acc = core.eval_matching(sampled_max)

	records_tuple.append((bp_min_fp, bp_min_fn, bp_min_acc))
	records_tuple.append((bp_max_fp, bp_max_fn, bp_max_acc))
	records_tuple.append((naive_min_fp, naive_min_fn, naive_min_acc))
	records_tuple.append((naive_max_fp, naive_max_fn, naive_max_acc))
	# records_tuple.append((sampled_min_fp, sampled_min_fn, sampled_min_acc))
	# records_tuple.append((sampled_max_fp, sampled_max_fn, sampled_max_acc))

	# print(records_tuple)
	# print((records_tuple[0][0], records_tuple[0][1], records_tuple[0][2]))
	# print((records_tuple[1][0], records_tuple[1][1], records_tuple[1][2])) 
	# print((records_tuple[2][0], records_tuple[2][1], records_tuple[2][2])) 
	# print((records_tuple[3][0], records_tuple[3][1], records_tuple[3][2]))
	# print((records_tuple[4][0], records_tuple[4][1], records_tuple[4][2]))
	# print((records_tuple[5][0], records_tuple[5][1], records_tuple[5][2]))
	return records_tuple


def real_data_1_to_n_sum_results(file1, file2, perf_match_file, result_filename, bp_sim, naive_sim, actual_n, bp_n, naive_n, table_a_length):
	experiment_funcs.create_csv_table(result_filename)
	table_a_non_duplicated, table_b, tables_map, data1_map, data2_map, name_to_id_dict_1, name_to_id_dict_2 = data_to_df(file1, file2)

	result_perfmatch, joined_table, perf_match_dict, name_to_id_dict = create_perfect_mapping(perf_match_file, file1, file2)

	# Find perfect matching sum outcome for evaluation of the model (NOTE: current 1-1 limitations.)
	perfect_mapping_sum_result = find_perfect_sum_result(perf_match_file, file1, file2)
	
	# Bipartite Matching Script
	total_max, total_min, bip_min_matching_time, bip_max_matching_time, out_max, out_min = realdata_sum_bip_script(table_a_non_duplicated, table_b, "name", bp_sim, bp_n, data1_map, data2_map)

	# Run Naive Matching Script
	naive_total_max, naive_total_min, naive_min_matching_time, naive_max_matching_time, naive_max, naive_min, res_naive_eval_max, res_naive_eval_min = realdata_sum_naive_script(naive_sim, file1, file2, table_a_non_duplicated, naive_n, "naive_dup")
		
	# Run Random Matching Script
	# sampled_total_max, sampled_total_min, sampled_min_matching_time, sampled_max_matching_time, sampled_max, sampled_min, res_sampled_eval_max, res_sampled_eval_min = realdata_sum_random_sample_script(sampled_sim, sample_num, file1, file2, table_a_non_duplicated, sampled_n, "random_dup")
	# print("res_naive_eval_max: ", res_naive_eval_max)
	# print("res_naive_eval_min: ", res_naive_eval_min)
	# Run Accuracy Evaluation
	eval_records = full_evaluation(out_min, out_max, res_naive_eval_min, res_naive_eval_max, result_perfmatch, perf_match_dict, joined_table, name_to_id_dict, name_to_id_dict_1, name_to_id_dict_2, perf_match_file)

	# Record Experiment Results
	experiment_funcs.table_csv_output(total_min, total_max, naive_total_min, naive_total_max, perfect_mapping_sum_result, result_filename, bip_min_matching_time, bip_max_matching_time, naive_min_matching_time, naive_max_matching_time, eval_records)


def exp1_helper(results_file):
	df = pd.read_csv(results_file)

	bp_min_med = df['Bipartite Min Matching'].median()
	bp_max_med = df['Bipartite Max Matching'].median()
	naive_min_med = df['Naive Min Matching'].median()
	naive_max_med = df['Naive Max Matching'].median()
	# sampled_min_med = df['Sampled Min Matching'].median()
	# sampled_max_med = df['Sampled Max Matching'].median()

	perf_threshold = df['Perfect Matching SUM'].median()

	return bp_min_med, bp_max_med, naive_min_med, naive_max_med, perf_threshold
def show_experiment_1_sum(file1, file2, perf_match_file, experiment_name, sim_thresh, true_n, max_n):
	real_data_1_to_n_sum_results(file1, file2, perf_match_file,"realdata_exp1_n1", sim_thresh,sim_thresh,true_n,max_n,max_n,100)
	# n = 1
	bp_min_med, bp_max_med, naive_min_med, naive_max_med, perf_threshold = exp1_helper("realdata_exp1_n1.csv")
	minThreshold = [bp_min_med, naive_min_med]

	maxThreshold = [bp_max_med, naive_max_med]

	indices = np.arange(len(maxThreshold))
	
	width = 0.8

	fig = plt.figure()

	ax = fig.add_subplot(111)

	max_limit = max(max(maxThreshold),perf_threshold)

	ax.set_title(experiment_name)
	ax.set_ylabel('Median Sum')
	ax.set_xlabel('Experiment Type')
	label_list = ['Bipartite Matching', 'Naive Matching']
	ax.bar(indices, maxThreshold, width=width, color='paleturquoise', label='Max Outcome', align='center')
	for index, value in enumerate(maxThreshold):
		ax.text(index, value, str(value))
	ax.bar([i for i in indices], minThreshold, width=width, color='white', alpha=1, label='Min Outcome', align='center')
	for index, value in enumerate(minThreshold):
		ax.text(index, value, str(value))
	ax.legend(loc=9, bbox_to_anchor=(0.5,-0.2))
	ax.set_xticks(indices, minor=False)
	ax.set_xticklabels(label_list, fontdict=None, minor=False)

	ax.axhline(perf_threshold, color="red")
	ax.text(1.02, perf_threshold, str(perf_threshold), va='center', ha="left", bbox=dict(facecolor="w",alpha=0.5),transform=ax.get_yaxis_transform())
	plt.savefig(experiment_name, dpi=300)
	plt.show()


def show_experiment_2(file1, file2, perf_match_file, experiment_name, sim_thresh, is_age_skewed=False, num_swaps=None):
	
	real_data_1_to_n_sum_results(file1, file2, perf_match_file, "realdata_exp2_n1", sim_thresh,sim_thresh,1,1,1,100)
	real_data_1_to_n_sum_results(file1, file2, perf_match_file, "realdata_exp2_n2", sim_thresh,sim_thresh,2,2,2,100)
	real_data_1_to_n_sum_results(file1, file2, perf_match_file, "realdata_exp2_n3", sim_thresh,sim_thresh,3,3,3,100)
	real_data_1_to_n_sum_results(file1, file2, perf_match_file, "realdata_exp2_n4", sim_thresh,sim_thresh,4,4,4,100)
	real_data_1_to_n_sum_results(file1, file2, perf_match_file, "realdata_exp2_n5", sim_thresh,sim_thresh,5,5,5,100)
	real_data_1_to_n_sum_results(file1, file2, perf_match_file, "realdata_exp2_n6", sim_thresh,sim_thresh,6,6,6,100)
	# 1-1
	bp_width1, naive_width1, sampled_width1 = model_overestimation_calc_helper("realdata_exp2_n1")
	#1-2
	bp_width2, naive_width2, sampled_width2 = model_overestimation_calc_helper("realdata_exp2_n2")
	#1-3
	bp_width3, naive_width3, sampled_width3 = model_overestimation_calc_helper("realdata_exp2_n3")
	#1-4
	bp_width4, naive_width4, sampled_width4 = model_overestimation_calc_helper("realdata_exp2_n4")
	#1-5
	bp_width5, naive_width5, sampled_width5 = model_overestimation_calc_helper("realdata_exp2_n5")
	#1-6
	bp_width6, naive_width6, sampled_width6 = model_overestimation_calc_helper("realdata_exp2_n6")

	bp_widths = [bp_width1, bp_width2, bp_width3, bp_width4, bp_width5, bp_width6]
	naive_widths = [naive_width1, naive_width2, naive_width3, naive_width4, naive_width5, naive_width6]
	sample_widths = [sampled_width1, sampled_width2, sampled_width3, sampled_width4, sampled_width5, sampled_width6]
	n = [1,2,3,4,5,6]

	# plt.scatter(n,bp_widths, color='k')
	plt.plot(n, bp_widths, '-o', color='r', label='Bipartite')
	# plt.scatter(n,naive_widths, color='g')
	plt.plot(n, naive_widths, '-o', color='g', label='PC')
	# plt.scatter(n,sample_widths, color='b')
	plt.plot(n, sample_widths, '-o', color='b', label='Sampled')
	plt.xlabel('N (User inputted Number of Matches)')
	plt.ylabel('Median Over-estimation Rate')
	plt.title(experiment_name)
	plt.xticks(n,n)
	plt.legend()
	plt.savefig(experiment_name, dpi=300)
	plt.show()


def model_overestimation_calc_helper(results_file):
	df1 = pd.read_csv('{}.csv'.format(results_file))
	# bp_min_med = df1['Bipartite Min Matching'].median()
	bp_max_med = df1['Bipartite Max Matching'].median()
	# naive_min_med = df1['Naive Min Matching'].median()
	naive_max_med = df1['Naive Max Matching'].median()
	# sampled_min_med = df1['Sampled Min Matching'].median()
	sampled_max_med = df1['Sampled Max Matching'].median()
	perf_threshold = df1['Perfect Matching SUM'].median()
	bp_acc_metric = (bp_max_med - perf_threshold) / perf_threshold
	naive_acc_metric = (naive_max_med - perf_threshold) / perf_threshold
	sampled_acc_metric = (sampled_max_med - perf_threshold) / perf_threshold

	return bp_acc_metric, naive_acc_metric, sampled_acc_metric

# EXPERIMENT 3 - INAPPLICABLE TO THIS REAL DATA INSTANCE
# def show_experiment_3(experiment_name, sim_thresh, is_age_skewed=False, num_swaps=None):
# 	create_skewed_1_to_n_sum_results("skewed_3_skew1", sim_thresh,sim_thresh,sim_thresh,1,1,1,1,100,100, is_age_skewed, num_swaps)
# 	create_skewed_1_to_n_sum_results("skewed_3_skew3", sim_thresh,sim_thresh,sim_thresh,3,3,3,3,100,150, is_age_skewed, num_swaps)
# 	create_skewed_1_to_n_sum_results("skewed_3_skew5", sim_thresh,sim_thresh,sim_thresh,5,5,5,5,100,200, is_age_skewed, num_swaps)
# 	create_skewed_1_to_n_sum_results("skewed_3_skew7", sim_thresh,sim_thresh,sim_thresh,7,7,7,7,100,250, is_age_skewed, num_swaps)
# 	create_skewed_1_to_n_sum_results("skewed_3_skew9", sim_thresh,sim_thresh,sim_thresh,9,9,9,9,100,300, is_age_skewed, num_swaps)
# 	create_skewed_1_to_n_sum_results("skewed_3_skew11", sim_thresh,sim_thresh,sim_thresh,11,11,11,11,100,300, is_age_skewed, num_swaps)

# 	# skew level = 1
# 	bp_width1, naive_width1, sampled_width1 = model_overestimation_calc_helper("skewed_3_skew1")
# 	# skew level = 3
# 	bp_width2, naive_width2, sampled_width2 = model_overestimation_calc_helper("skewed_3_skew3")
# 	# skew level = 5
# 	bp_width3, naive_width3, sampled_width3 = model_overestimation_calc_helper("skewed_3_skew5")
# 	# skew level = 7
# 	bp_width4, naive_width4, sampled_width4 = model_overestimation_calc_helper("skewed_3_skew7")
# 	# skew level = 9
# 	bp_width5, naive_width5, sampled_width5 = model_overestimation_calc_helper("skewed_3_skew9")
# 	# skew level = 11
# 	bp_width6, naive_width6, sampled_width6 = model_overestimation_calc_helper("skewed_3_skew11")

# 	bp_widths = [bp_width1, bp_width2, bp_width3, bp_width4, bp_width5, bp_width6]
# 	naive_widths = [naive_width1, naive_width2, naive_width3, naive_width4, naive_width5, naive_width6]
# 	sample_widths = [sampled_width1, sampled_width2, sampled_width3, sampled_width4, sampled_width5, sampled_width6]
# 	n = [1,3,5,7,9,11]

# 	# plt.scatter(n,bp_widths, color='k')
# 	plt.plot(n, bp_widths, '-o', color='r', label='Bipartite')
# 	# plt.scatter(n,naive_widths, color='g')
# 	plt.plot(n, naive_widths, '-o', color='g', label='Naive')
# 	# plt.scatter(n,sample_widths, color='b')
# 	plt.plot(n, sample_widths, '-o', color='b', label='Sampled')
# 	plt.xlabel('Skew Degree (Skew varies from 1-N)')
# 	plt.ylabel('Median Over-estimation Rate')
# 	plt.title(experiment_name)
# 	plt.xticks(n,n)
# 	plt.legend()
# 	plt.savefig(experiment_name, dpi=300)
# 	plt.show()


def show_experiment_4(file1, file2, experiment_name, user_n, is_age_skewed=False, num_swaps=None):
	
	real_data_1_to_n_sum_results(file1, file2, "1_exp4_realdata", 0.2,0.2,0.2,user_n,user_n,user_n,user_n,100,100, is_age_skewed, num_swaps)
	real_data_1_to_n_sum_results(file1, file2, "2_exp4_realdata", 0.17,0.17,0.17,user_n,user_n,user_n,user_n,100,150, is_age_skewed, num_swaps)
	real_data_1_to_n_sum_results(file1, file2, "3_exp4_realdata", 0.14,0.14,0.14,user_n,user_n,user_n,user_n,100,200, is_age_skewed, num_swaps)
	real_data_1_to_n_sum_results(file1, file2, "4_exp4_realdata", 0.11,0.11,0.11,user_n,user_n,user_n,user_n,100,250, is_age_skewed, num_swaps)
	real_data_1_to_n_sum_results(file1, file2, "5_exp4_realdata", 0.08,0.08,0.08,user_n,user_n,user_n,user_n,100,300, is_age_skewed, num_swaps)

	# do not uncomment here
	# create_1_to_n_sum_results("6_exp4_balanced", 0.06,0.06,0.06,user_n,user_n,user_n,user_n,100,300, is_age_skewed, num_swaps)
	# create_1_to_n_sum_results("6_exp4_balanced", 0.02,0.02,0.02,user_n,user_n,user_n,user_n,100,300, is_age_skewed, num_swaps)

	# sim thresh level = 1, optimal
	bp_width1, naive_width1, sampled_width1 = model_overestimation_calc_helper("1_exp4_balanced")
	# sim thresh level = 2
	bp_width2, naive_width2, sampled_width2 = model_overestimation_calc_helper("2_exp4_balanced")
	# sim thresh level = 3
	bp_width3, naive_width3, sampled_width3 = model_overestimation_calc_helper("3_exp4_balanced")
	# sim thresh level = 4
	bp_width4, naive_width4, sampled_width4 = model_overestimation_calc_helper("4_exp4_balanced")
	# sim thresh level = 5, worst
	bp_width5, naive_width5, sampled_width5 = model_overestimation_calc_helper("5_exp4_balanced")
	# sim thresh level = 6
	# bp_width6, naive_width6, sampled_width6 = model_overestimation_calc_helper("6_exp4_balanced")

	# # sim thresh level = 7, worst
	# bp_width7, naive_width7, sampled_width7 = model_overestimation_calc_helper("6_exp4_balanced")

	bp_widths = [bp_width1, bp_width2, bp_width3, bp_width4, bp_width5]
	naive_widths = [naive_width1, naive_width2, naive_width3, naive_width4, naive_width5]
	sample_widths = [sampled_width1, sampled_width2, sampled_width3, sampled_width4, sampled_width5]
	n = [0.2,0.17,0.14,0.11,0.08]

	# plt.scatter(n,bp_widths, color='k')
	plt.plot(n, bp_widths, '-o', color='r', label='Bipartite')
	# plt.scatter(n,naive_widths, color='g')
	plt.plot(n, naive_widths, '-o', color='g', label='Naive')
	# plt.scatter(n,sample_widths, color='b')
	plt.plot(n, sample_widths, '-o', color='b', label='Sampled')
	plt.xlabel('Quality of Similarity Metric (Worst to Best)')
	plt.ylabel('Median Over-estimation Rate')
	plt.title(experiment_name)
	plt.xticks(n,n)
	plt.legend()
	plt.savefig(experiment_name, dpi=300)
	plt.show()

"""

Collapse the results further to a dictionary format. I've made it optional as a separate function for now.
Although it is quite possible to merge this function with collapsed() function

Input: (id_match1, id_match2) formatted list of tuples (either from bipartite or naive result)
Output: A dictionary that has the key as the de-duplicated entries and the values as the "n" table values that they got matched
"""
def collapse_dict_for_evaluation(res):

    out = collections.defaultdict(set)
    for (key, val) in res:
        # print("\n\n", val, key)
        out[key].add(str(val))
    return out

def show_one_to_n_histogram_maxes(file1, file2, title, bp_sim, naive_sim, bp_n, naive_n):
	"""
	ONE_TO_N CALCULATIONS
	"""
	# Bipartite Max calc
	proposed_matches = []
	table_a_non_duplicated, table_b, tables_map, data1_map, data2_map, name_to_id_dict_1, name_to_id_dict_2 = data_to_df(file1, file2)
	now = datetime.datetime.now()
	bipartite_graph_result = one_to_n.realdata_keycomp_treshold_updated_maximal_construct_graph(table_a_non_duplicated, table_b, 'name', bp_sim, bp_n)
	timing_tresh = (datetime.datetime.now()-now).total_seconds()
	sum_weighted_graph = SUM_edit_edge_weight(bipartite_graph_result, data1_map, data2_map)
	timing_match_maximal = (datetime.datetime.now()-now).total_seconds()
	matching_set_maximal = nx.algorithms.matching.max_weight_matching(sum_weighted_graph)
	print("The Matching Set is:", matching_set_maximal, "\n")
	out_max = fetch_sum(sum_weighted_graph, matching_set_maximal)
	formatted_proposed_matching = experiment_funcs.fix_form_bp(out_max)
	print("\n\n\n FORMATTED PROPOSED MATCHING: ", formatted_proposed_matching, "\n\n\n")
	for (m1,m2,w) in formatted_proposed_matching:
		if m1 in name_to_id_dict_1:
			id1 = name_to_id_dict_1[m1]
			id2 = name_to_id_dict_2[m2]
		else:
			id1 = name_to_id_dict_2[m1]
			id2 = name_to_id_dict_1[m2]
		proposed_matches.append((id1,id2))
	print("Proposed matches: ", proposed_matches)
	bipartite_res_dict = collapse_dict_for_evaluation(proposed_matches)

	# Naive Max calc
	table_a_unprocessed = one_to_n.lat_convert_df(file1)
	table_a_dup = one_to_n.create_duplicates(table_a_unprocessed, "name", naive_n)
	table_a_dup.to_csv("table_a_dup_histogram", index = False, header=True)
	cat_table1_dup = core.data_catalog("table_a_dup_histogram")
	cat_table1 = core.data_catalog(file1)
	cat_table2 = core.data_catalog(file2)
	now = datetime.datetime.now()
	print('Performing compare all match (jaccard distance)...')
	now = datetime.datetime.now()
	max_compare_all_jaccard_match, res_for_eval_max = matcher.realdata_matcher_updated(naive_n, True, cat_table1,cat_table2,one_to_n.calc_jaccard, matcher.all, naive_sim)
	naive_time_jaccard_max = (datetime.datetime.now()-now).total_seconds()
	print("Naive Jaccard Matching computation time taken: ", naive_time_jaccard_max, " seconds", "\n")
	print(res_for_eval_max)

	naive_res_dict = collapse_dict_for_evaluation(res_for_eval_max)
	print("The de-duplicated final result is:", naive_res_dict, "\n")


	"""
	Find max n
	"""
	# max_n = 0
	# for key,vals in naive_res_dict.items():
	# 	if len(vals) > max_n:
	# 		max_n = len(vals)

	# for key,vals in bipartite_res_dict.items():
	# 	if len(vals) > max_n:
	# 		max_n = len(vals)

	# Hardcode max_n for now so that computation time is less.

	"""
	Find values for Histogram (1-1: # of matches, 1-2: # of matches, ..., 1-max_n: # of matches)
	"""
	count1, count2, count3, count4, count5 = 0,0,0,0,0
	for key, vals in bipartite_res_dict.items():
		if len(vals) == 1:
			count1 +=1
		if len(vals) == 2:
			count2 +=1
		if len(vals) == 3:
			count3 +=1
		if len(vals) == 4:
			count4 +=1
		if len(vals) == 5:
			count5 +=1

	bipartite_value_correlation = {'1-1': count1, '1-2': count2,'1-3': count3, '1-4': count4, '1-5': count5}


	count1, count2, count3, count4, count5 = 0,0,0,0,0

	for key, vals in naive_res_dict.items():
		if len(vals) == 1:
			count1 +=1
		if len(vals) == 2:
			count2 +=1
		if len(vals) == 3:
			count3 +=1
		if len(vals) == 4:
			count4 +=1
		if len(vals) == 5:
			count5 +=1

	naive_value_correlation = {'1-1': count1, '1-2': count2,'1-3': count3, '1-4': count4, '1-5': count5}

	"""
	PLOTTING THE HISTOGRAM
	"""

	bipartiteThreshold = [bipartite_value_correlation['1-1'], bipartite_value_correlation['1-2'],bipartite_value_correlation['1-3'],bipartite_value_correlation['1-4'],bipartite_value_correlation['1-5']]
	naiveThreshold = [naive_value_correlation['1-1'], naive_value_correlation['1-2'],naive_value_correlation['1-3'],naive_value_correlation['1-4'],naive_value_correlation['1-5']]
	indices = np.arange(len(bipartiteThreshold))
	
	width = 0.8

	fig = plt.figure()

	ax = fig.add_subplot(111)

	max_limit = max(max(bipartiteThreshold), max(naiveThreshold))

	ax.set_title(title)
	ax.set_ylabel('Frequency of Matches')
	ax.set_xlabel('Number of Matches when Max_n = 5')
	label_list = ['1', '2', '3', '4', '5']
	ax.bar(indices, bipartiteThreshold, width=width, color='paleturquoise', label='Bipartite Outcome', align='center')
	ax.bar([i for i in indices], naiveThreshold, width=width/2, color='red', alpha=0.5, label='PC Outcome', align='center')
	# for index, value in enumerate(bipartiteThreshold):
	# 	ax.text(index, value, str(value))
	# for index, value in enumerate(naiveThreshold):
	# 	ax.text(index, value, str(value))
	ax.legend(loc=9, bbox_to_anchor=(0.5,-0.2))
	ax.set_xticks(indices, minor=False)
	ax.set_xticklabels(label_list, fontdict=None, minor=False)

	plt.savefig("histogram_trial1", dpi=300)
	plt.show()

def show_sim_metric_distribution_maxes(file1, file2, title, bp_sim, naive_sim, bp_n, naive_n):
	naive_sim_outcomes = []
	bipartite_sim_outcomes = []

	# Bipartite Max calc
	table_a_non_duplicated, table_b, tables_map, data1_map, data2_map, name_to_id_dict_1, name_to_id_dict_2 = data_to_df(file1, file2)
	now = datetime.datetime.now()
	bipartite_graph_result = one_to_n.realdata_keycomp_treshold_updated_maximal_construct_graph(table_a_non_duplicated, table_b, 'name', bp_sim, bp_n)
	timing_tresh = (datetime.datetime.now()-now).total_seconds()
	sum_weighted_graph = SUM_edit_edge_weight(bipartite_graph_result, data1_map, data2_map)
	timing_match_maximal = (datetime.datetime.now()-now).total_seconds()
	matching_set_maximal = nx.algorithms.matching.max_weight_matching(sum_weighted_graph)
	# print("The Matching Set is:", matching_set_maximal, "\n")
	out_max = fetch_sum(sum_weighted_graph, matching_set_maximal)
	formatted_proposed_matching = experiment_funcs.fix_form_bp(out_max)
	# for (m1,m2,w) in formatted_proposed_matching:
	# 	if m1 in name_to_id_dict_1:
	# 		id1 = name_to_id_dict_1[m1]
	# 		id2 = name_to_id_dict_2[m2]
	# 	else:
	# 		id1 = name_to_id_dict_2[m1]
	# 		id2 = name_to_id_dict_1[m2]
	# 	proposed_matches.append((id1,id2))
	# print("Proposed matches: ", proposed_matches)
	# bipartite_res_dict = collapse_dict_for_evaluation(proposed_matches)

	# (match1, match2, sim)

	# Naive Max calc
	table_a_unprocessed = one_to_n.lat_convert_df(file1)
	table_a_dup = one_to_n.create_duplicates(table_a_unprocessed, "name", naive_n)
	table_a_dup.to_csv("table_a_dup_histogram", index = False, header=True)
	cat_table1_dup = core.data_catalog("table_a_dup_histogram")
	cat_table1 = core.data_catalog(file1)
	cat_table2 = core.data_catalog(file2)
	now = datetime.datetime.now()
	print('Performing compare all match (jaccard distance)...')
	now = datetime.datetime.now()
	max_compare_all_jaccard_match, res_for_eval_max = matcher.realdata_matcher_updated(naive_n, True, cat_table1,cat_table2,one_to_n.calc_jaccard, matcher.all, naive_sim)
	naive_time_jaccard_max = (datetime.datetime.now()-now).total_seconds()
	# print("\n\n\n", max_compare_all_jaccard_match)


	"""
	Compile an array of similarity values for bipartite and naive outcomes.
	"""

	for (m1,m2,w) in formatted_proposed_matching:
		dist = one_to_n.calc_max_weight(str(m1).lower(),str(m2).lower())
		bipartite_sim_outcomes.append(dist)


	for (m1,m2,ind1,ind2,outcome) in max_compare_all_jaccard_match:
		dist = one_to_n.calc_max_weight(str(m1).lower(),str(m2).lower())
		naive_sim_outcomes.append(dist)
	print(bipartite_sim_outcomes)
	print(naive_sim_outcomes)

	"""
	Plot Bipartite and Naive outcomes in 2 separate histograms. 
	"""
	mpl.use('tkagg')
	plt.figure(figsize=(8,6))
	plt.hist(bipartite_sim_outcomes, bins=100, alpha=0.8, color='green', label="Bipartite Matching Outcome")
	plt.hist(naive_sim_outcomes, bins=100, alpha=0.5, color='red', label="PC Matching Outcome")

	plt.xlabel("Distribution of Similarity Distances", size=14)
	plt.ylabel("Count", size=14)
	plt.title(title)
	plt.legend(loc='upper right')
	plt.savefig("sim_distribution_trial1.png")
	plt.show()

# show_sim_metric_distribution_maxes('../Background_Demonstration_Data/company_a_inventory.csv', '../Background_Demonstration_Data/company_b_inventory.csv', 'Distribution of Similarity Distances', 0.8, 0.8, 5, 5)


def show_candidate_nmatches_histogram(file1, file2, title, sim):
	"""
	ONE_TO_N CALCULATIONS
	"""
	cat_table1 = core.data_catalog(file1)
	cat_table2 = core.data_catalog(file2)
	now = datetime.datetime.now()
	print('Performing compare all match (jaccard distance)...')
	now = datetime.datetime.now()
	match_map = matcher.realdata_matcher_for_n_histogram(cat_table1,cat_table2,one_to_n.calc_jaccard, matcher.all, sim)
	naive_time_jaccard_max = (datetime.datetime.now()-now).total_seconds()
	print("\n\n\n", match_map)
	match_count = 0
	stored_counts = []
	for key, vals in match_map.items():
		match_count = len(vals)
		stored_counts.append(match_count)

	"""
	PLOTTING THE HISTOGRAM
	"""

	mpl.use('tkagg')
	plt.figure(figsize=(8,6))
	plt.hist(stored_counts, bins=100, alpha=0.8, color='green')

	plt.xlabel("Number of Candidate Matches (N) ", size=14)
	plt.ylabel("Count", size=14)
	plt.title(title)
	plt.legend(loc='upper right')
	plt.savefig("all_n_candidate_matches_dist.png")
	plt.show()

def show_candidate_avg_sim_metric_distribution(file1, file2, title, sim):
	"""
	ONE_TO_N CALCULATIONS
	"""
	cat_table1 = core.data_catalog(file1)
	cat_table2 = core.data_catalog(file2)
	now = datetime.datetime.now()
	print('Performing compare all match (jaccard distance)...')
	now = datetime.datetime.now()
	match_map = matcher.realdata_matcher_for_simdist_histogram(cat_table1,cat_table2,one_to_n.calc_jaccard, matcher.all, sim)
	naive_time_jaccard_max = (datetime.datetime.now()-now).total_seconds()
	print("\n\n\n", match_map)
	candidate_sim_dists = []
	stored_dists = []
	for key, vals in match_map.items():
		for val in vals:
			candidate_sim_dists.append(float(val[-1]))
		avg_sim = np.mean(candidate_sim_dists)
		stored_dists.append(avg_sim)
		candidate_sim_dists = []

	"""
	PLOTTING THE HISTOGRAM
	"""
	print(stored_dists)
	mpl.use('tkagg')
	plt.figure(figsize=(8,6))
	plt.hist(stored_dists, bins=100, alpha=0.8, color='red')

	plt.xlabel("Average Similarity Distance for Candidate Matches", size=14)
	plt.ylabel("Count", size=14)
	plt.title(title)
	plt.legend(loc='upper right')
	plt.savefig("candidate_avg_sim_dist_hist.png")
	plt.show()

# show_candidate_nmatches_histogram('../Background_Demonstration_Data/company_a_inventory.csv', '../Background_Demonstration_Data/company_b_inventory.csv', 'Distribution of All Number of Candidate Matches', 0.8)

show_candidate_avg_sim_metric_distribution('../Background_Demonstration_Data/company_a_inventory.csv', '../Background_Demonstration_Data/company_b_inventory.csv', 'Distribution of Average Similarity Distance for Candidate Matches', 0.8)
