"""
COUNT
"""
import sys
from os import path
import pandas as pd
import re
import csv
import datetime
import matplotlib.transforms as transforms
# import matplotlib.axes.Axes as ax
import matplotlib.pyplot as plt
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

def sum_total_weights(max_min_list):
	# print("MAX-MIN LIST ", max_min_list)
	if max_min_list == [] or max_min_list == None:
		print("ERROR: NO SIMILARITY FOUND IN BIPARTITE, NAIVE OR RANDOM SAMPLING APPROACH. Suggestion: Decrease Similarity Matching Threshold.")
		return None
	total = 0
	for i in max_min_list:
		total += i[-1]
	return total

def find_perfect_count_result(perf_matching_file, file1_amazon, file2_google):
	res_sum = 0

	result_perfmatch, joined_table, perf_match_dict = create_perfect_mapping(perf_matching_file, file1_amazon, file2_google)
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
	return match_count

def find_perfect_count_variation_result(perf_matching_file, file1_amazon, file2_google, filter_condition):
	res_sum = 0

	result_perfmatch, joined_table, perf_match_dict = create_perfect_mapping(perf_matching_file, file1_amazon, file2_google)
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
			if res_sum >= filter_condition:
			# if i[0] or i[1] == 'b000fjs8sc':
			# 	print("\n", "HERE IS TYPE: ", type(joined_table[i[0]][0][1]))
			# 	print(str(joined_table[i[0]][0][1]))
			# 	print(str(joined_table[i[0]][0][2]))
				match_count += 1
			# print("Google Item name: ", joined_table[i[1]][0][0],"GOOGLE PRICE", google_price, "Amazon Item Name", joined_table[i[0]][0][0],"AMAZON PRICE", amazon_price, "SUM: ", res_sum)
	print("Perf Matching Count: ", match_count)
	return match_count

def full_evaluation_count(bp_min,bp_max, naive_min,naive_max, sampled_min,sampled_max, perfect_mapping, perf_match_dict):
	formatted_max_bp = experiment_funcs.fix_form_bp(bp_max)
	formatted_min_bp = experiment_funcs.fix_form_bp(bp_min)

	records_tuple = []

	bp_min_fp, bp_min_fn, bp_min_acc = experiment_funcs.accuracy_eval(formatted_min_bp, perf_match_dict)
	bp_max_fp, bp_max_fn, bp_max_acc = experiment_funcs.accuracy_eval(formatted_max_bp, perf_match_dict)

	naive_min_fp, naive_min_fn, naive_min_acc = core_scholar.eval_matching(naive_min)
	naive_max_fp, naive_max_fn, naive_max_acc = core_scholar.eval_matching(naive_max)

	sampled_min_fp, sampled_min_fn, sampled_min_acc = core_scholar.eval_matching(sampled_min)
	sampled_max_fp, sampled_max_fn, sampled_max_acc = core_scholar.eval_matching(sampled_max)

	records_tuple.append((bp_min_fp, bp_min_fn, bp_min_acc))
	records_tuple.append((bp_max_fp, bp_max_fn, bp_max_acc))
	records_tuple.append((naive_min_fp, naive_min_fn, naive_min_acc))
	records_tuple.append((naive_max_fp, naive_max_fn, naive_max_acc))
	records_tuple.append((sampled_min_fp, sampled_min_fn, sampled_min_acc))
	records_tuple.append((sampled_max_fp, sampled_max_fn, sampled_max_acc))

	# print(records_tuple)
	# print((records_tuple[0][0], records_tuple[0][1], records_tuple[0][2]))
	# print((records_tuple[1][0], records_tuple[1][1], records_tuple[1][2])) 
	# print((records_tuple[2][0], records_tuple[2][1], records_tuple[2][2])) 
	# print((records_tuple[3][0], records_tuple[3][1], records_tuple[3][2]))
	# print((records_tuple[4][0], records_tuple[4][1], records_tuple[4][2]))
	# print((records_tuple[5][0], records_tuple[5][1], records_tuple[5][2]))
	return records_tuple

def COUNT_edit_edge_weight(bip_graph, lookup_table):
	for u,v,d in bip_graph.edges(data=True):
		# splitted_u = u.split("_")[1]
		# splitted_v = v.split("_")[1]

		# val_u = lookup_table[splitted_u]
		# val_v = lookup_table[splitted_v]

		# if isinstance(val_u, (np.floating, float, int)):
		# 	val1 = float(val_u)
		# else:
		# 	val1 = float(val_u.split(" ")[0])
		
		# if isinstance(val_v, (np.floating, float, int)):
		# 	val2 = float(val_v)
		# else:
		# 	val2 = float(val_v.split(" ")[0])

		# d['weight'] = val1 + val2
		d['weight'] = 1
		# print("splitted_u: ", splitted_u, "\n")
		# print("splitted_v: ", splitted_v, "\n")
		# print("val_u: ", val_u, "\n")
		# print("val_v: ", val_v, "\n")
		# print("val1: ", val1, "\n")
		# print("val2: ", val2, "\n")
		# print("weight: ", d['weight'], "\n")

	return bip_graph


def fetch_count(bip_graph, matching):
	output = []
	for u,v,d in bip_graph.edges(data=True):
		l = (u, v)
		k = (v, u)
		if l in matching:
			output.append([u,v, d['weight']])
		if k in matching:
			output.append([v,u, d['weight']])
	return output

def count_total_weights(max_min_list):
	# print("MAX-MIN LIST ", max_min_list)
	if max_min_list == [] or max_min_list == None:
		print("ERROR: NO SIMILARITY FOUND IN BIPARTITE, NAIVE OR RANDOM SAMPLING APPROACH. Suggestion: Decrease Similarity Matching Threshold.")
		return None
	total = 0
	for i in max_min_list:
		total += 1
	return total

def realdata_count_bip_script(table_a_non_duplicated, table_b, column_name, similarity_threshold, n_matches, tables_map, num_swaps=None):

	now = datetime.datetime.now()
	bipartite_graph_result = one_to_n.valcomp_treshold_updated_maximal_construct_graph(table_a_non_duplicated, table_b, column_name, similarity_threshold, n_matches)
	timing_tresh = (datetime.datetime.now()-now).total_seconds()
	# print("---- Timing for Graph Construction with Treshold Constraint ----")
	# print(timing_tresh,"seconds")

	if num_swaps != None:
		bipartite_graph_result = one_to_n.randomize_by_edge_swaps(bipartite_graph_result, num_swaps)
		sum_weighted_graph = COUNT_edit_edge_weight(bipartite_graph_result, tables_map)	
	else:
		sum_weighted_graph = COUNT_edit_edge_weight(bipartite_graph_result, tables_map)
	# print(bipartite_graph_result.edges(data=True))
	print(bipartite_graph_result.number_of_edges())
	# print("\n\n 'SUM' MAXIMAL MATCHING:")
	now = datetime.datetime.now()
	matching_set_maximal = nx.algorithms.matching.max_weight_matching(sum_weighted_graph)
	timing_match_maximal = (datetime.datetime.now()-now).total_seconds()

	now = datetime.datetime.now()
	min_bipartite_graph_result = one_to_n.valcomp_treshold_updated_maximal_construct_graph(table_a_non_duplicated, table_b, column_name, similarity_threshold, n_matches)
	min_timing_tresh = (datetime.datetime.now()-now).total_seconds()

	min_sum_weighted_graph = COUNT_edit_edge_weight(min_bipartite_graph_result, tables_map)

	now = datetime.datetime.now()
	matching_set_minimal = minimal_matching(min_sum_weighted_graph)
	timing_match_minimal = (datetime.datetime.now()-now).total_seconds()

	out_max = fetch_count(sum_weighted_graph, matching_set_maximal)
	out_min = fetch_count(min_sum_weighted_graph, matching_set_minimal)
	# print("OUT MIN: ", out_min)
	# form_output = formatted_output(out_max,out_min)
    
	total_max = count_total_weights(out_max)
	print("BP Matching: Highest bound for maximum:", total_max)

	total_min = count_total_weights(out_min)
	print("BP Matching: Lowest bound for minimum:", total_min)
	# print("BP MAX MATCHING OUTPUT WITH SUMS:", out_max)
	print("BP Max Match Count: ", len(out_max))
	# print("BP MIN MATCHING OUTPUT WITH SUMS:", out_min)
	print("BP Min Match Count: ", len(out_max))
	return total_max, total_min, timing_match_minimal+min_timing_tresh, timing_match_maximal+timing_tresh, out_max, out_min

def realdata_count_naive_script(sim_threshold, filename1, filename2, table_a_non_duplicated, n_matches, filename1_dup, num_swaps=None):
    # print(sim_threshold, filename1, filename2)
    table_a_unprocessed = one_to_n.lat_convert_df(filename1)
    table_a_dup = one_to_n.create_duplicates(table_a_unprocessed, "name", n_matches)

    table_a_dup.to_csv(filename1_dup, index = False, header=True)
    cat_table1_dup = core.data_catalog(filename1_dup)
    cat_table1 = core.data_catalog(filename1)
    cat_table2 = core.data_catalog(filename2)
    
    now = datetime.datetime.now()

    print('Performing compare all match (jaccard distance)...')
    now = datetime.datetime.now()
    if num_swaps != None:
    	max_compare_all_jaccard_match, res_for_eval_max = matcher.matching_with_random_swaps(num_swaps,n_matches, True, cat_table1,cat_table2,one_to_n.calc_jaccard, matcher.all, sim_threshold)
    else:
    	max_compare_all_jaccard_match, res_for_eval_max = matcher.realdata_matcher_count(n_matches, True, cat_table1,cat_table2,one_to_n.calc_jaccard, matcher.all, sim_threshold)
    naive_time_jaccard_max = (datetime.datetime.now()-now).total_seconds()
    print("Naive Jaccard Matching computation time taken: ", naive_time_jaccard_max, " seconds", "\n")
    # print(max_compare_all_jaccard_match)
    # print('Compare All Matcher (Jaccard Distance) Performance: ' + str(core_scholar.eval_matching(res_for_eval_max)))

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
    	min_compare_all_jaccard_match, res_for_eval_min = matcher.realdata_matcher_count(1, False, cat_table1,cat_table2,one_to_n.calc_jaccard, matcher.all, sim_threshold)
    naive_time_jaccard_min = (datetime.datetime.now()-now).total_seconds()
    print("Naive Jaccard Matching computation time taken: ", naive_time_jaccard_min, " seconds")
    # print('Compare All Matcher (Jaccard Distance) Performance: ' + str(core_scholar.eval_matching(res_for_eval_min)))


    naive_total_max = count_total_weights(max_compare_all_jaccard_match)
    naive_total_min = count_total_weights(min_compare_all_jaccard_match)
    print("NAIVE MAX Matching Bound: ", naive_total_max)
    print("NAIVE MIN Matching Bound: ", naive_total_min)
    # print("NAIVE MAX MATCHING WITH SUMS:", max_compare_all_jaccard_match)
    # print("NAIVE MIN MATCHING WITH SUMS:", min_compare_all_jaccard_match)
    print("Naive Max Match Count: ", len(max_compare_all_jaccard_match))
    print("Naive Min Match Count", len(min_compare_all_jaccard_match))
    return naive_total_max, naive_total_min, naive_time_jaccard_min, naive_time_jaccard_max, max_compare_all_jaccard_match, min_compare_all_jaccard_match, res_for_eval_max, res_for_eval_min


def realdata_count_random_sample_script(sim_threshold, sample_size, filename1, filename2, table_a_non_duplicated, n_matches, filename1_dup,num_swaps=None):

    table_a_dup = one_to_n.create_duplicates(table_a_non_duplicated, "id", n_matches)
    table_a_dup.to_csv(filename1_dup, index = False, header=True)
    cat_table1_dup = core_scholar.data_catalog(filename1_dup)
    cat_table1 = core_scholar.data_catalog(filename1)
    cat_table2 = core_scholar.data_catalog(filename2)
    # print('Loaded catalogs.')
    
    # RANDOM SAMPLING MAX MATCHING
    # print("RANDOM SAMPLE MAX MATCHING")
    # print('Performing random sample match (edit distance)...')
    # now = datetime.datetime.now()
    # if num_swaps != None:
    # 	max_compare_sampled_edit_match = matcher.matching_with_random_swaps(num_swaps,n_matches, True, cat_table1,cat_table2,editdistance.eval, matcher.random_sample, sim_threshold, sample_size)
    # else:
    # 	max_compare_sampled_edit_match = matcher.matcher_updated(n_matches, True, cat_table1,cat_table2,editdistance.eval, matcher.random_sample, sim_threshold, sample_size)
    # sim_time_edit_max = (datetime.datetime.now()-now).total_seconds()
    # print("Simulation-Based Edit Distance Matching computation time taken: ", sim_time_edit_max, " seconds")
    #print('Random Sample Matcher (Edit Distance) Performance: ' + str(core.eval_matching(compare_all_edit_match)))

    print('Performing random sample match (jaccard distance)...')
    now = datetime.datetime.now()
    if num_swaps != None:
    	max_compare_sampled_jaccard_match, res_for_eval_max = matcher.matching_with_random_swaps(num_swaps,n_matches, True, cat_table1,cat_table2,one_to_n.calc_jaccard, matcher.random_sample, sim_threshold, sample_size)
    else:
    	max_compare_sampled_jaccard_match, res_for_eval_max = matcher.realdata_matcher_count(n_matches, True, cat_table1,cat_table2,one_to_n.calc_jaccard, matcher.random_sample, sim_threshold, sample_size)
    sim_time_jaccard_max = (datetime.datetime.now()-now).total_seconds()
    print("Simulation-Based Jaccard Matching computation time taken: ", sim_time_jaccard_max, " seconds", "\n")
    # print('Random Sample Matcher (Jaccard Distance) Performance: ' + str(core_scholar.eval_matching(max_compare_sampled_jaccard_match)))


    # RANDOM SAMPLING MIN MATCHING
    print("RANDOM SAMPLE MIN MATCHING")
    print('Performing random sample match (edit distance)...')
    # now = datetime.datetime.now()
    # if num_swaps != None:
    # 	min_compare_sampled_edit_match = matcher.matching_with_random_swaps(num_swaps,1, True, cat_table1,cat_table2,editdistance.eval, matcher.random_sample, sim_threshold, sample_size)
    # else:
    # 	min_compare_sampled_edit_match = matcher.matcher_updated(1, False, cat_table1,cat_table2,editdistance.eval, matcher.random_sample, sim_threshold, sample_size)
    # sim_time_edit_min = (datetime.datetime.now()-now).total_seconds()
    # print("Simulation-Based Edit Distance Matching computation time taken: ", sim_time_edit_min, " seconds")
    #print('Random Sample Matcher (Edit Distance) Performance: ' + str(core.eval_matching(compare_all_edit_match)))

    print('Performing random sample match (jaccard distance)...')
    now = datetime.datetime.now()
    if num_swaps != None:
    	min_compare_sampled_jaccard_match, res_for_eval_min = matcher.matching_with_random_swaps(num_swaps,1, True, cat_table1,cat_table2,one_to_n.calc_jaccard, matcher.random_sample, sim_threshold, sample_size)
    else:
    	min_compare_sampled_jaccard_match, res_for_eval_min = matcher.realdata_matcher_count(1, False, cat_table1,cat_table2,one_to_n.calc_jaccard, matcher.random_sample, sim_threshold, sample_size)
    sim_time_jaccard_min = (datetime.datetime.now()-now).total_seconds()
    print("Simulation-Based Jaccard Matching computation time taken: ", sim_time_jaccard_min, " seconds")
    # print('Random Sample Matcher (Jaccard Distance) Performance: ' + str(core_scholar.eval_matching(min_compare_sampled_jaccard_match)))

    sampled_total_max = count_total_weights(max_compare_sampled_jaccard_match)
    sampled_total_min = count_total_weights(min_compare_sampled_jaccard_match)
    # print("SAMPLED MAX Matching Bound: ", sampled_total_max, "\n")
    # print("SAMPLED MIN Matching Bound: ", sampled_total_min)
    return sampled_total_max, sampled_total_min, sim_time_jaccard_min, sim_time_jaccard_max, max_compare_sampled_jaccard_match, min_compare_sampled_jaccard_match, res_for_eval_max, res_for_eval_min

def real_data_1_to_n_count_results(file1, file2, perf_match_file, result_filename, bp_sim, naive_sim, sampled_sim, actual_n, bp_n, naive_n, sampled_n, table_a_length, sample_num):
	experiment_funcs.create_csv_table(result_filename)
	table_a_non_duplicated, table_b, tables_map = data_to_df(file1, file2)

	result_perfmatch, joined_table, perf_match_dict = create_perfect_mapping(perf_match_file, file1, file2)

	# Find perfect matching sum outcome for evaluation of the model (NOTE: current 1-1 limitations.)
	perfect_mapping_sum_result = find_perfect_count_result(perf_match_file, file1, file2)
	
	# Bipartite Matching Script
	total_max, total_min, bip_min_matching_time, bip_max_matching_time, out_max, out_min = realdata_count_bip_script(table_a_non_duplicated, table_b, "name", bp_sim, bp_n, tables_map)

	# Run Naive Matching Script
	naive_total_max, naive_total_min, naive_min_matching_time, naive_max_matching_time, naive_max, naive_min, res_naive_eval_max, res_naive_eval_min = realdata_count_naive_script(naive_sim, file1, file2, table_a_non_duplicated, naive_n, "naive_dup")
		
	# Run Random Matching Script
	sampled_total_max, sampled_total_min, sampled_min_matching_time, sampled_max_matching_time, sampled_max, sampled_min, res_sampled_eval_max, res_sampled_eval_min = realdata_count_random_sample_script(sampled_sim, sample_num, file1, file2, table_a_non_duplicated, sampled_n, "random_dup")

	# Run Accuracy Evaluation
	eval_records = full_evaluation_count(out_min, out_max, res_naive_eval_min, res_naive_eval_max, res_sampled_eval_min, res_naive_eval_max, result_perfmatch, perf_match_dict)

	# Record Experiment Results
	experiment_funcs.table_csv_output(total_min, total_max, naive_total_min, naive_total_max, sampled_total_min, sampled_total_max, perfect_mapping_sum_result, result_filename, bip_min_matching_time, bip_max_matching_time, naive_min_matching_time, naive_max_matching_time, sampled_min_matching_time, sampled_max_matching_time, eval_records)

def show_experiment_1_count(file1, file2, perf_match_file, experiment_name, sim_thresh):
	real_data_1_to_n_count_results(file1, file2, perf_match_file,"realdata_exp1_n3_count", sim_thresh,sim_thresh,sim_thresh,3,3,3,3,100, 100)
	# n = 1
	bp_min_med, bp_max_med, naive_min_med, naive_max_med, sampled_min_med, sampled_max_med, perf_threshold = exp1_helper("realdata_exp1_n3_count.csv")
	minThreshold = [bp_min_med, naive_min_med, sampled_min_med]

	maxThreshold = [bp_max_med, naive_max_med, sampled_max_med]

	indices = np.arange(len(maxThreshold))
	
	width = 0.8

	fig = plt.figure()

	ax = fig.add_subplot(111)

	max_limit = max(max(maxThreshold),perf_threshold)

	ax.set_title(experiment_name)
	ax.set_ylabel('Median Count')
	ax.set_xlabel('Experiment Type')
	label_list = ['Bipartite Matching', 'Naive Matching', 'Sample Matching']
	ax.bar(indices, maxThreshold, width=width, color='paleturquoise', label='Max Outcome', align='center')
	for index, value in enumerate(maxThreshold):
		ax.text(index, value, str(value))
	ax.bar([i for i in indices], minThreshold, width=width, color='gold', alpha=0.5, label='Min Outcome', align='center')
	for index, value in enumerate(minThreshold):
		ax.text(index, value, str(value))
	ax.legend(loc=9, bbox_to_anchor=(0.5,-0.2))
	ax.set_xticks(indices, minor=False)
	ax.set_xticklabels(label_list, fontdict=None, minor=False)

	ax.axhline(perf_threshold, color="red")
	ax.text(1.02, perf_threshold, str(perf_threshold), va='center', ha="left", bbox=dict(facecolor="w",alpha=0.5),transform=ax.get_yaxis_transform())
	plt.savefig(experiment_name, dpi=300)
	plt.show()


def count_variation_bip_script(table_a_non_duplicated, table_b, column_name, similarity_threshold, n_matches, tables_map, filter_condition,num_swaps=None):

	now = datetime.datetime.now()
	bipartite_graph_result = one_to_n.valcomp_treshold_updated_maximal_construct_graph(table_a_non_duplicated, table_b, column_name, similarity_threshold, n_matches, filter_condition)
	timing_tresh = (datetime.datetime.now()-now).total_seconds()
	# print("---- Timing for Graph Construction with Treshold Constraint ----")
	# print(timing_tresh,"seconds")

	if num_swaps != None:
		bipartite_graph_result = one_to_n.randomize_by_edge_swaps(bipartite_graph_result, num_swaps)
		sum_weighted_graph = SUM_edit_edge_weight(bipartite_graph_result, tables_map)	
	else:
		sum_weighted_graph = SUM_edit_edge_weight(bipartite_graph_result, tables_map)
	# print(bipartite_graph_result.edges(data=True))
	print(bipartite_graph_result.number_of_edges())
	# print("\n\n 'SUM' MAXIMAL MATCHING:")
	now = datetime.datetime.now()
	matching_set_maximal = nx.algorithms.matching.max_weight_matching(sum_weighted_graph)
	timing_match_maximal = (datetime.datetime.now()-now).total_seconds()

	now = datetime.datetime.now()
	min_bipartite_graph_result = one_to_n.valcomp_treshold_updated_maximal_construct_graph(table_a_non_duplicated, table_b, column_name, similarity_threshold, n_matches, filter_condition)
	min_timing_tresh = (datetime.datetime.now()-now).total_seconds()
	# print("---- Timing for Graph Construction with Treshold Constraint ----")
	# print(timing_tresh,"seconds")
    
	min_sum_weighted_graph = SUM_edit_edge_weight(min_bipartite_graph_result, tables_map)
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

def count_variation_naive_script(sim_threshold, filename1, filename2, table_a_non_duplicated, n_matches, filename1_dup, filter_condition, num_swaps=None):
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
    	max_compare_all_jaccard_match, res_for_eval_max = matcher.matching_with_random_swaps(num_swaps,n_matches, True, cat_table1,cat_table2,one_to_n.calc_jaccard, matcher.all, sim_threshold, filter_condition)
    else:
    	max_compare_all_jaccard_match, res_for_eval_max = matcher.realdata_matcher_updated(n_matches, True, cat_table1,cat_table2,one_to_n.calc_jaccard, matcher.all, sim_threshold, filter_condition)
    naive_time_jaccard_max = (datetime.datetime.now()-now).total_seconds()
    print("Naive Jaccard Matching computation time taken: ", naive_time_jaccard_max, " seconds", "\n")
    # print(max_compare_all_jaccard_match)
    print('Compare All Matcher (Jaccard Distance) Performance: ' + str(core.eval_matching(res_for_eval_max)))

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
    	min_compare_all_jaccard_match, res_for_eval_min = matcher.matching_with_random_swaps(num_swaps,1, True, cat_table1,cat_table2,one_to_n.calc_jaccard, matcher.random_sample, sim_threshold, filter_condition)
    else:
    	min_compare_all_jaccard_match, res_for_eval_min = matcher.realdata_matcher_updated(1, False, cat_table1,cat_table2,one_to_n.calc_jaccard, matcher.all, sim_threshold, filter_condition)
    naive_time_jaccard_min = (datetime.datetime.now()-now).total_seconds()
    print("Naive Jaccard Matching computation time taken: ", naive_time_jaccard_min, " seconds")
    print('Compare All Matcher (Jaccard Distance) Performance: ' + str(core.eval_matching(res_for_eval_min)))


    naive_total_max = sum_total_weights(max_compare_all_jaccard_match)
    naive_total_min = sum_total_weights(min_compare_all_jaccard_match)
    print("NAIVE MAX Matching Bound: ", naive_total_max)
    print("NAIVE MIN Matching Bound: ", naive_total_min)
    # print("NAIVE MAX MATCHING WITH SUMS:", max_compare_all_jaccard_match)
    # print("NAIVE MIN MATCHING WITH SUMS:", min_compare_all_jaccard_match)
    print("Naive Max Match Count: ", len(max_compare_all_jaccard_match))
    print("Naive Min Match Count", len(min_compare_all_jaccard_match))
    return naive_total_max, naive_total_min, naive_time_jaccard_min, naive_time_jaccard_max, max_compare_all_jaccard_match, min_compare_all_jaccard_match, res_for_eval_max, res_for_eval_min


def count_variation_random_sample_script(sim_threshold, sample_size, filename1, filename2, table_a_non_duplicated, n_matches, filename1_dup,filter_condition,num_swaps=None):

    table_a_dup = one_to_n.create_duplicates(table_a_non_duplicated, "id", n_matches)
    table_a_dup.to_csv(filename1_dup, index = False, header=True)
    cat_table1_dup = core.data_catalog(filename1_dup)
    cat_table1 = core.data_catalog(filename1)
    cat_table2 = core.data_catalog(filename2)
    # print('Loaded catalogs.')
    
    # RANDOM SAMPLING MAX MATCHING
    # print("RANDOM SAMPLE MAX MATCHING")
    # print('Performing random sample match (edit distance)...')
    # now = datetime.datetime.now()
    # if num_swaps != None:
    # 	max_compare_sampled_edit_match = matcher.matching_with_random_swaps(num_swaps,n_matches, True, cat_table1,cat_table2,editdistance.eval, matcher.random_sample, sim_threshold, sample_size)
    # else:
    # 	max_compare_sampled_edit_match = matcher.matcher_updated(n_matches, True, cat_table1,cat_table2,editdistance.eval, matcher.random_sample, sim_threshold, sample_size)
    # sim_time_edit_max = (datetime.datetime.now()-now).total_seconds()
    # print("Simulation-Based Edit Distance Matching computation time taken: ", sim_time_edit_max, " seconds")
    #print('Random Sample Matcher (Edit Distance) Performance: ' + str(core.eval_matching(compare_all_edit_match)))

    print('Performing random sample match (jaccard distance)...')
    now = datetime.datetime.now()
    if num_swaps != None:
    	max_compare_sampled_jaccard_match, res_for_eval_max = matcher.matching_with_random_swaps(num_swaps,n_matches, True, cat_table1,cat_table2,one_to_n.calc_jaccard, matcher.random_sample, sim_threshold, filter_condition, sample_size)
    else:
    	max_compare_sampled_jaccard_match, res_for_eval_max = matcher.realdata_matcher_updated(n_matches, True, cat_table1,cat_table2,one_to_n.calc_jaccard, matcher.random_sample, sim_threshold, filter_condition,sample_size)
    sim_time_jaccard_max = (datetime.datetime.now()-now).total_seconds()
    print("Simulation-Based Jaccard Matching computation time taken: ", sim_time_jaccard_max, " seconds", "\n")
    print('Random Sample Matcher (Jaccard Distance) Performance: ' + str(core.eval_matching(max_compare_sampled_jaccard_match)))


    # RANDOM SAMPLING MIN MATCHING
    print("RANDOM SAMPLE MIN MATCHING")
    print('Performing random sample match (edit distance)...')
    # now = datetime.datetime.now()
    # if num_swaps != None:
    # 	min_compare_sampled_edit_match = matcher.matching_with_random_swaps(num_swaps,1, True, cat_table1,cat_table2,editdistance.eval, matcher.random_sample, sim_threshold, sample_size)
    # else:
    # 	min_compare_sampled_edit_match = matcher.matcher_updated(1, False, cat_table1,cat_table2,editdistance.eval, matcher.random_sample, sim_threshold, sample_size)
    # sim_time_edit_min = (datetime.datetime.now()-now).total_seconds()
    # print("Simulation-Based Edit Distance Matching computation time taken: ", sim_time_edit_min, " seconds")
    #print('Random Sample Matcher (Edit Distance) Performance: ' + str(core.eval_matching(compare_all_edit_match)))

    print('Performing random sample match (jaccard distance)...')
    now = datetime.datetime.now()
    if num_swaps != None:
    	min_compare_sampled_jaccard_match, res_for_eval_min = matcher.matching_with_random_swaps(num_swaps,1, True, cat_table1,cat_table2,one_to_n.calc_jaccard, matcher.random_sample, sim_threshold, filter_condition,sample_size)
    else:
    	min_compare_sampled_jaccard_match, res_for_eval_min = matcher.realdata_matcher_updated(1, False, cat_table1,cat_table2,one_to_n.calc_jaccard, matcher.random_sample, sim_threshold, filter_condition,sample_size)
    sim_time_jaccard_min = (datetime.datetime.now()-now).total_seconds()
    print("Simulation-Based Jaccard Matching computation time taken: ", sim_time_jaccard_min, " seconds")
    print('Random Sample Matcher (Jaccard Distance) Performance: ' + str(core.eval_matching(min_compare_sampled_jaccard_match)))

    sampled_total_max = sum_total_weights(max_compare_sampled_jaccard_match)
    sampled_total_min = sum_total_weights(min_compare_sampled_jaccard_match)
    # print("SAMPLED MAX Matching Bound: ", sampled_total_max, "\n")
    # print("SAMPLED MIN Matching Bound: ", sampled_total_min)
    return sampled_total_max, sampled_total_min, sim_time_jaccard_min, sim_time_jaccard_max, max_compare_sampled_jaccard_match, min_compare_sampled_jaccard_match, res_for_eval_max, res_for_eval_min

def real_data_1_to_n_count_variation_results(file1, file2, perf_match_file, result_filename, bp_sim, naive_sim, sampled_sim, actual_n, bp_n, naive_n, sampled_n, table_a_length, sample_num, filter_condition):
	experiment_funcs.create_csv_table(result_filename)
	table_a_non_duplicated, table_b, tables_map = data_to_df(file1, file2)

	result_perfmatch, joined_table, perf_match_dict = create_perfect_mapping(perf_match_file, file1, file2)

	perfect_mapping_count_result = find_perfect_count_variation_result(perf_match_file, file1, file2, filter_condition)
	
	# Bipartite Matching Script
	total_max, total_min, bip_min_matching_time, bip_max_matching_time, out_max, out_min = count_variation_bip_script(table_a_non_duplicated, table_b, "name", bp_sim, bp_n, tables_map, filter_condition)

	# Run Naive Matching Script
	naive_total_max, naive_total_min, naive_min_matching_time, naive_max_matching_time, naive_max, naive_min, res_naive_eval_max, res_naive_eval_min = count_variation_naive_script(naive_sim, file1, file2, table_a_non_duplicated, naive_n, "naive_dup", filter_condition)
		
	# Run Random Matching Script
	sampled_total_max, sampled_total_min, sampled_min_matching_time, sampled_max_matching_time, sampled_max, sampled_min, res_sampled_eval_max, res_sampled_eval_min = count_variation_random_sample_script(sampled_sim, sample_num, file1, file2, table_a_non_duplicated, sampled_n, "random_dup", filter_condition)

	# Run Accuracy Evaluation
	eval_records = full_evaluation(out_min, out_max, res_naive_eval_min, res_naive_eval_max, res_sampled_eval_min, res_naive_eval_max, result_perfmatch, perf_match_dict)

	perfect_count = count_total_weights(result_perfmatch)
	bp_max_count = count_total_weights(out_max)
	bp_min_count = count_total_weights(out_min)
	naive_max_count = count_total_weights(res_naive_eval_max)
	naive_min_count = count_total_weights(res_naive_eval_min)
	rs_max_count = count_total_weights(res_sampled_eval_max)
	rs_min_count = count_total_weights(res_sampled_eval_min)

	# Record Experiment Results
	experiment_funcs.table_csv_output(total_min, total_max, naive_total_min, naive_total_max, sampled_total_min, sampled_total_max, perfect_mapping_sum_result, result_filename, bip_min_matching_time, bip_max_matching_time, naive_min_matching_time, naive_max_matching_time, sampled_min_matching_time, sampled_max_matching_time, eval_records)


# Same as SUM, until the COUNT of all matchings at the end of calculations.
def show_experiment_1_count_variation(file1, file2, perf_match_file, experiment_name, sim_thresh):
	real_data_1_to_n_count_variation_results(file1, file2, perf_match_file,"realdata_exp1_n3_count_variation", sim_thresh,sim_thresh,sim_thresh,3,3,3,3,100, 100)
	# n = 1
	bp_min_med, bp_max_med, naive_min_med, naive_max_med, sampled_min_med, sampled_max_med, perf_threshold = exp1_helper("realdata_exp1_n3_count_variation.csv")
	minThreshold = [bp_min_med, naive_min_med, sampled_min_med]

	maxThreshold = [bp_max_med, naive_max_med, sampled_max_med]

	indices = np.arange(len(maxThreshold))
	
	width = 0.8

	fig = plt.figure()

	ax = fig.add_subplot(111)

	max_limit = max(max(maxThreshold),perf_threshold)

	ax.set_title(experiment_name)
	ax.set_ylabel('Median Count')
	ax.set_xlabel('Experiment Type')
	label_list = ['Bipartite Matching', 'Naive Matching', 'Sample Matching']
	ax.bar(indices, maxThreshold, width=width, color='paleturquoise', label='Max Outcome', align='center')
	for index, value in enumerate(maxThreshold):
		ax.text(index, value, str(value))
	ax.bar([i for i in indices], minThreshold, width=width, color='gold', alpha=0.5, label='Min Outcome', align='center')
	for index, value in enumerate(minThreshold):
		ax.text(index, value, str(value))
	ax.legend(loc=9, bbox_to_anchor=(0.5,-0.2))
	ax.set_xticks(indices, minor=False)
	ax.set_xticklabels(label_list, fontdict=None, minor=False)

	ax.axhline(perf_threshold, color="red")
	ax.text(1.02, perf_threshold, str(perf_threshold), va='center', ha="left", bbox=dict(facecolor="w",alpha=0.5),transform=ax.get_yaxis_transform())
	plt.savefig(experiment_name, dpi=300)
	plt.show()

real_data_exp_funcs.show_experiment_1_count_variation('../Amazon-GoogleProducts/Amazon.csv', '../Amazon-GoogleProducts/GoogleProducts.csv', '../Amazon-GoogleProducts/Amzon_GoogleProducts_perfectMapping.csv', 'COUNT: Max n = 3, True n <= 3', 0.3)