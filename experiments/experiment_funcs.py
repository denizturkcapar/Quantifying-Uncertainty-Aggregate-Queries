import sys
from os import path
sys.path.insert(0, '../src')
import one_to_n
sys.path.insert(0, '../tests')
import create_synthetic_dataset
import datetime
import textdistance
import editdistance
import pandas as pd
import networkx as nx
import re

import editdistance
from Matching import core2
from Matching import analyze
from Matching import matcher
import sys
import os
import datetime

"""
SUM
"""
"""
Converts dictionary output to dataframe. Use for experiment results
"""
def dict_to_df(output_dict):
    df = pd.DataFrame.from_dict(output_dict, orient='index',columns=['Matched Item', 'Max/Min', 'Value'])
    return df
"""
Converts experiment results to CSV format. Use for experiment results
"""
def exp_result_to_csv(filename, experiment_output_df):
    #create the CSV with custom filename describing the experiment
    df.to_csv(filename, index = False)

def create_synth_data(table1_rowcount, table2_rowcount, datafilename1, datafilename2, filename1_dup):
    table_a_non_duplicated = create_synthetic_dataset.create_first_df(table1_rowcount)

    table_b_non_typo = create_synthetic_dataset.create_second_df(table2_rowcount)

    table_b = create_synthetic_dataset.add_typo(table_a_non_duplicated, table_b_non_typo)
    
    table_a_non_duplicated.to_csv(datafilename1, index = False, header=True)
    
    table_b.to_csv(datafilename2, index = False, header=True)
    
    table_a_dup = one_to_n.create_duplicates(table_a_non_duplicated, "name", 3)
    
    table_a_dup.to_csv(filename1_dup, index = False, header=True)
    
    return table_a_non_duplicated, table_b, table_a_dup

def SUM_edit_edge_weight(bip_graph):
    for u,v,d in bip_graph.edges(data=True):
        val_tuple_1 = u.split("_")
        val_tuple_2 = v.split("_")
        
        if len(val_tuple_1) == 4:
            val1 = re.sub("[^0-9]", "", val_tuple_1[2])
        else: 
            val1 = re.sub("[^0-9]", "", val_tuple_1[1])
            
        if len(val_tuple_2) == 4:
            val2 =re.sub("[^0-9]", "", val_tuple_2[2])
        else:
            val2 =re.sub("[^0-9]", "", val_tuple_2[1])

        d['weight'] = float(val1) + float(val2)

    return bip_graph

def minimal_matching(sum_weighted_graph):
    new_graph = sum_weighted_graph.copy()
    max_weight = max([d['weight'] for u,v,d in new_graph.edges(data=True)])
    for u,v,d in new_graph.edges(data=True):
        d['weight'] = max_weight - d['weight']

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
    if max_min_list == [] or max_min_list == None:
        print("ERROR: NO SIMILARITY FOUND IN NAIVE OR RANDOM SAMPLING APPROACH. Suggestion: Decrease Similarity Matching Threshold.")
        return None
    total = 0
    for i in max_min_list:
        total += i[-1]
    return total

def sum_bip_script(table_a_non_duplicated, table_b, column_name, similarity_threshold, n_matches):
    now = datetime.datetime.now()
    bipartite_graph_result = one_to_n.keycomp_treshold_updated_maximal_construct_graph(table_a_non_duplicated, table_b, column_name, similarity_threshold, n_matches)
    timing_tresh = (datetime.datetime.now()-now).total_seconds()
    print("---- Timing for Graph Construction with Treshold Constraint ----")
    print(timing_tresh,"seconds")
    
    sum_weighted_graph = SUM_edit_edge_weight(bipartite_graph_result)
    
    print("\n\n 'SUM' MAXIMAL MATCHING:")
    now = datetime.datetime.now()
    matching_set_maximal = nx.algorithms.matching.max_weight_matching(sum_weighted_graph)
    timing_match_maximal = (datetime.datetime.now()-now).total_seconds()
#    print("The Maximal Matching Set is:", matching_set_maximal, "\n")
    print("---- Timing for Matching (Done on the graph constructed with the treshold constraint) ----")
    print(timing_match_maximal,"seconds")
    print("The number of edges in the graph is:", sum_weighted_graph.number_of_edges(), "\n")
    
    
    print("\n\n 'SUM' MINIMAL MATCHING RESULTS:")
    print(nx.bipartite.is_bipartite(sum_weighted_graph))
    now = datetime.datetime.now()
    matching_set_minimal = minimal_matching(sum_weighted_graph)
    timing_match_minimal = (datetime.datetime.now()-now).total_seconds()
#    print("The Minimal Matching Set is:", matching_set_minimal, "\n")
    print("---- Timing for Matching (Done on the graph constructed with the treshold constraint) ----")
    print(timing_match_minimal,"seconds")
    print("The number of edges in the graph is:", sum_weighted_graph.number_of_edges(), "\n")
    
    
    
    out_max = fetch_sum(sum_weighted_graph, matching_set_maximal)
    out_min = fetch_sum(sum_weighted_graph, matching_set_minimal)
    
    form_output = formatted_output(out_max,out_min)
    
    total_max = sum_total_weights(out_max)
    print("BP Matching: Highest bound for maximum:", total_max)

    total_min = sum_total_weights(out_min)
    print("BP Matching: Lowest bound for minimum:", total_min)
    
    return total_max, total_min, timing_match_minimal, timing_match_maximal


def sum_naive_script(n_matches_max, n_matches_min, sim_threshold, filename1_dup, filename2):
    cat_table1_dup = core2.data_catalog(filename1_dup)
    cat_table2 = core2.data_catalog(filename2)
    print('Loaded catalogs.')
    
    
    # NAIVE MAX MATCHING
    print("NAIVE MAX MATCHING")
    print('Performing compare all match (edit distance)...')
    now = datetime.datetime.now()
    max_compare_all_edit_match = matcher.matcher_dup_updated(n_matches_max, cat_table1_dup,cat_table2,editdistance.eval, matcher.all, sim_threshold)
    naive_time_edit_max = (datetime.datetime.now()-now).total_seconds()
    print("Naive Edit Distance Matching computation time taken: ", naive_time_edit_max, " seconds")
    #print('Compare All Matcher (Edit Distance) Performance: ' + str(core.eval_matching(compare_all_edit_match)))


#     print('Performing compare all match (jaccard distance)...')
#     now = datetime.datetime.now()
#     max_compare_all_jaccard_match = matcher.matcher_dup_updated(n_matches, cat_table1_dup,cat_table2,analyze.jaccard_calc, matcher.all, sim_threshold)
#     naive_time_jaccard = (datetime.datetime.now()-now).total_seconds()
#     print("Naive Jaccard Matching computation time taken: ", naive_time_jaccard, " seconds", "\n")
    #print('Compare All Matcher (Jaccard Distance) Performance: ' + str(core2.eval_matching(compare_all_jaccard_match)))

    # NAIVE MIN MATCHING
    print("NAIVE MIN MATCHING")
    print('Performing compare all match (edit distance)...')
    now = datetime.datetime.now()
    min_compare_all_edit_match = matcher.matcher_updated(n_matches_min, cat_table1_dup,cat_table2,editdistance.eval, matcher.all, sim_threshold)
    naive_time_edit_min = (datetime.datetime.now()-now).total_seconds()
    print("Naive Edit Distance Matching computation time taken: ", naive_time_edit_min, " seconds")
    #print('Compare All Matcher (Edit Distance) Performance: ' + str(core.eval_matching(compare_all_edit_match)))


#     print('Performing compare all match (jaccard distance)...')
#     now = datetime.datetime.now()
#     min_compare_all_jaccard_match = matcher.matcher_updated(n_matches, cat_table1,cat_table2,analyze.jaccard_calc, matcher.all, sim_threshold)
#     naive_time_jaccard = (datetime.datetime.now()-now).total_seconds()
#     print("Naive Jaccard Matching computation time taken: ", naive_time_jaccard, " seconds")
    #print('Compare All Matcher (Jaccard Distance) Performance: ' + str(core2.eval_matching(compare_all_jaccard_match)))

    naive_total_max = sum_total_weights(max_compare_all_edit_match)
    naive_total_min = sum_total_weights(min_compare_all_edit_match)
    print("NAIVE MAX Matching Bound: ", naive_total_max)
    print("NAIVE MIN Matching Bound: ", naive_total_min)
    return naive_total_max, naive_total_min, naive_time_edit_min, naive_time_edit_max


def sum_random_sample_script(n_matches_max, n_matches_min, sim_threshold, sample_size, filename1_dup, filename2):
    
    cat_table1_dup = core2.data_catalog(filename1_dup)
    cat_table2 = core2.data_catalog(filename2)
    print('Loaded catalogs.')
    
    # RANDOM SAMPLING MAX MATCHING
    print("RANDOM SAMPLE MAX MATCHING")
    print('Performing random sample match (edit distance)...')
    now = datetime.datetime.now()
    max_compare_sampled_edit_match = matcher.matcher_dup_updated(n_matches_max, cat_table1_dup,cat_table2,editdistance.eval, matcher.random_sample, sim_threshold, sample_size)
    sim_time_edit_max = (datetime.datetime.now()-now).total_seconds()
    print("Simulation-Based Edit Distance Matching computation time taken: ", sim_time_edit_max, " seconds")
    #print('Random Sample Matcher (Edit Distance) Performance: ' + str(core.eval_matching(compare_all_edit_match)))

    # print('Performing random sample match (jaccard distance)...')
    # now = datetime.datetime.now()
    # max_compare_sampled_jaccard_match = matcher.matcher_dup_updated(cat_table1_dup,cat_table2,analyze.jaccard_calc, matcher.random_sample, sim_threshold, sample_size)
    # sim_time_jaccard = (datetime.datetime.now()-now).total_seconds()
    # print("Simulation-Based Jaccard Matching computation time taken: ", sim_time_jaccard, " seconds", "\n")
    #print('Random Sample Matcher (Jaccard Distance) Performance: ' + str(core.eval_matching(compare_all_jaccard_match)))


    # RANDOM SAMPLING MIN MATCHING
    print("RANDOM SAMPLE MIN MATCHING")
    print('Performing random sample match (edit distance)...')
    now = datetime.datetime.now()
    min_compare_sampled_edit_match = matcher.matcher_updated(n_matches_min, cat_table1_dup,cat_table2,editdistance.eval, matcher.random_sample, sim_threshold, sample_size)
    sim_time_edit_min = (datetime.datetime.now()-now).total_seconds()
    print("Simulation-Based Edit Distance Matching computation time taken: ", sim_time_edit_min, " seconds")
    #print('Random Sample Matcher (Edit Distance) Performance: ' + str(core.eval_matching(compare_all_edit_match)))

    # print('Performing random sample match (jaccard distance)...')
    # now = datetime.datetime.now()
    # min_compare_sampled_jaccard_match = matcher.matcher_updated(cat_table1,cat_table2,analyze.jaccard_calc, matcher.random_sample, sim_threshold, sample_size)
    # sim_time_jaccard = (datetime.datetime.now()-now).total_seconds()
    # print("Simulation-Based Jaccard Matching computation time taken: ", sim_time_jaccard, " seconds")
    #print('Random Sample Matcher (Jaccard Distance) Performance: ' + str(core.eval_matching(compare_all_jaccard_match)))

    sampled_total_max = sum_total_weights(max_compare_sampled_edit_match)
    sampled_total_min = sum_total_weights(min_compare_sampled_edit_match)
    print("SAMPLED MAX Matching Bound: ", sampled_total_max, "\n")
    print("SAMPLED MIN Matching Bound: ", sampled_total_min)
    return sampled_total_max, sampled_total_min, sim_time_edit_min, sim_time_edit_max

def sum_results_summaries(bip_max, bip_min, naive_max, naive_min, random_max, random_min):
    print("MAX Matching Bound:")
    print("BP Matching: ", bip_sum_max)
    print("NAIVE Matching: ", naive_total_max)
    print("SAMPLED Matching: ", sampled_total_max, "\n")

    naive_total_min = sum_total_weights(min_compare_all_edit_match)
    sampled_total_min = sum_total_weights(min_compare_sampled_edit_match)
    bp_total_min = sum_total_weights(out_min)

    print("MIN Matching Bound:")
    print("BP Matching: ", bip_sum_min)
    print("NAIVE Matching: ", naive_total_min)
    print("SAMPLED Matching: ", sampled_total_min, "\n")

"""
COUNT
"""
def experiment_filter_count(filter_condition, matching_list):
    to_keep_list = []
    for (i,j,k) in matching_list:
        if int(k) >= filter_condition:
            to_keep_list.append((i,j,k))
        else:
            continue
    return to_keep_list

# Count all the matched people whose age is above 30.
def count_filter_condition(filter_condition, bip_graph):
    record_edges_to_delete = []
    
    for u,v,d in bip_graph.edges(data=True):
        val_tuple_1 = u.split("_")
        val_tuple_2 = v.split("_")
        
        if len(val_tuple_1) == 4:
            val1 = re.sub("[^0-9]", "", val_tuple_1[2])
        else: 
            val1 = re.sub("[^0-9]", "", val_tuple_1[1])
            
        if len(val_tuple_2) == 4:
            val2 =re.sub("[^0-9]", "", val_tuple_2[2])
        else:
            val2 =re.sub("[^0-9]", "", val_tuple_2[1])
            
        sum_vals = float(val1) + float(val2)
            
        if float(val1) < filter_condition or float(val2) < filter_condition or sum_vals < filter_condition:
            # record_edges_to_delete.append((u,v))
            d['weight'] = 0
        else:
            d['weight'] = sum_vals
            
    for i,j in record_edges_to_delete:
        bip_graph.remove_edge(i,j)

    return bip_graph

def count_bip_script(table_a_non_duplicated, table_b, column_name, similarity_threshold, n_matches, filter_condition):
    now = datetime.datetime.now()
    bipartite_graph_result = one_to_n.keycomp_treshold_updated_maximal_construct_graph(table_a_non_duplicated, table_b, column_name, similarity_threshold, n_matches)
    timing_tresh = (datetime.datetime.now()-now).total_seconds()
    print("---- Timing for Graph Construction with Treshold Constraint ----")
    print(timing_tresh,"seconds")
    print("The number of edges in the graph is:", bipartite_graph_result.number_of_edges(), "\n")
    
    count_edited_bip_graph = count_filter_condition(filter_condition, bipartite_graph_result)
	#count_edited_bip_graph = one_to_n.COUNT_edit_edge_weight(bipartite_graph_result)

	out_max_count = fetch_sum(count_edited_bip_graph, count_matching_set_maximal)
	out_min_count = fetch_sum(count_edited_bip_graph, count_matching_set_minimal)
	form_output_count = formatted_output(out_max_count,out_min_count)

    print("\n\n 'SUM' MAXIMAL MATCHING:")
    now = datetime.datetime.now()
	# Maximum matching for Count
	count_matching_set_maximal = nx.algorithms.matching.max_weight_matching(count_edited_bip_graph)
    timing_match_maximal = (datetime.datetime.now()-now).total_seconds()
#    print("The Maximal Matching Set is:", matching_set_maximal, "\n")
    print("---- Timing for Matching (Done on the graph constructed with the treshold constraint) ----")
    print(timing_match_maximal,"seconds")

    
    print("\n\n 'SUM' MINIMAL MATCHING RESULTS:")
    print(nx.bipartite.is_bipartite(sum_weighted_graph))
    now = datetime.datetime.now()
	count_matching_set_minimal = minimal_matching(count_edited_bip_graph)
    timing_match_minimal = (datetime.datetime.now()-now).total_seconds()
#    print("The Minimal Matching Set is:", matching_set_minimal, "\n")
    print("---- Timing for Matching (Done on the graph constructed with the treshold constraint) ----")
    print(timing_match_minimal,"seconds")
    print("The number of edges in the graph is:", count_matching_set_minimal.number_of_edges(), "\n")
    
    total_max_count = sum_total_weights(out_max_count)
	print("BP Matching: Highest bound for maximum:", total_max_count)

	total_min_count = sum_total_weights(out_min_count)
	print("BP Matching: Lowest bound for minimum:", total_min_count, "\n")
    
    return total_max_count, total_min_count, timing_match_minimal, timing_match_maximal

def count_naive_script(n_matches_max, n_matches_min, sim_threshold, filename1_dup, filename2, filter_condition):

    cat_table1_dup = core2.data_catalog(filename1_dup)
    cat_table2 = core2.data_catalog(filename2)
    print('Loaded catalogs.')
    
    # NAIVE MAX MATCHING
    print("NAIVE MAX MATCHING")
    print('Performing compare all match (edit distance)...')
    now = datetime.datetime.now()
    max_compare_all_edit_match = matcher.matcher_dup_updated(n_matches_max, cat_table1_dup,cat_table2,editdistance.eval, matcher.all, sim_threshold)
    naive_time_edit_max = (datetime.datetime.now()-now).total_seconds()
    print("Naive Edit Distance Matching computation time taken: ", naive_time_edit_max, " seconds")

    # NAIVE MIN MATCHING
    print("NAIVE MIN MATCHING")
    print('Performing compare all match (edit distance)...')
    now = datetime.datetime.now()
    min_compare_all_edit_match = matcher.matcher_updated(n_matches_min, cat_table1_dup,cat_table2,editdistance.eval, matcher.all, sim_threshold)
    naive_time_edit_min = (datetime.datetime.now()-now).total_seconds()
    print("Naive Edit Distance Matching computation time taken: ", naive_time_edit_min, " seconds")

	# Apply predicate constraint to count for max matching
	filtered_naive_max = experiment_filter_count(filter_condition, min_compare_all_edit_match)
	naive_total_max_count = sum_total_weights(filtered_naive_max)

	# Apply predicate constraint to count for min matching
	filtered_naive_min = experiment_filter_count(filter_condition, max_compare_all_edit_match)
	naive_total_min_count = sum_total_weights(filtered_naive_min)
    print("NAIVE MAX Matching Bound: ", naive_total_max_count)
    print("NAIVE MIN Matching Bound: ", naive_total_min_count)
    return naive_total_max_count, naive_total_min_count, naive_time_edit_min, naive_time_edit_max

def count_random_sample_script(n_matches_max, n_matches_min, sim_threshold, sample_size, filename1_dup, filename2, filter_condition):
	cat_table1_dup = core2.data_catalog(filename1_dup)
    cat_table2 = core2.data_catalog(filename2)
    print('Loaded catalogs.')
    
    # RANDOM SAMPLING MAX MATCHING
    print("RANDOM SAMPLE MAX MATCHING")
    print('Performing random sample match (edit distance)...')
    now = datetime.datetime.now()
    max_compare_sampled_edit_match = matcher.matcher_dup_updated(n_matches_max, cat_table1_dup,cat_table2,editdistance.eval, matcher.random_sample, sim_threshold, sample_size)
    sim_time_edit_max = (datetime.datetime.now()-now).total_seconds()
    print("Simulation-Based Edit Distance Matching computation time taken: ", sim_time_edit_max, " seconds")

    # RANDOM SAMPLING MIN MATCHING
    print("RANDOM SAMPLE MIN MATCHING")
    print('Performing random sample match (edit distance)...')
    now = datetime.datetime.now()
    min_compare_sampled_edit_match = matcher.matcher_updated(n_matches_min, cat_table1_dup,cat_table2,editdistance.eval, matcher.random_sample, sim_threshold, sample_size)
    sim_time_edit_min = (datetime.datetime.now()-now).total_seconds()
    print("Simulation-Based Edit Distance Matching computation time taken: ", sim_time_edit_min, " seconds")

    sampled_total_max = sum_total_weights(max_compare_sampled_edit_match)
    sampled_total_min = sum_total_weights(min_compare_sampled_edit_match)


	# Apply predicate constraint to count for max matching
	filtered_sampled_max = experiment_filter_count(filter_condition, min_compare_sampled_edit_match)
	sampled_total_max_count = sum_total_weights(filtered_sampled_max)

	# Apply predicate constraint to count for min matching
	filtered_sampled_min = experiment_filter_count(filter_condition, max_compare_sampled_edit_match)
	sampled_total_min_count = sum_total_weights(filtered_sampled_min)

    print("SAMPLED MAX Matching Bound: ", sampled_total_max_count, "\n")
    print("SAMPLED MIN Matching Bound: ", sampled_total_min_count)
    return sampled_total_max_count, sampled_total_min_count, sim_time_edit_min, sim_time_edit_max

import csv

def create_csv_table(exp_name):
	with open('{}.csv'.format(exp_name), mode='w') as exp_filename:
	    experiment_writer = csv.writer(exp_filename, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
	    experiment_writer.writerow(['Bipartite Min Matching', 'Bipartite Max Matching', 'Naive Min Matching', 'Naive Max Matching', 'Sampled Min Matching', 'Sampled Max Matching', 'TIMING Bipartite Min Matching', 'TIMING Bipartite Max Matching', 'TIMING Naive Min Matching', 'TIMING Naive Max Matching', 'TIMING Sampled Min Matching', 'TIMING Sampled Max Matching'])
def table_csv_output(bip_max, bip_min, naive_max, naive_min, sampled_max, sampled_min, exp_name, timing_match_minimal, timing_match_maximal, naive_time_edit_min, naive_time_edit_max, sim_time_edit_min, sim_time_edit_max):
	with open('{}.csv'.format(exp_name), mode='a') as exp_filename:
		experiment_writer = csv.writer(exp_filename, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
		experiment_writer.writerow([bip_max, bip_min, naive_max, naive_min, sampled_max, sampled_min, timing_match_minimal, timing_match_maximal, naive_time_edit_min, naive_time_edit_max, sim_time_edit_min, sim_time_edit_max])
