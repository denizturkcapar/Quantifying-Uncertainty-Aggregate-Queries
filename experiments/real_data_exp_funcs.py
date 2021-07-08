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
import experiment_funcs


def data_to_df(file1, file2):
	table_a = one_to_n.lat_convert_df("../Amazon-GoogleProducts/Amazon.csv")
	table_b = one_to_n.lat_convert_df("../Amazon-GoogleProducts/GoogleProducts.csv")


def real_data_1_to_n_sum_results(file1, file2, result_filename, bp_sim, naive_sim, sampled_sim, actual_n, bp_n, naive_n, sampled_n, table_a_length, sample_num, is_age_skewed=False, num_swaps=None):
	experiment_funcs.create_csv_table(result_filename)
	table_a_non_duplicated, table_b = data_to_df(file1, file2)
	if num_swaps != None:
		# Bipartite Matching Script
		total_max, total_min, bip_min_matching_time, bip_max_matching_time, out_max, out_min = experiment_funcs.sum_bip_script(table_a_non_duplicated, table_b, "name", bp_sim, bp_n, tables_map, num_swaps)

		# Run Naive Matching Script
		naive_total_max, naive_total_min, naive_min_matching_time, naive_max_matching_time, naive_max, naive_min = experiment_funcs.sum_naive_script(naive_sim, "table1", "table2", table_a_non_duplicated, naive_n, "naive_dup", num_swaps)
		
		# Run Random Matching Script
		sampled_total_max, sampled_total_min, sampled_min_matching_time, sampled_max_matching_time, sampled_max, sampled_min = experiment_funcs.sum_random_sample_script(sampled_sim, sample_num, "table1", "table2", table_a_non_duplicated, sampled_n, "random_dup", num_swaps)
	else:
		# Bipartite Matching Script
		total_max, total_min, bip_min_matching_time, bip_max_matching_time, out_max, out_min = experiment_funcs.sum_bip_script(table_a_non_duplicated, table_b, "name", bp_sim, bp_n, tables_map)

		# Run Naive Matching Script
		naive_total_max, naive_total_min, naive_min_matching_time, naive_max_matching_time, naive_max, naive_min = experiment_funcs.sum_naive_script(naive_sim, "table1", "table2", table_a_non_duplicated, naive_n, "naive_dup")
		
		# Run Random Matching Script
		sampled_total_max, sampled_total_min, sampled_min_matching_time, sampled_max_matching_time, sampled_max, sampled_min = experiment_funcs.sum_random_sample_script(sampled_sim, sample_num, "table1", "table2", table_a_non_duplicated, sampled_n, "random_dup")

	# Run Accuracy Evaluation
	eval_records = experiment_funcs.full_evaluation(out_min, out_max, naive_min, naive_max, sampled_min, sampled_max, perfect_mapping)

	# Record Experiment Results
	experiment_funcs.table_csv_output(total_min, total_max, naive_total_min, naive_total_max, sampled_total_min, sampled_total_max, perfect_mapping_sum_result, result_filename, bip_min_matching_time, bip_max_matching_time, naive_min_matching_time, naive_max_matching_time, sampled_min_matching_time, sampled_max_matching_time, eval_records)



def show_experiment_1(file1, file2, experiment_name, sim_thresh):
	real_data_1_to_n_sum_results(file1, file2, "realdata_exp2_n1", sim_thresh,sim_thresh,sim_thresh,1,1,1,1,100,100, is_age_skewed)
	df = pd.read_csv(results_file)

	bp_min_med = df['Bipartite Min Matching'].median()
	bp_max_med = df['Bipartite Max Matching'].median()
	naive_min_med = df['Naive Min Matching'].median()
	naive_max_med = df['Naive Max Matching'].median()
	sampled_min_med = df['Sampled Min Matching'].median()
	sampled_max_med = df['Sampled Max Matching'].median()

	perf_threshold = df['Perfect Matching SUM'].median()

	minThreshold = [bp_min_med, naive_min_med, sampled_min_med]

	maxThreshold = [bp_max_med, naive_max_med, sampled_max_med]

	indices = np.arange(len(maxThreshold))
	
	width = 0.8

	fig = plt.figure()

	ax = fig.add_subplot(111)

	max_limit = max(max(maxThreshold),perf_threshold)

	ax.set_title(experiment_name)
	ax.set_ylabel('Median Sum')
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


def show_experiment_2(file1, file2, experiment_name, sim_thresh, is_age_skewed=False, num_swaps=None):
	
	real_data_1_to_n_sum_results(file1, file2, "realdata_exp2_n1", sim_thresh,sim_thresh,sim_thresh,1,1,1,1,100,100, is_age_skewed)
	real_data_1_to_n_sum_results(file1, file2, "realdata_exp2_n2", sim_thresh,sim_thresh,sim_thresh,2,2,2,2,100,150, is_age_skewed)
	real_data_1_to_n_sum_results(file1, file2, "realdata_exp2_n3", sim_thresh,sim_thresh,sim_thresh,3,3,3,3,100,200, is_age_skewed)
	real_data_1_to_n_sum_results(file1, file2, "realdata_exp2_n4", sim_thresh,sim_thresh,sim_thresh,4,4,4,4,100,250, is_age_skewed)
	real_data_1_to_n_sum_results(file1, file2, "realdata_exp2_n5", sim_thresh,sim_thresh,sim_thresh,5,5,5,5,100,300, is_age_skewed)
	real_data_1_to_n_sum_results(file1, file2, "realdata_exp2_n6", sim_thresh,sim_thresh,sim_thresh,6,6,6,6,100,300, is_age_skewed)
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
	plt.plot(n, naive_widths, '-o', color='g', label='Naive')
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
	bp_width1 = (bp_max_med - perf_threshold) / perf_threshold
	naive_width1 = (naive_max_med - perf_threshold) / perf_threshold
	sampled_width1 = (sampled_max_med - perf_threshold) / perf_threshold

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


def show_experiment_4(experiment_name, user_n, is_age_skewed=False, num_swaps=None):
	
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

# EXPERIMENT 5 - INAPPLICABLE TO THIS REAL DATA INSTANCE

# def show_experiment_5_balanced(experiment_name, user_n, sim_thresh, is_age_skewed=False):
# 	# create_1_to_n_sum_results("25_exp5_skewed", sim_thresh,sim_thresh,sim_thresh,user_n,user_n,user_n,user_n,100,100, is_age_skewed, 25)
# 	# create_1_to_n_sum_results("50_exp5_skewed", sim_thresh,sim_thresh,sim_thresh,user_n,user_n,user_n,user_n,100,150, is_age_skewed, 50)
# 	# create_1_to_n_sum_results("75_exp5_skewed", sim_thresh,sim_thresh,sim_thresh,user_n,user_n,user_n,user_n,100,200, is_age_skewed, 75)
# 	# create_1_to_n_sum_results("100_exp5_skewed", sim_thresh,sim_thresh,sim_thresh,user_n,user_n,user_n,user_n,100,250, is_age_skewed, 100)
# 	# create_1_to_n_sum_results("125_exp5_skewed", sim_thresh,sim_thresh,sim_thresh,user_n,user_n,user_n,user_n,100,225, is_age_skewed, 125)
# 	# create_1_to_n_sum_results("150_exp4_skewed", sim_thresh,sim_thresh,sim_thresh,user_n,user_n,user_n,user_n,100,300, is_age_skewed, 150)
# 	# create_1_to_n_sum_results("175_exp4_skewed", sim_thresh,sim_thresh,sim_thresh,user_n,user_n,user_n,user_n,100,300, is_age_skewed, 175)

# 	# sim thresh level = 1, optimal
# 	bp_width1, naive_width1, sampled_width1 = model_overestimation_calc_helper("25_exp5_skewed")
# 	# sim thresh level = 2
# 	bp_width2, naive_width2, sampled_width2 = model_overestimation_calc_helper("50_exp5_skewed")
# 	# sim thresh level = 3
# 	bp_width3, naive_width3, sampled_width3 = model_overestimation_calc_helper("75_exp5_skewed")
# 	# sim thresh level = 4
# 	bp_width4, naive_width4, sampled_width4 = model_overestimation_calc_helper("100_exp5_skewed")
# 	# sim thresh level = 5, worst
# 	bp_width5, naive_width5, sampled_width5 = model_overestimation_calc_helper("125_exp5_skewed")
# 	# sim thresh level = 6, worst
# 	bp_width6, naive_width6, sampled_width6 = model_overestimation_calc_helper("150_exp4_skewed")
# 	# sim thresh level = 7, worst
# 	bp_width7, naive_width7, sampled_width7 = model_overestimation_calc_helper("175_exp4_skewed")

# 	bp_widths = [bp_width1, bp_width2, bp_width3, bp_width4, bp_width5, bp_width6, bp_width7]
# 	naive_widths = [naive_width1, naive_width2, naive_width3, naive_width4, naive_width5, naive_width6, naive_width7]
# 	sample_widths = [sampled_width1, sampled_width2, sampled_width3, sampled_width4, sampled_width5, sampled_width6, sampled_width7]
# 	n = [25,50,75,100,125,150,175]

# 	# plt.scatter(n,bp_widths, color='k')
# 	plt.plot(n, bp_widths, '-o', color='r', label='Bipartite')
# 	# plt.scatter(n,naive_widths, color='g')
# 	plt.plot(n, naive_widths, '-o', color='g', label='Naive')
# 	# plt.scatter(n,sample_widths, color='b')
# 	plt.plot(n, sample_widths, '-o', color='b', label='Sampled')
# 	plt.xlabel('Quality of the Similarity Metric (Worst to Best)')
# 	plt.ylabel('Median Over-estimation Rate')
# 	plt.title(experiment_name)
# 	plt.xticks(n,n)
# 	plt.legend()
# 	plt.savefig(experiment_name, dpi=300)
# 	plt.show()
