from Matching import core2
import editdistance
import math
import random

"""
Calculates maximum weight for the matching

Input: keys from 2 tables
Output: weight for each matching to be used in the weight part of constructing the graph
"""
def calc_max_weight_edit(key1, key2, distance_func):
    weight = (1)/(1+distance_func(key1,key2))
    # print(weight)
    return weight


def matcher(d1, d2, distance_fn, sampler_fn, sample_size=100):

	match = []	

	for e1 in d1:
		min_dist = math.inf
		min_dist_id = None
		for e2 in sampler_fn(d2, sample_size):
			dist = distance_fn(e1['name'],e2['name'])
			if dist < min_dist:
				min_dist = dist
				min_dist_id = e2['name']
				sum_total = int(e1['age']) + int(e2['age'])
		match.append((e1['name'],min_dist_id, sum_total))

	return match

def matcher_dup(d1, d2, distance_fn, sampler_fn, sample_size=100):

	match = []
	for e1 in d1:
		# Note that d1 is always the duplicated table
		# So the entries of names need to cleaned for the "_number" adjustment during the matching
		cleaned_e1 = e1['name'].split("_")[0]
		min_dist = math.inf
		min_dist_id = None
		for e2 in sampler_fn(d2, sample_size):
			dist = distance_fn(cleaned_e1,e2['name'])
			if dist < min_dist:
				min_dist = dist
				min_dist_id = e2['name']
				sum_total = int(e1['age']) + int(e2['age'])
		match.append((e1['name'],min_dist_id, sum_total))

	return match

def matcher_updated(d1, d2, distance_fn, sampler_fn, required_distance, sample_size=100):

	match = []	

	for e1 in d1:
		for e2 in sampler_fn(d2, sample_size):
			distance = calc_max_weight_edit(e1['name'], e2['name'], distance_fn)
#			print("first data: ", e1['name'], "second data: ", e2['name'], "distance: ", distance)
			if distance >= required_distance:
				sum_total = int(e1['age']) + int(e2['age'])
				match.append((e1['name'],e2['name'], sum_total))
				continue
			else:
				continue
		continue
	return match


def matcher_dup_updated(d1, d2, distance_fn, sampler_fn, required_distance, sample_size=100):

	match = []	

	for e1 in d1:
		# Note that d1 is always the duplicated table
		# So the entries of names need to cleaned for the "_number" adjustment during the matching
		cleaned_e1 = e1['name'].split("_")[0]
		for e2 in sampler_fn(d2, sample_size):
			distance = calc_max_weight_edit(cleaned_e1, e2['name'], distance_fn)
#			print("first data: ", e1['name'], "second data: ", e2['name'], "distance: ", distance)
			if distance >= required_distance:
				sum_total = int(e1['age']) + int(e2['age'])
				match.append((e1['name'],e2['name'], sum_total))
				continue
			else:
				continue
			continue
	return match

def all(catalog, k=None):
	return catalog

def random_sample(catalog, k):
	k = min(k,catalog.length)
	return random.sample(list(catalog),k)


