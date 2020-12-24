from Matching import core
import editdistance
import math
import random

def matcher(d1, d2, distance_fn, sampler_fn, sample_size=100):

	match = []	

	for e1 in d1:
		min_dist = math.inf
		min_dist_id = None
		for e2 in sampler_fn(d2, sample_size):
			dist = distance_fn(e1['title'],e2['title'])
			if dist < min_dist:
				min_dist = dist
				min_dist_id = e2['id']
		match.append((e1['id'],min_dist_id))

	return match

def all(catalog, k=None):
	return catalog

def random_sample(catalog, k):
	k = min(k,catalog.length)
	return random.sample(list(catalog),k)


