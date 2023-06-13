import math
from datetime import timedelta
import numpy as np

def std_dev(samples):
	N = len(samples)
	sample_mean = sum(samples) / N
	res = math.sqrt(sum([(x - sample_mean)**2 for x in samples]) / N)
	return res

def entropy(values):
	sum_values = sum(values)
	if sum_values == 0:
		return 1
	new_values = [x/sum_values for x in values]
	res = 0
	for x in new_values:
		res += x * np.log2(x)
	return -res

def time_to_str(seconds):
	str_time = str(timedelta(seconds=seconds))
	res = ""
	hms = str_time.split()[-1].split(":")
	res = f"{hms[0]}H : {hms[1]}M : {hms[2]}S"
	if "day" in str_time:
		days = str_time.split()[0]
		hms = str_time.split()[-1].split(":")
		res = f"{days}D : " + res
	return res

def eucl_dist(x, y):
	sum_diff = 0
	for idx in range(len(x)):
		sum_diff += (y[idx] - x[idx])**2
	return math.sqrt(sum_diff)

def solution_error(input_opt, input_opt_found):
	if input_opt == None:
		return ""
	else:
		if type(input_opt[0]) != list:
			res = eucl_dist(input_opt_found, input_opt)
			return res
		else:
			results = []
			for inp in input_opt:
				results.append(eucl_dist(input_opt_found, inp))
			res = min(results)
			return res

def objective_function_error(opt, opt_found):
	if opt == 0:
		return abs(opt_found)
	else:
		return abs((opt_found - opt) / opt)

