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

def write_log_file(path, string_res):
	with open(path, "w") as f:
		f.write(string_res)
		f.close()

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

def find_crs_thresholds(errors):
	errors = sorted(errors)
	N = len(errors)
	ratios = [0.25, 0.50, 0.75, 1.0]
	res = {0.0 : 0.0}
	i = 0
	idx_errors = 0
	n_corrects = 1
	while i < len(ratios):
		while n_corrects/N < ratios[i]:
			n_corrects += 1
			idx_errors += 1
		res[ratios[i]] = errors[idx_errors]
		i += 1
	return res

def correctness_ratios(errors):
	errors = sorted(errors)
	i = 0
	new_errors = []
	while i < len(errors):
		temp_errors = []
		current_err = errors[i]
		new_errors.append(current_err)
		temp_errors.append(current_err)
		i += 1
		while i < len(errors) and entropy(temp_errors) < 0.5:
			temp_errors.append(errors[i])
			i += 1
	errors = new_errors
	thresholds = [0] + sorted(list(set(errors)))
	N = len(errors)
	res = []
	i = 0
	n_corrects = 0
	last_ratio = 0
	for threshold in thresholds:
		while i < N and errors[i] <= threshold:
				n_corrects += 1
				i += 1
		ratio = n_corrects/N
		if ratio > last_ratio:
			if len(res) != 0 and threshold == res[-1][0]:
				res = res[:-1]
			res.append((threshold, ratio))
			last_ratio = ratio
		
	# add last value to avoid domain error in latex
	last_thr = thresholds[-1]
	if "e" in str(last_thr):
		power = int(str(last_thr).split("e-")[1])
		int_part = int(str(last_thr).split(".")[0])
		new_thr = float(str(int_part) + "e-" + str(power-1))
		res.append((new_thr, 1.0))

	else:
		first_n = int(str(last_thr)[0])
		last_thr = str(first_n+1) + str(last_thr)[1:]
		res.append((last_thr, 1.0))
	return res
	
def compute_perc_values(list_values, decimal_precision):
	new_list_values = []
	for val in list_values:
		str_val = str(val).split(".")
		if len(str_val) > 1 and len(str_val[1]) > decimal_precision:
			val = float(str_val[0] + "." + str_val[1][:decimal_precision])
		new_list_values.append(val)
	list_values = new_list_values
	N = len(list_values)
	uniq_val = sorted(list(set(list_values)))
	list_values = sorted(list_values)
	val_dict = {}
	res = []
	for val in uniq_val:
		val_dict[val] = 0
	for val in list_values:
		val_dict[val] += 1
	for val in uniq_val:
		perc_val = float((val_dict[val] / N) * 100)
		str_perc_val = str(perc_val).split(".")
		if len(str_perc_val[1]) > 4:
			perc_val = float(str_perc_val[0] + "." + str_perc_val[1][:4])
		res.append((val, perc_val))
	return res


def cumulative_exact_errors_plot(errors):
	errors = sorted(errors)
	# remove errors with similar values to decrease the number of elements in the csv
	# this reduction is done only for the latex error plots
	# all the errors returned during the runs are saved in the log_runs.csv file
	i = 0
	new_errors = []
	while i < len(errors):
		current_err = errors[i]
		new_errors.append(current_err)
		temp_errors = [current_err]
		i += 1
		while i < len(errors) and entropy(temp_errors) < 0.5:
			temp_errors.append(errors[i])
			i += 1
	errors = new_errors

	N = len(errors)
	uniq_ord_errors = sorted(list(set(errors)))
	n_corrects = 0
	i = 0
	res = []
	uniq_res = []
	prev_err = None
	prev_uniq_err = None
	dict_bar_err = {}

	for error in uniq_ord_errors:
		n_uniq_err = 0
		while i < N and errors[i] <= error:
			n_corrects += 1
			n_uniq_err += 1
			i += 1
		perc_err = (n_corrects / N) * 100
		perc_uniq_err = (n_uniq_err / N) * 100

		if prev_err != None:
			res.append((error, prev_err))
		prev_err = float(perc_err)
		str_prev_err = str(prev_err).split(".")
		if len(str_prev_err[1]) > 4:	
			prev_err = float(str_prev_err[0] + "." + str_prev_err[1][:4])
		res.append((error, prev_err))
		if "e-" in str(error):
			new_uniq_err = float("1e-" + str(error).split("e-")[1])
		elif "." in str(error):
			# if integral part is <= 9
			if len(str(error).split(".")[0]) == 1:
				new_uniq_err = float(str(error).split(".")[0])
				if new_uniq_err == 0 and error != 0:
					new_uniq_err = 0.01
			else:
				str_int_part = str(error).split(".")[0]
				len_precision = len(str_int_part)-1
				if int(str_int_part[1]) < 5:
					new_uniq_err = str_int_part[0]
					new_uniq_err = new_uniq_err + "0"*len_precision
				else:
					new_uniq_err= str(int(str_int_part[0])+1)
					new_uniq_err = new_uniq_err+ "0"*len_precision
					
				new_uniq_err = float(new_uniq_err)
		else:
			new_uniq_err = float(error)

		if new_uniq_err in dict_bar_err.keys():
			dict_bar_err[new_uniq_err] += n_uniq_err
		else:
			dict_bar_err[new_uniq_err] = n_uniq_err

	for err in dict_bar_err:
		perc_uniq = (dict_bar_err[err] / N) * 100
		str_perc_uniq_err = str(perc_uniq).split(".")
		if len(str_perc_uniq_err[1]) > 4:
			perc_uniq = float(str_perc_uniq_err[0] + "." + str_perc_uniq_err[1][:4])
		dict_bar_err[err] = perc_uniq

	uniq_res = [(err, dict_bar_err[err]) for err in sorted(list(dict_bar_err.keys()))]

	# add last value to avoid domain error in latex
	last_error = uniq_ord_errors[-1]
	if "e" in str(last_error):
		power = int(str(last_error).split("e-")[1])
		int_part = int(str(last_error).split(".")[0])
		new_err = float(str(int_part) + "e-" + str(power-1))
		res.append((new_err, 100.00))
	else:
		first_n = int(str(last_error)[0])
		last_error = str(first_n+1) + str(last_error)[1:]
		res.append((last_error, 100.00))
	return (res, uniq_res)