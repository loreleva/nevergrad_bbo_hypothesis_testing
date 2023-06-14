import os, math
from utils import *

csv_sep = ";"

# create all the csv files with the corresponding header
def init_log_files(filepath):
	# single runs log
	filepath_log = os.path.join(filepath, "log_runs.csv")
	header = (
				f"Index{csv_sep} "
				f"Run{csv_sep} "
				f"Iteration{csv_sep} "
				f"Optimum Found{csv_sep} "
				f"Global Optimum{csv_sep} "
				f"Objective Function Error{csv_sep} "
				f"Input Optimum Found{csv_sep} "
				f"Input Optimum{csv_sep} "
				f"Solution Error{csv_sep} "
				f"S{csv_sep} "
				f"Number of Asks{csv_sep} "
				f"Time{csv_sep} "
				f"Max RAM Megabyte Usage\n"
			)
	with open(filepath_log, "w") as f:
		f.write(header)
		f.close()


	# final result log
	filepath_log = os.path.join(filepath, "log_results.csv")
	header = (
				f"Runs of nevergrad{csv_sep} "
				f"Number external iterations{csv_sep} "
				f"Optimum Found{csv_sep} "
				f"Function Optimum{csv_sep} "
				f"Input Optimum Found{csv_sep} "
				f"Function Input Optimum{csv_sep} "
				f"Objective function error{csv_sep} "
				f"Solution Error{csv_sep}"
				f"Correctness ratio{csv_sep}"
				f"Multi core execution time{csv_sep} "
				f"Single core execution time{csv_sep} "
				f"Mean time per run{csv_sep} "
				f"Standard deviation time per run{csv_sep} "
				f"Total number of asks{csv_sep} "
				f"Mean number of asks per run{csv_sep} "
				f"Standard deviation number of asks per run{csv_sep} "
				f"Max RAM Megabyte usage{csv_sep} "
				f"Mean ram usage{csv_sep} "
				f"Standard deviation ram usage{csv_sep} "
				f"Speedup{csv_sep} "
				f"Efficiency\n"
			)
	with open(filepath_log, "w") as f:
		f.write(header)
		f.close()


	# S values log
	filepath_log = os.path.join(filepath, "log_s_values.csv")
	header = (
			f"Run{csv_sep} "
			f"Internal iteration{csv_sep} "
			f"N{csv_sep} "
			f"S value{csv_sep} "
			f"Function Optimum{csv_sep} "
			f"Objective function error{csv_sep} "
			f"Input S value{csv_sep} "
			f"Function Input Optimum{csv_sep} "
			f"Solution Error\n"
		)
	with open(filepath_log, "w") as f:
		f.write(header)
		f.close()


	# S values plot log
	filepath_log = os.path.join(filepath, "log_s_values_plot.csv")
	header = (
			f"Run{csv_sep} "
			f"S\n"
		)
	with open(filepath_log, "w") as f:
		f.write(header)
		f.close()


	# correctness ratio plot
	filepath_log = os.path.join(filepath, "log_correctness_ratio.csv")
	header = (
			f"Run{csv_sep} "
			f"Error{csv_sep} "
			f"Num Correct Ratio 0.00{csv_sep} "
			f"Num Correct Ratio 0.25{csv_sep} "
			f"Num Correct Ratio 0.50{csv_sep} "
			f"Num Correct Ratio 0.75{csv_sep} "
			f"Num Correct Ratio 1.00\n"
		)
	with open(filepath_log, "w") as f:
		f.write(header)
		f.close()

	# correctness ratio plot complete
	filepath_log = os.path.join(filepath, "log_correctness_ratio_complete.csv")
	with open(filepath_log, "w") as f:
		f.write(header)
		f.close()


	# correctness ratio plot legend
	filepath_log = os.path.join(filepath, "log_thresholds_correctness_plot.csv")
	header = (
			f"Ratio 0.00{csv_sep} " 
			f"Threshold Ratio 0.00{csv_sep} "
			f"Ratio 0.25{csv_sep} "
			f"Threshold Ratio 0.25{csv_sep} "
			f"Ratio 0.50{csv_sep} "
			f"Threshold Ratio 0.50{csv_sep} "
			f"Ratio 0.75{csv_sep} "
			f"Threshold Ratio 0.75{csv_sep} "
			f"Ratio 1.00{csv_sep} "
			f"Threshold Ratio 1.00\n"
		)
	with open(filepath_log, "w") as f:
		f.write(header)
		f.close()


	# correctness ratios plot
	filepath_log = os.path.join(filepath, "log_correctness_ratios.csv")
	header = (
			f"Index{csv_sep} "
			f"Threshold{csv_sep} "
			f"Correctness ratio\n"
		)
	with open(filepath_log, "w") as f:
		f.write(header)
		f.close()

	# correctness ratios plot complete
	filepath_log = os.path.join(filepath, "log_correctness_ratios_complete.csv")
	with open(filepath_log, "w") as f:
		f.write(header)
		f.close()


	# percentage time plot
	filepath_log = os.path.join(filepath, "log_exec_time_plot.csv")
	header = (
			f"Time{csv_sep} "
			f"Percentage\n"
		)
	with open(filepath_log, "w") as f:
		f.write(header)
		f.close()


	# percentage ram usage plot
	filepath_log = os.path.join(filepath, "log_ram_plot.csv")
	header = (
			f"Ram{csv_sep} "
			f"Percentage\n"
		)
	with open(filepath_log, "w") as f:
		f.write(header)
		f.close()


	# percentage num function evaluations plot
	filepath_log = os.path.join(filepath, "log_num_f_evals_plot.csv")
	header = (
			f"f evals{csv_sep} "
			f"Percentage\n"
		)
	with open(filepath_log, "w") as f:
		f.write(header)
		f.close()


	# errors plot
	filepath_log = os.path.join(filepath, "log_errors_plot.csv")
	header = (
			f"Error{csv_sep} "
			f"Percentage\n"
		)
	with open(filepath_log, "w") as f:
		f.write(header)
		f.close()

	filepath_log = os.path.join(filepath, "log_errors_bar_plot.csv")
	with open(filepath_log, "w") as f:
		f.write(header)
		f.close()


def write_single_run_log(filepath, res, num_run, iteration, N, opt, input_opt, S, verbose=False):
	filepath_log = os.path.join(filepath, "log_runs.csv")
	log_str = (
				f"{num_run-1}{csv_sep} "
				f"{num_run}{csv_sep} "
				f"{iteration}/{N}{csv_sep} "
				f"{res['opt']}{csv_sep} "
				f"{opt}{csv_sep} "
				f"{objective_function_error(opt, res['opt'])}{csv_sep} "
				f"{res['input opt']}{csv_sep} "
				f"{input_opt}{csv_sep} "
				f"{solution_error(input_opt, res['input opt'])}{csv_sep} "
				f"{S}{csv_sep} "
				f"{res['num evaluations']}{csv_sep} "
				f"{res['process time']}{csv_sep} "
				f"{res['max ram usage']}\n"	
			)

	if verbose:
		print("Single Run Log:\n" + log_str)
	
	with open(filepath_log, "a") as f:
		f.write(log_str)
		f.close()


def write_final_result_log(filepath, num_run, num_external_iterations, opt_found, opt, input_opt_found, input_opt, correctness_ratio, multicore_time, processes_runs_time, num_proc, num_asks, ram_usage, max_ram_usage, verbose):
	filepath_log = os.path.join(filepath, "log_results.csv")

	# compute results
	single_core_time = sum(processes_runs_time)
	speedup = single_core_time/multicore_time
	efficiency = speedup/num_proc
	mean_time_per_run = sum(processes_runs_time) / len(processes_runs_time)
	std_dev_time_per_run = std_dev(processes_runs_time)
	total_num_asks = sum(num_asks)
	mean_number_asks = total_num_asks/len(num_asks)
	std_dev_number_asks = std_dev(num_asks)
	mean_ram_usage = sum(ram_usage)/len(ram_usage)
	std_dev_ram_usage = std_dev(ram_usage)

	log_str = (
			f"{num_run}{csv_sep} "
			f"{num_external_iterations}{csv_sep} "
			f"{opt_found}{csv_sep} "
			f"{opt}{csv_sep} "
			f"{input_opt_found}{csv_sep} "
			f"{input_opt}{csv_sep} "
			f"{objective_function_error(opt, opt_found)}{csv_sep} "
			f"{solution_error(input_opt, input_opt_found)}{csv_sep} "
			f"{correctness_ratio}{csv_sep} "
			f"{time_to_str(multicore_time)}{csv_sep} "
			f"{time_to_str(single_core_time)}{csv_sep} "
			f"{mean_time_per_run}{csv_sep} "
			f"{std_dev_time_per_run}{csv_sep} "
			f"{total_num_asks}{csv_sep} "
			f"{mean_number_asks}{csv_sep} "
			f"{std_dev_number_asks}{csv_sep} "
			f"{max_ram_usage}{csv_sep} "
			f"{mean_ram_usage}{csv_sep} "
			f"{std_dev_ram_usage}{csv_sep} "
			f"{speedup}{csv_sep} "
			f"{efficiency}\n"
		)

	if verbose:
		print("Final Result Log:\n" + log_str)

	with open(filepath_log, "a") as f:
		f.write(log_str)
		f.close()


def write_s_values_log(filepath, s_values, opt, input_opt, N, verbose):
	filepath_log = os.path.join(filepath, "log_s_values.csv")

	log_str = ""
	for s_value in s_values:
		log_str += (
				f"{s_value['run number']}{csv_sep} "
				f"{s_value['internal iteration']}{csv_sep} "
				f"{N}{csv_sep} "
				f"{s_value['S value']}{csv_sep} "
				f"{opt}{csv_sep} "
				f"{objective_function_error(opt, s_value['S value'])}{csv_sep} "
				f"{s_value['input']}{csv_sep} "
				f"{input_opt}{csv_sep} "
				f"{solution_error(input_opt, s_value['input'])}\n"
			)

	if verbose:
		print("S values log:\n" + log_str)

	with open(filepath_log, "a") as f:
		f.write(log_str)
		f.close()


def write_s_values_plot_log(filepath, s_values, num_run):
	filepath_log = os.path.join(filepath, "log_s_values_plot.csv")

	log_str = ""
	
	prev_s_value = None
	for s_value in s_values:
		# if the current s value is not the first, write the old value with idx as current_idx-1
		if prev_s_value != None:
			log_str += (
					f"{s_value['run number']-1}{csv_sep} "
					f"{prev_s_value}\n"
				)
		log_str += (
				f"{s_value['run number']}{csv_sep} "
				f"{s_value['S value']}\n"
			)

	# write last result until index num_run
	if s_value['run number'] != num_run:
		log_str += (
				f"{num_run}{csv_sep} "
				f"{s_value['S value']}\n"
			)

	with open(filepath_log, "a") as f:
		f.write(log_str)
		f.close()


def write_correctness_ratio_plot_log(filepath, errors, num_run):
	filepath_log = os.path.join(filepath, "log_correctness_ratio.csv")

	# compute thresholds to obtain correctness ratios of 0, 0.25, 0.50, 0.75 and 1
	ratios_thresholds = find_crs_thresholds(errors)
	ratios = [0.0, 0.25, 0.50, 0.75, 1.0]
	# counter of correct results for each ratio
	num_correct = [0, 0, 0, 0, 0]

	log_str = ""

	# if number of runs too large, sample only a fraction of values
	if num_run > 7000:
		# iterate over a fraction of errors
		for idx in range(0, num_run, math.ceil(num_run/7000)):
			# iterate for each ratio
			for idx_ratio in range(len(ratios)):
				# if error is "counted" as correct for the corresponding threshold of the ratio, count it
				if errors[idx] <= ratios_thresholds[ratios[idx_ratio]]:
					num_correct[idx_ratio] += 1

			log_str += (f"{idx+1}{csv_sep} "
						f"{errors[idx]}{csv_sep} "
						f"{num_correct[0]}{csv_sep} "
						f"{num_correct[1]}{csv_sep} "
						f"{num_correct[2]}{csv_sep} "
						f"{num_correct[3]}{csv_sep} "
						f"{num_correct[4]}\n"
					)

	# otherwise sample all the errors
	else:
		run = 1
		for error in errors:
			for idx_ratio in range(len(ratios)):
				if error <= ratios_thresholds[ratios[idx_ratio]]:
					num_correct[idx_ratio] += 1

			log_str += (f"{run}{csv_sep} "
						f"{error}{csv_sep} "
						f"{num_correct[0]}{csv_sep} "
						f"{num_correct[1]}{csv_sep} "
						f"{num_correct[2]}{csv_sep} "
						f"{num_correct[3]}{csv_sep} "
						f"{num_correct[4]}\n"
					)
			run += 1

	with open(filepath_log, "a") as f:
		f.write(log_str)
		f.close()


	# write the complete versions
	filepath_log = os.path.join(filepath, "log_correctness_ratio_complete.csv")
	
	num_correct = [0, 0, 0, 0, 0]
	log_str = ""
	run = 1
	for error in errors:
		for idx_ratio in range(len(ratios)):
			if error <= ratios_thresholds[ratios[idx_ratio]]:
				num_correct[idx_ratio] += 1

		log_str += (f"{run}{csv_sep} "
					f"{error}{csv_sep} "
					f"{num_correct[0]}{csv_sep} "
					f"{num_correct[1]}{csv_sep} "
					f"{num_correct[2]}{csv_sep} "
					f"{num_correct[3]}{csv_sep} "
					f"{num_correct[4]}\n"
				)
		run += 1

	with open(filepath_log, "a") as f:
		f.write(log_str)
		f.close()


	# write thresholds for the legend of the plot
	filepath_log = os.path.join(filepath, "log_thresholds_correctness_plot.csv")

	log_str = (
			f"{str(num_correct[0]/num_run).split('.')[0]}.{str(num_correct[0]/num_run).split('.')[1][:2]}{csv_sep} {ratios_thresholds[0.0]}{csv_sep} "
			f"{str(num_correct[1]/num_run).split('.')[0]}.{str(num_correct[1]/num_run).split('.')[1][:2]}{csv_sep} {ratios_thresholds[0.25]}{csv_sep} "
			f"{str(num_correct[2]/num_run).split('.')[0]}.{str(num_correct[2]/num_run).split('.')[1][:2]}{csv_sep} {ratios_thresholds[0.50]}{csv_sep} "
			f"{str(num_correct[3]/num_run).split('.')[0]}.{str(num_correct[3]/num_run).split('.')[1][:2]}{csv_sep} {ratios_thresholds[0.75]}{csv_sep} "
			f"{str(num_correct[4]/num_run).split('.')[0]}.{str(num_correct[4]/num_run).split('.')[1][:2]}{csv_sep} {ratios_thresholds[1.0]}\n"
		)

	with open(filepath_log, "a") as f:
		f.write(log_str)
		f.close()


def write_correctness_ratios_plot_log(filepath, errors, num_run):
	filepath_log = os.path.join(filepath, "log_correctness_ratios.csv")
	# obtain list of tuples (correctness ratio, threshold), ordered by threshold
	corr_ratio_thr = correctness_ratios(errors)

	log_str = ""
	idx_csv = 0
	if num_run > 3500:
		prev_ratio = None
		for idx in range(0, num_run, math.ceil(num_run/3500)):
			if prev_ratio != None:
				log_str += (
						f"{idx_csv}{csv_sep} "
						f"{corr_ratio_thr[idx][1]}{csv_sep} "
						f"{prev_ratio}{csv_sep}\n"
					)
				idx_csv += 1

			log_str += (
					f"{idx_csv}{csv_sep} "
					f"{corr_ratio_thr[idx][1]}{csv_sep} "
					f"{corr_ratio_thr[idx][0]}\n"
				)

			prev_ratio = corr_ratio_thr[idx][0]
			idx_csv += 1
	else:
		prev_ratio = None
		for idx in range(0, num_run):
			if prev_ratio != None:
				log_str += (
						f"{idx_csv}{csv_sep} "
						f"{corr_ratio_thr[idx][1]}{csv_sep} "
						f"{prev_ratio}{csv_sep}\n"
					)
				idx_csv += 1

			log_str += (
					f"{idx_csv}{csv_sep} "
					f"{corr_ratio_thr[idx][1]}{csv_sep} "
					f"{corr_ratio_thr[idx][0]}\n"
				)

			prev_ratio = corr_ratio_thr[idx][0]
			idx_csv += 1

	# add last value
	log_str += (
			f"{idx_csv}{csv_sep} "
			f"{corr_ratio_thr[-1][1]}{csv_sep} "
			f"{corr_ratio_thr[-1][0]}\n"
		)
	idx_csv += 1

	# add bigger threshold to avoid domain error
	log_str += (
			f"{idx_csv}{csv_sep} "
			f"{corr_ratio_thr[-1][1]+1}{csv_sep} "
			f"{corr_ratio_thr[-1][0]}\n"
		)

	with open(filepath_log, "a") as f:
		f.write(log_str)
		f.close()


	# write complete log
	filepath_log = os.path.join(filepath, "log_correctness_ratios_complete.csv")

	log_str = ""
	idx_csv = 0
	prev_ratio = None
	for idx in range(0, num_run):
		if prev_ratio != None:
			log_str += (
					f"{idx_csv}{csv_sep} "
					f"{corr_ratio_thr[idx][1]}{csv_sep} "
					f"{prev_ratio}{csv_sep}\n"
				)
			idx_csv += 1

		log_str += (
			f"{idx_csv}{csv_sep} "
			f"{corr_ratio_thr[idx][1]}{csv_sep} "
			f"{corr_ratio_thr[idx][0]}\n"
		)

		prev_ratio = corr_ratio_thr[idx][0]
		idx_csv += 1

	# add last value
	log_str += (
			f"{idx_csv}{csv_sep} "
			f"{corr_ratio_thr[-1][1]}{csv_sep} "
			f"{corr_ratio_thr[-1][0]}\n"
		)
	idx_csv += 1

	# add bigger threshold to avoid domain error
	log_str += (
			f"{idx_csv}{csv_sep} "
			f"{corr_ratio_thr[-1][1]+1}{csv_sep} "
			f"{corr_ratio_thr[-1][0]}\n"
		)

	with open(filepath_log, "a") as f:
		f.write(log_str)
		f.close()



def write_time_percentage_plot(filepath, processes_runs_time):
	filepath_log = os.path.join(filepath, "log_exec_time_plot.csv")

	exec_time_plot = compute_perc_values(processes_runs_time, decimal_precision=1)
	log_str = ""
	for plot_val in exec_time_plot:
		log_str += f"{plot_val[0]}{csv_sep} {plot_val[1]}\n"

	with open(filepath_log, "a") as f:
		f.write(log_str)
		f.close()


def write_ram_percentage_plot(filepath, ram_usage):
	filepath_log = os.path.join(filepath, "log_ram_plot.csv")

	ram_plot = compute_perc_values(ram_usage, decimal_precision=0)
	log_str = ""
	for plot_val in ram_plot:
		log_str += f"{plot_val[0]}{csv_sep} {plot_val[1]}\n"

	with open(filepath_log, "a") as f:
		f.write(log_str)
		f.close()


def write_f_evals_percentage_plot(filepath, num_evals):
	filepath_log = os.path.join(filepath, "log_num_f_evals_plot.csv")

	num_f_evals_plot = compute_perc_values(num_evals, decimal_precision=0)
	log_str = ""
	for plot_val in num_f_evals_plot:
		log_str += f"{plot_val[0]}{csv_sep} {plot_val[1]}\n"

	with open(filepath_log, "a") as f:
		f.write(log_str)
		f.close()


def write_error_plot(filepath, errors):
	filepath_log = os.path.join(filepath, "log_errors_plot.csv")

	cumulative_errors_plot, exact_errors_plot = cumulative_exact_errors_plot(errors)
	log_str = ""
	for error in cumulative_errors_plot:
		log_str += f"{error[0]}{csv_sep} {error[1]}\n"

	with open(filepath_log, "a") as f:
		f.write(log_str)
		f.close()

	filepath_log = os.path.join(filepath, "log_errors_bar_plot.csv")
	log_str = ""
	for error in exact_errors_plot:
		log_str += f"{error[0]}{csv_sep} {error[1]}\n"

	with open(filepath_log, "a") as f:
		f.write(log_str)
		f.close()



# compute thresholds of error in order to obtain correctness ratios of the list "ratios"
def find_crs_thresholds(errors):
	errors = sorted(errors)
	N = len(errors)
	ratios = [0.25, 0.50, 0.75, 1.0]
	# threshold for correctness of 0 is min error - 1e-6
	min_thr = errors[0] - 1e-6
	res = {0.0 : min_thr if min_thr >= 0 else 0}
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
	N = len(errors)
	# list of tuples (correctness ratio, threshold)
	correctness_ratios = [(0, errors[0] - 1e-6)]
	correct = 0
	for error in errors:
		correct += 1
		correctness_ratios.append((correct/N, error))
	return correctness_ratios


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