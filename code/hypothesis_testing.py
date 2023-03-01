import sfu.evaluation_code.objective_function_class as ev
import time, sys, os, math, itertools, random
from memory_profiler import memory_usage
import nevergrad_utils as ng_util
from utils import *

# execute N run of nevergrad, in each run we have num_workers parallel evaluation of the function, then we take the minim evaluation among num_workers as temp result of nevergrad

num_proc = 0
num_points = 0
function_obj = None
q_inp = None
q_res = None
path_dir_log_file = None
csv_sep = ";"

def multiproc_function(q_inp, q_res, function_obj, num_points):
	"""Function runned by the concurrent processes. Each process executes a run of nevergrad if
	it can pop from the queue "q_inp" the value 1. If the popped value is None, then it stops.
	After each execution of nevergrad, each process push in the queue "q_res" a dictionary with the results of the execution, which has the keys:
	- "process time" : process time in seconds of the execution
	- "max ram usage" : maximum ram usage during the run
	- "num evaluations" : number of objective function evaluations
	- "input_opt" : input values of the objective function that nevergrad thinks are the ones producing the optimum
	- "opt" : optimum found by nevergrad
	"""
	inp = q_inp.get()
	while (inp != None):
		# init time and ram tracking variables
		start_time = time.process_time()
		mem_usage = memory_usage(os.getpid(), interval=.1)
		# run nevergrad
		res = ng_util.run_nevergrad(function_obj, range_stopping_criteria=20, num_points=num_points)
		# return results
		res.update({"process time" : time.process_time() - start_time, "max ram usage" : max(mem_usage)})
		q_res.put(res)
		# wait for a new run
		inp = q_inp.get()

def hypothesis_testing(input_opt, opt, delta, epsilon, path_dir_log_file, num_proc, q_inp, q_res, tolerance = 1e-6, correct_thr = 1e-6):
	# init N
	N = 5#math.ceil((math.log(delta)/math.log(1-epsilon)))

	# track execution time of algorithm
	start_time = time.time()
	
	print(f"N: {N}")
	
	# request execution of one run of nevergrad and obtain result
	q_inp.put(1)
	res = q_res.get()
	
	# init list of values to update for all the runs
	processes_runs_time = [res["process time"]]
	number_of_asks = [res["num evaluations"]]
	ram_usage = [res["max ram usage"]]
	
	max_ram_usage = res["max ram usage"]

	num_run = 1

	# init log runs
	runs_infos = [{"index" : num_run-1,
				  "run number" : num_run,
				  "iteration" : f"0/{N}",
				  "opt found" : res["opt"],
				  "obj funct error" : objective_function_error(opt, res["opt"]),
				  "input opt found" : res["input opt"],
				  "solution error" : solution_error(input_opt, res["input opt"]),
				  "S" : "",
				  "num evals" : res["num evaluations"],
				  "time" : res["process time"],
				  "max ram usage" : res["max ram usage"]
				}
			]
	print(runs_infos[-1])
	
	# init S and S prime
	# S_values contains tuple (run number, input_opt, opt)
	S_values = [{"run number" : 1, 
				"internal iteration" : 0,
				"S value" : res["opt"],
				"input" : res["input opt"]
				}
	]
	S_prime = res["opt"]

	num_correct_opts = 0
	if objective_function_error(opt, res["opt"]) <= correct_thr:
		num_correct_opts += 1
	
	write_string = ""
	num_external_iter = 1
	while (1):
		S = S_prime
		
		# if size of the queue is less than N, add requests for nevergrad runs
		q_sizes = q_inp.qsize() + q_res.qsize()
		for x in range(N - q_sizes):
			q_inp.put(1)

		# check results of N nevergrad runs, or exit from the loop if value better than S is found
		internal_iter = 0
		while(internal_iter < N):
			internal_iter += 1
			num_run += 1

			# update results
			res = q_res.get()
			processes_runs_time.append(res["process time"])
			number_of_asks.append(res["num evaluations"])
			ram_usage.append(res["max ram usage"])
			if max_ram_usage < res["max ram usage"]:
				max_ram_usage = res["max ram usage"]

			# update runs infos
			runs_infos.append({"index" : num_run-1,
								"run number" : num_run,
				  				"iteration" : f"{internal_iter}/{N}",
				 				"opt found" : res["opt"],
				  				"obj funct error" : objective_function_error(opt, res["opt"]),
				  				"input opt found" : res["input opt"],
				  				"solution error" : solution_error(input_opt, res["input opt"]),
								"S" : S,
								"num evals" : res["num evaluations"],
								"time" : res["process time"],
								"max ram usage" : res["max ram usage"]
				}
			)
			print(runs_infos[-1])

			# count correct results
			if objective_function_error(opt, res["opt"]) <= correct_thr:
				num_correct_opts += 1

			# if result smaller than S, restart
			if (res["opt"] + tolerance) < S:
				S_prime = res["opt"]
				S_values.append({"run number" : num_run, 
								"internal iteration" : internal_iter,
								"S value" : res["opt"],
								"input" : res["input opt"]
								}
				)
				write_string = ""
				break

		# if after the loop starting S is equal to S_prime, end algorithm
		if S == S_prime:
			break
		num_external_iter += 1

	# write log of the results
	multi_core_time = time.time() - start_time
	single_core_time = sum(processes_runs_time)
	speedup = single_core_time/multi_core_time
	efficiency = speedup/num_proc
	mean_time_per_run = sum(processes_runs_time) / len(processes_runs_time)
	std_dev_time_per_run = std_dev(processes_runs_time)
	mean_number_of_asks = sum(number_of_asks)/len(number_of_asks)
	std_dev_number_of_asks = std_dev(number_of_asks)
	mean_ram_usage = sum(ram_usage)/len(ram_usage)
	std_dev_ram_usage = std_dev(ram_usage)

	log_result_string = (
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
							f"Mean number of asks per run{csv_sep} "
							f"Standard deviation number of asks per run{csv_sep} "
							f"Max RAM Megabyte usage{csv_sep} "
							f"Mean ram usage{csv_sep} "
							f"Standard deviation ram usage{csv_sep} "
							f"Speedup{csv_sep} "
							f"Efficiency\n"
						)
	print(log_result_string)
	temp_string = (
					f"{num_run}{csv_sep} "
					f"{num_external_iter}{csv_sep} "
					f"{S_values[-1]['S value']}{csv_sep} "
					f"{opt}{csv_sep} "
					f"{S_values[-1]['input']}{csv_sep} "
					f"{input_opt}{csv_sep} "
					f"{objective_function_error(opt, S_values[-1]['S value'])}{csv_sep} "
					f"{solution_error(input_opt, S_values[-1]['input'])}{csv_sep} "
					f"{num_correct_opts/num_run}{csv_sep} "
					f"{time_to_str(multi_core_time)}{csv_sep} "
					f"{time_to_str(single_core_time)}{csv_sep} "
					f"{mean_time_per_run}{csv_sep} "
					f"{std_dev_time_per_run}{csv_sep} "
					f"{mean_number_of_asks}{csv_sep} "
					f"{std_dev_number_of_asks}{csv_sep} "
					f"{max_ram_usage}{csv_sep} "
					f"{mean_ram_usage}{csv_sep} "
					f"{std_dev_ram_usage}{csv_sep} "
					f"{speedup}{csv_sep} "
					f"{efficiency}\n"
				)
	print(temp_string)
	log_result_string = log_result_string + temp_string
	write_log_file(os.path.join(path_dir_log_file, "log_results.csv"), log_result_string)

	
	# write log of the runs
	log_runs_string = (
						f"Index{csv_sep} "
						f"Run{csv_sep} "
						f"Iteration{csv_sep} "
						f"Optimum Found{csv_sep} "
						f"S{csv_sep} "
						f"Number of Asks{csv_sep} "
						f"Time{csv_sep} "
						f"Max RAM Megabyte Usage\n"
					)
	for run_info in runs_infos:
		log_runs_string = log_runs_string + (
												f"{run_info['index']}{csv_sep} "
												f"{run_info['run number']}{csv_sep} "
												f"{run_info['iteration']}{csv_sep} "
												f"{run_info['opt found']}{csv_sep} "
												f"{run_info['S']}{csv_sep} "
												f"{run_info['num evals']}{csv_sep} "
												f"{run_info['time']}{csv_sep} "
												f"{run_info['max ram usage']}\n"	
										)
	write_log_file(os.path.join(path_dir_log_file, "log_runs.csv"), log_runs_string)

	

	# write s values log, for plot and table
	# table
	log_s_values_string = (
							f"Run{csv_sep} "
							f"Internal iteration{csv_sep} "
							f"N{csv_sep} "
							f"S value\n"
		)
	for s_value in S_values:
		log_s_values_string = log_s_values_string + (
														f"{s_value['run number']}{csv_sep} "
														f"{s_value['internal iteration']}{csv_sep} "
														f"{N}{csv_sep} "
														f"{s_value['S value']}\n"
												)
	print(log_s_values_string)
	write_log_file(os.path.join(path_dir_log_file, "log_s_values.csv"), log_s_values_string)
	
	# plot
	log_s_values_string = (
							f"Run{csv_sep} "
							f"S\n"
		)
	prev_s_value = None
	for s_value in S_values:
		if prev_s_value != None:
			log_s_values_string = log_s_values_string + (
															f"{s_value['run number']-1}{csv_sep} "
															f"{prev_s_value}\n"
													)
		log_s_values_string = log_s_values_string + (
														f"{s_value['run number']}{csv_sep} "
														f"{s_value['S value']}\n"
												)
	log_s_values_string = log_s_values_string + (
													f"{N}{csv_sep} "
													f"{s_value['S value']}\n"
											)
	write_log_file(os.path.join(path_dir_log_file, "log_s_values_plot.csv"), log_s_values_string)

	# create list of errors
	errors = [run_info["obj funct error"] for run_info in runs_infos]

	# write log correctness ratio i.e., shows the behaviour of the number of correct results during the runs, according to 5 different values for correctness ratio
	# obtain dictionary with correctness ratio values as keys and the corresponding error threshold as value
	ratios_thresholds = find_crs_thresholds(errors)
	ratios = [0.0, 0.25, 0.50, 0.75, 1.0]
	n_corrects = [0, 0, 0, 0, 0]
	log_correctness_ratio_string = (f"Run{csv_sep} "
									f"Error{csv_sep} "
									f"Num Correct Ratio 0.00{csv_sep} "
									f"Num Correct Ratio 0.25{csv_sep} "
									f"Num Correct Ratio 0.50{csv_sep} "
									f"Num Correct Ratio 0.75{csv_sep} "
									f"Num Correct Ratio 1.00\n"
								)
	i = 1
	num_errors = len(errors)
	idx_err = 0
	for error in errors:
		for idx_ratio in range(len(ratios)):
			if error <= ratios_thresholds[ratios[idx_ratio]]:
				n_corrects[idx_ratio] += 1
		# half the number of results when number of errors is too large
		if num_errors > 7000:
			if idx_err % 2 == 0:
				log_correctness_ratio_string = log_correctness_ratio_string + (f"{i}{csv_sep} "
																			   f"{error}{csv_sep} "
																			   f"{n_corrects[0]}{csv_sep} "
																			   f"{n_corrects[1]}{csv_sep} "
																			   f"{n_corrects[2]}{csv_sep} "
																			   f"{n_corrects[3]}{csv_sep} "
																			   f"{n_corrects[4]}\n"
																			)
		else:
			log_correctness_ratio_string = log_correctness_ratio_string + (f"{i}{csv_sep} "
																			   f"{error}{csv_sep} "
																			   f"{n_corrects[0]}{csv_sep} "
																			   f"{n_corrects[1]}{csv_sep} "
																			   f"{n_corrects[2]}{csv_sep} "
																			   f"{n_corrects[3]}{csv_sep} "
																			   f"{n_corrects[4]}\n"
																			)
		i += 1
		idx_err += 1
	write_log_file(os.path.join(path_dir_log_file, "log_correctness_ratio.csv"), log_correctness_ratio_string)


	# write log of the threshold values for each correctness ratio, shows the computed ratio (not always the exact ratio is possible) and its corresponding threshold
	N = len(errors)
	log_thr_correctness_ratio_string = (f"Ratio 0.00{csv_sep} " 
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
	log_thr_correctness_ratio_string = log_thr_correctness_ratio_string + \
	f"{str(n_corrects[0]/N).split('.')[0]}.{str(n_corrects[0]/N).split('.')[1][:2]}{csv_sep} {ratios_thresholds[0.0]}{csv_sep} "+\
	f"{str(n_corrects[1]/N).split('.')[0]}.{str(n_corrects[1]/N).split('.')[1][:2]}{csv_sep} {ratios_thresholds[0.25]}{csv_sep} "+\
	f"{str(n_corrects[2]/N).split('.')[0]}.{str(n_corrects[2]/N).split('.')[1][:2]}{csv_sep} {ratios_thresholds[0.50]}{csv_sep} "+\
	f"{str(n_corrects[3]/N).split('.')[0]}.{str(n_corrects[3]/N).split('.')[1][:2]}{csv_sep} {ratios_thresholds[0.75]}{csv_sep} "+\
	f"{str(n_corrects[4]/N).split('.')[0]}.{str(n_corrects[4]/N).split('.')[1][:2]}{csv_sep} {ratios_thresholds[1.0]}\n"
	write_log_file(os.path.join(path_dir_log_file, "log_thresholds_correctness_plot.csv"), log_thr_correctness_ratio_string)


	# write log for plot of the values of correctness ratio for different value of threshold
	thresholds = correctness_ratios(errors)
	log_correctness_ratios_string = 'Index; Threshold; Correctness ratio\n'
	idx = 0
	past_cr = None
	for threshold in thresholds:
		if idx % 1 == 0:
			if idx != 0 and idx != ((len(thresholds)-1)*2 -1):
				log_correctness_ratios_string = log_correctness_ratios_string + f'{idx}{csv_sep} {threshold[0]}{csv_sep} {past_cr}\n'
				idx += 1
			log_correctness_ratios_string = log_correctness_ratios_string + f'{idx}{csv_sep} {threshold[0]}{csv_sep} {threshold[1]}\n'
			past_cr = threshold[1]
		idx += 1
	write_log_file(os.path.join(path_dir_log_file, "log_correctness_ratios.csv"), log_correctness_ratios_string)


	# write log for plot of execution times
	exec_time_plot = compute_perc_values([run_info["time"] for run_info in runs_infos], decimal_precision=1)
	log_time_plot_string = f"Time{csv_sep} Percentage\n"
	for plot_val in exec_time_plot:
		log_time_plot_string = log_time_plot_string + f"{plot_val[0]}{csv_sep} {plot_val[1]}\n"
	write_log_file(os.path.join(path_dir_log_file, "log_exec_time_plot.csv"), log_time_plot_string)


	# write log for plot of ram usage
	ram_plot = compute_perc_values([run_info["max ram usage"] for run_info in runs_infos], decimal_precision=0)
	log_ram_plot_string = f"Ram{csv_sep} Percentage\n"
	for plot_val in ram_plot:
		log_ram_plot_string = log_ram_plot_string + f"{plot_val[0]}{csv_sep} {plot_val[1]}\n"
	write_log_file(os.path.join(path_dir_log_file, "log_ram_plot.csv"), log_ram_plot_string)



	# write log for plot of num function evals
	num_f_evals_plot = compute_perc_values([run_info["num evals"] for run_info in runs_infos], decimal_precision=0)
	log_f_evals_string = f"f evals{csv_sep} Percentage\n"
	for plot_val in num_f_evals_plot:
		log_f_evals_string = log_f_evals_string + f"{plot_val[0]}{csv_sep} {plot_val[1]}\n"
	write_log_file(os.path.join(path_dir_log_file, "log_num_f_evals_plot.csv"), log_f_evals_string)

	# write log errors plot
	cumulative_errors_plot, exact_errors_plot = cumulative_exact_errors_plot(errors)
	# cumulative error plot
	cumul_err_string = f"Error{csv_sep} Percentage\n"
	for error in cumulative_errors_plot:
		cumul_err_string = cumul_err_string + f"{error[0]}{csv_sep} {error[1]}\n"
	write_log_file(os.path.join(path_dir_log_file, "log_errors_plot.csv"), cumul_err_string)
	# bar plot errors
	exact_err_string = f"Error{csv_sep} Percentage\n"
	for error in exact_errors_plot:
		exact_err_string = exact_err_string + f"{error[0]}{csv_sep} {error[1]}\n"	
	write_log_file(os.path.join(path_dir_log_file, "log_errors_bar_plot.csv"), exact_err_string)

	return S_values[-1]