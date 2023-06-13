import sys, os, argparse
from hypothesis_testing import *
from multi_processes import *
import sfu.objective_function_class as obj_func


def main(argv):
	# argv[0] : function name, argv[1] : # points to evaluate, argv[2] : # parallel processes
	global num_proc, num_points, function_obj, range_stopping_criteria, path_dir_log_file
	
	# define parser
	parser = argparse.ArgumentParser(description="run hypothesis testing using Nevergrad")
	parser.add_argument("function_name", help="objective function name")
	parser.add_argument("points", type=int, help="number of points to be evaluated in parallel")
	parser.add_argument("processes", type=int, help="number of processes that runs optimization in parallel")
	parser.add_argument("--parameters", "-p", action="store_true", help="run the tests also for different values of the parameters")
	parser.add_argument("--dimensions", "-d", type=int, nargs="+", help="explicitly define dimensions on which perform the tests")
	parser.add_argument("--verbose", "-v", action="store_true")
	args = parser.parse_args()

	# number of points to evaluate in parallel
	num_points = args.points
	# number of concurrent processes
	num_proc = args.processes
	# hypothesis testing parameters delta and epsilon
	delta = 1e-3
	epsilon = 1e-3

	# create directory of the results logs
	path_dir_res = os.path.join(os.path.dirname(__file__),"log_results")
	if not os.path.exists(path_dir_res):
		os.mkdir(path_dir_res)

	# modify path of json of functions
	obj_func.json_filepath = os.path.join("sfu", obj_func.json_filepath)

	# create the objective function object
	function_name = args.function_name
	function_json = obj_func.get_json_functions(name=function_name)

	# path of the function results
	path_dir_res = os.path.join(path_dir_res, f"{function_name}")
	if not os.path.exists(path_dir_res):
		os.mkdir(path_dir_res)

	# if dimensions are specified on input, use those
	if args.dimensions != None:
		dimensions = args.dimensions
	# otherwise obtain the list of dimensions on which to test it	
	else:
		dimensions = get_test_dimensions(function_json["dimension"], function_json["minimum_f"])
		
	# test the function on all the selected dimensions
	for dim in dimensions:

		# path of the results for the dimension dim
		path_dir_res_dim = os.path.join(path_dir_res, f"dimension_{dim}")
		if not os.path.exists(path_dir_res_dim):
			os.mkdir(path_dir_res_dim)

		# create object of the objective function class
		function_obj = obj_func.objective_function(function_name, dim=dim)

		# if parameters testing is activated and function accepts parameters create the list of parameters on which to test the function
		if args.parameters and function_obj.has_parameters == True:
			# get dictionary of parameters with their values
			comb_parameters = get_test_parameters(function_obj)

			# for each possible combination of the parameters values
			for param_values_list in itertools.product(*[comb_parameters[param_name] for param_name in function_obj.parameters_names]):
				# obtain combination of parameters
				i=0
				temp_param_values = {}
				for param_name in function_obj.parameters_names:
					temp_param_values[param_name] = param_values_list[i]
					i+=1
					# update function's parameters values
				function_obj.set_parameters(temp_param_values)

				if args.verbose:
					print(f"TESTING FUNCTION DIM: {dim} PARAMETERS: {[ (param_name, str(temp_param_values[param_name])) for param_name in function_obj.parameters_names]} OPT_POINT: {function_obj.opt}\n")
					
				path_dir_log_file = os.path.join(path_dir_res_dim, f"function_{function_name}_dimension_{dim}_parameters_{[ (param_name, temp_param_values[param_name]) for param_name in function_obj.parameters_names ]}")
				if not os.path.exists(path_dir_log_file):
					os.mkdir(path_dir_log_file)

				# create concurrent processes and run hypothesis testing
				q_inp, q_res, processes = init_processes(num_proc, multiproc_function, function_obj, num_points)
				# run hypothesis testing
				hypothesis_testing(function_obj.input_opt, function_obj.opt, delta, epsilon, path_dir_log_file, num_proc, q_inp, q_res, verbose=args.verbose)
				# kill concurrent processes
				kill_processes(processes)

		# if not parameters testing not activated or the function does not accept parameters
		else:

			if args.verbose:
				print(f"TESTING FUNCTION DIM: {dim} OPT_POINT: {function_obj.opt}\n")
				
			path_dir_log_file = os.path.join(path_dir_res_dim, f"function_{function_name}_dimension_{dim}")
			if not os.path.exists(path_dir_log_file):
				os.mkdir(path_dir_log_file)

			# create concurrent processes and obtain queues for input and output communication
			q_inp, q_res, processes = init_processes(num_proc, multiproc_function, function_obj, num_points)
			# run hypothesis testing
			hypothesis_testing(function_obj.input_opt, function_obj.opt, delta, epsilon, path_dir_log_file, num_proc, q_inp, q_res, verbose=args.verbose)
			# kill concurrent processes
			kill_processes(processes)
				

def get_test_dimensions(dim, min_f):
	"""Returns the list of dimension on which the function must be tested.

	Parameters
	----------
	dim: str
		Json value for the field "dimension" of the functions' definition json. (e.g., "d", "2", etc...)
	min_f: dict, float
		Json value for the field "minimum_f" of the functions' definition json. 
		It is a dict when the function's global minimum is defined only for specific dimensions (e.g., dict = {"dimension" : {"2" : -1, "5" : 0}}).
		Otherwise it is a float number.

	Returns
	-------
	list
		List of integers representing the dimensions on which the function must be tested
	
	"""
	if dim == "d":
		# if global optimum is defined only for certain dimensions, select them
		if type(min_f) == dict and list(min_f)[0] == "dimension":
			dimensions = [int(x) for x in list(min_f["dimension"])]
		# otherwise select samples from 2 to 100, with an interval of 10
		else:
			dimensions = [2] + [10*(x+1) for x in range(10)]
	else:
		dimensions = [int(dim)]
	return dimensions

def get_test_parameters(function_obj, coeff_parameters=5):
	"""Returns a dictionary with parameter names as key and its values as value.

	Parameters
	----------
	function_obj: objective_function
		Objective function object of the objective_function class
	coeff_parameters: int, float
		Coefficient used to multiply the default value of a parameters to generate a new parameter value

	Returns
	-------
	dict
		Dictionary with parameter names as key and a list of 2 values as value, i.e., [default value, coeff_parameters*default value], 
		or a list as value if the function's optimum is defined only for specific values.
	"""
	comb_parameters = {}
	for param_name in function_obj.parameters_names:
		# if optimum depends on a parameter value, add values of that parameter for which optimum is defined
		if function_obj.param_opt != None and function_obj.param_opt[0] == param_name:
			comb_parameters[param_name] = function_obj.param_opt[1]
		else:
			param_value = function_obj.parameters[param_name]
			if type(param_value) == list:
				# if parameter's value is a matrix, multiply each element by the coefficient
				if type(param_value[0]) == list:
					temp_matrix = []
					for row in param_value:
						temp_matrix.append([x*coeff_parameters for x in row])
						comb_parameters[param_name] = [param_value, temp_matrix]
				else:
					# if parameter's value is a vector, multiply each element by the coefficient
					comb_parameters[param_name] = [param_value, [x*coeff_parameters for x in param_value]]
			else:
				# parameter's value is a float number
				comb_parameters[param_name] = [param_value, param_value*coeff_parameters]
	return comb_parameters

if __name__ == "__main__":
	main(sys.argv[1:])