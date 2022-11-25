import json, math, os
from rpy2 import robjects
from subprocess import check_output
import shutil

json_functions = None
function_name = None
function_informations = None
path_implementation = None
R_code = None
function_impl_name = None

class JsonNotLoaded(Exception):
	pass

class sfuFunctionError(Exception):
	pass

def load_json(filepath):
	global json_functions
	f = open(filepath)
	json_functions = json.load(f)
	f.close()
	return json_functions

def select_function(name):
	global json_functions
	if not json_functions:
		raise JsonNotLoaded("The json file has not been loaded")
	if not name in json_functions.keys():
		raise sfuFunctionError("The function selected does not exists")
	
	global function_name, function_informations, path_implementation, R_code, function_impl_name
	function_name = name
	function_informations = json_functions[name]
	path_implementation = os.path.join(os.path.dirname(os.path.dirname(__file__)), "functions/") + json_functions[name]["filepath_r"]
	with open(path_implementation, 'r') as f:
		R_code = f.read()
		function_impl_name = R_code.split('\n')[0].split()[0]
		f.close()
	

def search_function(filters=None):
	global json_functions
	if not json_functions:
		raise JsonNotLoaded("The json file has not been loaded")

	results = []
	for function in json_functions:
		ok = 1
		if filters!=None:
			for filt in filters:
				if type(filters[filt]) == bool:
					temp_value_filter = False
					if type(json_functions[function][filt]) != bool and json_functions[function][filt] != None:
						temp_value_filter = True
					if temp_value_filter != filters[filt]:
						ok = 0
				elif json_functions[function][filt] != str(filters[filt]):
					ok = 0
		if ok == 1:
			results.append(function)
	return results

def dimension():
	global function_name, function_informations
	if not function_name:
		raise sfuFunctionError("No function has been selected")
	
	try:
		dimension = int(function_informations["dimension"])
		return dimension
	except ValueError:
		return "d"

def minimum_point(dim=None):
	# return the point coordinates of the minimum value of the function
	global function_name, function_informations
	if not function_name:
		raise sfuFunctionError("No function has been selected")

	if type(function_informations["minimum_x"]) == str:
		local_var = {}
		if dimension() != "d":
			exec(function_informations["minimum_x"], globals(), local_var)
			return local_var["minimum_x"]
		else:
			if not dim:
				raise sfuFunctionError("The function needs the dimension value")
			
			exec(function_informations["minimum_x"], globals(), local_var)
			return local_var["minimum_x"]
	elif type(function_informations["minimum_x"]) == list:
		return function_informations["minimum_x"]
	elif function_informations["minimum_x"] == None:
		return None
	else:
		return [function_informations["minimum_x"] for x in range(dim)]

def minimum_value(dim=None):
	# returns the global minimum of the function

	global function_name, function_informations
	if not function_name:
		raise sfuFunctionError("No function has been selected")

	if type(function_informations["minimum_f"]) == str:
		local_var = {}
		if dimension() != "d":
			exec(function_informations["minimum_f"], globals(), local_var)
			return local_var["minimum_f"]
		else:
			if not dim:
				raise sfuFunctionError("The function needs the dimension value")

			global d
			d = dim
			exec(function_informations["minimum_f"], globals(), local_var)
			return local_var["minimum_f"]
	elif type(function_informations["minimum_f"]) == list:
		return function_informations["minimum_f"]
	elif function_informations["minimum_f"] == None:
		return None
	else:
		return function_informations["minimum_f"]

def parameters():
	# returns a description of the parameters

	global function_name, function_informations
	if not function_name:
		raise sfuFunctionError("No function has been selected")

	if not function_informations["parameters"]:
		return None
	else:
		return function_informations["parameters"]

def input_domain(dim=None):
	# returns an interval of the input domain

	global function_name, function_informations
	if not function_name:
		raise sfuFunctionError("No function has been selected")

	input_domain = function_informations["input_domain"]
	result = []
	
	# if we have an interval for each input dimension
	if len(input_domain) == dimension():
		for interval in input_domain:
			if type(interval[0]) == str:
				temp = eval(interval[0], globals())
				result.append([temp[0], temp[1]])
			else:
				result.append(interval)

	# if the dimension of the function is d (the interval of d described with only 1 interval)
	elif dimension() == "d":
		if dim == None:
			raise sfuFunctionError("The function needs the dimension value")

		# if the input domain is defined as str
		if type(input_domain[0][0]) == str:
			for i in range(dim):
				local_var = {"d" : dim}
				temp = eval(input_domain[0][0], globals(), local_var)
				result.append([temp[0], temp[1]])
		else:
			for i in range(dim):
				result.append(input_domain[0])
	elif len(input_domain) == 1:
		for i in range(dimension()):
			result.append(input_domain[0])
	return result


def evaluate(inp, param=None):
	global function_name, R_code
	function_dimension = dimension()
	if not function_name:
		raise sfuFunctionError("No function has been selected")
	if function_dimension == 1 and type(inp) != int and type(inp) != float:
		raise sfuFunctionError("Function input must be int or float")
	if function_dimension != "d" and function_dimension != 1 and len(inp) != function_dimension:
		raise sfuFunctionError("Function input does not match function dimension")
	if type(inp) != list:
		inp = [inp]
	if function_dimension == "d":
		function_dimension = len(inp)
	call = "\n" + function_impl_name
	if function_dimension == 1:
		call = call + "({}".format(inp[0])
	else:
		inp = [str(x) for x in inp]
		call = call + "(c(" + ",".join(tuple(inp)) + ")"
	if param != None:
		for par in param:
			call = call + ",{}={}".format(par, param[par])
	call = call + ")"
	return robjects.r(R_code + call)[0]