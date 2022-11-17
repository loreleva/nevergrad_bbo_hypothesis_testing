import sys
from functions import *

if __name__ == "__main__":
	load_json("../functions/functions.json")
	print("Functions with dimension d:{}\n".format(search_function({"dimension" : "d"})))
	print(f"SEarch result: {search_function({'dimension' : 1})}")
	function_selected = "trid_function"
	function_dimension = 1
	select_function(function_selected)
	print("Function selected: {}".format(function_selected))
	print("Parameters of {}: {}".format(function_selected, parameters()))
	print("Dimension of {}: {}".format(function_selected, dimension()))
	print("Input domain of {} with dimension {}: {}".format(function_selected, function_dimension ,input_domain(function_dimension)))
	input_x = 12
	print("{} value of dimension {} at point {}: {}".format(function_selected, function_dimension, input_x, evaluate(input_x)))
	