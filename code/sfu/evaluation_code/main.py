import sys, time
from functions import *

if __name__ == "__main__":
	test = objective_function("schwefel_function", dim=5)
	print(f"dimension: {test.dimension}")
	print(f"minimum_x: {test.minimum_x}")
	print(f"minimum_f: {test.minimum_f}")
	print(f"Input Lower Bound: {test.input_lb}\nInput Upper Bound: {test.input_ub}")
	#print("Functions with dimension d:{}\n".format(search_function({"dimension" : "d"})))
	#function_selected = "trid_function"
	#function_dimension = 5
	#select_function(function_selected)
	#print("Function selected: {}".format(function_selected))
	#print("Parameters of {}: {}".format(function_selected, parameters()))
	#print("Dimension of {}: {}".format(function_selected, dimension()))
	#print("Input domain of {} with dimension {}: {}".format(function_selected, function_dimension ,input_domain(function_dimension)))
	#input_x = [0.5, 0.5, 0.5, 0.5, 0.5]
	##evaluate_test(input_x)
	#start_time = time.time()
	#for x in range(50):
	#	print("{} value of dimension {} at point {}: {}".format(function_selected, function_dimension, input_x, evaluate(input_x)))
	#print(f"TIME: {time.time() - start_time}")
	