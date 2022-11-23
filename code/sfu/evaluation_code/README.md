# Introduction
This python module contains the functions necessary to search and evaluate functions of the [Surjanovic and Bingham Optimization Test Problems](https://www.sfu.ca/~ssurjano/optimization.html).
`main.py` shows an example of usage of the module.

# Modules function description

## List of functions
- [load_json()](#json_load)
- [select_function()](#select_function)
- [search_function()](#search_function)
- [dimension()](#dimension)
- [minimum_point()](#minimum_point)
- [minimum_value()](#minimum_value)
- [parameters()](#parameters)
- [input_domain()](#input_domain)
- [evaluate()](#evaluate)

## List of variables
- [json_functions](#json_functions)
- [function_name](#function_name)
- [function_informations](#function_informations)
- [path_implementation](#path_implementation)

## Functions

<a name="load_json"></a>
### `load_json(filepath)`
Load the json object at `filepath` and returns it.

<a name="select_function"></a>
### `select_function(name)`
Prepare the module to evaluate the function named `name`.  
The function `name` will be referred through the rest of this documentation as "selected function".

<a name="search_function"></a>
### `search_function(filters=None)`
Returns the names of the functions.  
If `filters` is provided, it returns the names of functions that satisfy the filters. `filters` is a dictionary with the information fields of the functions as key (e.g., `filters = { "dimension" : 2, "minimum_f" = True }`).

<a name="dimension"></a>
### `dimension()`
Returns the dimension of the selected function.

<a name="minimum_point"></a>
### `minimum_point(dim=None)`
Returns the input coordinates of the global minimum value of the selected function.  
The `dim` argument is passed when the selected function has dimension `d`.  
Returns `None` if no such value is available.

<a name="minimum_value"></a>
### `minimum_value(dim=None)`
Returns the global minimum value of the selected function.  
Returns `None` if no such value is available.

<a name="parameters"></a>
### `parameters()`
Returns a description of the parameters accepted by the selected function.  
Returns `None` if the function does not accepts parameters.

<a name="input_domain"></a>
### `input_domain(dim=None)`
Returns a list of ranges for the input dimension of each feature's dimension.  
The `dim` argument is passed when the selected function has dimension `d`.

<a name="evaluate"></a>
### `evaluate(inp, param=None)`
Returns the selected function value at point `inp`, which is a float value or a list of floats.  
The `param` dictionary is a dictionary of selected function's parameters.  
Returns `None` if there has been an error on the function computation.

## Variables

<a name="json_functions"></a>
### `json_functions`
Contains the object that is inside the json file, loaded with [`load_json()`](#load_json).

<a name="function_name"></a>
### `function_name`
Contains the name selected function.

<a name="function_informations"></a>
### `function_informations`
Contains a dictionary with the information fields as keys, of the selected function.

<a name="path_implementation"></a>
### `path_implementation`
Contains the string of the file path of the R implementation of the selected function.