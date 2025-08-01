"""keras2c_main.py
This file is part of keras2c
Copyright 2020 Rory Conlin
Licensed under LGPLv3
https://github.com/f0uriest/keras2c

Converts keras model to C code
"""

# Imports
from keras2c.layer2c import Layers2C
from keras2c.weights2c import Weights2C
from keras2c.io_parsing import get_model_io_names
from keras2c.check_model import check_model
from keras2c.make_test_suite import make_test_suite
from keras2c.types import Keras2CConfig
import subprocess
from .backend import keras
import os
import shutil
import pkg_resources


__author__ = "Rory Conlin"
__copyright__ = "Copyright 2020, Rory Conlin"
__license__ = "MIT"
__maintainer__ = "Rory Conlin, https://github.com/f0uriest/keras2c"
__email__ = "wconlin@princeton.edu"


def model2c(model, function_name, malloc=False, verbose=True, output_path='.'):
    """Generates C code for model

    Writes main function definition to "function_name.c" and a public header
    with declarations to "function_name.h"

    Args:
        model (keras.Model): model to convert
        function_name (str): name of C function
        malloc (bool): whether to allocate variables on the stack or heap
        verbose (bool): whether to print info to stdout
        output_path (str): directory to save the generated C files

    Returns:
        malloc_vars (list): names of variables loaded at runtime and stored on the heap
        stateful (bool): whether the model must maintain state between calls
    """

    model_inputs, model_outputs = get_model_io_names(model)
    includes = '#include <math.h>\n'
    includes += '#include <string.h>\n'
    includes += '#include "include/k2c_include.h"\n'
    includes += '#include "include/k2c_tensor_include.h"\n'
    includes += f'#include "{function_name}.h"\n'
    includes += '\n \n'

    if verbose:
        print('Gathering Weights')
    stack_vars, malloc_vars, static_vars = Weights2C(
        model, function_name, malloc).write_weights(verbose)
    stateful = len(static_vars) > 0
    layers = Layers2C(model, malloc).write_layers(verbose)

    function_signature = 'void ' + function_name + '('
    function_signature += ', '.join(['k2c_tensor* ' +
                                     in_nm + '_input' for in_nm in model_inputs]) + ', '
    function_signature += ', '.join(['k2c_tensor* ' +
                                     out_nm + '_output' for out_nm in model_outputs])
    if len(malloc_vars.keys()):
        function_signature += ',' + ','.join(['float* ' +
                                              key for key in malloc_vars.keys()])
    function_signature += ')'

    init_sig, init_fun = gen_function_initialize(function_name, malloc_vars)
    term_sig, term_fun = gen_function_terminate(function_name, malloc_vars)
    reset_sig, reset_fun = gen_function_reset(function_name)

    c_file = os.path.join(output_path, function_name + '.c')
    h_file = os.path.join(output_path, function_name + '.h')

    with open(c_file, 'w') as source:
        source.write('#ifdef __cplusplus\n')
        source.write('extern "C" {\n')
        source.write('#endif\n\n')
        source.write(includes)
        source.write(static_vars + '\n\n')
        source.write(function_signature)
        source.write(' { \n\n')
        source.write(stack_vars)
        source.write(layers)
        source.write('\n } \n\n')
        source.write(init_fun)
        source.write(term_fun)
        source.write('\n#ifdef __cplusplus\n')
        source.write('}\n')
        source.write('#endif\n\n')
        if stateful:
            source.write(reset_fun)

    with open(h_file, 'w') as header:
        header.write(f'#ifndef {function_name.upper()}_H\n')
        header.write(f'#define {function_name.upper()}_H\n\n')
        header.write('#ifdef __cplusplus\n')
        header.write('extern "C" {\n')
        header.write('#endif\n\n')
        header.write('#include "include/k2c_tensor_include.h" \n')
        header.write(function_signature + '; \n')
        header.write(init_sig + '; \n')
        header.write(term_sig + '; \n')
        if stateful:
            header.write(reset_sig + '; \n')
        header.write('\n#ifdef __cplusplus\n')
        header.write('}\n')
        header.write('#endif\n')
        header.write(f'\n#endif /* {function_name.upper()}_H */\n')
    try:
        subprocess.run(['astyle', '-n', h_file])
        subprocess.run(['astyle', '-n', c_file])
    except FileNotFoundError:
        print("astyle not found, {} and {} will not be auto-formatted".format(h_file, c_file))

    return malloc_vars.keys(), stateful


def gen_function_reset(function_name):
    """Writes a reset function for stateful models

    Reset function is used to clear internal state of the model

    Args:
        function_name (str): name of main function

    Returns:
        signature (str): declaration of the reset function
        function (str): definition of the reset function
    """

    reset_sig = 'void ' + function_name + '_reset_states()'

    reset_fun = reset_sig
    reset_fun += ' { \n\n'
    reset_fun += 'memset(&' + function_name + \
                 '_states,0,sizeof(' + function_name + '_states)); \n'
    reset_fun += "} \n\n"
    return reset_sig, reset_fun


def gen_function_initialize(function_name, malloc_vars):
    """Writes an initialize function

    Initialize function is used to load variables into memory and do other start-up tasks

    Args:
        function_name (str): name of main function
        malloc_vars (dict): variables to read in

    Returns:
        signature (str): declaration of the initialization function
        function (str): definition of the initialization function
    """

    init_sig = 'void ' + function_name + '_initialize('
    init_sig += ','.join(['float** ' +
                          key + ' \n' for key in malloc_vars.keys()])
    init_sig += ')'

    init_fun = init_sig
    init_fun += ' { \n\n'
    for key, value in malloc_vars.items():
        flat = value.flatten(order='C')
        init_fun += 'static const float ' + key + '_init[' + str(flat.size) + '] = {\n'
        for idx, val in enumerate(flat):
            init_fun += f'{val:+.8e}f,'
            if (idx + 1) % 5 == 0:
                init_fun += '\n'
        init_fun += '};\n'
        init_fun += '*' + key + ' = (float*) malloc(' + str(flat.size) + ' * sizeof(float)); \n'
        init_fun += 'memcpy(*' + key + ', ' + key + '_init, ' + str(flat.size) + ' * sizeof(float));\n'
    init_fun += "} \n\n"

    return init_sig, init_fun


def gen_function_terminate(function_name, malloc_vars):
    """Writes a terminate function

    Terminate function is used to deallocate memory after completion

    Args:
        function_name (str): name of main function
        malloc_vars (dict): variables to deallocate

    Returns:
        signature (str): declaration of the terminate function
        function (str): definition of the terminate function
    """

    term_sig = 'void ' + function_name + '_terminate('
    term_sig += ','.join(['float* ' +
                          key for key in malloc_vars.keys()])
    term_sig += ')'

    term_fun = term_sig
    term_fun += ' { \n\n'
    for key in malloc_vars.keys():
        term_fun += "free(" + key + "); \n"
    term_fun += "} \n\n"

    return term_sig, term_fun


def k2c(model, function_name, malloc=False, num_tests=10, verbose=True, output_path='.'):
    """Converts Keras model to C code and generates test suite

    Args:
        model (keras.Model or str): model to convert or path to saved .h5 file
        function_name (str): name of main function
        malloc (bool): whether to allocate variables on the stack or heap
        num_tests (int): how many tests to generate in the test suite
        verbose (bool): whether to print progress
        output_path (str): directory to save the generated C files

    Raises:
        ValueError: if model is not an instance of keras.Model

    Returns:
        None
    """

    cfg = Keras2CConfig(
        model=model,
        function_name=function_name,
        malloc=malloc,
        num_tests=num_tests,
        verbose=verbose,
    )

    model = cfg.model
    function_name = cfg.function_name
    malloc = cfg.malloc
    num_tests = cfg.num_tests
    verbose = cfg.verbose

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    include_dir = pkg_resources.resource_filename('keras2c', 'include')
    target_include_dir = os.path.join(output_path, 'include')
    if os.path.exists(target_include_dir):
        shutil.rmtree(target_include_dir)
    shutil.copytree(include_dir, target_include_dir)

    function_name = str(function_name)
    if isinstance(model, str):
        model = keras.load_model(model)
    elif not isinstance(model, keras.Model):
        raise ValueError(
            'Unknown model type. Model should either be an instance of keras.Model, '
            'or a filepath to a saved .h5 model'
        )

    # Check that the model can be converted
    check_model(model, function_name)
    if verbose:
        print('All checks passed')

    malloc_vars, stateful = model2c(
        model, function_name, malloc, verbose, output_path)

    s = 'Done \n'
    s += "C code is in '" + os.path.join(output_path, function_name + ".c") + \
        "' with header file '" + os.path.join(output_path, function_name + ".h") + "' \n"
    if num_tests > 0:
        make_test_suite(model, function_name, malloc_vars,
                        num_tests, stateful, verbose, output_path=output_path)
        s += "Tests are in '" + os.path.join(output_path, function_name + "_test_suite.c") + "' \n"
    if malloc:
        s += "Weight arrays are in .csv files of the form 'model_name_layer_name_array_type.csv' \n"
        s += "They should be placed in the directory from which the main program is run."
    if verbose:
        print(s)