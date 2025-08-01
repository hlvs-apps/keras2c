"""layer2c.py
This file is part of keras2c
Copyright 2020 Rory Conlin
Licensed under LGPLv3
https://github.com/f0uriest/keras2c

Writes individual layers to C code
"""

# Imports
from keras2c.io_parsing import (
    layer_type, get_model_io_names, get_all_io_names, get_layer_io_names, flatten
)
from keras2c.types import LayerIO

__author__ = "Rory Conlin"
__copyright__ = "Copyright 2020, Rory Conlin"
__license__ = "MIT"
__maintainer__ = "Rory Conlin, https://github.com/f0uriest/keras2c"
__email__ = "wconlin@princeton.edu"


class Layers2C():
    """Creates an object to parse and write layer functions.

    Args:
        model (keras.Model): model to parse
        malloc (bool): Whether to allocate variables on the heap using malloc.
    """

    def __init__(self, model, malloc):
        self.model = model
        self.model_inputs, self.model_outputs = get_model_io_names(self.model)
        self.layers = ''
        self.malloc = malloc

    def write_layers(self, verbose=True):
        """Writes layers in the correct graph order.

        Args:
            verbose (bool): whether to print progress

        Returns:
            layers (str): C code for calling layer functions in correct order
        """
        written_io = set(self.model_inputs)
        unwritten_io = set(get_all_io_names(self.model)) - written_io
        while len(unwritten_io) > 0:
            for layer in self.model.layers:
                layer_inputs, layer_outputs = get_layer_io_names(layer)
                if len(layer_outputs) == 0:
                    continue
                if len(layer_inputs) > 1:
                    inp = layer_inputs
                elif len(layer_inputs) == 1:
                    inp = layer_inputs[0]
                else:
                    inp = []
                for i, outp in enumerate(layer_outputs):
                    if (
                        set(flatten(inp)).issubset(written_io)
                        and set(flatten(outp)).issubset(unwritten_io)
                    ) or layer_type(layer) == 'InputLayer':
                        if verbose:
                            print('Writing layer ', outp)
                        method = getattr(self, '_write_layer_' + layer_type(layer))
                        method(layer, inp, outp, i)
                        written_io |= set(flatten(inp))
                        written_io |= set(flatten(outp))
                        unwritten_io -= set(flatten(inp))
                        unwritten_io -= set(flatten(outp))
        return self.layers

    def _format_io_names(self, layer, inp, outp, model_io=False) -> LayerIO:
        nm = layer.name
        pnm = '&' + nm
        is_model_input = False
        is_model_output = False
        if isinstance(inp, list):
            inp_nm = []
            for j in inp:
                if j in self.model_inputs or 'timeslice' in j:
                    inp_nm.append(j + '_input')
                    is_model_input = True
                else:
                    inp_nm.append('&' + j + '_output')
        else:
            if inp in self.model_inputs or 'timeslice' in inp:
                inp_nm = inp + '_input'
                is_model_input = True
            else:
                inp_nm = '&' + inp + '_output'
        if isinstance(outp, list):
            outp_nm = []
            for o in outp:
                if o in self.model_outputs or 'timeslice' in o:
                    outp_nm.append(o + '_output')
                    is_model_output = True
                else:
                    outp_nm.append('&' + o + '_output')
        else:
            if outp in self.model_outputs or 'timeslice' in outp:
                outp_nm = outp + '_output'
                is_model_output = True
            else:
                outp_nm = '&' + outp + '_output'

        return LayerIO(
            name=nm,
            pointer=pnm,
            inputs=inp_nm,
            outputs=outp_nm,
            is_model_input=is_model_input if model_io else False,
            is_model_output=is_model_output if model_io else False,
        )

    def _write_layer_TimeDistributed(self, layer, inputs, outputs, i):
        self.layers += f'for(size_t i=0; i<{layer.name}_timesteps; ++i) {{ \n'
        if inputs in self.model_inputs:
            self.layers += (
                f'{layer.layer.name}_timeslice_input.array = &{inputs}_input->array[i*{layer.name}_in_offset]; \n'
            )
        else:
            self.layers += (
                f'{layer.layer.name}_timeslice_input.array = &{inputs}_output.array[i*{layer.name}_in_offset]; \n'
            )
        if outputs in self.model_outputs:
            self.layers += (
                f'{layer.layer.name}_timeslice_output.array = &{outputs}_output->array[i*{layer.name}_out_offset]; \n'
            )
        else:
            self.layers += (
                f'{layer.layer.name}_timeslice_output.array = &{outputs}_output.array[i*{layer.name}_out_offset]; \n'
            )

        inp = '&' + layer.layer.name + '_timeslice'
        outp = '&' + layer.layer.name + '_timeslice'
        method = getattr(self, '_write_layer_' + layer_type(layer.layer))
        method(layer.layer, inp, outp, i)
        self.layers += '\n } \n'

    def _write_layer_Bidirectional(self, layer, inputs, outputs, i):
        # Get the wrapped layer type (e.g., 'LSTM', 'GRU')
        wrapped_layer_type = layer_type(layer.forward_layer)
        # Generate a subname based on the wrapped layer type
        subname = wrapped_layer_type.lower()

        method = getattr(self, '_write_layer_' + wrapped_layer_type)

        # Process forward and backward layers
        method(layer.forward_layer, inputs, 'forward_' + subname, i)
        method(layer.backward_layer, inputs, 'backward_' + subname, i)

        mode = layer.merge_mode
        # Update inputs to be the outputs of the forward and backward layers
        inputs = ['forward_' + subname, 'backward_' + subname]

        if layer.forward_layer.return_sequences:
            self.layers += f'k2c_flip(&backward_{subname}_output,0); \n'

        if mode == 'sum':
            self._write_layer_Merge(layer, inputs, outputs, i, 'Add')
        elif mode == 'mul':
            self._write_layer_Merge(layer, inputs, outputs, i, 'Multiply')
        elif mode == 'ave':
            self._write_layer_Merge(layer, inputs, outputs, i, 'Average')
        elif mode == 'concat':
            self._write_layer_Concatenate(layer, inputs, outputs, i)

    def _write_layer_LSTM(self, layer, inputs, outputs, i):
        ctx = self._format_io_names(layer, inputs, outputs)
        self.layers += (
            'k2c_lstm('
            + ctx.outputs
            + ','
            + ctx.inputs
            + ','
            + ctx.name
            + '_state,'
            + ctx.pointer
            + '_kernel, \n\t'
            + ctx.pointer
            + '_recurrent_kernel,'
            + ctx.pointer
            + '_bias,'
            + ctx.name
            + '_fwork, \n\t'
            + ctx.name
            + '_go_backwards,'
            + ctx.name
            + '_return_sequences, \n\t'
            + 'k2c_'
            + layer.get_config()['recurrent_activation']
            + ','
            + 'k2c_'
            + layer.get_config()['activation']
            + '); \n'
        )

    def _write_layer_Dense(self, layer, inputs, outputs, i):
        ctx = self._format_io_names(layer, inputs, outputs)
        activation = 'k2c_' + layer.get_config()['activation']

        self.layers += (
            'k2c_dense('
            + ctx.outputs
            + ','
            + ctx.inputs
            + ','
            + ctx.pointer
            + '_kernel, \n\t'
            + ctx.pointer
            + '_bias,'
            + activation
            + ','
            + ctx.name
            + '_fwork); \n'
        )

    def _write_layer_Conv(self, layer, inputs, outputs, i):
        ctx = self._format_io_names(layer, inputs, outputs)
        activation = 'k2c_' + layer.get_config()['activation']
        if layer_type(layer)[-2:] == '1D':
            fname = 'k2c_conv1d('
        elif layer_type(layer)[-2:] == '2D':
            fname = 'k2c_conv2d('
        elif layer_type(layer)[-2:] == '3D':
            fname = 'k2c_conv3d('
        if layer.get_config()['padding'] == 'valid':
            self.layers += (
                fname
                + ctx.outputs
                + ','
                + ctx.inputs
                + ','
                + ctx.pointer
                + '_kernel, \n\t'
                + ctx.pointer
                + '_bias,'
                + ctx.name
                + '_stride,'
                + ctx.name
                + '_dilation,'
                + activation
                + '); \n'
            )
        else:
            # When padding is 'same', pad the already-formatted input tensor.
            # `inputs` here refers to the raw IO name, but `_format_io_names`
            # returns the correct pointer string in ``ctx.inputs``. Using the
            # raw name breaks compilation for model inputs such as
            # ``keras_tensor`` where the variable is actually called
            # ``keras_tensor_input``.
            self._write_layer_ZeroPad(layer, ctx.inputs, ctx.pointer + '_padded_input', i)
            self.layers += (
                fname
                + ctx.outputs
                + ','
                + ctx.pointer
                + '_padded_input,'
                + ctx.pointer
                + '_kernel, \n\t'
                + ctx.pointer
                + '_bias,'
                + ctx.name
                + '_stride,'
                + ctx.name
                + '_dilation,'
                + activation
                + '); \n'
            )

    def _write_layer_Conv1D(self, layer, inputs, outputs, i):
        self._write_layer_Conv(layer, inputs, outputs, i)

    def _write_layer_Conv2D(self, layer, inputs, outputs, i):
        self._write_layer_Conv(layer, inputs, outputs, i)

    def _write_layer_Conv3D(self, layer, inputs, outputs, i):
        self._write_layer_Conv(layer, inputs, outputs, i)

    def _write_layer_MaxPooling1D(self, layer, inputs, outputs, i):
        self._write_layer_Pooling(layer, inputs, outputs, i)

    def _write_layer_AveragePooling1D(self, layer, inputs, outputs, i):
        self._write_layer_Pooling(layer, inputs, outputs, i)

    def _write_layer_Pooling(self, layer, inputs, outputs, i):
        ctx = self._format_io_names(layer, inputs, outputs)
        if 'Max' in layer_type(layer):
            s = 'k2c_maxpool'
        else:
            s = 'k2c_avgpool'
        if layer_type(layer)[-2:] == '1D':
            s += '1d(' + ctx.outputs + ','
        elif layer_type(layer)[-2:] == '2D':
            s += '2d(' + ctx.outputs + ','

        if layer.get_config()['padding'] == 'valid':
            s += ctx.inputs + ','
        else:
            # Use the formatted input name from ``ctx.inputs`` when padding
            # is ``same``. Passing the raw name causes unresolved identifiers
            # for model inputs.
            self._write_layer_ZeroPad(layer, ctx.inputs, ctx.pointer + '_padded_input', i)
            s += ctx.pointer + '_padded_input,'
        s += ctx.name + '_pool_size, \n\t' + ctx.name + '_stride); \n'
        self.layers += s

    def _write_layer_MaxPooling2D(self, layer, inputs, outputs, i):
        self._write_layer_Pooling(layer, inputs, outputs, i)

    def _write_layer_AveragePooling2D(self, layer, inputs, outputs, i):
        self._write_layer_Pooling(layer, inputs, outputs, i)

    def _write_layer_GlobalMaxPooling1D(self, layer, inputs, outputs, i):
        self._write_layer_GlobalPooling(layer, inputs, outputs, i)

    def _write_layer_GlobalMaxPooling2D(self, layer, inputs, outputs, i):
        self._write_layer_GlobalPooling(layer, inputs, outputs, i)

    def _write_layer_GlobalMaxPooling3D(self, layer, inputs, outputs, i):
        self._write_layer_GlobalPooling(layer, inputs, outputs, i)

    def _write_layer_GlobalAveragePooling1D(self, layer, inputs, outputs, i):
        self._write_layer_GlobalPooling(layer, inputs, outputs, i)

    def _write_layer_GlobalAveragePooling2D(self, layer, inputs, outputs, i):
        self._write_layer_GlobalPooling(layer, inputs, outputs, i)

    def _write_layer_GlobalAveragePooling3D(self, layer, inputs, outputs, i):
        self._write_layer_GlobalPooling(layer, inputs, outputs, i)

    def _write_layer_GlobalPooling(self, layer, inputs, outputs, i):
        ctx = self._format_io_names(layer, inputs, outputs)
        inputs = ctx.inputs
        outputs = ctx.outputs
        if 'Max' in layer_type(layer):
            self.layers += 'k2c_global_max_pooling('
        else:
            self.layers += 'k2c_global_avg_pooling('
        self.layers += outputs + ',' + inputs + '); \n'

    def _write_layer_Add(self, layer, inputs, outputs, i):
        self._write_layer_Merge(layer, inputs, outputs, i, 'Add')

    def _write_layer_Subtract(self, layer, inputs, outputs, i):
        self._write_layer_Merge(layer, inputs, outputs, i, 'Subtract')

    def _write_layer_Multiply(self, layer, inputs, outputs, i):
        self._write_layer_Merge(layer, inputs, outputs, i, 'Multiply')

    def _write_layer_Maximum(self, layer, inputs, outputs, i):
        self._write_layer_Merge(layer, inputs, outputs, i, 'Maximum')

    def _write_layer_Minimum(self, layer, inputs, outputs, i):
        self._write_layer_Merge(layer, inputs, outputs, i, 'Minimum')

    def _write_layer_Average(self, layer, inputs, outputs, i):
        self._write_layer_Merge(layer, inputs, outputs, i, 'Average')

    def _write_layer_Merge(self, layer, inputs, outputs, i, mode):
        ctx = self._format_io_names(layer, inputs, outputs)
        nm = ctx.name
        inputs = ctx.inputs
        outputs = ctx.outputs
        if mode == 'Subtract':
            self.layers += 'k2c_subtract('
        elif mode == 'Add':
            self.layers += 'k2c_add('
        elif mode == 'Multiply':
            self.layers += 'k2c_multiply('
        elif mode == 'Average':
            self.layers += 'k2c_average('
        elif mode == 'Maximum':
            self.layers += 'k2c_max('
        elif mode == 'Minimum':
            self.layers += 'k2c_min('
        self.layers += outputs + ',' + nm + '_num_tensors' + str(i) + ','
        c = ','.join(inputs)
        self.layers += c + '); \n'

    def _write_layer_Concatenate(self, layer, inputs, outputs, i):
        ctx = self._format_io_names(layer, inputs, outputs)
        nm = ctx.name
        self.layers += (
            'k2c_concatenate('
            + ctx.outputs
            + ','
            + nm
            + '_axis'
            + ','
            + nm
            + '_num_tensors'
            + str(i)
            + ','
        )
        c = ','.join(ctx.inputs)
        self.layers += c + '); \n'

    def _write_layer_GRU(self, layer, inputs, outputs, i):
        ctx = self._format_io_names(layer, inputs, outputs)
        self.layers += (
            'k2c_gru('
            + ctx.outputs
            + ','
            + ctx.inputs
            + ','
            + ctx.name
            + '_state,'
            + ctx.pointer
            + '_kernel, \n\t'
            + ctx.pointer
            + '_recurrent_kernel,'
            + ctx.pointer
            + '_bias,'
            + ctx.name
            + '_fwork, \n\t'
            + ctx.name
            + '_reset_after,'
            + ctx.name
            + '_go_backwards,'
            + ctx.name
            + '_return_sequences, \n\t'
            + 'k2c_'
            + layer.get_config()['recurrent_activation']
            + ','
            + 'k2c_'
            + layer.get_config()['activation']
            + '); \n'
        )

    def _write_layer_SimpleRNN(self, layer, inputs, outputs, i):
        ctx = self._format_io_names(layer, inputs, outputs)
        self.layers += (
            'k2c_simpleRNN('
            + ctx.outputs
            + ','
            + ctx.inputs
            + ','
            + ctx.name
            + '_state,'
            + ctx.pointer
            + '_kernel, \n\t'
            + ctx.pointer
            + '_recurrent_kernel,'
            + ctx.pointer
            + '_bias,'
            + ctx.name
            + '_fwork, \n\t'
            + ctx.name
            + '_go_backwards,'
            + ctx.name
            + '_return_sequences,'
            + 'k2c_'
            + layer.get_config()['activation']
            + '); \n'
        )

    def _write_layer_Activation(self, layer, inputs, outputs, i):
        ctx = self._format_io_names(layer, inputs, outputs, True)
        inputs = ctx.inputs
        outputs = ctx.outputs
        is_model_input = ctx.is_model_input
        is_model_output = ctx.is_model_output
        activation = 'k2c_' + layer.get_config()['activation']
        if is_model_input:
            inp = inputs + '->'
        else:
            inp = inputs[1:] + '.'
        self.layers += activation + '(' + inp + 'array,' + inp + 'numel); \n'
        self._write_dummy_layer(layer, inputs, outputs, i,
                                is_model_input, is_model_output)

    def _write_layer_LeakyReLU(self, layer, inputs, outputs, i):
        self._write_layer_AdvancedActivation(layer, inputs, outputs, i)

    def _write_layer_PReLU(self, layer, inputs, outputs, i):
        self._write_layer_AdvancedActivation(layer, inputs, outputs, i)

    def _write_layer_ELU(self, layer, inputs, outputs, i):
        self._write_layer_AdvancedActivation(layer, inputs, outputs, i)

    def _write_layer_ThresholdedReLU(self, layer, inputs, outputs, i):
        self._write_layer_AdvancedActivation(layer, inputs, outputs, i)

    def _write_layer_ReLU(self, layer, inputs, outputs, i):
        self._write_layer_AdvancedActivation(layer, inputs, outputs, i)

    def _write_layer_Swish(self, layer, inputs, outputs, i):
        self._write_layer_AdvancedActivation(layer, inputs, outputs, i)

    def _write_layer_SILU(self, layer, inputs, outputs, i):
        self._write_layer_AdvancedActivation(layer, inputs, outputs, i)

    def _write_layer_AdvancedActivation(self, layer, inputs, outputs, i):
        ctx = self._format_io_names(layer, inputs, outputs, True)
        nm = ctx.name
        inputs = ctx.inputs
        outputs = ctx.outputs
        is_model_input = ctx.is_model_input
        is_model_output = ctx.is_model_output
        if is_model_input:
            inp = inputs + '->'
        else:
            inp = inputs + '.'

        if layer_type(layer) == 'LeakyReLU':
            self.layers += 'k2c_LeakyReLU(' + inp + 'array,' + \
                inp + 'numel,' + nm + '_negative_slope); \n'
        if layer_type(layer) == 'PReLU':
            self.layers += 'k2c_PReLU(' + inp + 'array,' + inp + \
                'numel,' + nm + '_alpha.array); \n'
        if layer_type(layer) == 'ELU':
            self.layers += 'k2c_ELU(' + inp + 'array,' + inp + \
                'numel,' + nm + '_alpha); \n'
        if layer_type(layer) == 'ThresholdedReLU':
            self.layers += 'k2c_ThresholdedReLU(' + inp + 'array,' + \
                inp + 'numel,' + nm + '_theta); \n'
        if layer_type(layer) == 'ReLU':
            self.layers += 'k2c_ReLU(' + inp + 'array,' + inp + \
                           'numel,' + nm + '_max_value, \n\t' + \
                           nm + '_negative_slope,' + nm + '_threshold); \n'
        if layer_type(layer) == 'Swish':
            self.layers += 'k2c_swish(' + inp + 'array,' + inp + 'numel); \n'
        if layer_type(layer) == 'SILU':
            self.layers += 'k2c_silu(' + inp + 'array,' + inp + 'numel); \n'
        self._write_dummy_layer(layer, inputs, outputs, i,
                                is_model_input, is_model_output)

    def _write_dummy_layer(self, layer, inputs, outputs, i, is_model_input, is_model_output):
        outputs = outputs.replace("&", "")
        inputs = inputs.replace("&", "")
        if is_model_input and is_model_output:
            self.layers += outputs + '->ndim = ' + \
                inputs + '->ndim; // copy data into output struct \n'
            self.layers += outputs + '->numel = ' + inputs + '->numel; \n'
            self.layers += 'memcpy(&' + outputs + '->shape,&' + inputs + \
                           '->shape,K2C_MAX_NDIM*sizeof(size_t));  \n'
            self.layers += 'memcpy(' + outputs + '->array,' + inputs + '->array,' + \
                           outputs + \
                '->numel*sizeof(' + outputs + '->array[0])); \n'
        elif is_model_input:
            self.layers += 'k2c_tensor ' + outputs + '; \n'
            self.layers += outputs + '.ndim = ' + \
                inputs + '->ndim; // copy data into output struct \n'
            self.layers += outputs + '.numel = ' + inputs + '->numel; \n'
            self.layers += 'memcpy(' + outputs + '.shape,' + inputs + \
                           '->shape,K2C_MAX_NDIM*sizeof(size_t));  \n'
            self.layers += outputs + '.array = &' + inputs + \
                '->array[0]; // rename for clarity \n'
        elif is_model_output:
            self.layers += outputs + '->ndim = ' + \
                inputs + '.ndim; // copy data into output struct \n'
            self.layers += outputs + '->numel = ' + inputs + '.numel; \n'
            self.layers += 'memcpy(' + outputs + '->shape,' + inputs + \
                           '.shape,K2C_MAX_NDIM*sizeof(size_t));  \n'
            self.layers += 'memcpy(' + outputs + '->array,' + inputs + '.array,' + \
                           outputs + \
                '->numel*sizeof(' + outputs + '->array[0])); \n'
        else:
            self.layers += 'k2c_tensor ' + outputs + '; \n'
            self.layers += outputs + '.ndim = ' + \
                inputs + '.ndim; // copy data into output struct \n'
            self.layers += outputs + '.numel = ' + inputs + '.numel; \n'
            self.layers += 'memcpy(' + outputs + '.shape,' + inputs + \
                           '.shape,K2C_MAX_NDIM*sizeof(size_t));  \n'
            self.layers += outputs + '.array = &' + inputs + \
                '.array[0]; // rename for clarity \n'

    def _write_layer_Reshape(self, layer, inputs, outputs, i):
        ctx = self._format_io_names(layer, inputs, outputs, True)
        nm = ctx.name
        inputs = ctx.inputs
        outputs = ctx.outputs
        self.layers += 'k2c_reshape(' + outputs + ',' + inputs + ',' + nm + \
            '_newshp,' + nm + '_newndim); \n'

    def _write_layer_Flatten(self, layer, inputs, outputs, i):
        ctx = self._format_io_names(layer, inputs, outputs, True)
        inputs = ctx.inputs
        outputs = ctx.outputs
        self.layers += 'k2c_flatten(' + outputs + ',' + inputs + '); \n'

    def _write_layer_Permute(self, layer, inputs, outputs, i):
        ctx = self._format_io_names(layer, inputs, outputs)
        nm = ctx.name
        self.layers += 'k2c_permute_dims(' + ctx.outputs + ',' + ctx.inputs + \
            ',' + nm + '_permute); \n'

    def _write_layer_RepeatVector(self, layer, inputs, outputs, i):
        ctx = self._format_io_names(layer, inputs, outputs)
        nm = ctx.name
        self.layers += 'k2c_repeat_vector(' + ctx.outputs + ',' + ctx.inputs + \
            ',' + nm + '_n); \n'

    def _write_layer_Dot(self, layer, inputs, outputs, i):
        ctx = self._format_io_names(layer, inputs, outputs)
        nm = ctx.name
        self.layers += 'k2c_dot(' + ctx.outputs + ',' + ctx.inputs[0] + \
                       ',' + ctx.inputs[1] + ',' + nm + '_axesA,' + \
                       '\n\t' + nm + '_axesB,' + nm + '_naxes,' + \
                       nm + '_normalize,' + nm + '_fwork); \n'

    def _write_layer_BatchNormalization(self, layer, inputs, outputs, i):
        ctx = self._format_io_names(layer, inputs, outputs)
        nm = ctx.name
        pnm = ctx.pointer
        self.layers += 'k2c_batch_norm(' + ctx.outputs + ',' + ctx.inputs + \
                       ',' + pnm + '_mean,' + pnm + '_stdev,' + pnm + \
                       '_gamma,' + pnm + '_beta,' + nm + '_axis); \n'

    def _write_layer_Embedding(self, layer, inputs, outputs, i):
        ctx = self._format_io_names(layer, inputs, outputs)
        self.layers += 'k2c_embedding(' + ctx.outputs + ',' + ctx.inputs + \
            ',' + ctx.pointer + '_kernel); \n'

    def _write_layer_UpSampling1D(self, layer, inputs, outputs, i):
        self._write_layer_UpSampling(layer, inputs, outputs, i)

    def _write_layer_UpSampling2D(self, layer, inputs, outputs, i):
        self._write_layer_UpSampling(layer, inputs, outputs, i)

    def _write_layer_UpSampling3D(self, layer, inputs, outputs, i):
        self._write_layer_UpSampling(layer, inputs, outputs, i)

    def _write_layer_UpSampling(self, layer, inputs, outputs, i):
        ctx = self._format_io_names(layer, inputs, outputs)
        nm = ctx.name
        if layer_type(layer)[-2:] == '1D':
            self.layers += 'k2c_upsampling1d('
        elif layer_type(layer)[-2:] == '2D':
            self.layers += 'k2c_upsampling2d('
        elif layer_type(layer)[-2:] == '3D':
            self.layers += 'k2c_upsampling3d('
        self.layers += ctx.outputs + ',' + ctx.inputs + ',' + nm + '_size); \n'

    def _write_layer_Cropping1D(self, layer, inputs, outputs, i):
        self._write_layer_Cropping(layer, inputs, outputs, i)

    def _write_layer_Cropping2D(self, layer, inputs, outputs, i):
        self._write_layer_Cropping(layer, inputs, outputs, i)

    def _write_layer_Cropping3D(self, layer, inputs, outputs, i):
        self._write_layer_Cropping(layer, inputs, outputs, i)

    def _write_layer_Cropping(self, layer, inputs, outputs, i):
        ctx = self._format_io_names(layer, inputs, outputs)
        nm = ctx.name
        inputs = ctx.inputs
        outputs = ctx.outputs
        if layer_type(layer)[-2:] == '1D':
            self.layers += 'k2c_crop1d('
        elif layer_type(layer)[-2:] == '2D':
            self.layers += 'k2c_crop2d('
        elif layer_type(layer)[-2:] == '3D':
            self.layers += 'k2c_crop3d('
        self.layers += outputs + ',' + inputs + ',' + nm + '_crop); \n'

    def _write_layer_ZeroPadding1D(self, layer, inputs, outputs, i):
        self._write_layer_ZeroPad(layer, inputs, outputs, i)

    def _write_layer_ZeroPadding2D(self, layer, inputs, outputs, i):
        self._write_layer_ZeroPad(layer, inputs, outputs, i)

    def _write_layer_ZeroPadding3D(self, layer, inputs, outputs, i):
        self._write_layer_ZeroPad(layer, inputs, outputs, i)

    def _write_layer_ZeroPad(self, layer, inputs, outputs, i):
        if 'Zero' in layer_type(layer):
            ctx = self._format_io_names(layer, inputs, outputs)
            nm = ctx.name
            inputs = ctx.inputs
            outputs = ctx.outputs
        else:
            nm = layer.name
        if layer_type(layer)[-2:] == '1D':
            self.layers += 'k2c_pad1d('
        elif layer_type(layer)[-2:] == '2D':
            self.layers += 'k2c_pad2d('
        elif layer_type(layer)[-2:] == '3D':
            self.layers += 'k2c_pad3d('
        self.layers += outputs + ',' + inputs + ',' + nm + \
            '_fill, \n\t' + nm + '_pad); \n'

    def _write_layer_Dropout(self, layer, inputs, outputs, i):
        ctx = self._format_io_names(layer, inputs, outputs, True)
        inputs = ctx.inputs
        outputs = ctx.outputs
        is_model_input = ctx.is_model_input
        is_model_output = ctx.is_model_output
        self._write_dummy_layer(layer, inputs, outputs, i,
                                is_model_input, is_model_output)

    def _write_layer_SpatialDropout1D(self, layer, inputs, outputs, i):
        ctx = self._format_io_names(layer, inputs, outputs, True)
        self._write_dummy_layer(layer, ctx.inputs, ctx.outputs, i,
                                ctx.is_model_input, ctx.is_model_output)

    def _write_layer_SpatialDropout2D(self, layer, inputs, outputs, i):
        ctx = self._format_io_names(layer, inputs, outputs, True)
        self._write_dummy_layer(layer, ctx.inputs, ctx.outputs, i,
                                ctx.is_model_input, ctx.is_model_output)

    def _write_layer_SpatialDropout3D(self, layer, inputs, outputs, i):
        ctx = self._format_io_names(layer, inputs, outputs, True)
        self._write_dummy_layer(layer, ctx.inputs, ctx.outputs, i,
                                ctx.is_model_input, ctx.is_model_output)

    def _write_layer_ActivityRegularization(self, layer, inputs, outputs, i):
        ctx = self._format_io_names(layer, inputs, outputs, True)
        self._write_dummy_layer(layer, ctx.inputs, ctx.outputs, i,
                                ctx.is_model_input, ctx.is_model_output)

    def _write_layer_GaussianNoise(self, layer, inputs, outputs, i):
        ctx = self._format_io_names(layer, inputs, outputs, True)
        self._write_dummy_layer(layer, ctx.inputs, ctx.outputs, i,
                                ctx.is_model_input, ctx.is_model_output)

    def _write_layer_GaussianDropout(self, layer, inputs, outputs, i):
        ctx = self._format_io_names(layer, inputs, outputs, True)
        self._write_dummy_layer(layer, ctx.inputs, ctx.outputs, i,
                                ctx.is_model_input, ctx.is_model_output)

    def _write_layer_AlphaDropout(self, layer, inputs, outputs, i):
        ctx = self._format_io_names(layer, inputs, outputs, True)
        self._write_dummy_layer(layer, ctx.inputs, ctx.outputs, i,
                                ctx.is_model_input, ctx.is_model_output)

    def _write_layer_Input(self, layer, inputs, outputs, i):
        self.layers += ''

    def _write_layer_InputLayer(self, layer, inputs, outputs, i):
        self.layers += ''
