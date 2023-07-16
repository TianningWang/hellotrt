import onnx
import struct

static_model_path = './model/mnist.onnx' # batch = 1
dynamic_model_path = './model/mnist_dynamic2.onnx'

def rebatch(model):
    dyn_batch = 'N'
    graph = model.graph

    # change batch size in input, output and value_info
    for tensor in list(graph.input) + list(graph.value_info) + list(graph.output):
        print('=== ', tensor.type.tensor_type.shape)
        tensor.type.tensor_type.shape.dim[0].dim_param = dyn_batch

    for node in graph.node:
        if node.op_type != 'Reshape':
            continue
        for init in graph.initializer:
            if init.name != node.input[1]:
                continue
            if len(init.int64_data) > 0:
                init.int64_data[0] = -1
            elif len(init.raw_data) > 0:
                shape = bytearray(init.raw_data)
                struct.pack_into('q', shape, 0, -1)
                init.raw_data = bytes(shape)

def apply(transform, infile, outfile):
    model = onnx.load(infile)
    transform(model)
    onnx.save(model, outfile)

apply(rebatch, static_model_path, dynamic_model_path)
