import keras.backend as K
from tensorflow.python.framework import graph_util, graph_io
from tensorflow.python.tools import optimize_for_inference
import os
import argparse
from keras.models import model_from_json

def convert_keras_to_tf_graph(kmodeljson, kmodelweight, outtfpb):

    # disable learning
    K.set_learning_phase(0)

    sess = K.get_session()

    with open(kmodeljson) as f:
        net_model = model_from_json(f.read())
    net_model.load_weights(kmodelweight)

    output_node_names=[]
    for output in net_model.outputs:
        output_node_names.append(output.op.name)

    print("output", output_node_names)

    const_graph = graph_util.convert_variables_to_constants(
        sess,
        sess.graph.as_graph_def(),
        output_node_names
    )

    graph_io.write_graph(
        const_graph,
        '.',
        outtfpb,
        as_text=False
    )


def main(args):
    convert_keras_to_tf_graph(args.input_model_json,
                              args.input_model_weights,
                              args.out_tfpb)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--gpuID", default=0, type=int, help='gpu id')
    parser.add_argument("--input_model_json", required=True, help="model json file")
    parser.add_argument("--input_model_weights", required=True, help="model weight file")
    parser.add_argument("--out_tfpb", required=True, help="place to store converted tf pb")

    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpuID)

    main(args)
