import threading

import torch
import torch.distributed as dist

from megatron import initialize_megatron, get_args, print_rank_0
from megatron.checkpointing import load_checkpoint
from megatron.core import mpu
from megatron.model import GPTModel
from megatron.training import get_model
from flask import Flask, request, jsonify, current_app
from flask_restful import Resource, Api

lock = threading.Lock()


def evaluate(model, input_ids=None):
    pass


class MegatronEval(Resource):
    def __init__(self, model):
        self.model = model

    def put(self):
        args = get_args()

        input_ids = request.get_json()["input_ids"]
        if not isinstance(input_ids, list):
            return "prompts is not a list of strings", 400
        input_ids = torch.LongTensor(input_ids)
        with lock:
            dist.barrier()  # sync with other workers
            return jsonify({"log_probs": evaluate(self.model, input_ids)})


def model_provider(pre_process=True, post_process=True):
    """Build the model."""

    print_rank_0('building GPT model ...')
    model = GPTModel(num_tokentypes=0, parallel_output=False, pre_process=pre_process, post_process=post_process)

    return model


class MegatronServer(object):
    def __init__(self, model):
        self.app = Flask(__name__, static_url_path='')
        api = Api(self.app)
        api.add_resource(MegatronEval, '/api', resource_class_args=[model])

    def run(self, url):
        self.app.run(url, threaded=True, debug=False)


if __name__ == '__main__':
    initialize_megatron(args_defaults={'tokenizer_type': 'GPT2BPETokenizer',
                                       'no_load_rng': True,
                                       'no_load_optim': True})
    args = get_args()
    if args.num_layers_per_virtual_pipeline_stage is not None:
        print("Interleaved pipeline schedule is not yet supported for text generation.")
        exit()
    # Set up model and load checkpoint
    model = get_model(model_provider, wrap_with_ddp=False)

    if args.load is not None:
        _ = load_checkpoint(model, None, None)

    assert len(model) == 1, "Above condition should have caught this"
    model = model[0]
    if mpu.is_pipeline_first_stage() and mpu.get_tensor_model_parallel_rank() == 0:
        server = MegatronServer(model)
        server.run("0.0.0.0")
    else:
        while True:
            dist.barrier()
            evaluate(model)
