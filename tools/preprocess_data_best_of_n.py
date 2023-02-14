import argparse
import json
import multiprocessing
import sys
from datetime import time

import torch

from megatron.data import indexed_dataset
from megatron.tokenizer import build_tokenizer


class Encoder:
    def __init__(self, args):
        self.args = args
        self._text_extractors = []
        self._text_extractors.append(("prompt", lambda x: x["prompt"]))
        for i in range(self.args.best_of_n):
            self._text_extractors.append(("sample_{}".format(i), lambda x: x["samples"][i]))

    def initializer(self):
        Encoder.tokenizer = build_tokenizer(self.args)

    def encode(self, json_line):
        data = json.loads(json_line)
        ids = {"label": data["label"]}
        for key, extractor in self._text_extractors:
            text = extractor(data)
            doc_ids = Encoder.tokenizer.tokenize(text)
            ids[key] = doc_ids
        return ids, len(json_line)


def get_args():
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group(title='input data')
    group.add_argument('--input', type=str, required=True,
                       help='Path to input JSON')
    group.add_argument('--best-of-n', type=int, required=True,
                       help='Number of samples per prompt')

    group = parser.add_argument_group(title='tokenizer')
    group.add_argument('--tokenizer-type', type=str, required=True,
                       choices=['BertWordPieceLowerCase', 'BertWordPieceCase',
                                'GPT2BPETokenizer', 'SentencePieceTokenizer',
                                'HFTokenizer'],
                       help='What type of tokenizer to use.')
    group.add_argument('--vocab-file', type=str, default=None,
                       help='Path to the vocab file')
    group.add_argument('--merge-file', type=str, default=None,
                       help='Path to the BPE merge file (if necessary).')

    group = parser.add_argument_group(title='output data')
    group.add_argument('--output-prefix', type=str, required=True,
                       help='Path to binary output file without suffix')
    group.add_argument('--dataset-impl', type=str, default='mmap',
                       choices=['lazy', 'cached', 'mmap'])

    group = parser.add_argument_group(title='runtime')
    group.add_argument('--workers', type=int, required=True,
                       help='Number of worker processes to launch')
    group.add_argument('--chunk-size', type=int, required=True,
                       help='Chunk size assigned to each worker process')
    group.add_argument('--log-interval', type=int, default=100,
                       help='Interval between progress updates')
    args = parser.parse_args()
    args.keep_empty = False

    if args.tokenizer_type.lower().startswith('bert'):
        if not args.split_sentences:
            print("Bert tokenizer detected, are you sure you don't want to split sentences?")

    # some default/dummy values for the tokenizer
    args.rank = 0
    args.make_vocab_size_divisible_by = 128
    args.tensor_model_parallel_size = 1
    args.vocab_extra_ids = 0

    return args


def main():
    args = get_args()
    startup_start = time.time()
    print("Opening", args.input)
    fin = open(args.input, 'r', encoding='utf-8')
    encoder = Encoder(args)
    tokenizer = build_tokenizer(args)
    pool = multiprocessing.Pool(args.workers, initializer=encoder.initializer)
    encoded_docs = pool.imap(encoder.encode, fin, args.chunk_size)

    output_bin_files = {}
    output_idx_files = {}
    builders = {}
    for key in ["label", "prompt"] + [f"sample_{i}" for i in range(args.best_of_n)]:
        output_bin_files[key] = "{}_{}.bin".format(args.output_prefix, key)
        output_idx_files[key] = "{}_{}.idx".format(args.output_prefix, key)
        builders[key] = indexed_dataset.make_builder(
            output_bin_files[key],
            impl=args.dataset_impl,
            vocab_size=tokenizer.vocab_size if key != "label" else args.best_of_n)

    startup_end = time.time()
    proc_start = time.time()
    total_bytes_processed = 0
    print("Time to startup:", startup_end - startup_start)

    for i, (doc, bytes_processed) in enumerate(encoded_docs, start=1):
        total_bytes_processed += bytes_processed
        for key, sentences in doc.items():
            if len(sentences) == 0:
                continue
            for sentence in sentences:
                builders[key].add_item(torch.IntTensor(sentence))
            builders[key].end_document()
        if i % args.log_interval == 0:
            current = time.time()
            elapsed = current - proc_start
            mbs = total_bytes_processed / elapsed / 1024 / 1024
            print(f"Processed {i} documents",
                  f"({i / elapsed} docs/s, {mbs} MB/s).",
                  file=sys.stderr)
    print("Done! Now finalizing.")

    for key in args.json_keys:
        builders[key].finalize(output_idx_files[key])


if __name__ == '__main__':
    main()
