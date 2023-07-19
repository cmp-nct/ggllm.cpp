# Based on: https://github.com/KerfuffleV2/ggml-falcon/blob/feat-improve-falcon-convert-hf/examples/falcon/convert-hf-to-ggml.py
# Convert Hugging Face fine-tuned bloom-like models to ggml format
#
# Usage:
#
#   python3 models/convert-h5-to-ggml.py
#
# This script is similar to "convert-pt-to-ggml.py"
#

import io
import os
import sys
import struct
import json
import code
import torch
import numpy as np

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

# ref: https://github.com/openai/gpt-2/blob/master/src/encoder.py
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a significant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.
    """
    bs = list(range(ord("!"), ord("~")+1))+list(range(ord("¡"), ord("¬")+1))+list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8+n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))

if len(sys.argv) < 3:
    print("INFO: GGML V1 files produced are meant to be finalized through examples/falcon_quantize which will bring them to latest version and precision of choice");
    print("Usage: python falcon_convert.py model_directory output_directory [use-f32]")
    # print("  num_parts: number of pytorch parts, use 0 if not a multipart model. example: 9")
    print("  model_directory: name of the directory and model you convert (it should be a subdirectory)")
    print("  output-directory: directory where the output file will be written")
    print("  use-f32:    if present, use float32 instead of float16 (f32 is recommended)")
    sys.exit(1)

# num_parts = int(sys.argv[1])
dir_model = sys.argv[1] # name and dir of model
dir_out = sys.argv[2]   # output directory

# make sure the output directory exists
os.makedirs(dir_out, exist_ok=True)

num_parts = 0 
for file_name in os.listdir(dir_model): 
    if "pytorch_model" in file_name and '-' in file_name and '.' in file_name: 
        num_parts = max(num_parts, int(file_name.split('-')[-1].split('.')[0]))
num_parts = int(num_parts)

# possible data types
#   ftype == 0 -> float32
#   ftype == 1 -> float16
#
# map from ftype to string
ftype_str = ["f32", "f16"]
ftype = 1
if len(sys.argv) > 3:
    ftype = 0

tokenizer = AutoTokenizer.from_pretrained(dir_model)
# print(tokenizer)
config = AutoConfig.from_pretrained(dir_model, trust_remote_code=True)
hparams = config.to_dict()

n_head = hparams["n_head"]
n_head_kv = hparams["n_head_kv"] if "n_head_kv" in hparams else 1
head_dim = hparams["hidden_size"] // n_head
print("* Loading model from: ", dir_model)

fname_out = dir_out + f"/ggml-model-{dir_model.split('/')[-1]}-{ftype_str[ftype]}.bin"
fout = open(fname_out, "wb")
fout.write(struct.pack("i", 0x67676d66)) # magic: ggmf in hex (version 1) - possibly change to ggfc ?
fout.write(struct.pack("i", 1))
fout.write(struct.pack("i", hparams["vocab_size"]))
fout.write(struct.pack("i", hparams["hidden_size"]))
fout.write(struct.pack("i", n_head))
fout.write(struct.pack("i", n_head_kv))
fout.write(struct.pack("i", hparams["n_layer"]))
fout.write(struct.pack("i", 40 if "n_head_kv" in hparams else 7)) # obsolete field that breaks ggml compatibility - todo again remove one day
fout.write(struct.pack("i", ftype))

print(f'Vocab size: {hparams["vocab_size"]}')
print(f'Hidden size: {hparams["hidden_size"]}')
print(f'Number of heads: {n_head}')
print(f'Number of layers: {hparams["n_layer"]}')
print(f'Number of head_kv: {n_head_kv}')
print(f'Number of head_dim: {head_dim}')

reverse_vocab = {id: encoded_tok for encoded_tok, id in tokenizer.vocab.items()}
byte_encoder = bytes_to_unicode()
byte_decoder = {v:k for k, v in byte_encoder.items()}

for i in range(hparams["vocab_size"]):
    if i in reverse_vocab:
        try:
            text = bytearray([byte_decoder[c] for c in reverse_vocab[i]])
        except KeyError:
            text = bytearray()
            for c in reverse_vocab[i]:
                if ord(c) < 256:  # single byte character
                    text.append(byte_decoder[ord(c)])
                else:  # multibyte special token character
                    text.extend(c.encode('utf-8'))
    else:
        print(f"Key {i} not in tokenizer vocabulary. Padding with an arbitrary token.")
        padding_token = f"[PAD{i}]".encode("utf8")
        text = bytearray(padding_token)
    fout.write(struct.pack("i", len(text)))
    fout.write(text)
    fout.write(struct.pack("f", 0.0)) # falcon uses bpe on RefinedWeb - no probability scores used

if num_parts == 0:
    partnames= ('pytorch_model.bin',)
else:
    partnames = (f'pytorch_model-{n:05}-of-{num_parts:05}.bin' for n in range(1, num_parts + 1))
for partname in partnames:
    filename = f'{dir_model}/{partname}'
    print(f'\n* Loading part: {partname}')
    model = torch.load(filename, map_location = 'cpu')
    for name in model.keys():
        src = name
        # The original query_key_value tensor contains n_head_kv "kv groups",
        # each consisting of n_head/n_head_kv query weights followed by one key
        # and one value weight (shared by all query heads in the kv group).
        # This layout makes it a big pain to work with in GGML.
        # So we rearrange them here,, so that we have n_head query weights
        # followed by n_head_kv key weights followed by n_head_kv value weights,
        # in contiguous fashion.

        if "query_key_value" in src:
            qkv = model[src].view(
                n_head_kv, n_head // n_head_kv + 2, head_dim, head_dim * n_head)

            q = qkv[:, :-2 ].reshape(n_head * head_dim, head_dim * n_head)
            k = qkv[:, [-2]].reshape(n_head_kv * head_dim, head_dim * n_head)
            v = qkv[:, [-1]].reshape(n_head_kv * head_dim, head_dim * n_head)

            model[src] = torch.cat((q,k,v)).reshape_as(model[src])
        data = model[src].squeeze()
        n_dims = len(data.shape)
        # default type is fp32
        ftype_cur = 1 if ftype == 1 and n_dims > 1 else 0
        data = data.to(dtype = torch.float16 if ftype_cur == 1 else torch.float32).numpy()
        print(f'  |', name, data.shape, '->', data.dtype)
        # header
        str = name.encode('utf-8')
        fout.write(struct.pack("iii", n_dims, len(str), ftype_cur))
        for i in range(n_dims):
            fout.write(struct.pack("i", data.shape[n_dims - 1 - i]))
        fout.write(str)

        # data
        data.tofile(fout)

fout.close()

print("Done. Output file: " + fname_out)
print("")