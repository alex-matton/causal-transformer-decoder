import torch
import torch.nn as nn
import time
from causal_transformer_decoder import (
    CausalTransformerDecoder,
    CausalTransformerDecoderLayer,
)
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--type_generation", default="short", type=str,
)
parser.add_argument("--vocab_size", default=30000, type=int)
parser.add_argument("--bsz", default=8, type=int)
parser.add_argument("--input_len_long_gen", default=500, type=int)
args = parser.parse_args()

hdim = 512
nhead = 8
dim_feedforward = hdim * 4
num_layers = 6
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device used: {device}")

bsz = args.bsz
vocab_size = args.vocab_size

if args.type_generation == "short":
    input_lens = [10, 25, 50, 100, 200, 300, 400, 500]
    output_lens = input_lens
    n_experiments = [10, 10, 5, 1, 1, 1, 1, 1]
elif args.type_generation == "long":
    output_lens = [500, 1000, 1500, 2000]
    input_lens = [args.input_len_long_gen] * len(output_lens)
    n_experiments = [1] * len(output_lens)


def generate_square_subsequent_mask(sz: int, device: str = "cpu") -> torch.Tensor:
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = (
        mask.float()
        .masked_fill(mask == 0, float("-inf"))
        .masked_fill(mask == 1, float(0.0))
    ).to(device=device)
    return mask


# Naive way to use transformers
transformer = nn.Transformer(
    d_model=hdim,
    nhead=nhead,
    num_encoder_layers=num_layers,
    num_decoder_layers=num_layers,
    dim_feedforward=dim_feedforward,
).to(device=device)
transformer.eval()

# Decoupling encoder and decoder
encoder = nn.TransformerEncoder(
    nn.TransformerEncoderLayer(
        d_model=hdim, nhead=nhead, dim_feedforward=dim_feedforward
    ),
    num_layers=num_layers,
).to(device=device)
encoder.eval()

torch.manual_seed(42)
decoder = nn.TransformerDecoder(
    nn.TransformerDecoderLayer(
        d_model=hdim, nhead=nhead, dim_feedforward=dim_feedforward
    ),
    num_layers=num_layers,
).to(device=device)
decoder.eval()

# Causal Decoder
torch.manual_seed(42)
causal_decoder = CausalTransformerDecoder(
    CausalTransformerDecoderLayer(
        d_model=hdim, nhead=nhead, dim_feedforward=dim_feedforward,
    ),
    num_layers=num_layers,
).to(device=device)
causal_decoder.eval()

to_vocab = nn.Linear(hdim, vocab_size).to(device=device)
to_vocab.eval()
embedding = nn.Embedding(vocab_size, hdim).to(device=device)
embedding.eval()

time_exp_regular_transf = [[] for _ in range(len(input_lens))]
time_exp_enc_dec = [[] for _ in range(len(input_lens))]
time_exp_causal_end_dec = [[] for _ in range(len(input_lens))]

with torch.no_grad():

    for len_index, (input_len, output_len) in tqdm(
        enumerate(zip(input_lens, output_lens))
    ):
        for _ in range(n_experiments[len_index]):
            src = torch.rand(input_len, bsz, hdim).to(device=device)
            first_token = torch.zeros((1, bsz)).long().to(device=device)

            # Inference loops for the three models
            t = time.time()
            decoded_tokens = first_token
            for i in range(output_len):
                mask_dec = generate_square_subsequent_mask(
                    i + 1, device=first_token.device
                )  # create mask for autoregressive decoding
                decoded_embeddings = embedding(decoded_tokens)
                output = transformer(src, decoded_embeddings, tgt_mask=mask_dec)
                logits = to_vocab(output)  # projection to vocab size

                # keep most likely tokens
                top_indices = torch.argmax(logits, dim=-1)
                # we only care about the last token that was decoded
                top_indices_last_token = top_indices[-1:]

                # add most likely token to the already decoded tokens
                decoded_tokens = torch.cat(
                    [decoded_tokens, top_indices_last_token], dim=0
                )
            time_exp_regular_transf[len_index].append(time.time() - t)

            t = time.time()
            decoded_tokens = first_token
            src_embeddings = encoder(src)
            for i in range(output_len):
                mask_dec = generate_square_subsequent_mask(
                    i + 1, device=first_token.device
                )  # create mask for autoregressive decoding
                decoded_embeddings = embedding(decoded_tokens)
                output = decoder(decoded_embeddings, src_embeddings, tgt_mask=mask_dec)
                logits = to_vocab(output)  # projection to vocab size

                # keep most likely tokens
                top_indices = torch.argmax(logits, dim=-1)
                # we only care about the last token that was decoded
                top_indices_last_token = top_indices[-1:]

                # add most likely token to the already decoded tokens
                decoded_tokens = torch.cat(
                    [decoded_tokens, top_indices_last_token], dim=0
                )
            time_exp_enc_dec[len_index].append(time.time() - t)
            logits_enc_dec = logits

            t = time.time()
            decoded_tokens = first_token
            src_embeddings = encoder(src)
            cache = None
            for i in range(output_len):
                decoded_embeddings = embedding(decoded_tokens)
                output, cache = causal_decoder(
                    decoded_embeddings, src_embeddings, cache
                )
                logits = to_vocab(output)  # projection to vocab size

                # keep most likely tokens
                top_indices = torch.argmax(logits, dim=-1)
                # we only care about the last token that was decoded
                top_indices_last_token = top_indices[-1:]

                # add most likely token to the already decoded tokens
                decoded_tokens = torch.cat(
                    [decoded_tokens, top_indices_last_token], dim=0
                )
            time_exp_causal_end_dec[len_index].append(time.time() - t)
            logits_causal = logits

time_exp_regular_transf = [sum(sub) / len(sub) for sub in time_exp_regular_transf]
time_exp_enc_dec = [sum(sub) / len(sub) for sub in time_exp_enc_dec]
time_exp_causal_end_dec = [sum(sub) / len(sub) for sub in time_exp_causal_end_dec]

print(
    "Bsz, hdim, Vocab size, Len input, Len output,"
    " Regular Transf, Enc/Dec, Causal Enc/Dec"
)
for (input_len, output_len, time_transf, time_enc_dec, time_causal,) in zip(
    input_lens,
    output_lens,
    time_exp_regular_transf,
    time_exp_enc_dec,
    time_exp_causal_end_dec,
):
    print(
        f"{bsz}, {hdim}, {vocab_size}, {input_len}, {output_len}, "
        f"{time_transf:.4}, {time_enc_dec:.4}, {time_causal:.4}"
    )
