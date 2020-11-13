import pytest
import torch
import torch.nn as nn
from causal_transformer_decoder import (
    CausalTransformerDecoder,
    CausalTransformerDecoderLayer,
)

hdim = 512
nhead = 8
dim_feedforward = hdim * 4
num_layers = 6
device = "cuda" if torch.cuda.is_available() else "cpu"


def generate_square_subsequent_mask(
    sz: int, device: str = "cpu"
) -> torch.Tensor:
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = (
        mask.float()
        .masked_fill(mask == 0, float("-inf"))
        .masked_fill(mask == 1, float(0.0))
    ).to(device=device)
    return mask


@pytest.fixture
def torch_encoder():
    encoder = nn.TransformerEncoder(
        nn.TransformerEncoderLayer(
            d_model=hdim, nhead=nhead, dim_feedforward=dim_feedforward
        ),
        num_layers=num_layers,
    ).to(device=device)
    encoder.eval()
    return encoder


@pytest.fixture
def torch_decoder():
    torch.manual_seed(42)
    decoder = nn.TransformerDecoder(
        nn.TransformerDecoderLayer(
            d_model=hdim, nhead=nhead, dim_feedforward=dim_feedforward
        ),
        num_layers=num_layers,
    ).to(device=device)
    decoder.eval()
    return decoder


@pytest.fixture
def causal_decoder():
    causal_decoder = CausalTransformerDecoder(
        CausalTransformerDecoderLayer(
            d_model=hdim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
        ),
        num_layers=num_layers,
    ).to(device=device)
    causal_decoder.eval()
    return causal_decoder


@pytest.mark.parametrize("bsz", [1, 8])
def test_consistency_with_torch_implementation(
    bsz, torch_encoder, torch_decoder, causal_decoder
):
    input_len = 10
    output_len = 20

    first_output_emb = torch.rand(1, bsz, hdim).to(device=device)
    src = torch.rand(input_len, bsz, hdim).to(device=device)

    src_embeddings = torch_encoder(src)
    decoded_embeddings = first_output_emb
    for i in range(output_len - 1):
        mask_dec = generate_square_subsequent_mask(
            i + 1, device=src_embeddings.device
        )  # create mask for autoregressive decoding
        output = torch_decoder(
            decoded_embeddings, src_embeddings, tgt_mask=mask_dec
        )
        decoded_embeddings = torch.cat(
            [decoded_embeddings, output[-1:, :, :]], dim=0
        )

    decoded_embeddings_2 = first_output_emb
    cache = None
    for i in range(output_len - 1):
        output, cache = causal_decoder(
            decoded_embeddings_2, src_embeddings, cache
        )
        decoded_embeddings_2 = torch.cat(
            [decoded_embeddings_2, output[-1:, :, :]], dim=0
        )

    assert decoded_embeddings.size() == (output_len, bsz, hdim)
    torch.eq(decoded_embeddings, decoded_embeddings_2)
