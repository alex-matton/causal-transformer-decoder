import torch
import torch.nn as nn
from typing import Optional
from torch import Tensor


class CausalTransformerDecoder(nn.TransformerDecoder):
    """Implementation of a transformer decoder based on torch implementation but
    more efficient. The difference is that it doesn't need to recompute the
    embedding of all the past decoded tokens but instead uses a cache to
    store them. This makes use of the fact that the attention of a decoder is
    causal, so new predicted tokens don't affect the old tokens' embedding bc
    the corresponding attention cells are masked.
    The complexity goes from seq_len^3 to seq_len^2.
    """

    def forward(
        self,
        tgt: Tensor,
        memory: Optional[Tensor] = None,
        cache: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            tgt (Tensor): current_len_output x bsz x hidden_dim
            memory (Tensor): len_encoded_seq x bsz x hidden_dim
            cache (Optional[Tensor], optional):
                n_layers x (current_len_output - 1) x bsz x hidden_dim
                If current_len_output == 1, nothing is cached yet, so cache
                should be None
        Returns:
            output (Tensor): current_len_output x bsz x hidden_dim
            cache (Tensor): n_layers x current_len_output x bsz x hidden_dim
        """

        output = tgt

        new_token_cache = []
        for i, mod in enumerate(self.layers):
            output = mod(output, memory)
            new_token_cache.append(output)
            if cache is not None:
                output = torch.cat([cache[i], output], dim=0)

        if cache is not None:
            new_cache = torch.cat(
                [cache, torch.stack(new_token_cache, dim=0)], dim=1
            )
        else:
            new_cache = torch.stack(new_token_cache, dim=0)

        return output, new_cache


class CausalTransformerDecoderLayer(nn.TransformerDecoderLayer):
    def forward(
        self,
        tgt: Tensor,
        memory: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            tgt (Tensor): see CausalTransformerDecoder
            memory (Tensor, optional): see CausalTransformerDecoder

        Returns:
            Tensor: embedding of last token of dim: 1 x bsz x hidden_dim
        """

        # This part is adapted from the official Pytorch implementation
        # So that only the last token gets modified and returned.

        tgt_last_tok = tgt[-1:, :, :]

        # self attention part
        tmp_tgt = self.self_attn(tgt_last_tok, tgt, tgt, attn_mask=None)[0]
        tgt_last_tok = tgt_last_tok + self.dropout1(tmp_tgt)
        tgt_last_tok = self.norm1(tgt_last_tok)

        # encoder-decoder attention
        if memory is not None:
            tmp_tgt = self.multihead_attn(
                tgt_last_tok,
                memory,
                memory,
            )[0]
            tgt_last_tok = tgt_last_tok + self.dropout2(tmp_tgt)
            tgt_last_tok = self.norm2(tgt_last_tok)

        # final feed-forward network
        tmp_tgt = self.linear2(
            self.dropout(self.activation(self.linear1(tgt_last_tok)))
        )
        tgt_last_tok = tgt_last_tok + self.dropout3(tmp_tgt)
        tgt_last_tok = self.norm3(tgt_last_tok)
        return tgt_last_tok


def generate_square_subsequent_mask(
    sz: int, device: str = "cpu"
) -> torch.Tensor:
    """ Generate the attention mask for causal decoding """
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = (
        mask.float()
        .masked_fill(mask == 0, float("-inf"))
        .masked_fill(mask == 1, float(0.0))
    ).to(device=device)
    return mask
