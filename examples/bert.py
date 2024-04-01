from typing import Tuple

import numpy as np

from easygrad.nn import Dropout, Embedding, LayerNorm, Linear
from easygrad.tensor import Tensor


class Bert:
    def __init__(
        self,
        vocab_size: int = 30522,
        max_position_embeddings: int = 512,
        type_vocab_size: int = 2,
        num_hidden_layers: int = 12,
        hidden_size: int = 768,
        intermediate_size: int = 3072,
        num_attention_heads: int = 12,
        attention_probs_dropout_prob: float = 0.0,
        layer_norm_eps: float = 1e-12,
        hidden_dropout_prob: float = 0.1,
    ):
        self.embeddings = BertEmbedding(
            vocab_size=vocab_size,
            max_sequence_len=max_position_embeddings,
            type_vocab_size=type_vocab_size,
            embed_dim=hidden_size,
            hidden_dropout_prob=hidden_dropout_prob,
        )
        self.encoder = BertEncoder(
            num_hidden_layers,
            hidden_size,
            intermediate_size,
            num_attention_heads,
            attention_probs_dropout_prob,
            layer_norm_eps,
            hidden_dropout_prob,
        )
        self.pooler = BertPooler(hidden_size)

    def __call__(self, input_ids: np.ndarray, token_type_ids: np.ndarray, attention_mask: np.ndarray) -> Tensor:
        embedding_output = self.embeddings(input_ids, token_type_ids)
        encoded_output = self.encoder(embedding_output, attention_mask)
        pooled_output = self.pooler(encoded_output)
        return encoded_output, pooled_output


class BertEmbedding:
    def __init__(
        self,
        vocab_size: int = 30522,
        max_sequence_len: int = 512,
        type_vocab_size: int = 2,
        embed_dim: int = 768,
        hidden_dropout_prob: float = 0.1,
    ):
        self.word_embeddings = Embedding(vocab_size, embed_dim)
        self.position_embeddings = Embedding(max_sequence_len, embed_dim)
        self.token_type_embeddings = Embedding(type_vocab_size, embed_dim)

        self.LayerNorm = LayerNorm(normalized_shape=embed_dim)
        self.dropout = Dropout(p=hidden_dropout_prob)

    def __call__(
        self,
        input_ids: np.ndarray,  # [BS, max_text_len]
        token_type_ids: np.ndarray,  # [BS, max_text_len]
    ) -> Tensor:
        batch_size, max_seq_len = input_ids.shape

        word_embed = self.word_embeddings(input_ids)
        position_embed = self.position_embeddings(np.broadcast_to(np.arange(max_seq_len), (batch_size, max_seq_len)))
        token_type_embed = self.token_type_embeddings(token_type_ids)

        final_embeddings = word_embed.add(position_embed).add(token_type_embed)
        out = self.LayerNorm(final_embeddings)
        out = self.dropout(out)
        return out


class BertEncoder:
    def __init__(
        self,
        num_hidden_layers: int = 12,
        hidden_size: int = 768,
        intermediate_size: int = 3072,
        num_attention_heads: int = 12,
        attention_probs_dropout_prob: float = 0.0,
        layer_norm_eps: float = 1e-12,
        hidden_dropout_prob: float = 0.1,
    ):
        self.layer = [
            BertLayer(hidden_size, intermediate_size, num_attention_heads, attention_probs_dropout_prob, layer_norm_eps, hidden_dropout_prob)
            for _ in range(num_hidden_layers)
        ]

    def __call__(self, hidden_states: Tensor, attention_mask: np.ndarray) -> Tensor:
        for layer in self.layer:
            hidden_states = layer(hidden_states, attention_mask)
        return hidden_states


class BertLayer:
    def __init__(
        self,
        hidden_size: int = 768,
        intermediate_size:int = 3072,
        num_attention_heads: int = 12,
        attention_probs_dropout_prob: float = 0.0,
        layer_norm_eps: float = 1e-12,
        hidden_dropout_prob: float = 0.1,
    ):
        self.attention = BertAttention(hidden_size, num_attention_heads, attention_probs_dropout_prob, layer_norm_eps, hidden_dropout_prob)
        self.intermediate = BertIntermediate(hidden_size, intermediate_size)
        self.output = BertOutput(hidden_size, intermediate_size, layer_norm_eps, hidden_dropout_prob)

    def __call__(self, hidden_states: Tensor, attention_mask: np.ndarray) -> Tensor:
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class BertIntermediate:
    def __init__(self, hidden_size: int = 768, intermediate_size: int = 3072):
        self.dense = Linear(hidden_size, intermediate_size)

    def __call__(self, hidden_states: Tensor) -> Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = hidden_states.gelu_original()
        return hidden_states


class BertOutput:
    def __init__(
        self,
        hidden_size: int = 768,
        intermediate_size:int = 3072,
        layer_norm_eps: float = 1e-12,
        hidden_dropout_prob: float = 0.1,
    ):
        self.dense = Linear(intermediate_size, hidden_size)
        self.LayerNorm = LayerNorm(hidden_size, layer_norm_eps)
        self.dropout = Dropout(hidden_dropout_prob)

    def __call__(self, hidden_states: Tensor, input_tensor: Tensor) -> Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention:
    def __init__(
        self,
        hidden_size: int = 768,
        num_attention_heads: int = 12,
        attention_probs_dropout_prob: float = 0.0,
        layer_norm_eps: float = 1e-12,
        hidden_dropout_prob: float = 0.1,
    ):
        self.self = BertSelfAttention(hidden_size, num_attention_heads, attention_probs_dropout_prob)
        self.output = BertSelfOutput(hidden_size, layer_norm_eps, hidden_dropout_prob)

    def __call__(self, hidden_states: Tensor, attention_mask: np.ndarray) -> Tensor:
        self_output = self.self(hidden_states, attention_mask)
        attention_output = self.output(self_output, hidden_states)
        return attention_output


class BertSelfAttention:
    def __init__(
        self, hidden_size: int = 768, num_attention_heads: int = 12, attention_probs_dropout_prob: float = 0.1
    ):
        if hidden_size % num_attention_heads != 0:
            raise ValueError(f"The hidden dimention ({hidden_size}) is not a multiple of number of attention heads ({num_attention_heads})")

        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_heads_size = self.num_attention_heads * self.attention_head_size  # = hidden_size

        self.query = Linear(hidden_size, hidden_size)
        self.key = Linear(hidden_size, hidden_size)
        self.value = Linear(hidden_size, hidden_size)

        self.dropout = Dropout(attention_probs_dropout_prob)

    def transpose_for_scores(self, x: Tensor) -> Tensor:
        new_shape = x.shape[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.reshape(shape=new_shape)
        return x.permute(dims=(0, 2, 1, 3))

    def get_extended_attention_mask(self, attention_mask: np.ndarray, shape: Tuple[int], dtype=np.float32) -> Tensor:
        if len(attention_mask.shape) != 2:
            raise ValueError(f"Attention mask must be 2-dimensional, got {len(attention_mask.shape)} dimensions.")
        extended_attention_mask = attention_mask[:, None, None, :]
        extended_attention_mask = (1.0 - extended_attention_mask) * np.finfo(np.float32).min
        extended_attention_mask = Tensor(extended_attention_mask.astype(dtype)).expand(shape=shape)
        return extended_attention_mask

    def __call__(self, hidden_states: Tensor, attention_mask: np.ndarray = None) -> Tensor:
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

        attention_scores = query_layer.dot(key_layer.permute(dims=(0, 1, 3, 2)))
        div_ = Tensor(np.ones_like(attention_scores.data) * np.sqrt(self.attention_head_size))
        attention_scores = attention_scores.div(div_)

        if attention_mask is not None:
            extended_attention_mask = self.get_extended_attention_mask(attention_mask, attention_scores.shape)
            attention_scores = attention_scores.add(extended_attention_mask)

        attention_probs = attention_scores.softmax(dim=3)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = attention_probs.dot(value_layer)

        context_layer = context_layer.permute(dims=(0, 2, 1, 3))
        new_context_layer_shape = context_layer.shape[:2] + (self.all_heads_size,)
        context_layer = context_layer.reshape(shape=new_context_layer_shape)

        return context_layer


class BertSelfOutput:
    def __init__(
        self,
        hidden_size: int = 768,
        layer_norm_eps: float = 1e-12,
        hidden_dropout_prob: float = 0.1,
    ):
        self.dense = Linear(hidden_size, hidden_size)
        self.LayerNorm = LayerNorm(hidden_size, layer_norm_eps)
        self.dropout = Dropout(hidden_dropout_prob)

    def __call__(self, hidden_states: Tensor, input_tensor: Tensor) -> Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertPooler:
    def __init__(self, hidden_size: int = 768):
        self.dense = Linear(hidden_size, hidden_size)

    def __call__(self, hidden_states: Tensor) -> Tensor:
        # Very weirdly take first token - we have no tensor slicing as of now
        # We want to achieve first_token_tensor = hidden_states[:, 0, :]
        mask = np.zeros(hidden_states.shape)
        mask[:, 0, :] = 1
        mask = Tensor(mask)
        first_token_tensor = (hidden_states * mask).sum(axis=1)
        pooled_output = self.dense(first_token_tensor)
        pooled_output = pooled_output.tanh()
        return pooled_output


if __name__ == "__main__":
    from transformers import BertModel, BertTokenizer

    from examples.hf_weight_transfer import compare_easy_to_hf_outputs, transfer_huggingface_weights

    # Usage
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    bert_easy = Bert(
        vocab_size=30522,
        max_position_embeddings=512,
        type_vocab_size=2,
        num_hidden_layers=12,
        hidden_size=768,
        intermediate_size=3072,
        num_attention_heads=12,
        attention_probs_dropout_prob=0.0,
        layer_norm_eps=1e-12,
        hidden_dropout_prob=0.1,
    )

    sentences = [
        "This is an example sentence.",
        "This is another example sentence longer.",
    ]
    encoded_input = tokenizer(sentences, return_tensors="np", padding=True)
    last_hidden_state, pooler_output = bert_easy(**encoded_input)
    
    # Compare to huggingface output
    bert_hf = BertModel.from_pretrained("bert-base-uncased")
    encoded_input_hf = tokenizer(sentences, return_tensors="pt", padding=True)
    out_hf = bert_hf(**encoded_input_hf)

    to_transpose = ["query.weight", "key.weight", "value.weight", "dense.weight"]
    transfer_huggingface_weights(bert_easy, bert_hf, to_transpose)
    last_hidden_state, pooler_output = bert_easy(**encoded_input)

    last_hidden_state_results = compare_easy_to_hf_outputs(last_hidden_state.data, out_hf["last_hidden_state"].detach().numpy())
    pooling_layer_results = compare_easy_to_hf_outputs(pooler_output.data, out_hf["pooler_output"].detach().numpy())
    print("Last hidden states errors, easy output compared to hf output:")
    [print(f"{k}: {v:.4f}") for k, v in last_hidden_state_results.items()]
    print("\nPooling layer errors, easy output compared to hf output:")
    [print(f"{k}: {v:.4f}") for k, v in pooling_layer_results.items()]
