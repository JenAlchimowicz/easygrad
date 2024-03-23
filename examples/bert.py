import numpy as np
from transformers import BertTokenizer

from easygrad.tensor import Tensor
from easygrad.nn import Embedding, Dropout, LayerNorm, Linear

from typing import Tuple


class BertEmbedding:
    def __init__(self, vocab_size: int = 30522, max_sequence_len: int = 512, embed_dim: int = 768):
        self.word_embeddings = Embedding(vocab_size, embed_dim)
        self.position_embeddings = Embedding(max_sequence_len, embed_dim)
        self.token_type_embeddings = Embedding(2, embed_dim)  # 2 fixed for BERT

        self.layer_norm = LayerNorm(normalized_shape=embed_dim)
        self.dropout = Dropout(p=0.1)

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
        out = self.layer_norm(final_embeddings)
        out = self.dropout(out)
        return out


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
        extended_attention_mask = -(1.0 - extended_attention_mask) * np.finfo(np.float32).min
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


if __name__ == "__main__":
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    easy_bert_embedding = BertEmbedding(vocab_size=30522, max_sequence_len=512, embed_dim=768)
    easy_bert_self_attention = BertSelfAttention(
        hidden_size=768,
        num_attention_heads=12,
        attention_probs_dropout_prob=0.1
    )

    sentences = [
        "This is an example sentence.",
        "This is another example sentence longer.",
    ]
    encoded_input = tokenizer(sentences, return_tensors="np", padding=True)
    # For now
    attention_mask = encoded_input["attention_mask"]
    del encoded_input["attention_mask"]

    out_embed = easy_bert_embedding(**encoded_input)
    out_self_attention = easy_bert_self_attention(out_embed, attention_mask)
    print(out_self_attention.shape)
