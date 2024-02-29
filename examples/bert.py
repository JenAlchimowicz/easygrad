import numpy as np
from transformers import BertTokenizer

from easygrad.tensor import Tensor
from easygrad.nn import Embedding, LayerNorm, Dropout


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


if __name__ == "__main__":
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    easy_bert_embedding = BertEmbedding(vocab_size = 30522, max_sequence_len = 512, embed_dim = 768)

    sentences = [
        "This is an example sentence.",
        "This is another example sentence longer.",
    ]
    encoded_input = tokenizer(sentences, return_tensors="np", padding=True)
    del encoded_input["attention_mask"]  # For now

    out = easy_bert_embedding(**encoded_input)
    print(out.shape)
