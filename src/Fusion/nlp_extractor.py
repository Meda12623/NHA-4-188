import os
import torch
import torch.nn as nn

import config


class BioBERTTokenExtractor(nn.Module):

    def __init__(
        self,
        model_name=config.NLP_MODEL_NAME,
        max_length=config.NLP_MAX_LENGTH,
        freeze=config.NLP_FREEZE_BACKBONE,
        unfreeze_last_n_layers=config.NLP_UNFREEZE_LAST_N_LAYERS,
    ):
        super().__init__()
        from transformers import AutoModel
        from tokenizers import BertWordPieceTokenizer
        from huggingface_hub import hf_hub_download

        vocab_path = hf_hub_download(repo_id=model_name, filename="vocab.txt")
        self.tokenizer = BertWordPieceTokenizer(vocab_path, lowercase=False)
        self.tokenizer.enable_truncation(max_length=max_length)

        from transformers import BertModel
        self.bert = BertModel.from_pretrained(model_name)
        self.hidden_size = self.bert.config.hidden_size  # 768

        if freeze:
            for p in self.bert.parameters():
                p.requires_grad = False
            if unfreeze_last_n_layers > 0:
                for layer in self.bert.encoder.layer[-unfreeze_last_n_layers:]:
                    for p in layer.parameters():
                        p.requires_grad = True

    def _encode_batch(self, texts):
        self.tokenizer.enable_padding(length=None)
        encodings = self.tokenizer.encode_batch(texts)
        max_len = max(len(e.ids) for e in encodings)
        input_ids = torch.tensor(
            [e.ids + [0] * (max_len - len(e.ids)) for e in encodings]
        )
        attention_mask = torch.tensor(
            [e.attention_mask + [0] * (max_len - len(e.attention_mask)) for e in encodings]
        )
        return input_ids, attention_mask

    @staticmethod
    def _mean_pool(last_hidden_state, attention_mask):
        mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        summed = torch.sum(last_hidden_state * mask, dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)
        return summed / counts

    def forward(self, texts):
        device = next(self.bert.parameters()).device
        input_ids, attention_mask = self._encode_batch(texts)
        input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)

        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        token_states = outputs.last_hidden_state                 # (B, L, 768)
        pooled_embedding = self._mean_pool(token_states, attention_mask)

        return token_states, attention_mask, pooled_embedding


class MockBioBERTTokenExtractor(nn.Module):

    def __init__(self, hidden_size=config.NLP_EMBEDDING_DIM, vocab_size=30000):
        super().__init__()
        self.hidden_size = hidden_size
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(hidden_size, nhead=8, batch_first=True),
            num_layers=2,
        )

    def forward(self, texts):
        device = self.embed.weight.device
        lengths = [max(3, min(20, len(t.split()))) for t in texts]
        max_len = max(lengths)
        input_ids = torch.randint(0, self.embed.num_embeddings, (len(texts), max_len), device=device)
        attention_mask = torch.zeros(len(texts), max_len, dtype=torch.long, device=device)
        for i, l in enumerate(lengths):
            attention_mask[i, :l] = 1

        token_states = self.encoder(self.embed(input_ids))
        mask = attention_mask.unsqueeze(-1).expand(token_states.size()).float()
        pooled_embedding = (token_states * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        return token_states, attention_mask, pooled_embedding
