from transformers import AutoModel, AutoTokenizer
import torch.nn as nn   

class TextAdapterDistilbert(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = AutoModel.from_pretrained("distilbert-base-uncased")
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        self.proj = nn.Linear(768, 128)

    def forward(self, texts):
        tokens = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
        out = self.model(**tokens)
        pooled = out.last_hidden_state.mean(dim=1)
        return self.proj(pooled)  # Project to 128-dim


class TextAdapterGeneralized(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(100, 128)  # fake text embedding

    def forward(self, x):
        return self.encoder(x)