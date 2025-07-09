from transformers import AutoModel, AutoTokenizer

class TextAdapterDistilbert(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = AutoModel.from_pretrained("distilbert-base-uncased")
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    def forward(self, texts):
        tokens = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
        out = self.model(**tokens)
        return out.last_hidden_state.mean(dim=1)  # Mean pooled embedding


class TextAdapterGeneralized(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Linear(100, 128)  # fake text embedding

    def forward(self, x):
        return self.encoder(x)