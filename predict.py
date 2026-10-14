import torch
from nltk.tokenize import word_tokenize
from model import LSTMModel

SEQ_LEN = 10


class NextWordPredictor:
    def __init__(self, path="models/lstm.pt"):
        ck = torch.load(path, weights_only=False, map_location="cpu")
        self.itos = ck["itos"]
        self.stoi = {w: i for i, w in enumerate(self.itos)}
        self.unk = self.stoi["<unk>"]
        self.eos = self.stoi["<eos>"]

        self.model = LSTMModel(len(self.itos))
        self.model.load_state_dict(ck["model"])
        self.model.eval()

    def predict(self, seed, n_words=30, temperature=0.8, top_k=10):
        words = word_tokenize(seed.lower())
        ids = [self.stoi.get(w, self.unk) for w in words]
        ids = [self.unk] * SEQ_LEN + ids

        generated = []
        with torch.no_grad():
            for _ in range(n_words):
                logits = self.model(torch.tensor([ids[-SEQ_LEN:]]))[0] / temperature
                logits[self.unk] = float("-inf")
                top = torch.topk(logits, top_k)
                probs = torch.softmax(top.values, dim=-1)
                next_id = top.indices[torch.multinomial(probs, 1)].item()
                if next_id == self.eos:
                    break
                generated.append(self.itos[next_id])
                ids.append(next_id)

        return " ".join(words + generated)