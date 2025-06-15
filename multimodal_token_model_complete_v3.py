# multimodal_token_model.py

"""
Designprincip för tokenisering:

"I ett system där det finns parallell input över flera sensorer,
ska man välja token size efter den parallella inputens största storlek."

Varje token representerar ett ögonblick av parallell sensorisk information (t.ex. RGBC från en pixel,
eller 4 bytes i en textsträng). Detta möjliggör enhetlig inläsning, jämförbar representation,
och multimodal inlärning.
"""

from typing import List
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torchaudio.datasets import SPEECHCOMMANDS
from PIL import Image, ImageOps
import torchaudio
import random
import os
import numpy as np
from sklearn.cluster import KMeans
import umap
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd
import time
import string


def text_to_tokens(text: str) -> List[List[int]]:
    byte_stream = list(text.encode("utf-8"))
    while len(byte_stream) % 4 != 0:
        byte_stream.append(0x00)
    return [byte_stream[i:i+4] for i in range(0, len(byte_stream), 4)]


def corrupt_text(text: str) -> str:
    new_text = ""
    for c in text:
        if random.random() < 0.1:
            c = random.choice(string.ascii_letters)
        if random.random() < 0.3:
            c = c.upper() if random.random() < 0.5 else c.lower()
        new_text += c
    return new_text


def augment_image(image: Image.Image) -> List[Image.Image]:
    variants = []
    for angle in [0, 90, 180, 270]:
        rotated = image.rotate(angle)
        variants.append(rotated)
        variants.append(ImageOps.mirror(rotated))
    return variants


def augment_audio(waveform, sample_rate):
    factor = random.uniform(0.9, 1.1)
    resampled = torchaudio.functional.resample(waveform, sample_rate, int(sample_rate * factor))
    return resampled


def image_to_tokens(image: Image.Image) -> List[List[int]]:
    image = image.convert("RGB").resize((32, 32))
    pixels = np.array(image)
    tokens = []
    for row in pixels:
        for r, g, b in row:
            luminance = int(0.299*r + 0.587*g + 0.114*b)
            tokens.append([r, g, b, luminance])
    return tokens


def audio_to_tokens(waveform, sample_rate) -> List[List[int]]:
    samples = waveform[0].numpy()
    tokens = []
    for i in range(0, len(samples), 4):
        chunk = samples[i:i+4]
        bytes_chunk = [int(((x + 1) / 2) * 255) for x in chunk]
        while len(bytes_chunk) < 4:
            bytes_chunk.append(0)
        tokens.append(bytes_chunk)
    return tokens[:64]


def tokens_to_text(tokens: List[List[int]]) -> str:
    flat_bytes = [b for token in tokens for b in token if b != 0x00]
    return bytes(flat_bytes).decode("utf-8")


class TokenSample:
    def __init__(self, tokens: List[List[int]], modality: str, name: str, concept: str = ""):
        self.tokens = tokens
        self.modality = modality
        self.name = name
        self.concept = concept


class TokenEmbedder(nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        self.linear = nn.Linear(4, dim)

    def forward(self, x):
        return self.linear(x.float() / 255.0)


class MultimodalTransformer(nn.Module):
    def __init__(self, dim=128, depth=4, heads=4):
        super().__init__()
        self.embedding = TokenEmbedder(dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.to_output = nn.Linear(dim, 4)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        return self.to_output(x[:, 0])


class MultimodalTokenDataset(Dataset):
    def __init__(self, max_samples=100):
        self.samples = []

        cifar = CIFAR10(root="./data", train=True, download=True)
        for i in range(len(cifar)):
            img, label = cifar[i]
            for variant in augment_image(img):
                tokens = image_to_tokens(variant)
                self.samples.append(TokenSample(tokens, modality="image", name=str(label), concept="vision"))

        speech = SPEECHCOMMANDS("./data", download=True)
        speech = list(speech)
        random.shuffle(speech)
        for i in range(len(speech)):
            waveform, sample_rate, label, *_ = speech[i]
            waveform = augment_audio(waveform, sample_rate)
            tokens = audio_to_tokens(waveform, sample_rate)
            self.samples.append(TokenSample(tokens, modality="audio", name=label, concept="ljud"))

        base_texts = ["hej världen", "jag heter gish", "det här är en test"]
        for text in base_texts:
            for _ in range(5):
                noisy = corrupt_text(text)
                tokens = text_to_tokens(noisy)
                self.samples.append(TokenSample(tokens, modality="text", name=text, concept="språk"))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return torch.tensor(sample.tokens), sample.modality, sample.name, sample.concept


def run(mode="train", resume=False, save_path="checkpoint.pt", epochs=5, query=None, input_text=None, input_image=None):
    if mode == "train" and os.path.exists(save_path) and not resume:
        answer = input(f"🔁 En checkpoint hittades på '{save_path}'. Vill du återuppta träning? (ja/nej): ").strip().lower()
        if answer == "ja":
            resume = True
        else:
            print("🆕 Ny träning påbörjas (checkpoint ignoreras).")
    model = MultimodalTransformer()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    if resume and os.path.exists(save_path):
        checkpoint = torch.load(save_path)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        print("✅ Checkpoint återläst.")

    dataset = MultimodalTokenDataset()
    loader = DataLoader(dataset, batch_size=8, shuffle=True)

    if mode == "train":
        try:
            for epoch in range(epochs):
                model.train()
                total_loss = 0.0
                for batch in loader:
                    tokens, modality, name, concept = batch
                    output = model(tokens)
                    target = tokens[:, 0]  # förenklad loss
                    loss = loss_fn(output, target)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                print(f"📚 Epoch {epoch+1}: Loss = {total_loss/len(loader):.4f}")
                torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict()}, save_path)
        except KeyboardInterrupt:
            print("🛑 Träning avbröts. Sparar modell...")
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict()}, save_path)

    elif mode == "visualize":
        model.eval()
        all_embeddings = []
        all_labels = []
        for batch in loader:
            tokens, modality, name, concept = batch
            with torch.no_grad():
                emb = model.embedding(tokens)
            emb_mean = emb.mean(dim=1).numpy()
            all_embeddings.extend(emb_mean)
            all_labels.extend(name)

        reducer = umap.UMAP()
        reduced = reducer.fit_transform(all_embeddings)
        df = pd.DataFrame(reduced, columns=["x", "y"])
        df["label"] = all_labels
        px.scatter(df, x="x", y="y", color="label", title="Multimodal Token Clustering").show()

    elif mode == "query" and query:
        model.eval()
        if os.path.exists(save_path):
            checkpoint = torch.load(save_path)
            model.load_state_dict(checkpoint["model"])
        else:
            print("❌ Ingen modell hittades. Träna modellen först.")
            return

        question_tokens = torch.tensor(text_to_tokens(query)).unsqueeze(0)
        with torch.no_grad():
            qvec = model.embedding(question_tokens).mean(dim=1)

        if input_text:
            input_tokens = torch.tensor(text_to_tokens(input_text)).unsqueeze(0)
        elif input_image:
            image = Image.open(input_image)
            input_tokens = torch.tensor(image_to_tokens(image)).unsqueeze(0)
        else:
            print("⚠️ Inget jämförelseinput angivet.")
            return

        with torch.no_grad():
            dvec = model.embedding(input_tokens).mean(dim=1)

        sim = F.cosine_similarity(qvec, dvec).item()
        print(f"🤖 Likhet med frågan: {sim:.4f}")
        print("✅ Svar: JA" if sim > 0.85 else "❌ Svar: NEJ")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, help="Fråga modellen, t.ex. 'Är detta en katt?'")
    parser.add_argument("--input_text", type=str, help="Text att jämföra med frågan")
    parser.add_argument("--input_image", type=str, help="Bildfil att jämföra med frågan")
    parser.add_argument("--mode", type=str, choices=["train", "visualize", "query"], required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--save_path", type=str, default="checkpoint.pt")
    args = parser.parse_args()

    run(mode=args.mode, resume=args.resume, save_path=args.save_path, epochs=args.epochs, query=args.query, input_text=args.input_text, input_image=args.input_image)
