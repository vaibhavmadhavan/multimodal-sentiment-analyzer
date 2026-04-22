"""
Run this ONCE before starting app.py to pre-cache both models.
Saves ~10 min on first app launch.
"""
from transformers import pipeline
from deepface import DeepFace
import numpy as np

print("Downloading NLP model (~265MB)...")
pipe = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    device="cpu"
)
print("NLP model ready:", pipe("test")[0])

print("\nDownloading Vision model (~580MB)...")
dummy = np.zeros((100, 100, 3), dtype=np.uint8)
try:
    DeepFace.analyze(dummy, actions=["emotion"], enforce_detection=False)
except:
    pass
print("Vision model pre-cached ✓")
print("\n✅ All models ready. You can now run: streamlit run app.py")