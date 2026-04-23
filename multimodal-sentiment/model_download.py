from transformers import pipeline
from deepface import DeepFace
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification


from transformers import AutoTokenizer, AutoModelForSequenceClassification

print("Downloading NLP model (~500MB)...")
model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
AutoTokenizer.from_pretrained(model_name)
AutoModelForSequenceClassification.from_pretrained(model_name)
print("NLP model ready ✓")

print("\nDownloading Vision model (DeepFace + RetinaFace)...")
dummy = np.zeros((100, 100, 3), dtype=np.uint8)
try:
    DeepFace.analyze(
        dummy,
        actions=["emotion"],
        detector_backend="retinaface",
        enforce_detection=False,
        silent=True
    )
except:
    pass
print("Vision model pre-cached ✓")
print("\n✅ All models ready. You can now run: streamlit run app.py")