# app.py
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

# Load BLIP model + processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Load an image (put a sample image in your project folder, e.g. sample.jpg)
image_path = "sample.jpg"
raw_image = Image.open(image_path).convert("RGB")

# Prepare inputs
inputs = processor(raw_image, return_tensors="pt")

# Generate caption
out = model.generate(**inputs)
caption = processor.decode(out[0], skip_special_tokens=True)

print("Image description:", caption)