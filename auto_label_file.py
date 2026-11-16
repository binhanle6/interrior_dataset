from pathlib import Path
import pandas as pd
from PIL import Image
import torch
from transformers import CLIPModel, CLIPProcessor

csv_path = "furniture.data.csv"
df = pd.read_csv(csv_path, sep=None, engine="python")
df.columns = df.columns.str.strip()

print(f"Đã tải thành công {len(df)} sản phẩm từ {csv_path}")

model_id = "openai/clip-vit-base-patch32"
print("Đang tải mô hình CLIP để phân loại style...")
model = CLIPModel.from_pretrained(model_id)
processor = CLIPProcessor.from_pretrained(model_id)

device = "mps" if torch.backends.mps.is_available() else "cpu"
model.to(device)

styles = [
    "Scandinavian",
    "Minimalist",
    "Modern",
    "Industrial",
    "Japandi",
    "Classic Luxury",
]
style_prompts = [f"a photo of {s} style interior furniture" for s in styles]

text_inputs = processor(text=style_prompts, return_tensors="pt", padding=True).to(
    device
)
with torch.no_grad():
  text_features = model.get_text_features(**text_inputs)
  text_features = text_features / text_features.norm(dim=-1, keepdim=True)

predicted_styles = []

print("\nBắt đầu quét ảnh để tự động gán style...")

for idx, row in df.iterrows():
  img_path_str = str(row["image_path"]).strip()
  img_path = Path(img_path_str)

  if img_path.exists():
    image = Image.open(img_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
      img_feat = model.get_image_features(**inputs)
      img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)

      similarity = (img_feat @ text_features.T).squeeze(0)
      best_style_idx = similarity.argmax().item()
      best_style = styles[best_style_idx]

    predicted_styles.append(best_style)

    product_name = (
        row.get("Name") if "Name" in df.columns else f"Sản phẩm {idx+1}"
    )
    print(
        f"[{idx+1}/{len(df)}] {product_name} ---> AI đoán Style: {best_style}"
    )
  else:
    print(f"Cảnh báo: Không tìm thấy ảnh tại đường dẫn: '{img_path}'")
    predicted_styles.append("Unknown")

df["style"] = predicted_styles
df.to_csv("furniture_processed.csv", index=False)

print("\nHOÀN TẤT GÁN STYLE!")
print("- File kết quả đã lưu tại: furniture_processed.csv")