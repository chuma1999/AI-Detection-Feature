from pathlib import Path

root = Path("C:/Users/info/OneDrive/Desktop/AI Detection")
img_dir = root / "dataset" / "images" / "train_balanced"
label_dir = root / "dataset" / "labels" / "train_balanced"

images = sorted(img_dir.glob("*.jpg"))
labels = sorted(label_dir.glob("*.txt"))
missing = []
extra = []

for img in images:
    label = label_dir / f"{img.stem}.txt"
    if not label.exists():
        missing.append(str(label))

for label in labels:
    image = img_dir / f"{label.stem}.jpg"
    if not image.exists():
        extra.append(str(label))

print(f"Images: {len(images)}")
print(f"Labels: {len(labels)}")
print(f"Missing labels: {len(missing)}")
print(f"Extra labels: {len(extra)}")

if missing:
    print("Missing:")
    for p in missing:
        print(p)
else:
    print("Missing: none")

if extra:
    print("Extra:")
    for p in extra:
        print(p)
else:
    print("Extra: none")
