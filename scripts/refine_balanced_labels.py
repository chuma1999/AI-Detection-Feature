from pathlib import Path

root = Path("C:/Users/info/OneDrive/Desktop/AI Detection")
label_dir = root / "dataset" / "labels" / "train_balanced"
image_dir = root / "dataset" / "images" / "train_balanced"

files = sorted(label_dir.glob("*.txt"))
refined = 0
for label_path in files:
    lines = []
    for raw in label_path.read_text(encoding="utf-8").strip().splitlines():
        if not raw.strip():
            continue
        parts = raw.split()
        if len(parts) != 5:
            raise ValueError(f"Malformed annotation in {label_path.name}: {raw}")
        cls, x, y, w, h = [float(v) for v in parts]
        cls = int(cls)

        # Tighten boxes modestly while keeping centers fixed.
        w = max(0.02, min(0.95, w * 0.88))
        h = max(0.02, min(0.95, h * 0.88))
        x = max(0.01, min(0.99, x))
        y = max(0.01, min(0.99, y))

        lines.append(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
    label_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    refined += 1

images = sorted(image_dir.glob("*.jpg"))
problems = []
for image_path in images:
    label_path = label_dir / f"{image_path.stem}.txt"
    if not label_path.exists():
        problems.append(f"Missing label: {image_path.name}")
        continue

    for raw in label_path.read_text(encoding="utf-8").splitlines():
        parts = raw.split()
        if len(parts) != 5:
            problems.append(f"Bad format: {label_path.name} -> {raw}")
            continue
        cls, x, y, w, h = [float(v) for v in parts]
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 and 0.0 <= w <= 1.0 and 0.0 <= h <= 1.0):
            problems.append(f"Out-of-range box: {label_path.name} -> {raw}")

if problems:
    print(f"Refined files: {refined}")
    print(f"Validation issues: {len(problems)}")
    for item in problems:
        print(item)
    raise SystemExit(1)

print(f"Refined files: {refined}")
print(f"Images checked: {len(images)}")
print("Validation passed: balanced set is structurally valid and label boxes were tightened.")
