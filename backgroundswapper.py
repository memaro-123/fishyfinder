import os
import random
from PIL import Image

# Paths
DATASET_ROOT = "/root/fishyfinder/fish_dataset"
ORIGINAL_DIR = os.path.join(DATASET_ROOT, "original_fish")
BACKGROUND_DIR = os.path.join(DATASET_ROOT, "backgrounds")
MODDED_DIR = os.path.join(DATASET_ROOT, "modded_fish")

# Parameters
IMG_SIZE = 224
VARIATIONS_PER_IMAGE = 20
MAX_TRANSLATION = 15
ZOOM_RANGE = (0.9, 1.1)


def ensure_dirs():
    classes = os.listdir(ORIGINAL_DIR)
    for cls in classes:
        os.makedirs(os.path.join(MODDED_DIR, cls), exist_ok=True)


def get_random_background():
    bg_file = random.choice(os.listdir(BACKGROUND_DIR))
    bg = Image.open(os.path.join(BACKGROUND_DIR, bg_file)).convert("RGB")
    return bg.resize((IMG_SIZE, IMG_SIZE))


def center_and_pad(img):
    img.thumbnail((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    canvas = Image.new("RGBA", (IMG_SIZE, IMG_SIZE), (0, 0, 0, 0))
    x = (IMG_SIZE - img.width) // 2
    y = (IMG_SIZE - img.height) // 2
    canvas.paste(img, (x, y), img)
    return canvas


def augment_and_save(fish_img, class_name, base_name):
    out_dir = os.path.join(MODDED_DIR, class_name)
    for i in range(VARIATIONS_PER_IMAGE):

        # zoom
        scale = random.uniform(*ZOOM_RANGE)
        w, h = fish_img.size
        resized = fish_img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

        # blank canvas
        canvas = Image.new("RGBA", (IMG_SIZE, IMG_SIZE), (0, 0, 0, 0))

        # translation
        tx = random.randint(-MAX_TRANSLATION, MAX_TRANSLATION)
        ty = random.randint(-MAX_TRANSLATION, MAX_TRANSLATION)

        x = (IMG_SIZE - resized.width) // 2 + tx
        y = (IMG_SIZE - resized.height) // 2 + ty

        canvas.paste(resized, (x, y), resized)

        # apply background
        bg = get_random_background().convert("RGBA")
        final = Image.alpha_composite(bg, canvas).convert("RGB")

        final.save(os.path.join(out_dir, f"{base_name}_{i}.jpg"), quality=95)


def main():
    ensure_dirs()
    classes = os.listdir(ORIGINAL_DIR)

    for cls in classes:
        class_path = os.path.join(ORIGINAL_DIR, cls)
        for file in os.listdir(class_path):
            path = os.path.join(class_path, file)
            fish = Image.open(path).convert("RGBA")
            fish = center_and_pad(fish)
            base_name = os.path.splitext(file)[0]
            augment_and_save(fish, cls, base_name)

    print("Dataset generation complete.")


if __name__ == "__main__":
    main()