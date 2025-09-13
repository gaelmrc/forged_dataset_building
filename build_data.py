# src/forged_dataset_building/cli.py
import os
import sys
import time
import json
import random
import logging
import argparse
from io import BytesIO
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, List

import numpy as np
import requests
from PIL import Image
from rembg import remove

# ---------------------------
# Config & Logging
# ---------------------------

def setup_logger(verbose: bool):
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

def get_token(cli_token: Optional[str]) -> str:
    token = cli_token or os.getenv("HF_TOKEN", "")
    if not token:
        logging.error("Aucun token HF. Fournis --hf-token ou exporte HF_TOKEN.")
        sys.exit(2)
    return token

# ---------------------------
# API clients (HF Inference)
# ---------------------------

IMG2TEXT_URL = "https://api-inference.huggingface.co/models/microsoft/git-large-r-coco"
TEXT2IMG_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"

def _post_with_retry(url: str, headers: Dict[str, str], **kwargs) -> requests.Response:

    delay = 2.0
    for attempt in range(6):
        try:
            resp = requests.post(url, headers=headers, timeout=60, **kwargs)
            if resp.status_code == 200:
                return resp
            if resp.status_code in (429, 500, 502, 503, 524, 529):
                logging.warning(f"{url} -> {resp.status_code}, retry in {delay:.1f}s…")
                time.sleep(delay)
                delay = min(delay * 1.6, 15.0)
                continue
            # autres codes: log + break
            logging.error(f"HTTP {resp.status_code} on {url}: {resp.text[:200]}")
            resp.raise_for_status()
            return resp
        except requests.RequestException as e:
            logging.warning(f"Network error: {e}; retry in {delay:.1f}s…")
            time.sleep(delay)
            delay = min(delay * 1.6, 15.0)
    raise RuntimeError(f"Failed POST to {url} after retries.")

def query_img2text(image_path: Path, hf_token: str) -> str:
    headers = {"Authorization": f"Bearer {hf_token}"}
    with open(image_path, "rb") as f:
        data = f.read()
    resp = _post_with_retry(IMG2TEXT_URL, headers, data=data)
    try:
        out = resp.json()
    except ValueError:
        raise RuntimeError("ERROR : No JSON for img2text.")
    if not isinstance(out, list) or not out or "generated_text" not in out[0]:
        raise RuntimeError(f"ERROR output img2text : {out}")
    return out[0]["generated_text"].strip()

def query_text2img(prompt: str, hf_token: str, seed: Optional[int] = None) -> Image.Image:
    headers = {"Authorization": f"Bearer {hf_token}"}
    payload = {"inputs": prompt} 
    resp = _post_with_retry(TEXT2IMG_URL, headers, json=payload)
    try:
        
        maybe_json = resp.json()
        if isinstance(maybe_json, dict) and "error" in maybe_json:
            raise RuntimeError(f"HF text2img error: {maybe_json.get('error')}")
    except ValueError:
        pass
    img = Image.open(BytesIO(resp.content))
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    return img

# ---------------------------
# Image ops
# ---------------------------

def square_select(image_path: Path, min_area_ratio=0.25, max_area_ratio=0.5) -> Tuple[int, int, int]:
    """Random square choice (between 25% and 50% of the total area)."""
    with Image.open(image_path) as im:
        w, h = im.size
    # aire -> côté ~ sqrt(ratio * w*h)
    min_side = int((min_area_ratio * w * h) ** 0.5)
    max_side = int((max_area_ratio * w * h) ** 0.5)
    if max_side <= 0 or min_side <= 0:
        raise ValueError("ERROR : invalid dim ")
    size = random.randint(max(1, min_side), max_side)
    max_coord_x = max(0, w - size)
    max_coord_y = max(0, h - size)
    x = random.randint(0, max_coord_x)
    y = random.randint(0, max_coord_y)
    return x, y, size

def crop_save_square(image_path: Path, x: int, y: int, size: int, out_path: Path) -> None:
    with Image.open(image_path) as im:
        sq = im.crop((x, y, x + size, y + size))
        sq.save(out_path)

def remove_bg_rgba(img_rgba: Image.Image, alpha_threshold: int = 127) -> Image.Image:
    """Uses rembg then applies alpha threshold."""
    if img_rgba.mode != "RGBA":
        img_rgba = img_rgba.convert("RGBA")
    arr = np.array(img_rgba)
    # rembg  uint8 HxWx{3,4}
    rmbg = remove(arr)
    if rmbg.shape[-1] == 3:
        # si pas d'alpha, on en crée un plein
        a = np.full(rmbg.shape[:2] + (1,), 255, dtype=np.uint8)
        rmbg = np.concatenate([rmbg, a], axis=-1)
    #  alpha thresholding
    rmbg[..., 3] = np.where(rmbg[..., 3] >= alpha_threshold, 255, 0).astype(np.uint8)
    return Image.fromarray(rmbg, "RGBA")

def paste_rgba_on_rgb(base_rgb: Image.Image, overlay_rgba: Image.Image, x: int, y: int) -> Image.Image:
    out = base_rgb.copy()
    out.paste(overlay_rgba, (x, y), overlay_rgba)
    return out

# ---------------------------
# Pipeline 
# ---------------------------

def build_one(image_path: Path, out_auth_sq: Path, out_forged_sq: Path, out_mask: Path,out_forged_img: Path,hf_token: str,alpha_threshold: int = 127,sleep_between_calls: float = 3.5,seed: Optional[int] = None,) -> str:

    # Square choice + save authentic square
    x, y, size = square_select(image_path)
    out_auth_sq.parent.mkdir(parents=True, exist_ok=True)
    crop_save_square(image_path, x, y, size, out_auth_sq)

    # Caption -> prompt
    prompt = query_img2text(out_auth_sq, hf_token=hf_token)
    logging.debug(f"Prompt: {prompt}")
    time.sleep(sleep_between_calls) 

    # Generation -> redimension -> remove BG
    gen = query_text2img(prompt, hf_token=hf_token, seed=seed)
    gen = gen.resize((size, size), resample=Image.LANCZOS)
    gen_rmbg = remove_bg_rgba(gen, alpha_threshold=alpha_threshold)
    out_forged_sq.parent.mkdir(parents=True, exist_ok=True)
    gen_rmbg.save(out_forged_sq)

    # Paste on authentic image
    with Image.open(image_path) as auth_im:
        base_rgb = auth_im.convert("RGB")
        forged_rgb = paste_rgba_on_rgb(base_rgb, gen_rmbg, x, y)
    out_forged_img.parent.mkdir(parents=True, exist_ok=True)
    forged_rgb.save(out_forged_img)

    # Mask (forged area = black, authentic = white)
    mask = Image.new("RGB", base_rgb.size, "white")
    black = Image.new("RGB", (size, size), "black")  
    mask.paste(black, (x, y), gen_rmbg)
    out_mask.parent.mkdir(parents=True, exist_ok=True)
    mask.save(out_mask)

    return prompt

# ---------------------------
# Batch / CLI
# ---------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        "forged-ds",
        description="Build a forged image dataset (caption GIT -> t2i SDXL -> remove BG -> paste).",
    )
    p.add_argument("--auth-dir", type=Path, required=True, help="path authentic img"
    p.add_argument("--out-dir", type=Path, required=True, help="path output")
    p.add_argument("--hf-token", type=str, default=None, help="Token HF (env variable HF_TOKEN).")
    p.add_argument("--limit", type=int, default=0, help="Limit img number (0 = all)")
    p.add_argument("--seed", type=int, default=None, help="randomness seed")
    p.add_argument("--alpha-th", type=int, default=127, help="mask alpha threshold (0-255).")
    p.add_argument("--sleep", type=float, default=3.5, help="pause between HF calls.")
    p.add_argument("--overwrite", action="store_true", help="overwriting existing files.")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)

def discover_images(root: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    return [p for p in root.rglob("*") if p.suffix.lower() in exts and p.is_file()]

def main(argv=None):
    args = parse_args(argv)
    setup_logger(args.verbose)
    hf_token = get_token(args.hf_token)

    out_root = args.out_dir
    paths = {
        "auth_sq": out_root / "authentic_squares",
        "forged_sq": out_root / "forged_squares",
        "masks": out_root / "masks",
        "forged": out_root / "forged",
        "meta": out_root / "metadata.jsonl",
        "prompts_pickle": out_root / "prompts.pkl",
    }
    for k, v in paths.items():
        if k in ("meta", "prompts_pickle"):
            v.parent.mkdir(parents=True, exist_ok=True)
        else:
            v.mkdir(parents=True, exist_ok=True)

    images = discover_images(args.auth_dir)
    if args.limit > 0:
        images = images[: args.limit]
    logging.info(f"{len(images)} images in {args.auth_dir}")

    old_prompts: Dict[str, str] = {}
    if paths["prompts_pickle"].exists():
        try:
            import pickle
            with open(paths["prompts_pickle"], "rb") as f:
                old_prompts = pickle.load(f)
            logging.info(f"Prompts loaded: {len(old_prompts)}")
        except Exception as e:
            logging.warning(f" not readable {paths['prompts_pickle'].name}: {e}")

    processed = 0
    new_prompts: Dict[str, str] = dict(old_prompts)

    with open(paths["meta"], "a", encoding="utf-8") as meta_out:
        for idx, img_path in enumerate(images, 1):
            key = str(img_path.relative_to(args.auth_dir))

            stem = img_path.stem
            auth_sq = paths["auth_sq"] / f"{stem}_auth_sq.png"
            forged_sq = paths["forged_sq"] / f"{stem}_forged_sq.png"
            mask_p = paths["masks"] / f"{stem}_mask.png"
            forged_img = paths["forged"] / f"{stem}_forged.png"

            if (
                not args.overwrite
                and auth_sq.exists()
                and forged_sq.exists()
                and mask_p.exists()
                and forged_img.exists()
                and key in new_prompts
            ):
                logging.debug(f"Skip {key} (déjà présent)")
                continue

            try:
                prompt = build_one(
                    image_path=img_path,
                    out_auth_sq=auth_sq,
                    out_forged_sq=forged_sq,
                    out_mask=mask_p,
                    out_forged_img=forged_img,
                    hf_token=hf_token,
                    alpha_threshold=args.alpha_th,
                    sleep_between_calls=args.sleep,
                    seed=args.seed,
                )
                new_prompts[key] = prompt
                record = {
                    "image": key,
                    "prompt": prompt,
                    "auth_square": str(auth_sq.relative_to(out_root)),
                    "forged_square": str(forged_sq.relative_to(out_root)),
                    "mask": str(mask_p.relative_to(out_root)),
                    "forged_image": str(forged_img.relative_to(out_root)),
                }
                meta_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                processed += 1
                if idx % 10 == 0:
                    logging.info(f"Progression {idx}/{len(images)}")
            except Exception as e:
                logging.exception(f"Fail on {key}: {e}")

    try:
        import pickle
        with open(paths["prompts_pickle"], "wb") as f:
            pickle.dump(new_prompts, f)
    except Exception as e:
        logging.warning(f"Fail writing {paths['prompts_pickle'].name}: {e}")

    logging.info(f"Success.  | new items: {processed}")

if __name__ == "__main__":
    main()
