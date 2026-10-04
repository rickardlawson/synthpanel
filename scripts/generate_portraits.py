"""Genererer portrettbiblioteket for persona-galleriet («Ti på gata»).

Portrettene er AI-genererte – de forestiller ingen virkelige personer. De lages én
gang og sjekkes inn som små WebP-filer, så panelet ikke trenger bildemodell for å kjøre.

Modell: Realistic Vision 5.1 (SD 1.5, CreativeML OpenRAIL-M) + LCM-LoRA (OpenRAIL++)
+ sd-vae-ft-mse (MIT). Kjører på CPU (~40 s per bilde).

    pip install torch --index-url https://download.pytorch.org/whl/cpu
    pip install diffusers transformers accelerate peft safetensors pillow
    python scripts/generate_portraits.py            # fortsetter der den slapp
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from synthpanel.personas.portraits import AGES, BACKGROUNDS, DIR as OUT, nationality  # noqa: E402

SETTINGS = ["in a kitchen at home", "on a street in a Norwegian town", "outdoors by a forest trail", "in a living room",
            "in a cafe", "at a bus stop", "in an office break room", "outside a wooden house", "by the sea on a cloudy day",
            "in a grocery store parking lot", "on a balcony of an apartment block", "in a small workshop"]
CLOTHES = ["a knitted sweater", "a rain jacket", "a fleece jacket", "a plain t-shirt", "a flannel shirt", "a hoodie",
           "a cardigan", "a casual blazer", "a wool coat", "a work jacket", "a denim jacket", "a simple blouse"]
EXTRAS = ["", "", "", "with glasses, ", "slight smile, ", "neutral expression, ", "tired eyes, ", "freckles, "]


def prompt(sex: str, age: int, nationality: str, rng: random.Random) -> str:
    who = "man" if sex == "mann" else "woman"
    # «Somali-Norwegian» trekker mot et nordisk utseende; «of Somali descent» holder opphavet.
    origin = "Norwegian " if nationality == "Norwegian" else ""
    # Modellen gjør middelaldrende kvinner for unge – gi tydelige alderstegn.
    cue = ("" if age < 40 else "middle-aged, natural skin texture, fine wrinkles, " if age < 58
           else "older, wrinkled skin, " if age < 75 else "elderly, deeply wrinkled skin, ")
    shown = {"Somali": "Somali (Black, East African)", "Eritrean": "Eritrean (Black, East African)",
             "Ethiopian": "Ethiopian (Black, East African)", "Nigerian": "Nigerian (Black, West African)"}.get(nationality, nationality)
    descent = "" if nationality == "Norwegian" else f" of {shown} descent, living in Norway"
    return (f"RAW candid portrait photo of a {age} year old {origin}{who}{descent}, {cue}{rng.choice(EXTRAS)}wearing {rng.choice(CLOTHES)}, "
            f"{rng.choice(SETTINGS)}, overcast natural light, head and shoulders, looking at camera, "
            f"ordinary everyday person, 50mm, sharp focus, film grain")


def plan() -> list[dict]:
    items = []
    maxv = max(n for n, _ in BACKGROUNDS.values())
    for v in range(maxv):  # runde for runde, så alle celler får første bilde tidlig
        for bg, (n, _) in BACKGROUNDS.items():
            if v >= n:
                continue
            for band, age in AGES.items():
                for sex in ("kvinne", "mann"):
                    key = f"{bg}_{sex}_{band}_{v}"
                    rng = random.Random(key)
                    nat = nationality(bg, band, v)
                    items.append({"file": f"{key}.webp", "bakgrunn": bg, "kjonn": sex, "alder": band, "variant": v,
                                  "prompt": prompt(sex, age + rng.randint(-2, 2), nat, rng), "seed": rng.randint(0, 2**31)})
    return items


def main(limit: int | None = None):
    import torch
    from diffusers import AutoencoderKL, LCMScheduler, StableDiffusionPipeline

    torch.set_num_threads(max(1, torch.get_num_threads()))
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
    pipe = StableDiffusionPipeline.from_pretrained("SG161222/Realistic_Vision_V5.1_noVAE", vae=vae,
                                                   safety_checker=None, requires_safety_checker=False)
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    pipe.load_lora_weights("latent-consistency/lcm-lora-sdv1-5")
    pipe.fuse_lora()

    OUT.mkdir(parents=True, exist_ok=True)
    items = plan()
    done = 0
    for it in items:
        path = OUT / it["file"]
        if not path.exists():
            img = pipe(it["prompt"], num_inference_steps=5, guidance_scale=1.0, height=512, width=512,
                       generator=torch.Generator().manual_seed(it["seed"])).images[0]
            img.resize((320, 320)).save(path, "WEBP", quality=80)
            done += 1
            print(f"{done} {it['file']}", flush=True)
        if limit and done >= limit:
            break


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else None)
