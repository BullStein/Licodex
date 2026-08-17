"""
Watch & Train - LIBRAS
--------------------------
Fica rodando em loop, vigiando a pasta "imgs/". A cada intervalo
(padrao 5s):
    1. Detecta fotos NOVAS (que ainda nao foram processadas -- usa
       um manifesto em imgs/.processed_manifest.json pra nao
       retreinar a mesma foto toda vez e nao inflar o pool).
    2. Extrai os landmarks, aplica augmentation/dedupe se ativado
       (mesma logica do batch_train_from_images.py) e atualiza
       data.json / data_raw.json / commands.json / commands_raw.json.
    3. Redesenha um painel no terminal mostrando, por letra/comando:
         - quantas amostras tem no pool agora, e quanto cresceu desde
           a ultima rodada (+N)
         - COESAO: distancia media das amostras daquela letra ate a
           propria media (menor = fotos mais consistentes entre si)
         - CONFUSAO: qual e a letra/comando mais parecido e a que
           distancia (maior = mais seguro o reconhecimento distinguir
           essa letra das outras)

Pode deixar rodando enquanto voce tira fotos com o photo_capture.py
em outro terminal -- a cada novas fotos salvas, esse script pega
automaticamente na proxima rodada do loop e retreina.

CTRL+C encerra a qualquer momento.

Como usar:
    python watch_and_train.py
    python watch_and_train.py --interval 3 --augment --dedupe 0.015
"""

import os
import sys
import json
import math
import time
import argparse
from collections import defaultdict

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from matching_improved import weighted_distance, LANDMARK_WEIGHTS

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "hand_landmarker.task")
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
DATA_PATH = os.path.join(BASE_DIR, "data.json")
RAW_DATA_PATH = os.path.join(BASE_DIR, "data_raw.json")
COMMANDS_DATA_PATH = os.path.join(BASE_DIR, "commands.json")
RAW_COMMANDS_DATA_PATH = os.path.join(BASE_DIR, "commands_raw.json")

WRIST = 0
MIDDLE_MCP = 9

LETTERS = {chr(c) for c in range(ord("A"), ord("Z") + 1)} | {"Ç", "CA"}
COMMANDS = ["BACKSPACE", "SPACE", "CLEAR"]
VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print("Baixando modelo hand_landmarker.task (primeira execucao)...")
        import urllib.request
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Download concluido:", MODEL_PATH)


def normalize_landmarks(landmarks):
    wrist = landmarks[WRIST]
    ref = landmarks[MIDDLE_MCP]
    scale = math.sqrt(
        (ref.x - wrist.x) ** 2 + (ref.y - wrist.y) ** 2 + (ref.z - wrist.z) ** 2
    )
    scale = scale if scale > 1e-6 else 1e-6
    normalized = []
    for lm in landmarks:
        normalized.append([
            (lm.x - wrist.x) / scale,
            (lm.y - wrist.y) / scale,
            (lm.z - wrist.z) / scale,
        ])
    return np.array(normalized, dtype=np.float64)


def resolve_label(folder_name):
    name = folder_name.strip().upper()
    if name in COMMANDS:
        return name, "command"
    if name == "CA":
        return "Ç", "letter"
    if name in LETTERS:
        return name, "letter"
    return None, None


def rotate_in_plane(pose, degrees):
    theta = math.radians(degrees)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    rot = np.array([[cos_t, -sin_t, 0.0], [sin_t, cos_t, 0.0], [0.0, 0.0, 1.0]])
    return pose @ rot.T


def jitter(pose, sigma):
    return pose + np.random.normal(loc=0.0, scale=sigma, size=pose.shape)


def augment_pose(pose, rotations_deg, noise_copies, noise_sigma):
    variants = [rotate_in_plane(pose, deg) for deg in rotations_deg]
    variants.extend(jitter(pose, noise_sigma) for _ in range(noise_copies))
    return variants


def load_json(path):
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def as_sample_list(value):
    arr = np.array(value)
    if arr.ndim == 2:
        return [arr.tolist()]
    return arr.tolist()


def min_dist_to_pool(pose, existing_raw):
    if not existing_raw:
        return float("inf")
    arr = np.array(existing_raw)
    return float(weighted_distance(pose, arr).min())


def load_manifest(imgs_dir):
    path = os.path.join(imgs_dir, ".processed_manifest.json")
    return load_json(path), path


def save_manifest(path, manifest):
    save_json(path, manifest)


def scan_new_images(imgs_dir, manifest):
    """Retorna dict {(folder, kind): [(fname, fpath), ...]} so com o que ainda nao foi processado ou mudou de mtime."""
    pending = defaultdict(list)
    if not os.path.isdir(imgs_dir):
        return pending
    for folder in sorted(os.listdir(imgs_dir)):
        folder_path = os.path.join(imgs_dir, folder)
        if not os.path.isdir(folder_path):
            continue
        label, kind = resolve_label(folder)
        if label is None:
            continue
        for fname in sorted(os.listdir(folder_path)):
            if not fname.lower().endswith(VALID_EXTENSIONS):
                continue
            fpath = os.path.join(folder_path, fname)
            rel_key = os.path.relpath(fpath, imgs_dir)
            mtime = os.path.getmtime(fpath)
            if manifest.get(rel_key) == mtime:
                continue
            pending[(label, kind)].append((fname, fpath, rel_key, mtime))
    return pending


def process_pending(pending, args, manifest):
    """Extrai landmarks das imagens pendentes, aplica augment/dedupe e devolve
    samples_letters, samples_commands (novos, prontos pra salvar) + atualiza manifest em memoria."""
    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.5,
    )
    landmarker = mp_vision.HandLandmarker.create_from_options(options)

    raw_pool_letters = load_json(RAW_DATA_PATH)
    raw_pool_commands = load_json(RAW_COMMANDS_DATA_PATH)

    samples_letters = defaultdict(list)
    samples_commands = defaultdict(list)
    no_hand = []

    for (label, kind), items in pending.items():
        pool = raw_pool_letters if kind == "letter" else raw_pool_commands
        existing_raw = as_sample_list(pool[label]) if label in pool else []
        target = samples_letters if kind == "letter" else samples_commands

        for fname, fpath, rel_key, mtime in items:
            img_bgr = cv2.imread(fpath)
            if img_bgr is None:
                manifest[rel_key] = mtime
                continue
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
            result = landmarker.detect(mp_image)

            manifest[rel_key] = mtime  # marca como processada de qualquer forma (nao reprocessa mais)

            if not result.hand_landmarks:
                no_hand.append(rel_key)
                continue

            best_idx = 0
            if len(result.handedness) > 1:
                best_idx = max(range(len(result.handedness)), key=lambda i: result.handedness[i][0].score)
            pose = normalize_landmarks(result.hand_landmarks[best_idx])

            candidates = [pose]
            if args.augment:
                candidates.extend(augment_pose(pose, args.rotations, args.aug_noise_copies, args.aug_noise_sigma))

            accepted = []
            for cand in candidates:
                if args.dedupe > 0:
                    d = min_dist_to_pool(cand, existing_raw + accepted)
                    if d < args.dedupe:
                        continue
                accepted.append(cand.tolist())

            if accepted:
                target[label].extend(accepted)
                existing_raw = existing_raw + accepted

    landmarker.close()
    return samples_letters, samples_commands, no_hand


def merge_and_save(samples_letters, samples_commands):
    for samples, output_path, raw_path in (
        (samples_letters, DATA_PATH, RAW_DATA_PATH),
        (samples_commands, COMMANDS_DATA_PATH, RAW_COMMANDS_DATA_PATH),
    ):
        targets = {name: s for name, s in samples.items() if s}
        if not targets:
            continue
        raw_pool = load_json(raw_path)
        output_data = load_json(output_path)
        for name, new_samples in targets.items():
            existing_raw = as_sample_list(raw_pool[name]) if name in raw_pool else []
            combined = existing_raw + [np.round(np.array(s), 5).tolist() for s in new_samples]
            raw_pool[name] = combined
            mean_pose = np.mean(np.array(combined), axis=0)
            output_data[name] = np.round(mean_pose, 5).tolist()
        save_json(raw_path, raw_pool)
        save_json(output_path, output_data)


def load_signatures(path):
    raw = load_json(path)
    sigs = {}
    for name, value in raw.items():
        arr = np.array(value, dtype=np.float64)
        sigs[name] = arr[np.newaxis, :, :] if arr.ndim == 2 else arr
    return sigs


def compute_dashboard_stats():
    """Coesao (media da distancia de cada amostra ate a media da propria classe)
    e Confusao (label mais proximo + distancia) para letras e comandos."""
    stats = {}
    for path, kind in ((DATA_PATH, "letter"), (COMMANDS_DATA_PATH, "command")):
        sigs = load_signatures(path)
        if not sigs:
            continue
        means = {name: poses.mean(axis=0) for name, poses in sigs.items()}
        names = list(means.keys())
        for name in names:
            poses = sigs[name]
            mean_pose = means[name]
            cohesion = float(weighted_distance(mean_pose, poses).mean()) if len(poses) > 1 else 0.0

            other_names = [n for n in names if n != name]
            if other_names:
                other_means = np.stack([means[n] for n in other_names])
                dists = weighted_distance(mean_pose, other_means)
                closest_idx = int(np.argmin(dists))
                closest_name = other_names[closest_idx]
                closest_dist = float(dists[closest_idx])
            else:
                closest_name, closest_dist = "-", float("inf")

            stats[name] = {
                "kind": kind,
                "n": len(poses),
                "cohesion": cohesion,
                "closest_name": closest_name,
                "closest_dist": closest_dist,
            }
    return stats


def bar(value, max_value, width=14):
    if max_value <= 0:
        return " " * width
    filled = int(round(width * min(value, max_value) / max_value))
    return "#" * filled + "." * (width - filled)


def render_dashboard(stats, prev_counts, iteration, last_summary):
    os.system("cls" if os.name == "nt" else "clear")
    print("=" * 72)
    print(" WATCH & TRAIN - LIBRAS  (CTRL+C para sair)")
    print("=" * 72)
    print(f"Rodada #{iteration}   |   {last_summary}")
    print("-" * 72)

    if not stats:
        print("Nenhuma amostra em data.json/commands.json ainda. Tire fotos com")
        print("photo_capture.py e/ou rode batch_train_from_images.py.")
        return

    max_n = max(s["n"] for s in stats.values())

    def print_section(title, kind):
        items = {k: v for k, v in stats.items() if v["kind"] == kind}
        if not items:
            return
        print(f"\n{title}")
        for name in sorted(items.keys()):
            s = items[name]
            delta = s["n"] - prev_counts.get(name, s["n"])
            delta_txt = f"+{delta}" if delta > 0 else ("0" if delta == 0 else str(delta))
            print(
                f"  {name:>3}  [{bar(s['n'], max_n)}] {s['n']:>3} amostras ({delta_txt:>3})  "
                f"coesao={s['cohesion']:.3f}  confunde com '{s['closest_name']}' (dist={s['closest_dist']:.3f})"
            )

    print_section("LETRAS", "letter")
    print_section("COMANDOS", "command")
    print("\n(coesao menor = fotos mais consistentes | dist. de confusao maior = mais seguro)")


def main():
    parser = argparse.ArgumentParser(description="Loop de treino automatico + painel ao vivo, vigiando imgs/.")
    parser.add_argument("--imgs-dir", default=os.path.join(BASE_DIR, "imgs"))
    parser.add_argument("--interval", type=float, default=5.0, help="Segundos entre cada varredura (padrao: 5)")
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--aug-rotations", default="5,10")
    parser.add_argument("--aug-noise-copies", type=int, default=2)
    parser.add_argument("--aug-noise-sigma", type=float, default=0.01)
    parser.add_argument("--dedupe", type=float, default=0.015)
    args = parser.parse_args()

    rot_degs = [float(d.strip()) for d in args.aug_rotations.split(",") if d.strip()]
    args.rotations = []
    for d in rot_degs:
        args.rotations.extend([d, -d])

    if not os.path.isdir(args.imgs_dir):
        print(f"Erro: pasta '{args.imgs_dir}' nao encontrada. Rode setup_imgs_folders.py primeiro.")
        sys.exit(1)

    ensure_model()

    manifest, manifest_path = load_manifest(args.imgs_dir)
    iteration = 0
    prev_counts = {}
    last_summary = "iniciando..."

    try:
        while True:
            iteration += 1
            pending = scan_new_images(args.imgs_dir, manifest)
            n_pending = sum(len(v) for v in pending.values())

            if n_pending > 0:
                samples_letters, samples_commands, no_hand = process_pending(pending, args, manifest)
                merge_and_save(samples_letters, samples_commands)
                save_manifest(manifest_path, manifest)
                accepted = sum(len(v) for v in samples_letters.values()) + sum(len(v) for v in samples_commands.values())
                last_summary = f"{n_pending} foto(s) nova(s) -> {accepted} amostra(s) aceitas, {len(no_hand)} sem mao detectada"
            else:
                last_summary = "nenhuma foto nova desde a ultima rodada"

            stats = compute_dashboard_stats()
            render_dashboard(stats, prev_counts, iteration, last_summary)
            prev_counts = {name: s["n"] for name, s in stats.items()}

            time.sleep(args.interval)

    except KeyboardInterrupt:
        print("\nEncerrado pelo usuario.")


if __name__ == "__main__":
    main()