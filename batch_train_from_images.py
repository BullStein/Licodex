"""
Treino Autonomo a partir de Imagens - LIBRAS
----------------------------------------------
Processa TODAS as imagens dentro de uma pasta (padrao: "imgs/"),
sem precisar de webcam nem de teclas. Estrutura esperada: uma
subpasta por alvo (letra ou comando):

    imgs/
      A/
        foto1.jpg
        foto2.png
      B/
        ...
      BACKSPACE/
        ...
      SPACE/
      CLEAR/

O nome da subpasta = o nome do alvo. Se bater com um item de
COMMANDS (ver abaixo), vai para commands.json/commands_raw.json.
Caso contrario (letra A-Z ou CA), vai para data.json/data_raw.json.

Para cada imagem:
    1. Detecta a mao (MediaPipe Hand Landmarker, modo IMAGEM).
       Se mais de uma mao for encontrada, usa a de maior confianca.
    2. Normaliza os landmarks (mesma tecnica dos outros scripts:
       origem no pulso, escala pela distancia pulso -> base do dedo
       medio).
    3. (Opcional, --augment) gera variacoes sinteticas daquela pose
       -- pequenas rotacoes no plano + ruido gaussiano leve -- para
       aumentar a robustez do pool sem precisar de mais fotos.
    4. (Opcional, --dedupe DIST) ignora poses quase identicas a
       alguma ja existente no pool daquele alvo, evitando inflar o
       arquivo com amostras redundantes.
    5. Acrescenta ao pool bruto (data_raw.json / commands_raw.json,
       que nunca eh sobrescrito, so cresce) e recalcula a media
       (data.json / commands.json) -- mesma logica do
       capture_signatures.py.

Imagens sem mao detectavel sao puladas e reportadas no final; o
processo nao trava.

Como usar:
    python batch_train_from_images.py
    python batch_train_from_images.py --imgs-dir outra_pasta
    python batch_train_from_images.py --augment
    python batch_train_from_images.py --dedupe 0.02
    python batch_train_from_images.py --dry-run
"""

import os
import sys
import json
import math
import argparse
import random
from collections import defaultdict

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

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

LETTERS = {chr(c) for c in range(ord("A"), ord("Z") + 1)} | {"Ç", "CA"}  # "CA" = pasta sem cedilha
COMMANDS = ["BACKSPACE", "SPACE", "CLEAR"]  # deve bater com capture_signatures.py

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
    """Mapeia o nome da subpasta para um alvo valido, ou None se nao reconhecido."""
    name = folder_name.strip().upper()
    if name in COMMANDS:
        return name, "command"
    if name == "CA":
        return "Ç", "letter"
    if name in LETTERS:
        return name, "letter"
    return None, None


def rotate_in_plane(pose, degrees):
    """Roda a pose (ja normalizada, origem no pulso) em torno do eixo Z."""
    theta = math.radians(degrees)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    rot = np.array([
        [cos_t, -sin_t, 0.0],
        [sin_t, cos_t, 0.0],
        [0.0, 0.0, 1.0],
    ])
    return pose @ rot.T


def jitter(pose, sigma):
    noise = np.random.normal(loc=0.0, scale=sigma, size=pose.shape)
    return pose + noise


def augment_pose(pose, rotations_deg, noise_copies, noise_sigma):
    variants = []
    for deg in rotations_deg:
        variants.append(rotate_in_plane(pose, deg))
    for _ in range(noise_copies):
        variants.append(jitter(pose, noise_sigma))
    return variants


def load_json(path):
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"Aviso: nao foi possivel ler {os.path.basename(path)} ({e}). Comecando do zero.")
        return {}


def as_sample_list(value):
    arr = np.array(value)
    if arr.ndim == 2:
        return [arr.tolist()]
    return arr.tolist()


def min_dist_to_pool(pose, existing_raw):
    if not existing_raw:
        return float("inf")
    arr = np.array(existing_raw)  # (n, 21, 3)
    dists = np.linalg.norm(arr - pose[np.newaxis, :, :], axis=2).mean(axis=1)
    return float(dists.min())


def save_pool(samples_by_label, output_path, raw_path, dry_run=False):
    targets = {name: s for name, s in samples_by_label.items() if s}
    if not targets:
        print(f"  Nenhuma amostra nova para {os.path.basename(output_path)}.")
        return

    raw_pool = load_json(raw_path)
    output_data = load_json(output_path)

    for name, new_samples in targets.items():
        existing_raw = as_sample_list(raw_pool[name]) if name in raw_pool else []
        combined_raw = existing_raw + [np.round(s, 5).tolist() for s in new_samples]
        raw_pool[name] = combined_raw
        mean_pose = np.mean(np.array(combined_raw), axis=0)
        output_data[name] = np.round(mean_pose, 5).tolist()

    if dry_run:
        print(f"  [DRY-RUN] Nao gravou nada em disco. Alvos que seriam atualizados: {sorted(targets.keys())}")
        return

    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(raw_pool, f, ensure_ascii=False, indent=2)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"  {os.path.basename(output_path)} atualizado.")
    for name in sorted(targets.keys()):
        print(f"    {name}: {len(raw_pool[name])} amostras no pool total")


def main():
    parser = argparse.ArgumentParser(description="Treino autonomo LIBRAS a partir de pasta de imagens.")
    parser.add_argument("--imgs-dir", default=os.path.join(BASE_DIR, "imgs"),
                         help="Pasta com subpastas por alvo (padrao: ./imgs)")
    parser.add_argument("--min-detection-confidence", type=float, default=0.5)
    parser.add_argument("--augment", action="store_true",
                         help="Ativa augmentation (rotacoes +/-5 e +/-10 graus + 2 copias com ruido leve)")
    parser.add_argument("--aug-rotations", default="5,10",
                         help="Graus de rotacao para augmentation, aplicados em +/- (padrao: 5,10)")
    parser.add_argument("--aug-noise-copies", type=int, default=2,
                         help="Quantas copias com ruido gaussiano leve gerar por imagem quando --augment (padrao: 2)")
    parser.add_argument("--aug-noise-sigma", type=float, default=0.01)
    parser.add_argument("--dedupe", type=float, default=0.0,
                         help="Se > 0, ignora poses (originais ou aumentadas) com distancia media menor que esse valor a alguma amostra ja existente no pool daquele alvo")
    parser.add_argument("--dry-run", action="store_true",
                         help="Processa e mostra o relatorio, mas nao grava nada em disco")
    args = parser.parse_args()

    if not os.path.isdir(args.imgs_dir):
        print(f"Erro: pasta '{args.imgs_dir}' nao encontrada.")
        sys.exit(1)

    ensure_model()

    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.IMAGE,
        num_hands=2,  # detecta ate 2, mas ficamos com a de maior confianca
        min_hand_detection_confidence=args.min_detection_confidence,
        min_hand_presence_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_detection_confidence,
    )
    landmarker = mp_vision.HandLandmarker.create_from_options(options)

    rotations = []
    if args.augment:
        degs = [float(d.strip()) for d in args.aug_rotations.split(",") if d.strip()]
        for d in degs:
            rotations.extend([d, -d])

    samples_letters = defaultdict(list)
    samples_commands = defaultdict(list)
    raw_pool_letters = load_json(RAW_DATA_PATH)
    raw_pool_commands = load_json(RAW_COMMANDS_DATA_PATH)

    stats_ok = defaultdict(int)
    stats_skipped_no_hand = []
    stats_skipped_dedupe = defaultdict(int)
    stats_unknown_folders = []
    total_images = 0
    total_augmented = 0

    subfolders = sorted(
        d for d in os.listdir(args.imgs_dir)
        if os.path.isdir(os.path.join(args.imgs_dir, d))
    )
    if not subfolders:
        print(f"Nenhuma subpasta encontrada em '{args.imgs_dir}'. Crie uma subpasta por letra/comando.")
        sys.exit(1)

    print(f"Processando '{args.imgs_dir}'... ({len(subfolders)} subpastas encontradas)\n")

    for folder in subfolders:
        label, kind = resolve_label(folder)
        folder_path = os.path.join(args.imgs_dir, folder)
        images = sorted(
            f for f in os.listdir(folder_path)
            if f.lower().endswith(VALID_EXTENSIONS)
        )

        if label is None:
            if images:
                stats_unknown_folders.append(folder)
            continue

        pool = raw_pool_letters if kind == "letter" else raw_pool_commands
        existing_raw = as_sample_list(pool[label]) if label in pool else []
        target_buffer = samples_letters if kind == "letter" else samples_commands

        for fname in images:
            total_images += 1
            fpath = os.path.join(folder_path, fname)
            img_bgr = cv2.imread(fpath)
            if img_bgr is None:
                stats_skipped_no_hand.append(f"{folder}/{fname} (falha ao ler imagem)")
                continue

            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
            result = landmarker.detect(mp_image)

            if not result.hand_landmarks:
                stats_skipped_no_hand.append(f"{folder}/{fname}")
                continue

            # escolhe a mao de maior confianca, se houver mais de uma
            best_idx = 0
            if len(result.handedness) > 1:
                best_idx = max(
                    range(len(result.handedness)),
                    key=lambda i: result.handedness[i][0].score,
                )

            pose = normalize_landmarks(result.hand_landmarks[best_idx])

            candidates = [pose]
            if args.augment:
                candidates.extend(
                    augment_pose(pose, rotations, args.aug_noise_copies, args.aug_noise_sigma)
                )

            accepted_here = []
            for cand in candidates:
                if args.dedupe > 0:
                    d = min_dist_to_pool(cand, existing_raw + accepted_here)
                    if d < args.dedupe:
                        stats_skipped_dedupe[label] += 1
                        continue
                accepted_here.append(cand.tolist())

            if accepted_here:
                target_buffer[label].extend(accepted_here)
                stats_ok[label] += 1
                total_augmented += len(accepted_here) - 1  # a original nao conta como "extra"
                existing_raw = existing_raw + accepted_here  # dedupe considera o que acabou de entrar tambem

    print("--- Relatorio ---")
    print(f"Imagens processadas: {total_images}")
    print(f"Amostras extras geradas por augmentation: {total_augmented}")
    if stats_ok:
        print("Amostras-base aceitas por alvo (imagens com mao detectada, antes de augmentation):")
        for label in sorted(stats_ok.keys()):
            print(f"  {label}: {stats_ok[label]}")
    if stats_skipped_dedupe:
        print("Poses ignoradas por dedupe (quase identicas a alguma ja existente):")
        for label, n in sorted(stats_skipped_dedupe.items()):
            print(f"  {label}: {n}")
    if stats_skipped_no_hand:
        print(f"Imagens sem mao detectada ({len(stats_skipped_no_hand)}):")
        for item in stats_skipped_no_hand[:20]:
            print(f"  - {item}")
        if len(stats_skipped_no_hand) > 20:
            print(f"  ... e mais {len(stats_skipped_no_hand) - 20}")
    if stats_unknown_folders:
        print(f"Subpastas com imagens mas nome nao reconhecido (ignoradas): {stats_unknown_folders}")

    print("\n--- Gravando ---")
    print("Letras:")
    save_pool(samples_letters, DATA_PATH, RAW_DATA_PATH, dry_run=args.dry_run)
    print("Comandos:")
    save_pool(samples_commands, COMMANDS_DATA_PATH, RAW_COMMANDS_DATA_PATH, dry_run=args.dry_run)

    landmarker.close()
    print("\nConcluido.")


if __name__ == "__main__":
    main()