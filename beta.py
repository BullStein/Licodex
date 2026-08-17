"""
Beta Tester - LIBRAS (versão DEBUG)
------------------------------------
Lê `data.json` (letras, mão esquerda) e `commands.json` (comandos,
mão direita) e classifica as duas mãos por nearest neighbor,
mostrando o ranking de distâncias das duas (não só das letras).

Mostra:
- Esqueleto das duas mãos desenhado
- Ranking das 5 letras mais próximas (mão esquerda), com distância
- Ranking dos comandos mais próximos (mão direita), com distância
- Lista de comandos disponíveis carregados do commands.json
- FPS

Abre em tela cheia, em resolução alta.

Pré-requisitos:
    - data.json (rode capture_signatures.py em modo letra)
    - commands.json (rode capture_signatures.py em modo comando,
      tecla "8"). Se não existir, o painel de comando fica vazio
      com um aviso, mas o resto funciona normalmente.

Como usar:
    python beta_tester.py

Controles:
    - "0" ou ESC para sair
"""

import os
import json
import math
import time
from collections import deque, Counter

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data.json")
COMMANDS_DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "commands.json")

WINDOW_NAME = "Beta Tester - LIBRAS (DEBUG)"
WINDOW_WIDTH, WINDOW_HEIGHT = 1100, 680  # janela de tamanho médio, não fullscreen

TARGET_FPS = 60
MIN_FRAME_INTERVAL = 1.0 / TARGET_FPS

LETTER_HAND_LABEL = "Left"
COMMAND_HAND_LABEL = "Right"

WRIST = 0
MIDDLE_MCP = 9

LETTER_CONFIDENCE_THRESHOLD = 0.55
COMMAND_CONFIDENCE_THRESHOLD = 0.6

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print("Baixando modelo hand_landmarker.task (primeira execução)...")
        import urllib.request
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Download concluído:", MODEL_PATH)


def load_signatures(path, label):
    if not os.path.exists(path):
        print(f"Aviso: {os.path.basename(path)} não encontrado ({label} ficará sem reconhecer nada).")
        return {}
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    signatures = {}
    for name, value in raw.items():
        arr = np.array(value, dtype=np.float64)
        signatures[name] = arr[np.newaxis, :, :] if arr.ndim == 2 else arr
    return signatures


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
    return np.array(normalized)


def classify_ranked(normalized_pose, signatures):
    """Retorna [(nome, distancia), ...] ordenado do mais parecido pro menos."""
    results = []
    for name, poses in signatures.items():
        dists = np.linalg.norm(poses - normalized_pose[np.newaxis, :, :], axis=2)
        best = dists.mean(axis=1).min()
        results.append((name, float(best)))
    results.sort(key=lambda x: x[1])
    return results


def draw_landmarks(frame, landmarks, color=(0, 200, 0)):
    h, w, _ = frame.shape
    points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for start, end in HAND_CONNECTIONS:
        cv2.line(frame, points[start], points[end], (255, 255, 255), 2)
    for x, y in points:
        cv2.circle(frame, (x, y), 4, color, -1)


def smooth_prediction(buffer, new_value, maxlen=8):
    buffer.append(new_value)
    if len(buffer) > maxlen:
        buffer.popleft()
    most_common, _ = Counter(buffer).most_common(1)[0]
    return most_common


def draw_ranking_panel(frame, top_left, title, ranking, threshold, max_items=5):
    x, y = top_left
    panel_h = 30 + max_items * 22 + 25
    cv2.rectangle(frame, (x, y), (x + 220, y + panel_h), (30, 30, 30), -1)
    cv2.putText(frame, title, (x + 10, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    for i, (name, d) in enumerate(ranking[:max_items]):
        color = (0, 255, 0) if i == 0 and d <= threshold else (200, 200, 200)
        cv2.putText(frame, f"{name}: {d:.3f}", (x + 10, y + 45 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    cv2.putText(frame, f"Limiar: {threshold}", (x + 10, y + panel_h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1, cv2.LINE_AA)


def main():
    ensure_model()
    letter_signatures = load_signatures(DATA_PATH, "letras")
    command_signatures = load_signatures(COMMANDS_DATA_PATH, "comandos")
    print(f"data.json: {len(letter_signatures)} letras -> {sorted(letter_signatures.keys())}")
    print(f"commands.json: {len(command_signatures)} comandos -> {sorted(command_signatures.keys())}")

    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.6,
    )
    landmarker = mp_vision.HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Erro: não foi possível acessar a webcam.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, WINDOW_WIDTH, WINDOW_HEIGHT)

    letter_buffer = deque(maxlen=8)
    frame_timestamp_ms = 0
    start_time = time.time()
    prev_time = time.time()

    print("Beta tester rodando. Aperte '0' ou ESC para sair.")

    while cap.isOpened():
        loop_start = time.time()

        success, frame = cap.read()
        if not success:
            continue

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        frame_timestamp_ms = int((time.time() - start_time) * 1000)
        result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

        letter_ranking = []
        command_ranking = []
        letter = "?"

        if result.hand_landmarks:
            for hand_landmarks, handedness in zip(result.hand_landmarks, result.handedness):
                label = handedness[0].category_name

                if label == LETTER_HAND_LABEL:
                    draw_landmarks(frame, hand_landmarks, color=(0, 200, 0))
                    normalized_pose = normalize_landmarks(hand_landmarks)
                    letter_ranking = classify_ranked(normalized_pose, letter_signatures)
                    raw_letter = letter_ranking[0][0] if (letter_ranking and letter_ranking[0][1] <= LETTER_CONFIDENCE_THRESHOLD) else "?"
                    letter = smooth_prediction(letter_buffer, raw_letter)

                elif label == COMMAND_HAND_LABEL:
                    draw_landmarks(frame, hand_landmarks, color=(0, 140, 255))
                    normalized_pose = normalize_landmarks(hand_landmarks)
                    command_ranking = classify_ranked(normalized_pose, command_signatures)

        now = time.time()
        fps = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now

        h, w, _ = frame.shape

        # Overlay da letra
        cv2.rectangle(frame, (0, 0), (260, 100), (245, 117, 16), -1)
        cv2.putText(frame, "Letra:", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, letter, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(frame, f"FPS: {fps:.1f}", (150, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # Ranking de letras (canto superior direito)
        draw_ranking_panel(frame, (w - 220, 0), "Letras (dist):", letter_ranking, LETTER_CONFIDENCE_THRESHOLD)

        # Ranking de comandos (abaixo do painel de letras, mesmo canto)
        draw_ranking_panel(frame, (w - 220, 190), "Comandos (dist):", command_ranking, COMMAND_CONFIDENCE_THRESHOLD, max_items=len(command_signatures) or 1)

        cv2.imshow(WINDOW_NAME, frame)

        key = cv2.waitKey(5) & 0xFF
        if key == ord("0") or key == 27:
            break

        # --- Limitador de FPS: garante um ritmo estável (não passa de TARGET_FPS) ---
        elapsed = time.time() - loop_start
        if elapsed < MIN_FRAME_INTERVAL:
            time.sleep(MIN_FRAME_INTERVAL - elapsed)

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()


if __name__ == "__main__":
    main()