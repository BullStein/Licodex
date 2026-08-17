"""
Tradutor LIBRAS -> Letras (versão FINAL, com classificação reforçada)
------------------------------------------------------------------------
Lê `data.json` (letras, mão esquerda) e `commands.json` (comandos,
mão direita), ambos gerados pelo capture_signatures.py (ou pelo
batch_train_from_images.py), e classifica as duas mãos usando
`matching_improved.py`:
    - distância ponderada por landmark (pontas dos dedos pesam mais
      que o pulso/base da palma)
    - voto k-NN por classe (média das K amostras mais próximas, não
      só a mais próxima)
    - margem de confiança (só aceita o resultado se o 2º colocado
      estiver claramente mais longe que o 1º, evitando trocas em
      poses ambíguas)

Essa é a versão "de produção": sem esqueleto desenhado, sem ranking
de debug, sem prints. Mostra a letra detectada, o histórico de
letras/comandos e a lista de comandos disponíveis (pra você não
precisar decorar o que cada gesto faz), num painel preto
semitransparente. Abre em tela cheia, em resolução alta.

Divisão das mãos:
    - Mão ESQUERDA -> letras (data.json)
    - Mão DIREITA  -> comandos (commands.json)

Nota sobre espelhamento:
    O frame é espelhado (cv2.flip) pra parecer um espelho na tela.
    Isso pode inverter o rótulo Left/Right do MediaPipe. Se a mão de
    comando ficar "trocada" com a de letras no seu setup, troque
    LETTER_HAND_LABEL / COMMAND_HAND_LABEL logo abaixo.

Pré-requisitos:
    - Ter rodado capture_signatures.py e gerado data.json (letras)
    - Ter rodado capture_signatures.py em modo comando ("8") e
      gerado commands.json (opcional: se não existir, a mão de
      comando fica simplesmente desativada e um aviso é mostrado)
    - matching_improved.py na mesma pasta

Como usar:
    python translator.py

Controles (teclado, além dos gestos da mão direita):
    - "0" ou ESC para sair
    - "9" para limpar o histórico manualmente
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

from matching_improved import classify_ranked_weighted, is_confident

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data.json")
COMMANDS_DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "commands.json")

WINDOW_NAME = "Tradutor LIBRAS"
WINDOW_WIDTH, WINDOW_HEIGHT = 1000, 620  # janela de tamanho médio, não fullscreen

TARGET_FPS = 60
MIN_FRAME_INTERVAL = 1.0 / TARGET_FPS

# Qual rótulo de mão faz o quê (ver nota sobre espelhamento acima)
LETTER_HAND_LABEL = "Left"
COMMAND_HAND_LABEL = "Right"

WRIST = 0
MIDDLE_MCP = 9

# Limiar de distância (mesmo papel de antes) + margem mínima entre o
# 1º e o 2º colocado no ranking, pra só aceitar o resultado quando
# ele não está "empatando" com outra classe parecida.
LETTER_CONFIDENCE_THRESHOLD = 0.55
LETTER_MIN_MARGIN = 0.04

COMMAND_CONFIDENCE_THRESHOLD = 0.6  # comandos toleram um pouco mais de variação
COMMAND_MIN_MARGIN = 0.04

# Quantos vizinhos por classe entram no voto k-NN da classificação
KNN_K = 3

# Quantos caracteres manter no histórico exibido na tela
HISTORY_MAX_LEN = 30


def ensure_model():
    if not os.path.exists(MODEL_PATH):
        import urllib.request
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)


def load_signatures(path, label):
    """
    Carrega um arquivo de assinaturas (data.json ou commands.json) no
    formato {nome: [[x,y,z]*21]} ou {nome: [[[x,y,z]*21], ...]}.
    Retorna dict nome -> ndarray de shape (n_amostras, 21, 3).
    Se o arquivo não existir, retorna {} e avisa no console (não
    quebra o programa: aquela mão simplesmente fica sem reconhecer
    nada até o arquivo existir).
    """
    if not os.path.exists(path):
        print(f"Aviso: {os.path.basename(path)} não encontrado ({label} desativado até você gerá-lo "
              f"com capture_signatures.py).")
        return {}

    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    signatures = {}
    for name, value in raw.items():
        arr = np.array(value, dtype=np.float64)
        if arr.ndim == 2:  # uma única pose: (21, 3)
            signatures[name] = arr[np.newaxis, :, :]
        else:  # múltiplas poses: (n, 21, 3)
            signatures[name] = arr
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


def classify_hand(normalized_pose, signatures, threshold, min_margin):
    """
    Substitui o antigo classify_nearest: agora usa distância
    ponderada por landmark + voto k-NN (matching_improved.py) e só
    aceita o resultado se ele for confiante o bastante (limiar de
    distância + margem clara pro 2º colocado).
    Retorna o nome do alvo reconhecido, ou None.
    """
    if not signatures:
        return None
    ranking = classify_ranked_weighted(normalized_pose, signatures, k=KNN_K)
    if not is_confident(ranking, threshold, min_margin):
        return None
    return ranking[0][0]


def smooth_prediction(buffer, new_value, maxlen=8):
    buffer.append(new_value)
    if len(buffer) > maxlen:
        buffer.popleft()
    most_common, _ = Counter(buffer).most_common(1)[0]
    return most_common


def draw_transparent_box(frame, top_left, bottom_right, color=(0, 0, 0), alpha=0.55):
    overlay = frame.copy()
    cv2.rectangle(overlay, top_left, bottom_right, color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def apply_command(command, letter_history):
    if command == "BACKSPACE":
        return letter_history[:-1]
    if command == "SPACE":
        new_history = letter_history + " "
        return new_history[-HISTORY_MAX_LEN:]
    if command == "CLEAR":
        return ""
    return letter_history


def main():
    ensure_model()
    letter_signatures = load_signatures(DATA_PATH, "letras")
    command_signatures = load_signatures(COMMANDS_DATA_PATH, "comandos")
    available_commands = sorted(command_signatures.keys())

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

    # Resolução moderada (720p): boa qualidade sem pesar demais no FPS
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, WINDOW_WIDTH, WINDOW_HEIGHT)

    letter_buffer = deque(maxlen=8)
    command_buffer = deque(maxlen=6)
    frame_timestamp_ms = 0
    start_time = time.time()

    letter_history = ""
    last_committed_letter = None
    last_committed_command = None
    last_command_seen = "-"

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

        letter = "-"
        raw_command = None

        if result.hand_landmarks:
            for hand_landmarks, handedness in zip(result.hand_landmarks, result.handedness):
                label = handedness[0].category_name

                if label == LETTER_HAND_LABEL:
                    normalized_pose = normalize_landmarks(hand_landmarks)
                    raw_letter = classify_hand(
                        normalized_pose, letter_signatures,
                        LETTER_CONFIDENCE_THRESHOLD, LETTER_MIN_MARGIN,
                    )
                    letter = smooth_prediction(letter_buffer, raw_letter or "?")

                elif label == COMMAND_HAND_LABEL:
                    normalized_pose = normalize_landmarks(hand_landmarks)
                    raw_command = classify_hand(
                        normalized_pose, command_signatures,
                        COMMAND_CONFIDENCE_THRESHOLD, COMMAND_MIN_MARGIN,
                    )

        # --- Letra: só adiciona ao histórico quando MUDA ---
        if letter not in ("-", "?") and letter != last_committed_letter:
            letter_history += letter
            letter_history = letter_history[-HISTORY_MAX_LEN:]
            last_committed_letter = letter
        elif letter in ("-", "?"):
            last_committed_letter = None

        # --- Comando: suaviza e só dispara quando MUDA ---
        command = smooth_prediction(command_buffer, raw_command if raw_command else "NONE")
        command = None if command == "NONE" else command
        if command:
            last_command_seen = command
        if command is not None and command != last_committed_command:
            letter_history = apply_command(command, letter_history)
            last_committed_command = command
        elif command is None:
            last_committed_command = None

        # --- Overlays ---
        h, w, _ = frame.shape

        # Painel da letra atual
        draw_transparent_box(frame, (0, 0), (200, 100), color=(0, 0, 0), alpha=0.55)
        cv2.putText(frame, "Letra:", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, letter, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (255, 255, 255), 3, cv2.LINE_AA)

        # Painel do comando (mão direita) + legenda dos comandos disponíveis
        legend_height = 60 + len(available_commands) * 22
        draw_transparent_box(frame, (w - 240, 0), (w, legend_height), color=(0, 0, 0), alpha=0.55)
        cv2.putText(frame, "Comando atual:", (w - 230, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(frame, last_command_seen, (w - 230, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        if available_commands:
            cv2.putText(frame, "Disponiveis:", (w - 230, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
            for i, cmd_name in enumerate(available_commands):
                cv2.putText(frame, f"- {cmd_name}", (w - 230, 98 + i * 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
        else:
            cv2.putText(frame, "(commands.json nao encontrado)", (w - 230, 78),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1, cv2.LINE_AA)

        # Painel de histórico, embaixo
        history_box_top = h - 60
        draw_transparent_box(frame, (0, history_box_top), (w, h), color=(0, 0, 0), alpha=0.55)
        cv2.putText(frame, "Historico:", (10, history_box_top + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
        history_text = letter_history if letter_history else "-"
        cv2.putText(frame, history_text, (10, history_box_top + 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow(WINDOW_NAME, frame)

        key = cv2.waitKey(5) & 0xFF
        if key == ord("0") or key == 27:
            break
        elif key == ord("9"):
            letter_history = ""

        # --- Limitador de FPS: garante um ritmo estável (não passa de TARGET_FPS) ---
        elapsed = time.time() - loop_start
        if elapsed < MIN_FRAME_INTERVAL:
            time.sleep(MIN_FRAME_INTERVAL - elapsed)

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()


if __name__ == "__main__":
    main()