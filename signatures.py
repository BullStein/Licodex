"""
Captura de Assinaturas de Landmarks - LIBRAS
---------------------------------------------
Gera/atualiza quatro arquivos:
    - data.json          -> assinaturas MÉDIAS das LETRAS (mão de LIBRAS), usado pelos outros scripts
    - data_raw.json       -> POOL de todas as amostras cruas de letras já capturadas (cresce a cada sessão)
    - commands.json       -> assinaturas MÉDIAS/POOL dos COMANDOS (mão de comando)
    - commands_raw.json   -> POOL de todas as amostras cruas de comandos já capturadas

A assinatura é a posição dos 21 landmarks da mão (x, y, z),
NORMALIZADOS em relação ao pulso e escalados pelo tamanho da mão —
assim funciona independente de distância/posição da mão na câmera.

MODOS (tecla "8" alterna entre eles):
    - Modo LETRA (padrão): aperte a tecla da letra (A-Z, Ç) pra
      capturar uma amostra daquela letra.
    - Modo COMANDO: em vez de teclar uma letra, você navega por uma
      LISTA de comandos pré-definidos (COMMANDS, lá embaixo no
      código) e captura amostra do que estiver selecionado:
          "4" = comando anterior da lista
          "6" = próximo comando da lista
          "5" ou ESPAÇO = capturar amostra do comando selecionado

Controles gerais (funcionam nos dois modos):
    - "8" = alternar entre modo LETRA e modo COMANDO
    - "1" = salvar a MÉDIA (letras -> data.json | comandos -> commands.json),
            calculada em cima de TODO o pool acumulado (sessões antigas + esta)
    - "2" = salvar TODAS as amostras do pool acumulado, sem tirar média
            (letras -> data.json | comandos -> commands.json)
    - "3" = resetar as amostras do alvo atual NESTA SESSÃO (não mexe no
            pool já salvo em disco — só limpa o que você capturou agora)
    - "0" (ou ESC) = sair sem salvar

IMPORTANTE - como funciona o acúmulo entre execuções:
    Toda vez que você aperta "1" ou "2", as amostras capturadas nesta
    sessão são ACRESCENTADAS ao pool bruto (data_raw.json / commands_raw.json),
    que nunca é sobrescrito — só cresce. O arquivo final (data.json /
    commands.json) é sempre recalculado a partir do pool completo (tudo
    que você já capturou em qualquer execução anterior + agora).
    Ou seja: rode o script quantas vezes quiser, capturando só algumas
    letras/comandos de cada vez, que a base vai ficando cada vez maior
    e mais robusta — igual já acontecia com os comandos, agora vale
    pras letras também.

Como usar:
    python capture_signatures.py
"""

import os
import json
import math
import time
from collections import defaultdict

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# ---------------------------------------------------------------------
# Modelo (reaproveita o mesmo hand_landmarker.task dos outros scripts)
# ---------------------------------------------------------------------
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data.json")
RAW_DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_raw.json")
COMMANDS_DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "commands.json")
RAW_COMMANDS_DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "commands_raw.json")

WINDOW_NAME = "Captura de Assinaturas - LIBRAS"
WINDOW_WIDTH, WINDOW_HEIGHT = 1000, 620  # janela de tamanho médio, não fullscreen

TARGET_FPS = 60          # teto de FPS
MIN_TARGET_FPS = 30      # piso aceitável (informativo, pra referência)
MIN_FRAME_INTERVAL = 1.0 / TARGET_FPS

WRIST = 0
MIDDLE_MCP = 9  # usado como referência de "tamanho da mão"

LETTERS = [chr(c) for c in range(ord("A"), ord("Z") + 1)] + ["Ç"]

# Lista de comandos disponíveis para captura no modo COMANDO.
# Edite essa lista se quiser adicionar/remover comandos.
COMMANDS = ["BACKSPACE", "SPACE", "CLEAR"]

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # polegar
    (0, 5), (5, 6), (6, 7), (7, 8),          # indicador
    (5, 9), (9, 10), (10, 11), (11, 12),     # médio
    (9, 13), (13, 14), (14, 15), (15, 16),   # anelar
    (13, 17), (17, 18), (18, 19), (19, 20),  # mínimo
    (0, 17),                                  # base da palma
]


def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print("Baixando modelo hand_landmarker.task (primeira execução)...")
        import urllib.request
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Download concluído:", MODEL_PATH)


def normalize_landmarks(landmarks):
    """
    Retorna lista de 21 pontos [x, y, z], normalizados:
    - Origem deslocada para o pulso (landmark 0)
    - Escala dividida pela distância pulso -> base do dedo médio
    """
    wrist = landmarks[WRIST]
    ref = landmarks[MIDDLE_MCP]
    scale = math.sqrt(
        (ref.x - wrist.x) ** 2 + (ref.y - wrist.y) ** 2 + (ref.z - wrist.z) ** 2
    )
    scale = scale if scale > 1e-6 else 1e-6

    normalized = []
    for lm in landmarks:
        normalized.append([
            round((lm.x - wrist.x) / scale, 5),
            round((lm.y - wrist.y) / scale, 5),
            round((lm.z - wrist.z) / scale, 5),
        ])
    return normalized


def draw_landmarks(frame, landmarks):
    h, w, _ = frame.shape
    points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for start, end in HAND_CONNECTIONS:
        cv2.line(frame, points[start], points[end], (255, 255, 255), 2)
    for x, y in points:
        cv2.circle(frame, (x, y), 4, (0, 200, 0), -1)


def draw_transparent_box(frame, top_left, bottom_right, color=(0, 0, 0), alpha=0.55):
    overlay = frame.copy()
    cv2.rectangle(overlay, top_left, bottom_right, color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def load_json(path):
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"Aviso: não foi possível ler {os.path.basename(path)} existente ({e}). "
              "Um novo arquivo será criado a partir do zero.")
        return {}


def as_sample_list(value):
    """
    Normaliza uma entrada de arquivo (que pode ser uma pose única
    (21,3) ou uma lista de poses (n,21,3)) para SEMPRE uma lista de
    poses (lista de listas de [x,y,z]), pra poder concatenar.
    """
    arr = np.array(value)
    if arr.ndim == 2:  # pose única -> vira lista com 1 amostra
        return [arr.tolist()]
    return arr.tolist()


def save_data(samples, output_path, raw_path, average=True):
    """
    Acrescenta as amostras capturadas nesta sessão ao POOL bruto
    (raw_path), que nunca é sobrescrito — só cresce entre execuções.
    Depois recalcula o arquivo final (output_path) a partir do pool
    completo (amostras antigas + desta sessão).
    """
    targets_with_samples = {name: s for name, s in samples.items() if s}
    if not targets_with_samples:
        print("Nenhuma amostra capturada ainda, nada foi salvo.")
        return

    raw_pool = load_json(raw_path)  # nome -> lista de poses cruas já acumuladas
    output_data = load_json(output_path)  # arquivo final (médias ou pool completo)

    updated_targets = []
    for name, new_samples in targets_with_samples.items():
        existing_raw = as_sample_list(raw_pool[name]) if name in raw_pool else []
        combined_raw = existing_raw + new_samples
        raw_pool[name] = combined_raw

        if average:
            arr = np.array(combined_raw)
            mean_pose = np.mean(arr, axis=0)
            output_data[name] = np.round(mean_pose, 5).tolist()
        else:
            output_data[name] = combined_raw

        updated_targets.append(name)

    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(raw_pool, f, ensure_ascii=False, indent=2)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"{os.path.basename(output_path)} atualizado em: {output_path}")
    for name in sorted(updated_targets):
        print(f"  {name}: {len(raw_pool[name])} amostras no pool total (pool salvo em {os.path.basename(raw_path)})")
    print(f"  Total de alvos no arquivo final: {len(output_data)} -> {sorted(output_data.keys())}")


def main():
    ensure_model()

    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=1,  # captura de propósito só com 1 mão por vez (mais limpo)
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.6,
    )
    landmarker = mp_vision.HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Erro: não foi possível acessar a webcam.")
        return

    # Resolução moderada (720p): suficiente pra qualidade e ajuda a
    # manter o FPS estável, sem pesar demais no processamento.
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, WINDOW_WIDTH, WINDOW_HEIGHT)

    # --- Estado de modo ---
    mode = "letter"  # ou "command"
    current_letter = None
    current_command_index = 0

    # amostras acumuladas nesta sessão (só o que foi capturado agora;
    # o pool completo entre sessões vive em data_raw.json/commands_raw.json)
    samples_letters = defaultdict(list)
    samples_commands = defaultdict(list)

    frame_timestamp_ms = 0
    start_time = time.time()
    last_landmarks = None

    print("Captura iniciada. Modo atual: LETRA")
    print("Letra: tecle A-Z para capturar | Comando: '4'/'6' navega, '5'/ESPAÇO captura")
    print("'8' = alternar modo | '1' = salvar médias | '2' = salvar tudo | '3' = resetar alvo atual (sessão) | '0'/ESC = sair")

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

        last_landmarks = None
        if result.hand_landmarks:
            hand_landmarks = result.hand_landmarks[0]
            draw_landmarks(frame, hand_landmarks)
            last_landmarks = hand_landmarks

        # --- HUD ---
        h, w, _ = frame.shape
        draw_transparent_box(frame, (0, 0), (460, 90), color=(0, 0, 0), alpha=0.55)

        if mode == "letter":
            n_samples = len(samples_letters[current_letter]) if current_letter else 0
            info = f"[MODO LETRA] Letra atual: {current_letter or '-'} | Amostras (sessao): {n_samples}"
            controls = "Letra=captura | 8=modo comando | 1/2=salvar | 3=resetar | 0=sair"
        else:
            cmd_name = COMMANDS[current_command_index]
            n_samples = len(samples_commands[cmd_name])
            info = f"[MODO COMANDO] Comando ({current_command_index + 1}/{len(COMMANDS)}): {cmd_name} | Amostras (sessao): {n_samples}"
            controls = "4/6=navega | 5/ESPACO=captura | 8=modo letra | 1/2=salvar | 3=resetar | 0=sair"

        cv2.putText(frame, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, controls, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        # --- Painel com a lista completa de comandos (modo comando) ---
        if mode == "command":
            list_top = 100
            list_height = 30 + len(COMMANDS) * 26
            draw_transparent_box(frame, (0, list_top), (260, list_top + list_height), color=(0, 0, 0), alpha=0.55)
            cv2.putText(frame, "Comandos disponiveis:", (10, list_top + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
            for i, cmd_option in enumerate(COMMANDS):
                is_selected = i == current_command_index
                color = (0, 255, 0) if is_selected else (200, 200, 200)
                prefix = "> " if is_selected else "   "
                n = len(samples_commands[cmd_option])
                cv2.putText(frame, f"{prefix}{cmd_option} ({n})", (10, list_top + 48 + i * 26),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2 if is_selected else 1, cv2.LINE_AA)

        cv2.imshow(WINDOW_NAME, frame)

        key = cv2.waitKey(5) & 0xFF

        if key == ord("0") or key == 27:  # "0" ou ESC
            break

        elif key == ord("8"):
            mode = "command" if mode == "letter" else "letter"
            print(f"Modo alternado para: {'COMANDO' if mode == 'command' else 'LETRA'}")

        elif key == ord("1"):
            if mode == "letter":
                save_data(samples_letters, DATA_PATH, RAW_DATA_PATH, average=True)
            else:
                save_data(samples_commands, COMMANDS_DATA_PATH, RAW_COMMANDS_DATA_PATH, average=True)

        elif key == ord("2"):
            if mode == "letter":
                save_data(samples_letters, DATA_PATH, RAW_DATA_PATH, average=False)
            else:
                save_data(samples_commands, COMMANDS_DATA_PATH, RAW_COMMANDS_DATA_PATH, average=False)

        elif key == ord("3"):
            if mode == "letter" and current_letter:
                samples_letters[current_letter] = []
                print(f"Amostras da letra {current_letter} resetadas (só as desta sessão).")
            elif mode == "command":
                cmd_name = COMMANDS[current_command_index]
                samples_commands[cmd_name] = []
                print(f"Amostras do comando {cmd_name} resetadas (só as desta sessão).")

        elif mode == "command" and key == ord("4"):
            current_command_index = (current_command_index - 1) % len(COMMANDS)

        elif mode == "command" and key == ord("6"):
            current_command_index = (current_command_index + 1) % len(COMMANDS)

        elif mode == "command" and key in (ord("5"), 32):  # "5" ou barra de espaço
            cmd_name = COMMANDS[current_command_index]
            if last_landmarks is not None:
                normalized = normalize_landmarks(last_landmarks)
                samples_commands[cmd_name].append(normalized)
                print(f"Amostra capturada para comando '{cmd_name}' (total nesta sessão: {len(samples_commands[cmd_name])})")
            else:
                print(f"Comando '{cmd_name}' selecionado, mas nenhuma mão detectada no frame.")

        elif mode == "letter":
            ch = chr(key).upper() if 32 <= key <= 126 else ""
            if ch in LETTERS:
                current_letter = ch
                if last_landmarks is not None:
                    normalized = normalize_landmarks(last_landmarks)
                    samples_letters[ch].append(normalized)
                    print(f"Amostra capturada para '{ch}' (total nesta sessão: {len(samples_letters[ch])})")
                else:
                    print(f"Letra '{ch}' selecionada, mas nenhuma mão detectada no frame.")

        # --- Limitador de FPS: garante um ritmo estável (não passa de TARGET_FPS) ---
        elapsed = time.time() - loop_start
        if elapsed < MIN_FRAME_INTERVAL:
            time.sleep(MIN_FRAME_INTERVAL - elapsed)

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()


if __name__ == "__main__":
    main()