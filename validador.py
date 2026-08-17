"""
Validador de Previsões - LIBRAS (com "foto", tempo de análise e correção)
----------------------------------------------------------------------------
Fluxo:
    1. Você posiciona a mão e aperta ESPAÇO -> tira uma "foto"
       (congela a pose da mão naquele instante exato, letra=esquerda,
       comando=direita, se estiverem visíveis).
    2. Durante um tempo configurável (ANALYSIS_SECONDS, ajustável na
       hora com "+"/"-"), a tela vai revelando o RANKING de
       candidatos mais próximos daquela pose (usando distância
       ponderada por landmark + voto k-NN, via matching_improved.py).
    3. Quando o tempo acaba, aparece a RESPOSTA FINAL. Aí você confirma:
           "S" = acertou -> essa pose vira amostra nova da classe
                 prevista, entra no pool acumulado e a média é
                 recalculada na hora.
           "N" = errou -> NÃO descarta mais! Ele te pergunta qual era
                 a resposta certa:
                   - Letra: tecle a letra certa (A-Z, "9" = Ç), ou
                     ESPAÇO pra pular sem corrigir.
                   - Comando: navegue com "4"/"6" na lista de
                     comandos, "5" ou ENTER pra confirmar, ou ESPAÇO
                     pra pular.
                 A pose congelada é salva na classe CORRETA que você
                 informou (não na que o sistema chutou errado).
    4. Volta pro estado inicial, pronto pra nova foto.

Todo o raciocínio (ranking completo) também é impresso no console.

Pré-requisitos:
    - data.json (rode capture_signatures.py ou batch_train_from_images.py em modo letra)
    - commands.json (idem, modo comando)
    - matching_improved.py na mesma pasta

Como usar:
    python validador.py

Controles:
    - ESPAÇO = tirar a foto e iniciar a análise (só funciona parado, sem foto pendente)
    - "+"/"-" = aumentar/diminuir o tempo de análise (só fora de uma análise em andamento)
    - "S" = confirmar resposta final como correta (reforça a base)
    - "N" = marcar resposta final como errada -> abre correção
    - (durante correção de letra) A-Z / "9"=Ç = informa a letra certa | ESPAÇO = pular
    - (durante correção de comando) "4"/"6" = navega | "5"/ENTER = confirma | ESPAÇO = pular
    - "0" ou ESC = sair
"""

import os
import json
import math
import time

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from matching_improved import classify_ranked_weighted

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data.json")
RAW_DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_raw.json")
COMMANDS_DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "commands.json")
RAW_COMMANDS_DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "commands_raw.json")

WINDOW_NAME = "Validador de Previsoes - LIBRAS"
WINDOW_WIDTH, WINDOW_HEIGHT = 1000, 640

TARGET_FPS = 60
MIN_FRAME_INTERVAL = 1.0 / TARGET_FPS

LETTER_HAND_LABEL = "Left"
COMMAND_HAND_LABEL = "Right"

WRIST = 0
MIDDLE_MCP = 9

LETTERS = {chr(c) for c in range(ord("A"), ord("Z") + 1)}
COMMANDS = ["BACKSPACE", "SPACE", "CLEAR"]  # mesma lista do capture_signatures.py

KNN_K = 3  # quantos vizinhos por classe entram no voto (matching_improved)

# --- Tempo de análise (configurável em runtime com "+"/"-") ---
ANALYSIS_SECONDS_DEFAULT = 2.5
ANALYSIS_SECONDS_MIN = 0.5
ANALYSIS_SECONDS_MAX = 6.0
ANALYSIS_SECONDS_STEP = 0.5

MAX_RANKING_SHOWN = 5  # quantos candidatos do ranking aparecem na tela

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]

STATE_IDLE = "idle"
STATE_THINKING = "thinking"
STATE_CONFIRM = "confirm"
STATE_CORRECT_LETTER = "correct_letter"
STATE_CORRECT_COMMAND = "correct_command"


def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print("Baixando modelo hand_landmarker.task (primeira execução)...")
        import urllib.request
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Download concluído:", MODEL_PATH)


def load_json(path):
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


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


def as_sample_list(value):
    arr = np.array(value)
    if arr.ndim == 2:
        return [arr.tolist()]
    return arr.tolist()


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
    """Retorna [(nome, distancia), ...] do MAIS parecido pro MENOS parecido,
    usando distância ponderada por landmark + voto k-NN (matching_improved.py)."""
    return classify_ranked_weighted(normalized_pose, signatures, k=KNN_K)


def draw_landmarks(frame, landmarks, color=(0, 200, 0)):
    h, w, _ = frame.shape
    points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for start, end in HAND_CONNECTIONS:
        cv2.line(frame, points[start], points[end], (255, 255, 255), 2)
    for x, y in points:
        cv2.circle(frame, (x, y), 4, color, -1)


def draw_transparent_box(frame, top_left, bottom_right, color=(0, 0, 0), alpha=0.6):
    overlay = frame.copy()
    cv2.rectangle(overlay, top_left, bottom_right, color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def confirm_sample(name, normalized_pose, output_path, raw_path):
    """
    Salva a pose congelada (a "foto") como amostra nova de `name`,
    acrescenta ao pool bruto persistente e recalcula a média na hora.
    Funciona tanto pra reforçar um acerto quanto pra registrar a
    correção de um erro -- é a mesma operação, só muda o rótulo usado.
    """
    raw_pool = load_json(raw_path)
    output_data = load_json(output_path)

    existing_raw = as_sample_list(raw_pool[name]) if name in raw_pool else []
    new_sample = np.round(normalized_pose, 5).tolist()
    combined_raw = existing_raw + [new_sample]
    raw_pool[name] = combined_raw

    mean_pose = np.mean(np.array(combined_raw), axis=0)
    output_data[name] = np.round(mean_pose, 5).tolist()

    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(raw_pool, f, ensure_ascii=False, indent=2)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    return len(combined_raw)


def print_reasoning(label, ranking):
    print(f"\n--- Caminho de pensamento ({label}) ---")
    if not ranking:
        print("  (mão não detectada no instante da foto)")
        return
    for i, (name, dist) in enumerate(ranking, start=1):
        marcador = " <-- ESCOLHIDO" if i == 1 else ""
        print(f"  {i}. {name}: distância {dist:.4f}{marcador}")


def main():
    ensure_model()
    letter_signatures = load_signatures(DATA_PATH, "letras")
    command_signatures = load_signatures(COMMANDS_DATA_PATH, "comandos")

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

    frame_timestamp_ms = 0
    start_time = time.time()

    analysis_seconds = ANALYSIS_SECONDS_DEFAULT
    state = STATE_IDLE

    # dados da "foto" congelada nesta rodada
    letter_ranking = []   # [(nome, dist), ...] completo
    command_ranking = []
    letter_pose = None    # pose normalizada congelada (pra salvar se confirmar/corrigir)
    command_pose = None
    thinking_start = 0.0

    correct_command_index = 0  # navegação durante a correção de comando

    stats = {"corretas": 0, "erradas": 0, "corrigidas": 0}
    last_feedback = ""
    last_feedback_until = 0

    print("Validador rodando. ESPACO tira a foto e inicia a analise. '+/-' ajusta o tempo.")
    print("'S/N' confirma. Se 'N', ele pergunta a resposta certa pra corrigir a base. '0'/ESC sai.")

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

        live_letter_landmarks = None
        live_command_landmarks = None
        if result.hand_landmarks:
            for hand_landmarks, handedness in zip(result.hand_landmarks, result.handedness):
                label = handedness[0].category_name
                if label == LETTER_HAND_LABEL:
                    live_letter_landmarks = hand_landmarks
                    draw_landmarks(frame, hand_landmarks, color=(0, 200, 0))
                elif label == COMMAND_HAND_LABEL:
                    live_command_landmarks = hand_landmarks
                    draw_landmarks(frame, hand_landmarks, color=(0, 140, 255))

        h, w, _ = frame.shape

        # --- HUD superior: estado + tempo de analise configurável ---
        draw_transparent_box(frame, (0, 0), (340, 55), color=(0, 0, 0), alpha=0.6)
        estado_txt = {
            STATE_IDLE: "Pronto (ESPACO = foto)",
            STATE_THINKING: "Analisando...",
            STATE_CONFIRM: "Aguardando S/N",
            STATE_CORRECT_LETTER: "Corrigindo letra",
            STATE_CORRECT_COMMAND: "Corrigindo comando",
        }[state]
        cv2.putText(frame, f"Estado: {estado_txt}", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Tempo de analise: {analysis_seconds:.1f}s  (+/- ajusta)", (10, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

        # =========================== ESTADO: THINKING ===========================
        if state == STATE_THINKING:
            elapsed = time.time() - thinking_start
            fraction = min(1.0, elapsed / analysis_seconds) if analysis_seconds > 0 else 1.0
            remaining = max(0.0, analysis_seconds - elapsed)

            def reveal_count(ranking):
                if not ranking:
                    return 0
                shown = min(MAX_RANKING_SHOWN, len(ranking))
                return max(1, math.ceil(fraction * shown))

            n_letter = reveal_count(letter_ranking)
            n_command = reveal_count(command_ranking)

            box_h = 30 + MAX_RANKING_SHOWN * 24 + 10
            draw_transparent_box(frame, (0, 65), (300, 65 + box_h), color=(0, 0, 0), alpha=0.6)
            cv2.putText(frame, f"Pensando (letra)... {remaining:.1f}s", (10, 65 + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
            if letter_ranking:
                for i in range(n_letter):
                    name, dist = letter_ranking[i]
                    color = (0, 255, 0) if i == 0 and fraction >= 1.0 else (200, 200, 200)
                    cv2.putText(frame, f"{i + 1}. {name} ({dist:.3f})", (10, 65 + 48 + i * 24),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
            else:
                cv2.putText(frame, "(mao esquerda nao detectada na foto)", (10, 65 + 48),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1, cv2.LINE_AA)

            draw_transparent_box(frame, (w - 300, 65), (w, 65 + box_h), color=(0, 0, 0), alpha=0.6)
            cv2.putText(frame, f"Pensando (comando)... {remaining:.1f}s", (w - 290, 65 + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
            if command_ranking:
                for i in range(n_command):
                    name, dist = command_ranking[i]
                    color = (0, 255, 0) if i == 0 and fraction >= 1.0 else (200, 200, 200)
                    cv2.putText(frame, f"{i + 1}. {name} ({dist:.3f})", (w - 290, 65 + 48 + i * 24),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
            else:
                cv2.putText(frame, "(mao direita nao detectada na foto)", (w - 290, 65 + 48),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1, cv2.LINE_AA)

            if elapsed >= analysis_seconds:
                print_reasoning("letra", letter_ranking)
                print_reasoning("comando", command_ranking)
                final_letter = letter_ranking[0][0] if letter_ranking else None
                final_command = command_ranking[0][0] if command_ranking else None
                print(f"\n=== RESPOSTA FINAL === letra: {final_letter or '-'} | comando: {final_command or '-'}\n")
                state = STATE_CONFIRM

        # =========================== ESTADO: CONFIRM ===========================
        elif state == STATE_CONFIRM:
            final_letter = letter_ranking[0][0] if letter_ranking else None
            final_command = command_ranking[0][0] if command_ranking else None

            draw_transparent_box(frame, (0, 65), (320, 200), color=(0, 0, 0), alpha=0.65)
            cv2.putText(frame, "RESPOSTA FINAL", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, f"Letra: {final_letter or '-'}", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, f"Comando: {final_command or '-'}", (10, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, "S = correto   N = errado (corrigir)", (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

        # =========================== ESTADO: CORRECT_LETTER ===========================
        elif state == STATE_CORRECT_LETTER:
            draw_transparent_box(frame, (0, 65), (400, 160), color=(0, 0, 40), alpha=0.7)
            cv2.putText(frame, "Qual era a letra certa?", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, "Tecle A-Z  |  '9' = C-cedilha", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, "ESPACO = pular (nao corrigir)", (10, 148), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

        # =========================== ESTADO: CORRECT_COMMAND ===========================
        elif state == STATE_CORRECT_COMMAND:
            list_top = 65
            list_height = 55 + len(COMMANDS) * 26
            draw_transparent_box(frame, (0, list_top), (320, list_top + list_height), color=(0, 0, 40), alpha=0.7)
            cv2.putText(frame, "Qual era o comando certo?", (10, list_top + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)
            for i, cmd_option in enumerate(COMMANDS):
                is_selected = i == correct_command_index
                color = (0, 255, 0) if is_selected else (200, 200, 200)
                prefix = "> " if is_selected else "   "
                cv2.putText(frame, f"{prefix}{cmd_option}", (10, list_top + 50 + i * 26),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2 if is_selected else 1, cv2.LINE_AA)
            cv2.putText(frame, "4/6=navega  5/ENTER=confirma  ESPACO=pular",
                        (10, list_top + list_height - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)

        # --- Estatísticas + último feedback ---
        total = stats["corretas"] + stats["erradas"]
        acc = (stats["corretas"] / total * 100) if total else 0.0
        cv2.putText(frame, f"Corretas: {stats['corretas']}  Erradas: {stats['erradas']}  "
                            f"Corrigidas: {stats['corrigidas']}  Acc: {acc:.0f}%",
                    (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        if last_feedback and time.time() < last_feedback_until:
            cv2.putText(frame, last_feedback, (10, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

        cv2.imshow(WINDOW_NAME, frame)

        key = cv2.waitKey(5) & 0xFF

        if key == ord("0") or key == 27:
            break

        elif key == 32 and state == STATE_IDLE:  # ESPAÇO: tira a foto
            letter_ranking = classify_ranked(normalize_landmarks(live_letter_landmarks), letter_signatures) if live_letter_landmarks is not None else []
            command_ranking = classify_ranked(normalize_landmarks(live_command_landmarks), command_signatures) if live_command_landmarks is not None else []
            letter_pose = normalize_landmarks(live_letter_landmarks) if live_letter_landmarks is not None else None
            command_pose = normalize_landmarks(live_command_landmarks) if live_command_landmarks is not None else None

            if not letter_ranking and not command_ranking:
                print("Foto tirada, mas nenhuma mao foi detectada no frame. Tente de novo.")
            else:
                print(f"\nFoto tirada. Analisando por {analysis_seconds:.1f}s...")
                thinking_start = time.time()
                state = STATE_THINKING

        elif key in (ord("+"), ord("=")) and state == STATE_IDLE:
            analysis_seconds = min(ANALYSIS_SECONDS_MAX, analysis_seconds + ANALYSIS_SECONDS_STEP)

        elif key == ord("-") and state == STATE_IDLE:
            analysis_seconds = max(ANALYSIS_SECONDS_MIN, analysis_seconds - ANALYSIS_SECONDS_STEP)

        elif key == ord("s") and state == STATE_CONFIRM:
            fed = []
            if letter_ranking and letter_pose is not None:
                name = letter_ranking[0][0]
                total_amostras = confirm_sample(name, letter_pose, DATA_PATH, RAW_DATA_PATH)
                letter_signatures = load_signatures(DATA_PATH, "letras")
                stats["corretas"] += 1
                fed.append(f"'{name}' confirmada ({total_amostras} amostras)")
            if command_ranking and command_pose is not None:
                name = command_ranking[0][0]
                total_amostras = confirm_sample(name, command_pose, COMMANDS_DATA_PATH, RAW_COMMANDS_DATA_PATH)
                command_signatures = load_signatures(COMMANDS_DATA_PATH, "comandos")
                stats["corretas"] += 1
                fed.append(f"'{name}' confirmado ({total_amostras} amostras)")
            msg = " | ".join(fed) if fed else "nada para confirmar"
            print(f"[OK] {msg}")
            last_feedback = "OK: " + msg
            last_feedback_until = time.time() + 2.5
            state = STATE_IDLE
            letter_ranking, command_ranking = [], []
            letter_pose, command_pose = None, None

        elif key == ord("n") and state == STATE_CONFIRM:
            marcados = []
            if letter_ranking:
                marcados.append(letter_ranking[0][0])
                stats["erradas"] += 1
            if command_ranking:
                marcados.append(command_ranking[0][0])
                stats["erradas"] += 1
            msg = f"Marcado(s) como errado: {', '.join(marcados)}" if marcados else "nada marcado"
            print(f"[X] {msg}")

            # Em vez de descartar, entra no fluxo de correção.
            if letter_pose is not None:
                state = STATE_CORRECT_LETTER
            elif command_pose is not None:
                correct_command_index = 0
                state = STATE_CORRECT_COMMAND
            else:
                state = STATE_IDLE

        # =========================== teclas durante correção de letra ===========================
        elif state == STATE_CORRECT_LETTER:
            corrected = None
            if key == 32:  # ESPAÇO = pular
                pass
            elif key == ord("9"):
                corrected = "Ç"
            elif 32 <= key <= 126:
                ch = chr(key).upper()
                if ch in LETTERS:
                    corrected = ch

            if corrected is not None:
                total_amostras = confirm_sample(corrected, letter_pose, DATA_PATH, RAW_DATA_PATH)
                letter_signatures = load_signatures(DATA_PATH, "letras")
                stats["corrigidas"] += 1
                last_feedback = f"Corrigido: letra certa era '{corrected}' ({total_amostras} amostras)"
                last_feedback_until = time.time() + 2.5
                print(f"[CORRIGIDO] letra -> '{corrected}' ({total_amostras} amostras)")

            if key == 32 or corrected is not None:
                if command_pose is not None:
                    correct_command_index = 0
                    state = STATE_CORRECT_COMMAND
                else:
                    state = STATE_IDLE
                    letter_ranking, command_ranking = [], []
                    letter_pose, command_pose = None, None

        # =========================== teclas durante correção de comando ===========================
        elif state == STATE_CORRECT_COMMAND:
            if key == ord("4"):
                correct_command_index = (correct_command_index - 1) % len(COMMANDS)
            elif key == ord("6"):
                correct_command_index = (correct_command_index + 1) % len(COMMANDS)
            elif key in (ord("5"), 13):  # "5" ou ENTER
                corrected = COMMANDS[correct_command_index]
                total_amostras = confirm_sample(corrected, command_pose, COMMANDS_DATA_PATH, RAW_COMMANDS_DATA_PATH)
                command_signatures = load_signatures(COMMANDS_DATA_PATH, "comandos")
                stats["corrigidas"] += 1
                last_feedback = f"Corrigido: comando certo era '{corrected}' ({total_amostras} amostras)"
                last_feedback_until = time.time() + 2.5
                print(f"[CORRIGIDO] comando -> '{corrected}' ({total_amostras} amostras)")
                state = STATE_IDLE
                letter_ranking, command_ranking = [], []
                letter_pose, command_pose = None, None
            elif key == 32:  # ESPAÇO = pular
                state = STATE_IDLE
                letter_ranking, command_ranking = [], []
                letter_pose, command_pose = None, None

        # --- Limitador de FPS ---
        elapsed = time.time() - loop_start
        if elapsed < MIN_FRAME_INTERVAL:
            time.sleep(MIN_FRAME_INTERVAL - elapsed)

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()

    total = stats["corretas"] + stats["erradas"]
    if total:
        acc = stats["corretas"] / total * 100
        print(f"\nSessão encerrada. {stats['corretas']} corretas / {stats['erradas']} erradas "
              f"({acc:.1f}% de acerto) | {stats['corrigidas']} correções salvas na base.")


if __name__ == "__main__":
    main()