"""
Captura de Fotos por Tecla - LIBRAS
--------------------------------------
Tira fotos da webcam e salva DIRETO na subpasta certa dentro de
"imgs/", pra depois voce rodar o batch_train_from_images.py e treinar
a base com elas.

MODOS (tecla "8" alterna):
    - Modo LETRA (padrao): aperte a letra (A-Z) pra fotografar e
      salvar em imgs/<LETRA>/. Para Ç, use a tecla "9" (salva em
      imgs/CA/, que o batch_train_from_images.py reconhece como Ç).
    - Modo COMANDO: navegue pela lista de comandos com "4"/"6" e
      aperte "5" ou ESPACO pra fotografar o comando selecionado,
      salvando em imgs/<COMANDO>/.

Cada foto entra com nome incremental (ex.: A_0001.jpg, A_0002.jpg,
...), sem sobrescrever fotos que ja existirem na pasta -- pode rodar
o script em varias sessoes que ele so vai empilhando.

Por padrao, so salva a foto se uma mao for detectada no frame no
instante do clique (evita foto "vazia" ir pro treino). Use
--allow-no-hand se quiser desativar essa checagem.

Ha um debounce (padrao 350ms) pra evitar fotos repetidas em
sequencia se voce segurar a tecla apertada.

Controles:
    - Letra (A-Z) = foto da letra (modo LETRA)
    - "9"         = foto de Ç (salva em imgs/CA)
    - "8"         = alterna modo LETRA / COMANDO
    - "4" / "6"   = (modo COMANDO) navega na lista de comandos
    - "5" / ESPACO = (modo COMANDO) foto do comando selecionado
    - "0" ou ESC  = sair

Como usar:
    python photo_capture.py
    python photo_capture.py --imgs-dir outra_pasta
    python photo_capture.py --allow-no-hand
    python photo_capture.py --debounce 0.5
"""

import os
import time
import argparse

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "hand_landmarker.task")
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)

WINDOW_NAME = "Captura de Fotos - LIBRAS"
WINDOW_WIDTH, WINDOW_HEIGHT = 1000, 620

TARGET_FPS = 60
MIN_FRAME_INTERVAL = 1.0 / TARGET_FPS

LETTERS = [chr(c) for c in range(ord("A"), ord("Z") + 1)]  # Ç tratado à parte, tecla "9"
COMMANDS = ["BACKSPACE", "SPACE", "CLEAR"]  # mesma lista do capture_signatures.py

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
        print("Baixando modelo hand_landmarker.task (primeira execucao)...")
        import urllib.request
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Download concluido:", MODEL_PATH)


def draw_landmarks(frame, landmarks, color=(0, 200, 0)):
    h, w, _ = frame.shape
    points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for start, end in HAND_CONNECTIONS:
        cv2.line(frame, points[start], points[end], (255, 255, 255), 2)
    for x, y in points:
        cv2.circle(frame, (x, y), 4, color, -1)


def draw_transparent_box(frame, top_left, bottom_right, color=(0, 0, 0), alpha=0.55):
    overlay = frame.copy()
    cv2.rectangle(overlay, top_left, bottom_right, color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def next_filename(folder_path, label):
    os.makedirs(folder_path, exist_ok=True)
    existing = [f for f in os.listdir(folder_path) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    n = len(existing) + 1
    while True:
        fname = f"{label}_{n:04d}.jpg"
        if not os.path.exists(os.path.join(folder_path, fname)):
            return fname, n
        n += 1


def save_photo(frame, imgs_dir, label):
    folder_path = os.path.join(imgs_dir, label)
    fname, n = next_filename(folder_path, label)
    fpath = os.path.join(folder_path, fname)
    cv2.imwrite(fpath, frame)
    return fpath, n


def main():
    parser = argparse.ArgumentParser(description="Captura fotos por tecla e salva em imgs/<LABEL>/.")
    parser.add_argument("--imgs-dir", default=os.path.join(BASE_DIR, "imgs"))
    parser.add_argument("--allow-no-hand", action="store_true",
                         help="Salva a foto mesmo se nenhuma mao for detectada no frame")
    parser.add_argument("--debounce", type=float, default=0.35,
                         help="Tempo minimo (segundos) entre duas capturas seguidas (evita repetir com a tecla presa)")
    args = parser.parse_args()

    ensure_model()

    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=1,
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

    mode = "letter"  # ou "command"
    current_command_index = 0

    frame_timestamp_ms = 0
    start_time = time.time()
    last_hand_landmarks = None
    hand_present = False

    last_capture_time = 0.0
    last_feedback = ""
    last_feedback_until = 0.0

    os.makedirs(args.imgs_dir, exist_ok=True)

    print("Captura de fotos iniciada. Modo atual: LETRA")
    print("Letra: tecle A-Z (foto) | '9' = foto de Ç | '8' = alterna modo")
    print("Modo comando: '4'/'6' navega, '5'/ESPACO tira foto | '0'/ESC sai")

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

        last_hand_landmarks = None
        hand_present = False
        if result.hand_landmarks:
            last_hand_landmarks = result.hand_landmarks[0]
            hand_present = True

        display_frame = frame.copy()
        if last_hand_landmarks is not None:
            draw_landmarks(display_frame, last_hand_landmarks)

        h, w, _ = display_frame.shape

        # --- HUD ---
        draw_transparent_box(display_frame, (0, 0), (460, 90), color=(0, 0, 0), alpha=0.55)
        if mode == "letter":
            info = "[MODO LETRA] Tecle A-Z para fotografar | '9' = foto de C-cedilha"
            controls = "8=modo comando | 0/ESC=sair"
        else:
            cmd_name = COMMANDS[current_command_index]
            info = f"[MODO COMANDO] Selecionado ({current_command_index + 1}/{len(COMMANDS)}): {cmd_name}"
            controls = "4/6=navega | 5/ESPACO=foto | 8=modo letra | 0/ESC=sair"
        cv2.putText(display_frame, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(display_frame, controls, (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        hand_txt = "Mao detectada" if hand_present else "Mao NAO detectada"
        hand_color = (0, 255, 0) if hand_present else (0, 0, 255)
        cv2.putText(display_frame, hand_txt, (10, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.5, hand_color, 1, cv2.LINE_AA)

        if mode == "command":
            list_top = 100
            list_height = 30 + len(COMMANDS) * 26
            draw_transparent_box(display_frame, (0, list_top), (260, list_top + list_height), color=(0, 0, 0), alpha=0.55)
            cv2.putText(display_frame, "Comandos:", (10, list_top + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
            for i, cmd_option in enumerate(COMMANDS):
                is_selected = i == current_command_index
                color = (0, 255, 0) if is_selected else (200, 200, 200)
                prefix = "> " if is_selected else "   "
                cv2.putText(display_frame, f"{prefix}{cmd_option}", (10, list_top + 48 + i * 26),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2 if is_selected else 1, cv2.LINE_AA)

        if last_feedback and time.time() < last_feedback_until:
            draw_transparent_box(display_frame, (0, h - 45), (w, h), color=(0, 100, 0), alpha=0.5)
            cv2.putText(display_frame, last_feedback, (10, h - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow(WINDOW_NAME, display_frame)

        key = cv2.waitKey(5) & 0xFF
        now = time.time()

        def try_capture(label):
            nonlocal last_capture_time, last_feedback, last_feedback_until
            if now - last_capture_time < args.debounce:
                return
            if not hand_present and not args.allow_no_hand:
                last_feedback = f"Sem mao detectada -- foto de '{label}' NAO salva"
                last_feedback_until = now + 1.2
                print(f"[SKIP] {label}: nenhuma mao detectada no frame.")
                last_capture_time = now
                return
            fpath, n = save_photo(frame, args.imgs_dir, label)
            last_capture_time = now
            last_feedback = f"Salvo: {label} (#{n}) -> {os.path.basename(fpath)}"
            last_feedback_until = now + 1.2
            print(f"[OK] {fpath}")

        if key == ord("0") or key == 27:
            break

        elif key == ord("8"):
            mode = "command" if mode == "letter" else "letter"
            print(f"Modo alternado para: {'COMANDO' if mode == 'command' else 'LETRA'}")

        elif mode == "letter" and key == ord("9"):
            try_capture("CA")  # Ç

        elif mode == "letter" and 32 <= key <= 126:
            ch = chr(key).upper()
            if ch in LETTERS:
                try_capture(ch)

        elif mode == "command" and key == ord("4"):
            current_command_index = (current_command_index - 1) % len(COMMANDS)

        elif mode == "command" and key == ord("6"):
            current_command_index = (current_command_index + 1) % len(COMMANDS)

        elif mode == "command" and key in (ord("5"), 32):
            try_capture(COMMANDS[current_command_index])

        elapsed = time.time() - loop_start
        if elapsed < MIN_FRAME_INTERVAL:
            time.sleep(MIN_FRAME_INTERVAL - elapsed)

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()
    print("Encerrado.")


if __name__ == "__main__":
    main()