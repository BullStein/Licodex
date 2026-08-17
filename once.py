"""
Cria a estrutura base de pastas dentro de "imgs/" -- uma subpasta
por letra (A-Z, CA para Ç) e uma por comando (BACKSPACE, SPACE,
CLEAR), prontas pra você jogar as fotos dentro.

Não sobrescreve nada: se a pasta já existir (com fotos ou não), ele
só pula e avisa. Seguro rodar de novo a qualquer momento.

Como usar:
    python setup_imgs_folders.py
    python setup_imgs_folders.py --imgs-dir outra_pasta
"""

import os
import argparse

LETTERS = [chr(c) for c in range(ord("A"), ord("Z") + 1)] + ["CA"]  # CA = Ç sem cedilha
COMMANDS = ["BACKSPACE", "SPACE", "CLEAR"]


def main():
    parser = argparse.ArgumentParser(description="Cria a estrutura de pastas base em imgs/.")
    parser.add_argument("--imgs-dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "imgs"))
    args = parser.parse_args()

    criadas = []
    ja_existiam = []

    for name in LETTERS + COMMANDS:
        path = os.path.join(args.imgs_dir, name)
        if os.path.isdir(path):
            ja_existiam.append(name)
            continue
        os.makedirs(path, exist_ok=True)
        criadas.append(name)

    print(f"Pasta base: {args.imgs_dir}\n")
    if criadas:
        print(f"Criadas ({len(criadas)}): {', '.join(criadas)}")
    if ja_existiam:
        print(f"Já existiam, não mexi ({len(ja_existiam)}): {', '.join(ja_existiam)}")
    print(f"\nTotal de subpastas: {len(LETTERS) + len(COMMANDS)} "
          f"({len(LETTERS)} letras + {len(COMMANDS)} comandos)")
    print("Agora é só jogar as fotos de cada gesto dentro da subpasta correspondente.")


if __name__ == "__main__":
    main()