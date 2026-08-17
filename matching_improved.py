"""
Classificacao mais forte para as maos LIBRAS
------------------------------------------------
Modulo compartilhado para deixar a previsao (letra/comando) mais
resistente a ambiguidade e ruido, sem mudar o formato dos arquivos
data.json / commands.json.

O que muda em relacao ao "nearest neighbor simples" usado hoje:

1. DISTANCIA PONDERADA POR LANDMARK
   Nem todo ponto da mao carrega a mesma informacao. As pontas dos
   dedos (8, 12, 16, 20) e o polegar (4) sao o que mais diferencia
   um gesto do outro; o pulso e a base da palma variam pouco entre
   letras e so acrescentam ruido. Pesos maiores nesses pontos fazem
   a distancia refletir melhor a forma da mao.

2. VOTO K-NN (nao so o vizinho mais proximo)
   Em vez de comparar so com a amostra mais parecida de cada classe,
   usamos a media das K amostras mais proximas por classe. Isso
   suaviza outliers (uma amostra ruim capturada durante o treino).

3. MARGEM DE CONFIANCA
   Alem do limiar de distancia, exige que a 2a colocada esteja
   suficientemente mais longe que a 1a. Sem isso, poses realmente
   ambiguas (ex.: M vs N) ficam trocando de rotulo a cada frame
   mesmo com filtro de suavizacao temporal.

Como usar (troca minima no validador.py / translator.py):

    from matching_improved import classify_ranked_weighted, is_confident

    ranking = classify_ranked_weighted(normalized_pose, letter_signatures)
    label, dist = ranking[0]
    ok = is_confident(ranking, threshold=LETTER_CONFIDENCE_THRESHOLD, min_margin=0.04)
    letter = label if ok else None

Isso substitui a funcao `classify_ranked` (validador.py, beta_tester.py)
ou `classify_nearest` (translator.py) linha a linha -- o formato de
retorno de `classify_ranked_weighted` eh o mesmo (lista ordenada de
tuplas (nome, distancia)), entao o resto do codigo (ranking na tela,
etc.) nao precisa mudar.
"""

import numpy as np

# Pesos por landmark (indice 0-20, mesma ordem do MediaPipe Hands).
# Pontas dos dedos e polegar pesam mais; pulso/base da palma pesam
# menos. Ajuste livremente se notar que alguma letra especifica
# depende mais de um ponto que o padrao nao cobre bem.
LANDMARK_WEIGHTS = np.array([
    0.5,                    # 0  pulso
    0.7, 0.8, 0.9, 1.4,     # 1-4  polegar (ponta = 4, pesa mais)
    0.6, 0.8, 1.0, 1.4,     # 5-8  indicador (ponta = 8)
    0.6, 0.8, 1.0, 1.4,     # 9-12 medio (ponta = 12)
    0.6, 0.8, 1.0, 1.4,     # 13-16 anelar (ponta = 16)
    0.6, 0.8, 1.0, 1.4,     # 17-20 minimo (ponta = 20)
])[:, np.newaxis]  # shape (21, 1) pra multiplicar contra (21, 3)


def weighted_distance(pose_a, poses_b):
    """
    pose_a: (21, 3)  -- uma pose
    poses_b: (n, 21, 3) -- n poses de uma classe
    Retorna array (n,) com a distancia media ponderada de pose_a a
    cada uma das n poses.
    """
    diff = poses_b - pose_a[np.newaxis, :, :]          # (n, 21, 3)
    dist_per_point = np.linalg.norm(diff, axis=2)        # (n, 21)
    weighted = dist_per_point * LANDMARK_WEIGHTS.T        # (n, 21)
    return weighted.sum(axis=1) / LANDMARK_WEIGHTS.sum()  # (n,) media ponderada


def classify_ranked_weighted(normalized_pose, signatures, k=3):
    """
    Mesma assinatura de retorno do classify_ranked original:
    lista [(nome, distancia), ...] do mais parecido pro menos.

    Mas em vez de pegar so a amostra mais proxima de cada classe,
    tira a media das K amostras mais proximas (k-NN por classe),
    usando a distancia ponderada por landmark.
    """
    results = []
    for name, poses in signatures.items():
        dists = weighted_distance(normalized_pose, poses)
        k_eff = min(k, len(dists))
        nearest_k = np.partition(dists, k_eff - 1)[:k_eff]
        score = float(nearest_k.mean())
        results.append((name, score))
    results.sort(key=lambda x: x[1])
    return results


def is_confident(ranking, threshold, min_margin=0.04):
    """
    Decide se o resultado top-1 do ranking deve ser aceito:
    - a distancia do 1o colocado precisa estar abaixo do limiar
    - E o 2o colocado precisa estar pelo menos `min_margin` mais
      longe que o 1o (evita alternar entre duas classes muito
      parecidas). Se so existir 1 classe no ranking, so o limiar
      de distancia vale.
    """
    if not ranking:
        return False
    best_name, best_dist = ranking[0]
    if best_dist > threshold:
        return False
    if len(ranking) < 2:
        return True
    _, second_dist = ranking[1]
    return (second_dist - best_dist) >= min_margin