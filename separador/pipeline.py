"""Pipeline: VAD -> embeddings por janela -> agrupamento -> rótulo por quadro."""

from dataclasses import dataclass

import numpy as np

from .vad import QUADRO_S, detectar_fala, trechos


@dataclass
class Resultado:
    rotulos: np.ndarray  # por quadro de 10 ms: -1 = silêncio, 0..N-1 = locutor
    alvo: int | None  # índice do locutor escolhido pela referência
    similaridade_ref: list[float] | None  # similaridade de cada locutor com a referência
    centroides: np.ndarray | None = None  # impressão vocal média de cada locutor
    ref: np.ndarray | None = None  # impressão vocal da referência
    margem: np.ndarray | None = None  # por quadro: diferença entre os dois locutores mais prováveis

    def segmentos(self, locutor):
        """Lista de (início_s, fim_s) em que `locutor` fala."""
        return [(i * QUADRO_S, f * QUADRO_S) for i, f in trechos(self.rotulos == locutor)]


def janelar(sinal, sr, fala, janela_s, passo_s):
    """Corta janelas sobre as regiões de fala. Retorna (janelas, quadro_inicial)."""
    q_jan = int(janela_s / QUADRO_S)
    q_passo = int(passo_s / QUADRO_S)
    amostras_q = int(sr * QUADRO_S)
    janelas, inicios = [], []
    for ini, fim in trechos(fala):
        if fim - ini < q_jan:
            # Trecho curto: uma janela centralizada nele (com contexto ao redor).
            centro = (ini + fim) // 2
            posicoes = [max(0, min(centro - q_jan // 2, len(fala) - q_jan))]
        else:
            posicoes = list(range(ini, fim - q_jan + 1, q_passo))
            if posicoes[-1] != fim - q_jan:
                posicoes.append(fim - q_jan)
        for p in posicoes:
            w = sinal[p * amostras_q : (p + q_jan) * amostras_q]
            if len(w) < q_jan * amostras_q:
                w = np.pad(w, (0, q_jan * amostras_q - len(w)))
            janelas.append(w.astype(np.float32))
            inicios.append(p)
    return janelas, np.array(inicios, dtype=int)


def janelas_referencia(sinais, sr, janela_s, passo_s):
    todas = []
    for s in sinais:
        fala = detectar_fala(s, sr)
        if not fala.any():
            fala = np.ones_like(fala)
        todas += janelar(s, sr, fala, janela_s, passo_s)[0]
    return todas


def agrupar(embs, n_locutores, ref=None, n_inicios=10, iteracoes=50, semente=0):
    """K-means esférico (similaridade de cosseno) sobre as impressões vocais.

    Com `ref`, um dos grupos começa na referência e os demais nas janelas menos
    parecidas com ela, o que ancora a separação nas vozes e impede que um punhado
    de janelas atípicas (vinheta, risadas, ruído) forme um grupo sozinho.
    """
    if len(embs) < n_locutores:
        return np.zeros(len(embs), dtype=int)
    rng = np.random.default_rng(semente)
    if ref is not None:
        sim = embs @ ref
        distantes = embs[sim <= np.quantile(sim, 0.3)]
        iniciais = [_kmeans_mais_mais(distantes, n_locutores - 1, rng, [ref])]
    else:
        iniciais = [_kmeans_mais_mais(embs, n_locutores, rng) for _ in range(n_inicios)]

    melhor, melhor_pont = None, -np.inf
    for centroides in iniciais:
        for _ in range(iteracoes):
            rotulos = np.argmax(embs @ centroides.T, axis=1)
            novos = _centroides(embs, rotulos, n_locutores, centroides)
            if np.allclose(novos, centroides):
                break
            centroides = novos
        pont = np.max(embs @ centroides.T, axis=1).sum()
        if pont > melhor_pont:
            melhor, melhor_pont = rotulos, pont
    return melhor


def _kmeans_mais_mais(embs, n, rng, fixos=()):
    """Sorteia centróides iniciais afastados entre si (k-means++)."""
    escolhidos = list(fixos)
    if not escolhidos:
        escolhidos.append(embs[rng.integers(len(embs))])
    while len(escolhidos) < n + len(fixos):
        dist = 1 - np.max(embs @ np.array(escolhidos).T, axis=1)
        p = np.clip(dist, 0, None) ** 2
        p = p / p.sum() if p.sum() > 0 else None
        escolhidos.append(embs[rng.choice(len(embs), p=p)])
    return _normalizar(np.array(escolhidos))


def _normalizar(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)


def _centroides(embs, rotulos, n, anteriores=None):
    c = []
    for k in range(n):
        if np.any(rotulos == k):
            c.append(embs[rotulos == k].mean(0))
        else:
            c.append(anteriores[k] if anteriores is not None else np.zeros(embs.shape[1]))
    return _normalizar(np.stack(c))


def separar(
    sinal,
    sr,
    embedder,
    n_locutores=2,
    referencias=None,
    janela_s=1.5,
    passo_s=0.25,
    min_segmento_s=0.4,
    margem_vad_db=12.0,
):
    """Rotula cada quadro de 10 ms com o locutor que está falando."""
    fala = detectar_fala(sinal, sr, margem_db=margem_vad_db)
    n_quadros = len(fala)
    janelas, inicios = janelar(sinal, sr, fala, janela_s, passo_s)
    if not janelas:
        return Resultado(np.full(n_quadros, -1), None, None, margem=np.zeros(n_quadros))

    janelas_ref = janelas_referencia(referencias or [], sr, janela_s, passo_s)
    # Um único lote garante que a referência e o podcast passem pela mesma normalização.
    embs_todos = embedder(janelas + janelas_ref)
    embs, embs_ref = embs_todos[: len(janelas)], embs_todos[len(janelas):]

    ref = None
    if len(embs_ref):
        ref = embs_ref.mean(0)
        ref /= np.linalg.norm(ref) + 1e-8
    rotulos_jan = agrupar(embs, n_locutores, ref)
    centroides = _centroides(embs, rotulos_jan, n_locutores)

    # Cada quadro recebe a média da similaridade das janelas que o cobrem.
    q_jan = int(janela_s / QUADRO_S)
    pontuacao = np.zeros((n_quadros, n_locutores))
    cobertura = np.zeros(n_quadros)
    sims = embs @ centroides.T
    for ini, s in zip(inicios, sims):
        pontuacao[ini : ini + q_jan] += s
        cobertura[ini : ini + q_jan] += 1
    rotulos = np.where(cobertura > 0, np.argmax(pontuacao, axis=1), -1)
    media = pontuacao / np.maximum(cobertura, 1)[:, None]
    ordenado = np.sort(media, axis=1)
    margem = ordenado[:, -1] - ordenado[:, -2] if n_locutores > 1 else ordenado[:, -1]
    rotulos[~fala] = -1
    rotulos = _suavizar(rotulos, int(min_segmento_s / QUADRO_S))

    alvo, sim_ref = None, None
    if ref is not None:
        sim_ref = (centroides @ ref).tolist()
        alvo = int(np.argmax(sim_ref))
    return Resultado(rotulos, alvo, sim_ref, centroides, ref, margem)


def _suavizar(rotulos, min_len):
    """Funde trechos de locutor mais curtos que `min_len` no vizinho anterior."""
    rotulos = rotulos.copy()
    fala = rotulos >= 0
    for ini, fim in trechos(fala):
        bloco = rotulos[ini:fim]
        mudou = True
        while mudou:
            mudou = False
            limites = np.flatnonzero(np.diff(bloco)) + 1
            cortes = np.concatenate([[0], limites, [len(bloco)]])
            for a, b in zip(cortes[:-1], cortes[1:]):
                if b - a < min_len and len(cortes) > 2:
                    vizinho = bloco[a - 1] if a > 0 else bloco[b]
                    bloco[a:b] = vizinho
                    mudou = True
                    break
        rotulos[ini:fim] = bloco
    return rotulos


def mascara_amostras(rotulos, locutor, n_amostras, sr, fade_s=0.015):
    """Converte rótulos por quadro em ganho por amostra, com fade para evitar cliques."""
    amostras_q = int(sr * QUADRO_S)
    ganho = np.repeat((rotulos == locutor).astype(np.float32), amostras_q)
    ganho = np.pad(ganho, (0, max(0, n_amostras - len(ganho))))[:n_amostras]
    n_fade = max(1, int(sr * fade_s))
    nucleo = np.ones(n_fade, dtype=np.float32) / n_fade
    return np.convolve(ganho, nucleo, mode="same")
