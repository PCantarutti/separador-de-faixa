"""Gera "vozes" sintéticas (fonte glotal + formantes) para testar o pipeline."""

import numpy as np
from scipy.signal import lfilter

VOGAIS = {  # formantes aproximados (F1, F2, F3)
    "a": (800, 1200, 2500), "e": (500, 1800, 2500), "i": (300, 2300, 3000),
    "o": (500, 900, 2400), "u": (320, 800, 2300),
}


def _ressonador(x, f, bw, sr):
    r = np.exp(-np.pi * bw / sr)
    a = [1, -2 * r * np.cos(2 * np.pi * f / sr), r * r]
    return lfilter([1 - r], a, x)


def voz(duracao, sr, f0, escala_formantes, rng):
    """Sequência de vogais com f0 variando levemente."""
    n = int(duracao * sr)
    saida = np.zeros(n)
    pos = 0
    while pos < n:
        dur = int(rng.uniform(0.12, 0.3) * sr)
        vogal = VOGAIS[rng.choice(list(VOGAIS))]
        t = np.arange(dur) / sr
        f0_inst = f0 * (1 + 0.05 * np.sin(2 * np.pi * rng.uniform(2, 5) * t))
        fase = 2 * np.pi * np.cumsum(f0_inst) / sr
        fonte = np.sign(np.sin(fase)) * 0.5 + 0.02 * rng.standard_normal(dur)
        seg = sum(_ressonador(fonte, f * escala_formantes, 80 + i * 40, sr)
                  for i, f in enumerate(vogal))
        seg *= np.hanning(dur)
        fim = min(n, pos + dur)
        saida[pos:fim] += seg[: fim - pos]
        pos += int(dur * 0.8)
    return saida / (np.abs(saida).max() + 1e-9) * 0.5


def podcast(sr=16000, turnos=12, semente=0, sobreposicao=False):
    """Retorna (sinal, verdade_por_quadro_10ms, ref_A, ref_B).

    Com `sobreposicao=True` retorna também (fonte_A, fonte_B) e a pessoa A solta
    interjeições por cima das falas de B; a verdade marca esses quadros com 2.
    """
    rng = np.random.default_rng(semente)
    locutores = [(110, 1.0), (210, 1.18)]  # (f0, escala de formantes)
    n_total = 0
    blocos = []  # (quem, início, sinal)
    verdade = []
    for k in range(turnos):
        quem = k % 2
        dur = rng.uniform(2.0, 5.0)
        s = voz(dur, sr, *locutores[quem], rng)
        blocos.append((quem, n_total, s))
        verdade += [quem] * (len(s) // (sr // 100))
        n_total += len(s)
        if sobreposicao and quem == 1:
            # Interjeição de A no meio da fala de B.
            ini = n_total - len(s) + int(rng.uniform(0.6, 1.2) * sr)
            blocos.append((0, ini, voz(rng.uniform(0.6, 0.9), sr, *locutores[0], rng)))
        pausa = int(rng.uniform(0.3, 0.8) * sr)
        verdade += [-1] * (pausa // (sr // 100))
        n_total += pausa
    fontes = np.zeros((2, n_total))
    for quem, ini, s in blocos:
        fontes[quem, ini : ini + len(s)] += s
    verdade = np.array(verdade[: n_total // (sr // 100)])
    q = sr // 100
    if sobreposicao:
        energia = (fontes[:, : len(verdade) * q].reshape(2, -1, q) ** 2).mean(2) > 1e-4
        verdade[energia[0] & energia[1]] = 2
    sinal = fontes.sum(0) + 0.003 * rng.standard_normal(n_total)
    refs = [voz(10, sr, *loc, rng) for loc in locutores]
    if sobreposicao:
        return sinal.astype(np.float32), verdade, refs, fontes.astype(np.float32)
    return sinal.astype(np.float32), verdade, refs
