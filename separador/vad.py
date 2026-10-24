"""Detecção de atividade de voz baseada em energia (sem dependências pesadas)."""

import numpy as np

QUADRO_S = 0.01  # resolução de 10 ms usada em todo o pipeline


def energia_db(sinal, sr, quadro_s=QUADRO_S):
    passo = int(sr * quadro_s)
    n = len(sinal) // passo
    quadros = sinal[: n * passo].reshape(n, passo)
    rms = np.sqrt(np.mean(quadros**2, axis=1) + 1e-12)
    return 20 * np.log10(rms)


def detectar_fala(sinal, sr, margem_db=12.0, min_fala_s=0.25, max_pausa_s=0.3):
    """Retorna uma máscara booleana por quadro de 10 ms indicando fala."""
    db = energia_db(sinal, sr)
    if len(db) == 0:
        return np.zeros(0, dtype=bool)
    # O piso de ruído é estimado pelos quadros mais silenciosos.
    piso = np.percentile(db, 10)
    teto = np.percentile(db, 95)
    limiar = max(piso + margem_db, teto - 40)
    fala = db > limiar
    fala = _preencher(fala, valor=False, max_len=int(max_pausa_s / QUADRO_S))
    fala = _preencher(fala, valor=True, max_len=int(min_fala_s / QUADRO_S))
    return fala


def _preencher(mascara, valor, max_len):
    """Inverte trechos curtos (<= max_len) iguais a `valor`."""
    mascara = mascara.copy()
    for ini, fim in trechos(mascara == valor):
        if fim - ini <= max_len:
            mascara[ini:fim] = not valor
    return mascara


def trechos(mascara):
    """Lista de (início, fim) dos trechos True contíguos."""
    m = np.concatenate([[False], mascara, [False]]).astype(np.int8)
    d = np.diff(m)
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))
