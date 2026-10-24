"""Separação de falas sobrepostas com SepFormer.

Nos trechos em que as duas pessoas podem estar falando ao mesmo tempo, o SepFormer
divide a mistura em duas fontes. A impressão vocal de cada fonte é comparada com a
do locutor desejado e, se uma delas for claramente ele, ela substitui o trecho
na faixa final (em vez de silenciar ou deixar a outra voz vazar).
"""

from dataclasses import dataclass

import numpy as np

from .audio import reamostrar
from .vad import QUADRO_S, trechos

MODELO_PADRAO = "speechbrain/sepformer-whamr16k"


class SeparadorSepFormer:
    """SepFormer do SpeechBrain. Na primeira execução baixa o modelo (~110 MB)."""

    def __init__(self, fonte=MODELO_PADRAO, dispositivo="cpu"):
        import torch
        from speechbrain.inference.separation import SepformerSeparation
        from speechbrain.utils.fetching import LocalStrategy

        self._torch = torch
        self.modelo = SepformerSeparation.from_hparams(
            source=fonte,
            savedir=f"modelos/{fonte.rsplit('/', 1)[-1]}",
            run_opts={"device": dispositivo},
            # Copia em vez de criar symlinks, que no Windows exigem administrador.
            local_strategy=LocalStrategy.COPY_SKIP_CACHE,
        )
        self.sr = int(self.modelo.hparams.sample_rate)

    def __call__(self, mistura, sr):
        """Recebe um trecho mono em `sr` e devolve as fontes com forma [2, n]."""
        x = reamostrar(mistura.astype(np.float32), sr, self.sr)
        with self._torch.no_grad():
            est = self.modelo.separate_batch(self._torch.from_numpy(x)[None])
        fontes = reamostrar(est[0].T.cpu().numpy(), self.sr, sr)
        return _ajustar(fontes, len(mistura))


@dataclass
class TrechoSeparado:
    ini: int  # quadro de 10 ms
    fim: int
    fontes: np.ndarray  # [2, n] na taxa de análise, já na escala da mistura
    embs: np.ndarray  # [2, d] impressão vocal de cada fonte
    energia: np.ndarray  # [2] energia de cada fonte relativa à mistura


def regioes_candidatas(rotulos, margem, modo="auto", percentil=20.0,
                       contexto_s=1.0, max_pausa_troca_s=0.5, max_s=15.0):
    """Trechos (início, fim) em quadros onde vale a pena rodar o SepFormer.

    modo "auto": trocas de locutor (sobreposição é mais comum ali) e os quadros em
    que a identificação de quem fala foi mais ambígua. modo "tudo": toda a fala.
    """
    fala = rotulos >= 0
    ctx = int(contexto_s / QUADRO_S)
    if modo == "tudo":
        cand = fala.copy()
    else:
        cand = np.zeros_like(fala)
        if fala.any():
            ambiguo = fala & (margem <= np.percentile(margem[fala], percentil))
            for a, b in trechos(ambiguo):
                cand[max(0, a - ctx // 2) : b + ctx // 2] = True
        pos = np.flatnonzero(fala)
        quem = rotulos[pos]
        troca = (quem[1:] != quem[:-1]) & (np.diff(pos) <= int(max_pausa_troca_s / QUADRO_S))
        for a, b in zip(pos[:-1][troca], pos[1:][troca]):
            cand[max(0, a - ctx) : b + ctx] = True
    # Une regiões próximas e quebra as longas em pedaços que caibam na memória.
    cand = ~_fechar_lacunas(~cand, int(0.5 / QUADRO_S))
    max_q = int(max_s / QUADRO_S)
    regioes = []
    for a, b in trechos(cand):
        n = int(np.ceil((b - a) / max_q))
        cortes = np.linspace(a, b, n + 1).astype(int)
        regioes += list(zip(cortes[:-1], cortes[1:]))
    return regioes


def _fechar_lacunas(mascara, max_len):
    mascara = mascara.copy()
    for a, b in trechos(mascara):
        if b - a <= max_len and a > 0 and b < len(mascara):
            mascara[a:b] = False
    return mascara


def separar_regioes(sinal, sr, regioes, separador, embedder, ao_progredir=None):
    q = int(sr * QUADRO_S)
    resultado = []
    for i, (ini, fim) in enumerate(regioes):
        mistura = sinal[ini * q : min(fim * q, len(sinal))]
        if len(mistura) >= sr // 2:
            fontes = separador(mistura, sr)
            # Ajusta o volume de cada fonte para a escala em que aparece na mistura.
            escala = (fontes @ mistura) / ((fontes**2).sum(1) + 1e-12)
            fontes = (fontes * np.clip(escala, 0, None)[:, None]).astype(np.float32)
            energia = (fontes**2).mean(1) / ((mistura**2).mean() + 1e-12)
            embs = embedder(list(fontes))
            resultado.append(TrechoSeparado(ini, fim, fontes, embs, energia))
        if ao_progredir:
            ao_progredir(i + 1, len(regioes))
    return resultado


def escolher_fontes(trechos_sep, alvo, outros, min_energia=0.05):
    """Para cada trecho, o índice da fonte que é o locutor alvo (ou None)."""
    escolhas = []
    for t in trechos_sep:
        sim_alvo = t.embs @ alvo
        sim_outro = (t.embs @ outros.T).max(axis=1) if len(outros) else np.zeros(2)
        pont = np.where(t.energia >= min_energia, sim_alvo - sim_outro, -np.inf)
        j = int(np.argmax(pont))
        escolhas.append(j if pont[j] > 0 else None)
    return escolhas


def aplicar(faixa, ativo, sr_saida, sr_analise, trechos_sep, escolhas, fade_s=0.02):
    """Substitui os trechos aceitos da faixa pela fonte separada (com crossfade)."""
    q = int(sr_saida * QUADRO_S)
    n_fade = max(1, int(sr_saida * fade_s))
    for t, j in zip(trechos_sep, escolhas):
        if j is None:
            continue
        a = t.ini * q
        fonte = reamostrar(t.fontes[j], sr_analise, sr_saida)
        n = min(len(fonte), len(faixa) - a, (t.fim - t.ini) * q)
        if n <= 0:
            continue
        peso = np.ones(n, dtype=np.float32)
        rampa = np.linspace(0, 1, min(n_fade, n // 2), dtype=np.float32)
        if len(rampa):
            peso[: len(rampa)] = rampa
            peso[n - len(rampa) :] = rampa[::-1]
        faixa[a : a + n] = faixa[a : a + n] * (1 - peso)[:, None] + (fonte[:n] * peso)[:, None]
        ativo[a : a + n] = True
    return faixa, ativo


def _ajustar(x, n):
    if x.shape[-1] >= n:
        return x[..., :n]
    return np.pad(x, [(0, 0)] * (x.ndim - 1) + [(0, n - x.shape[-1])])
