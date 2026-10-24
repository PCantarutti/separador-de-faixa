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


def aplicar(faixa, ativo, original, sr_saida, sr_analise, trechos_sep, escolhas, fade_s=0.02):
    """Nos trechos aceitos, filtra o áudio original com uma máscara tirada do SepFormer.

    O áudio gerado pelo SepFormer tem artefatos (voz metálica/"robótica"); usar só a
    proporção entre as fontes como máscara sobre o original preserva o som real da
    pessoa, os agudos e o estéreo, e remove apenas o que pertence à outra voz.
    """
    q = int(sr_saida * QUADRO_S)
    n_fade = max(1, int(sr_saida * fade_s))
    for t, j in zip(trechos_sep, escolhas):
        if j is None:
            continue
        a = t.ini * q
        n = min(len(faixa) - a, (t.fim - t.ini) * q)
        if n <= 0:
            continue
        outras = np.delete(t.fontes, j, axis=0).sum(0)
        filtrado = _filtrar(original[a : a + n], sr_saida, t.fontes[j], outras, sr_analise)
        peso = np.ones(n, dtype=np.float32)
        rampa = np.linspace(0, 1, min(n_fade, n // 2), dtype=np.float32)
        if len(rampa):
            peso[: len(rampa)] = rampa
            peso[n - len(rampa) :] = rampa[::-1]
        faixa[a : a + n] = faixa[a : a + n] * (1 - peso)[:, None] + filtrado * peso[:, None]
        ativo[a : a + n] |= _tem_voz(filtrado.mean(1), sr_saida)
    return faixa, ativo


def _filtrar(trecho, sr, alvo, outras, sr_analise, janela_s=0.032, passo_s=0.008):
    """Aplica ao `trecho` original ([n, canais], taxa `sr`) a máscara de Wiener
    |alvo|² / (|alvo|² + |outras|²), calculada na taxa de análise."""
    from scipy.ndimage import uniform_filter
    from scipy.signal import istft, stft

    def espectro(x, taxa):
        nper = int(round(janela_s * taxa))
        return stft(x, taxa, nperseg=nper, noverlap=nper - int(round(passo_s * taxa)),
                    axis=-1)

    _, _, a_ = espectro(alvo, sr_analise)
    freqs_a, tempos_a, o_ = espectro(outras, sr_analise)
    pa, po = np.abs(a_) ** 2, np.abs(o_) ** 2
    # Suaviza levemente para evitar "ruído musical" (chiados intermitentes).
    mascara = uniform_filter(pa, 3) / (uniform_filter(pa, 3) + uniform_filter(po, 3) + 1e-10)

    freqs, tempos, x = espectro(trecho.T, sr)  # x: [canais, freqs, tempos]
    # Leva a máscara para a grade de frequência/tempo do áudio original. Acima do
    # limite da análise usa a média da faixa mais aguda disponível.
    idx_t = np.clip(np.searchsorted(tempos_a, tempos), 0, len(tempos_a) - 1)
    m = mascara[:, idx_t]
    agudos = m[freqs_a >= freqs_a[-1] / 2].mean(0)
    m_saida = np.empty((len(freqs), len(tempos)))
    for i, f in enumerate(freqs):
        m_saida[i] = agudos if f > freqs_a[-1] else m[np.argmin(np.abs(freqs_a - f))]

    nper = int(round(janela_s * sr))
    _, y = istft(x * m_saida[None], sr, nperseg=nper,
                 noverlap=nper - int(round(passo_s * sr)), time_axis=-1, freq_axis=-2)
    y = y[:, : len(trecho)]
    if y.shape[1] < len(trecho):
        y = np.pad(y, ((0, 0), (0, len(trecho) - y.shape[1])))
    return y.T.astype(np.float32)


def _tem_voz(fonte, sr, abaixo_do_pico_db=35.0, min_pausa_s=0.3):
    """Marca as amostras em que a fonte separada tem voz (o resto é resíduo/silêncio)."""
    q = int(sr * QUADRO_S)
    n = len(fonte) // q
    if n == 0:
        return np.zeros(len(fonte), dtype=bool)
    db = 10 * np.log10((fonte[: n * q].reshape(n, q) ** 2).mean(1) + 1e-12)
    voz = db > max(db.max() - abaixo_do_pico_db, -60.0)
    # Não corta pausas curtas no meio de uma frase.
    for a, b in trechos(~voz):
        if b - a <= int(min_pausa_s / QUADRO_S) and a > 0 and b < n:
            voz[a:b] = True
    voz = np.repeat(voz, q)
    return np.pad(voz, (0, len(fonte) - len(voz)), constant_values=bool(voz[-1]))


def _ajustar(x, n):
    if x.shape[-1] >= n:
        return x[..., :n]
    return np.pad(x, [(0, 0)] * (x.ndim - 1) + [(0, n - x.shape[-1])])
