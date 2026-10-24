"""Extratores de "impressão vocal" (embeddings de locutor)."""

import numpy as np


class EmbedderECAPA:
    """ECAPA-TDNN treinado no VoxCeleb (SpeechBrain). Recomendado.

    Na primeira execução baixa ~80 MB do Hugging Face.
    """

    def __init__(self, dispositivo="cpu", pasta_modelo="modelos/ecapa"):
        import torch
        from speechbrain.inference.speaker import EncoderClassifier

        self._torch = torch
        self.modelo = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=pasta_modelo,
            run_opts={"device": dispositivo},
        )

    def __call__(self, janelas, lote=32):
        saidas = []
        with self._torch.no_grad():
            for i in range(0, len(janelas), lote):
                x = self._torch.from_numpy(np.stack(janelas[i : i + lote]))
                emb = self.modelo.encode_batch(x).squeeze(1).cpu().numpy()
                saidas.append(emb)
        return _normalizar(np.concatenate(saidas))


class EmbedderMFCC:
    """Estatísticas de MFCC. Leve e offline, mas bem menos preciso que o ECAPA.

    Útil para testes ou quando não há como baixar o modelo.
    """

    def __init__(self, sr=16000, n_mfcc=20):
        self.sr = sr
        self.n_mfcc = n_mfcc
        self._banco = _banco_mel(sr, n_fft=512, n_mels=40)

    def __call__(self, janelas):
        embs = []
        for w in janelas:
            mfcc = self._mfcc(w)
            embs.append(np.concatenate([mfcc.mean(0), mfcc.std(0)]))
        embs = np.array(embs)
        embs -= embs.mean(0, keepdims=True)
        embs /= embs.std(0, keepdims=True) + 1e-8
        return _normalizar(embs)

    def _mfcc(self, w):
        n_fft, passo = 512, 160
        w = np.append(w[0], w[1:] - 0.97 * w[:-1])
        n = 1 + (len(w) - n_fft) // passo
        idx = np.arange(n_fft)[None, :] + passo * np.arange(n)[:, None]
        espectro = np.abs(np.fft.rfft(w[idx] * np.hanning(n_fft), axis=1)) ** 2
        logmel = np.log(espectro @ self._banco.T + 1e-10)
        k = np.arange(logmel.shape[1])
        dct = np.cos(np.pi / logmel.shape[1] * (k[None, :] + 0.5) * np.arange(self.n_mfcc)[:, None])
        return logmel @ dct.T


def _banco_mel(sr, n_fft, n_mels):
    def hz_mel(f):
        return 2595 * np.log10(1 + f / 700)

    def mel_hz(m):
        return 700 * (10 ** (m / 2595) - 1)

    pontos = mel_hz(np.linspace(hz_mel(0), hz_mel(sr / 2), n_mels + 2))
    bins = np.floor((n_fft + 1) * pontos / sr).astype(int)
    banco = np.zeros((n_mels, n_fft // 2 + 1))
    for i in range(1, n_mels + 1):
        a, b, c = bins[i - 1], bins[i], bins[i + 1]
        banco[i - 1, a:b] = (np.arange(a, b) - a) / max(b - a, 1)
        banco[i - 1, b:c] = (c - np.arange(b, c)) / max(c - b, 1)
    return banco


def _normalizar(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)


def criar(nome, dispositivo="cpu"):
    if nome == "ecapa":
        return EmbedderECAPA(dispositivo=dispositivo)
    if nome == "mfcc":
        return EmbedderMFCC()
    raise ValueError(f"modelo desconhecido: {nome}")
