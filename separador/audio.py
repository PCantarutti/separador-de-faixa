"""Leitura e escrita de áudio usando ffmpeg (aceita mp3, m4a, wav, flac, ...)."""

import shutil
import subprocess

import numpy as np
import soundfile as sf

SR_ANALISE = 16000


def _exigir_ffmpeg():
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg não encontrado. Instale-o e deixe-o no PATH.")


def carregar(caminho, sr=None, mono=False, inicio=None, duracao=None):
    """Decodifica qualquer formato suportado pelo ffmpeg.

    Retorna (amostras float32 com forma [n, canais], taxa_de_amostragem).
    Sem `sr`, mantém a taxa original do arquivo. `inicio` e `duracao` (em
    segundos) leem só um pedaço do arquivo.
    """
    _exigir_ffmpeg()
    if sr is None:
        sr = _ffprobe(caminho, "sample_rate")
    canais = 1 if mono else _ffprobe(caminho, "channels")
    cmd = ["ffmpeg", "-v", "error"]
    if inicio is not None:
        cmd += ["-ss", f"{inicio:.3f}"]
    if duracao is not None:
        cmd += ["-t", f"{duracao:.3f}"]
    cmd += ["-i", str(caminho), "-vn",
            "-f", "f32le", "-ar", str(sr), "-ac", str(canais), "-"]
    bruto = subprocess.run(cmd, capture_output=True, check=True).stdout
    dados = np.frombuffer(bruto, dtype=np.float32).reshape(-1, canais)
    return dados.copy(), sr


def salvar(caminho, dados, sr):
    """Salva em qualquer formato; wav/flac/ogg direto, o resto via ffmpeg."""
    caminho = str(caminho)
    ext = caminho.rsplit(".", 1)[-1].lower()
    if ext in ("wav", "flac", "ogg"):
        sf.write(caminho, dados, sr)
        return
    _exigir_ffmpeg()
    canais = 1 if dados.ndim == 1 else dados.shape[1]
    cmd = [
        "ffmpeg", "-v", "error", "-y",
        "-f", "f32le", "-ar", str(sr), "-ac", str(canais), "-i", "-",
        caminho,
    ]
    subprocess.run(cmd, input=np.ascontiguousarray(dados, dtype=np.float32).tobytes(), check=True)


def duracao(caminho):
    """Duração do arquivo em segundos, segundo o ffprobe."""
    saida = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(caminho)],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    return float(saida) if saida and saida != "N/A" else 0.0


def _ffprobe(caminho, campo):
    saida = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0",
         "-show_entries", f"stream={campo}", "-of", "csv=p=0", str(caminho)],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    return int(saida.split(",")[0])


def reamostrar(sinal, sr_de, sr_para):
    """Reamostra ao longo do último eixo."""
    if sr_de == sr_para:
        return sinal
    from math import gcd

    from scipy.signal import resample_poly

    g = gcd(int(sr_de), int(sr_para))
    return resample_poly(sinal, sr_para // g, sr_de // g, axis=-1).astype(np.float32)
