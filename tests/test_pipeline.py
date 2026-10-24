import numpy as np
import soundfile as sf

from separador import cli
from separador.embeddings import EmbedderMFCC
from separador.pipeline import separar
from tests.sintetico import podcast

SR = 16000


def _acerto(previsto, verdade, mapeamento):
    n = min(len(previsto), len(verdade))
    p, v = previsto[:n], verdade[:n]
    fala = (v >= 0) & (p >= 0)
    return np.mean(np.array([mapeamento[x] for x in p[fala]]) == v[fala])


def test_referencia_identifica_locutor_certo():
    sinal, verdade, refs = podcast()
    for quem in (0, 1):
        res = separar(sinal, SR, EmbedderMFCC(), referencias=[refs[quem]])
        # O alvo escolhido deve ser o grupo que mais coincide com a pessoa `quem`.
        n = min(len(res.rotulos), len(verdade))
        coincide = np.mean(verdade[:n][res.rotulos[:n] == res.alvo] == quem)
        assert coincide > 0.9


def test_diarizacao_tem_alta_precisao():
    sinal, verdade, _ = podcast(semente=3)
    res = separar(sinal, SR, EmbedderMFCC())
    acerto = max(_acerto(res.rotulos, verdade, {0: 0, 1: 1}),
                 _acerto(res.rotulos, verdade, {0: 1, 1: 0}))
    assert acerto > 0.9


def test_cli_ponta_a_ponta(tmp_path, monkeypatch):
    sinal, verdade, refs = podcast(semente=5)
    estereo = np.stack([sinal, sinal * 0.8], axis=1)
    sf.write(tmp_path / "pod.wav", estereo, SR)
    sf.write(tmp_path / "ref.wav", refs[1], SR)
    saida = tmp_path / "so_b.wav"
    cli.main([str(tmp_path / "pod.wav"), "-r", str(tmp_path / "ref.wav"),
              "-o", str(saida), "--modelo", "mfcc", "--segmentos"])
    out, sr = sf.read(saida)
    assert sr == SR and out.shape == estereo.shape
    # Energia deve sobrar nos trechos da pessoa B e quase sumir nos da A.
    q = SR // 100
    n = min(len(out) // q, len(verdade))
    quadros = out[: n * q].reshape(n, q, 2)
    v = verdade[:n]
    energia = lambda m: np.mean(quadros[m] ** 2)
    assert energia(v == 1) > 50 * energia(v == 0)
    assert (tmp_path / "so_b.csv").read_text().startswith("inicio,fim")
