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


def _vozes_com_vinheta(rng, n=3000, d=192, n_atipicas=40):
    """Duas vozes parecidas (cos ~0.5) + janelas atípicas bem distantes."""
    base = rng.standard_normal(d)
    a, b = base + 0.9 * rng.standard_normal(d), base + 0.9 * rng.standard_normal(d)
    quem = rng.integers(0, 2, n)
    vozes = np.where(quem[:, None] == 0, a, b) + 1.5 * rng.standard_normal((n, d))
    atipicas = 3 * rng.standard_normal(d) + 1.5 * rng.standard_normal((n_atipicas, d))
    embs = np.vstack([vozes, atipicas])
    embs /= np.linalg.norm(embs, axis=1, keepdims=True)
    return embs, quem, a / np.linalg.norm(a)


def test_janelas_atipicas_nao_formam_grupo_proprio():
    # Antes, 40 janelas de vinheta viravam um "locutor" e as duas vozes ficavam juntas.
    from separador.pipeline import agrupar

    embs, quem, ref_a = _vozes_com_vinheta(np.random.default_rng(0))
    for ref in (None, ref_a):
        rotulos = agrupar(embs, 2, ref)[: len(quem)]
        acerto = max(np.mean(rotulos == quem), np.mean(rotulos != quem))
        assert acerto > 0.95
    # Com referência, o grupo 0 nasce dela: deve ser a pessoa A.
    assert np.mean(agrupar(embs, 2, ref_a)[: len(quem)] == quem) > 0.95


def test_agrupar_episodio_longo_e_rapido():
    import time

    from separador.pipeline import agrupar

    embs, quem, _ = _vozes_com_vinheta(np.random.default_rng(1), n=20000)
    t = time.time()
    rotulos = agrupar(embs, 2)[: len(quem)]
    assert time.time() - t < 10
    assert max(np.mean(rotulos == quem), np.mean(rotulos != quem)) > 0.95
