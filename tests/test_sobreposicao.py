import numpy as np
import soundfile as sf

from separador import cli
from separador.sobreposicao import regioes_candidatas
from tests.sintetico import podcast

SR = 16000


class SeparadorPerfeito:
    """Simula o SepFormer devolvendo as fontes verdadeiras do trecho pedido."""

    def __init__(self, mistura, fontes):
        self.mistura, self.fontes = mistura, fontes
        self.chamadas = 0

    def __call__(self, trecho, sr):
        self.chamadas += 1
        candidatos = np.flatnonzero(self.mistura == trecho[0])
        ini = next(c for c in candidatos
                   if np.array_equal(self.mistura[c : c + len(trecho)], trecho))
        # Ordem trocada e escala diferente, como num modelo real.
        return self.fontes[::-1, ini : ini + len(trecho)] * 3.0


def _rodar(tmp_path, args_extra, separador=None):
    sinal, verdade, refs, fontes = podcast(semente=2, sobreposicao=True)
    # Float para que o áudio lido pelo CLI seja idêntico ao da simulação.
    sf.write(tmp_path / "pod.wav", sinal, SR, subtype="FLOAT")
    sf.write(tmp_path / "ref.wav", refs[0], SR)
    saida = tmp_path / "a.wav"
    sep = SeparadorPerfeito(sinal, fontes) if separador else None
    cli.main([str(tmp_path / "pod.wav"), "-r", str(tmp_path / "ref.wav"), "-o", str(saida),
              "--modelo", "mfcc", *args_extra], separador=sep)
    out = sf.read(saida)[0]
    q = SR // 100
    n = min(len(out) // q, len(verdade))
    erro = ((out[: n * q] - fontes[0, : n * q]) ** 2).reshape(n, q).mean(1)
    sobreposto = verdade[:n] == 2
    return erro[sobreposto].mean() / (fontes[0, : n * q] ** 2).reshape(n, q).mean(1)[sobreposto].mean()


def test_sepformer_recupera_voz_nas_sobreposicoes(tmp_path):
    sem = _rodar(tmp_path, [])
    com = _rodar(tmp_path, ["--sobreposicao-tudo"], separador=True)
    # Sem separação, a interjeição de A some (ou vem junto com B): erro ~100%.
    assert sem > 0.5
    assert com < 0.1


def test_modo_auto_processa_so_parte_da_fala():
    rotulos = np.array([-1] * 50 + [0] * 500 + [1] * 500 + [-1] * 300 + [1] * 2000)
    margem = np.random.default_rng(0).uniform(0.5, 1.0, len(rotulos))
    margem[2000:2010] = 0.0  # trecho ambíguo
    regioes = regioes_candidatas(rotulos, margem, "auto", percentil=0.3)
    cobertos = np.zeros(len(rotulos), bool)
    for a, b in regioes:
        cobertos[a:b] = True
    assert cobertos[540:560].all()  # troca de locutor
    assert cobertos[2000:2010].all()  # trecho ambíguo
    assert not cobertos[100:400].any() and not cobertos[3000:].any()
    assert all(b - a <= 1500 for a, b in regioes)


def test_referencia_por_intervalo(tmp_path):
    sinal, verdade, _ = podcast(semente=4)
    sf.write(tmp_path / "pod.wav", sinal, SR)
    # Primeiro turno é sempre da pessoa 0; usa os 2 primeiros segundos como referência.
    for args in (["pod.wav", "-r", "0:00", "-", "0:02"], ["-r", "0:00-0:02", "pod.wav"]):
        saida = tmp_path / "x.wav"
        cli.main([str(tmp_path / a) if a == "pod.wav" else a for a in args]
                 + ["-o", str(saida), "--modelo", "mfcc"])
        out = sf.read(saida)[0]
        q = SR // 100
        n = min(len(out) // q, len(verdade))
        energia = (out[: n * q] ** 2).reshape(n, q).mean(1)
        assert energia[verdade[:n] == 0].mean() > 50 * energia[verdade[:n] == 1].mean()


def test_trecho_processa_so_um_pedaco(tmp_path):
    sinal, verdade, _ = podcast(semente=4)
    sf.write(tmp_path / "pod.wav", sinal, SR)
    saida = tmp_path / "x.wav"
    # Referência (0:00-0:02) fora do trecho processado: lida direto do arquivo.
    cli.main([str(tmp_path / "pod.wav"), "-r", "0:00-0:02", "--trecho", "0:10-0:30",
              "-o", str(saida), "--modelo", "mfcc"])
    out = sf.read(saida)[0]
    assert abs(len(out) - 20 * SR) < SR // 100
    q = SR // 100
    v = verdade[1000:3000]
    energia = (out[: len(v) * q] ** 2).reshape(len(v), q).mean(1)
    assert energia[v == 0].mean() > 50 * energia[v == 1].mean()


def test_concatenar_com_sobreposicao_remove_silencios(tmp_path):
    sinal, verdade, refs, fontes = podcast(semente=2, sobreposicao=True)
    sf.write(tmp_path / "pod.wav", sinal, SR, subtype="FLOAT")
    sf.write(tmp_path / "ref.wav", refs[0], SR)
    saida = tmp_path / "a.wav"
    cli.main([str(tmp_path / "pod.wav"), "-r", str(tmp_path / "ref.wav"), "-o", str(saida),
              "--modelo", "mfcc", "--sobreposicao-tudo", "--modo", "concatenar",
              "--pausa", "0.1"], separador=SeparadorPerfeito(sinal, fontes))
    out = sf.read(saida)[0]
    fala_a = np.sum((verdade == 0) | (verdade == 2)) / 100
    # Só a fala de A (incluindo as interjeições) + pausas curtas, sem os turnos de B.
    assert fala_a * 0.9 < len(out) / SR < fala_a + 4
    q = SR // 100
    n = len(out) // q
    db = 10 * np.log10((out[: n * q].reshape(n, q) ** 2).mean(1) + 1e-12)
    assert max(len(x) for x in np.split(db < -50, np.flatnonzero(np.diff(db < -50)) + 1)
               if x[0]) <= 40  # nenhum silêncio maior que ~0,4 s
