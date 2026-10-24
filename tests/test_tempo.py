import pytest

from separador.tempo import ler_intervalo, ler_tempo


@pytest.mark.parametrize("texto,esperado", [
    ("90", 90), ("1:30", 90), ("01:02:03", 3723), ("10:00.5", 600.5), ("0:05,25", 5.25),
])
def test_ler_tempo(texto, esperado):
    assert ler_tempo(texto) == pytest.approx(esperado)


@pytest.mark.parametrize("texto", ["10:00-14:45", "10:00 - 14:45", " 10:00 -14:45 "])
def test_ler_intervalo(texto):
    assert ler_intervalo(texto) == (600, 885)


def test_nao_intervalo():
    assert ler_intervalo("voz.wav") is None
    assert ler_intervalo("10:00") is None


def test_intervalo_invalido():
    with pytest.raises(ValueError):
        ler_intervalo("14:45 - 10:00")
    with pytest.raises(ValueError):
        ler_intervalo("10:75 - 11:00")
