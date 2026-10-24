"""Interpretação de tempos ("90", "1:30", "01:02:03.5") e intervalos ("10:00 - 14:45")."""

import re

_TEMPO = r"\d+(?::\d{1,2}){0,2}(?:[.,]\d+)?"
_INTERVALO = re.compile(rf"^\s*({_TEMPO})\s*-\s*({_TEMPO})\s*$")


def ler_tempo(texto):
    """Converte "SS", "MM:SS" ou "HH:MM:SS" (com decimais opcionais) em segundos."""
    partes = texto.strip().replace(",", ".").split(":")
    for p in partes[1:]:
        if float(p) >= 60:
            raise ValueError(f"tempo inválido: {texto!r}")
    return sum(float(p) * 60**i for i, p in enumerate(reversed(partes)))


def ler_intervalo(texto):
    """Retorna (início_s, fim_s) se `texto` for um intervalo, senão None."""
    m = _INTERVALO.match(texto)
    if not m:
        return None
    ini, fim = ler_tempo(m.group(1)), ler_tempo(m.group(2))
    if fim <= ini:
        raise ValueError(f"intervalo com fim antes do início: {texto!r}")
    return ini, fim
