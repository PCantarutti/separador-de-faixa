"""Linha de comando do separador de faixa."""

import argparse
import sys
from pathlib import Path

import numpy as np

from . import audio, embeddings
from .pipeline import mascara_amostras, separar


def criar_parser():
    p = argparse.ArgumentParser(
        prog="separador",
        description="Isola a voz de um locutor específico em um podcast/gravação.",
    )
    p.add_argument("entrada", help="arquivo de áudio (mp3, wav, m4a, flac...)")
    p.add_argument("-r", "--referencia", action="append", default=[],
                   help="trecho (5-30 s) só com a voz da pessoa desejada; pode repetir")
    p.add_argument("-n", "--locutores", type=int, default=2,
                   help="quantas pessoas falam na gravação (padrão: 2)")
    p.add_argument("-l", "--locutor", type=int,
                   help="sem referência: índice do locutor a extrair (0, 1, ...)")
    p.add_argument("-o", "--saida", help="arquivo de saída (padrão: <entrada>_locutor.wav)")
    p.add_argument("--todos", action="store_true",
                   help="exporta um arquivo para cada locutor")
    p.add_argument("--modo", choices=["mutar", "concatenar"], default="mutar",
                   help="mutar: mantém a duração e silencia os outros (padrão); "
                        "concatenar: junta só os trechos da pessoa")
    p.add_argument("--modelo", choices=["ecapa", "mfcc"], default="ecapa",
                   help="extrator de voz: ecapa (preciso, padrão) ou mfcc (leve, offline)")
    p.add_argument("--dispositivo", default="cpu", help="cpu ou cuda")
    p.add_argument("--janela", type=float, default=1.5, help="janela de análise em s")
    p.add_argument("--min-segmento", type=float, default=0.4,
                   help="trocas de locutor mais curtas que isso são ignoradas (s)")
    p.add_argument("--margem-vad", type=float, default=12.0,
                   help="dB acima do ruído de fundo para considerar fala")
    p.add_argument("--segmentos", action="store_true",
                   help="também salva um .csv com os tempos de fala de cada locutor")
    return p


def extrair(original, rotulos, locutor, sr, modo):
    ganho = mascara_amostras(rotulos, locutor, len(original), sr)
    if modo == "mutar":
        return original * ganho[:, None]
    # Concatena com pequenos respiros de silêncio entre os trechos.
    pausa = np.zeros((int(0.25 * sr), original.shape[1]), dtype=original.dtype)
    partes = []
    ativo = ganho > 0
    m = np.concatenate([[False], ativo, [False]]).astype(np.int8)
    d = np.diff(m)
    for ini, fim in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
        partes += [original[ini:fim] * ganho[ini:fim, None], pausa]
    return np.concatenate(partes) if partes else original[:0]


def formatar_tempo(s):
    m, s = divmod(s, 60)
    h, m = divmod(int(m), 60)
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def main(argv=None):
    args = criar_parser().parse_args(argv)
    entrada = Path(args.entrada)

    print(f"Carregando {entrada}...", file=sys.stderr)
    analise, sr_a = audio.carregar(entrada, sr=audio.SR_ANALISE, mono=True)
    refs = [audio.carregar(r, sr=audio.SR_ANALISE, mono=True)[0][:, 0] for r in args.referencia]

    print(f"Carregando modelo '{args.modelo}'...", file=sys.stderr)
    embedder = embeddings.criar(args.modelo, args.dispositivo)

    print("Identificando quem fala quando...", file=sys.stderr)
    res = separar(
        analise[:, 0], sr_a, embedder,
        n_locutores=args.locutores,
        referencias=refs,
        janela_s=args.janela,
        min_segmento_s=args.min_segmento,
        margem_vad_db=args.margem_vad,
    )

    for k in range(args.locutores):
        segs = res.segmentos(k)
        total = sum(f - i for i, f in segs)
        extra = ""
        if res.similaridade_ref is not None:
            extra = f"  similaridade com a referência: {res.similaridade_ref[k]:.2f}"
            if k == res.alvo:
                extra += "  <- alvo"
        print(f"  locutor {k}: {total:7.1f} s de fala em {len(segs)} trechos{extra}",
              file=sys.stderr)

    if args.todos:
        alvos = list(range(args.locutores))
    elif res.alvo is not None:
        alvos = [res.alvo]
    elif args.locutor is not None:
        alvos = [args.locutor]
    else:
        print("\nSem --referencia nem --locutor: exportando todos os locutores. "
              "Ouça os arquivos e use --locutor N, ou forneça uma referência.",
              file=sys.stderr)
        alvos = list(range(args.locutores))

    original, sr_o = audio.carregar(entrada)
    for k in alvos:
        if args.saida and len(alvos) == 1:
            destino = Path(args.saida)
        else:
            sufixo = Path(args.saida).suffix if args.saida else ".wav"
            base = Path(args.saida).with_suffix("") if args.saida else entrada.with_suffix("")
            destino = Path(f"{base}_locutor{k}{sufixo}")
        audio.salvar(destino, extrair(original, res.rotulos, k, sr_o, args.modo), sr_o)
        print(f"Salvo: {destino}", file=sys.stderr)

        if args.segmentos:
            csv = destino.with_suffix(".csv")
            with open(csv, "w", encoding="utf-8") as f:
                f.write("inicio,fim,inicio_s,fim_s\n")
                for i, fim in res.segmentos(k):
                    f.write(f"{formatar_tempo(i)},{formatar_tempo(fim)},{i:.2f},{fim:.2f}\n")
            print(f"Salvo: {csv}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
