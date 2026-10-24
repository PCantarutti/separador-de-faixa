"""Linha de comando do separador de faixa."""

import argparse
import sys
from pathlib import Path

import numpy as np

from . import audio, embeddings, sobreposicao
from .pipeline import mascara_amostras, separar
from .tempo import ler_intervalo


def criar_parser():
    p = argparse.ArgumentParser(
        prog="separador",
        description="Isola a voz de um locutor específico em um podcast/gravação.",
    )
    p.add_argument("entrada", nargs="?", help="arquivo de áudio (mp3, wav, m4a, flac...)")
    p.add_argument("-r", "--referencia", action="append", nargs="+", default=[],
                   metavar="ARQUIVO|INÍCIO-FIM",
                   help="voz da pessoa desejada (5-30 s): um arquivo só com ela, ou um "
                        "intervalo da própria gravação, ex.: -r 10:00-14:45; pode repetir")
    p.add_argument("-t", "--trecho", metavar="INÍCIO-FIM",
                   help="processa só esse pedaço da gravação, ex.: --trecho 20:00-25:00 "
                        "(os tempos de -r continuam contando do início do arquivo)")
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
    p.add_argument("-s", "--sobreposicao", action="store_true",
                   help="separa falas sobrepostas com SepFormer (nas trocas de locutor e "
                        "nos trechos ambíguos)")
    p.add_argument("--sobreposicao-tudo", action="store_true",
                   help="roda o SepFormer em toda a fala, não só nos trechos suspeitos "
                        "(recupera mais interjeições; lento, recomendado com GPU)")
    p.add_argument("--modelo-sobreposicao", default=sobreposicao.MODELO_PADRAO,
                   help="modelo SepFormer do Hugging Face (padrão: %(default)s)")
    p.add_argument("--percentil-ambiguo", type=float, default=20.0,
                   help="com -s, %% da fala (os trechos mais ambíguos) enviada ao SepFormer")
    return p


def ler_intervalo_ou_erro(parser, texto):
    try:
        return ler_intervalo(texto)
    except ValueError as e:
        parser.error(str(e))


def ler_referencias(parser, args, sr):
    """Converte os valores de -r em sinais (arquivos ou intervalos da entrada)."""
    sinais = []
    for grupo in args.referencia:
        texto = " ".join(grupo)
        intervalo = ler_intervalo_ou_erro(parser, texto)
        if intervalo:
            ini, fim = intervalo
            sinal = audio.carregar(args.entrada, sr=sr, mono=True,
                                   inicio=ini, duracao=fim - ini)[0][:, 0]
            if len(sinal) < sr:
                parser.error(f"referência {texto!r} tem menos de 1 s de áudio; o arquivo "
                             f"tem {audio.duracao(args.entrada):.1f} s")
            sinais.append(sinal)
            print(f"Referência: trecho {texto} da entrada ({len(sinal) / sr:.0f} s)",
                  file=sys.stderr)
            continue
        for item in grupo:
            if not Path(item).is_file():
                parser.error(f"referência não encontrada: {item!r} (use um arquivo ou um "
                             "intervalo como 10:00-14:45)")
            sinais.append(audio.carregar(item, sr=sr, mono=True)[0][:, 0])
    return sinais


def corrigir_entrada(parser, args):
    """Permite "separador -r 10:00-14:45 podcast.mp3": o -r engole a entrada."""
    if args.entrada is None and args.referencia:
        grupo = args.referencia[-1]
        if len(grupo) > 1 or not ler_intervalo(grupo[0]):
            args.entrada = grupo.pop()
            if not grupo:
                args.referencia.pop()
    if args.entrada is None:
        parser.error("informe o arquivo de áudio de entrada")


def extrair(original, rotulos, locutor, sr, modo, separados=None, escolhas=None, sr_analise=None):
    ganho = mascara_amostras(rotulos, locutor, len(original), sr)
    faixa = original * ganho[:, None]
    ativo = ganho > 0
    if separados:
        faixa, ativo = sobreposicao.aplicar(faixa, ativo, sr, sr_analise, separados, escolhas)
    if modo == "mutar":
        return faixa
    # Concatena com pequenos respiros de silêncio entre os trechos.
    pausa = np.zeros((int(0.25 * sr), original.shape[1]), dtype=original.dtype)
    partes = []
    m = np.concatenate([[False], ativo, [False]]).astype(np.int8)
    d = np.diff(m)
    for ini, fim in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
        partes += [faixa[ini:fim], pausa]
    return np.concatenate(partes) if partes else original[:0]


def formatar_tempo(s):
    m, s = divmod(s, 60)
    h, m = divmod(int(m), 60)
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def main(argv=None, separador=None):
    parser = criar_parser()
    args = parser.parse_args(argv)
    corrigir_entrada(parser, args)
    entrada = Path(args.entrada)

    corte = {}
    if args.trecho:
        intervalo = ler_intervalo_ou_erro(parser, args.trecho)
        if not intervalo:
            parser.error(f"--trecho inválido: {args.trecho!r} (use INÍCIO-FIM, ex.: 20:00-25:00)")
        corte = {"inicio": intervalo[0], "duracao": intervalo[1] - intervalo[0]}

    print(f"Carregando {entrada}...", file=sys.stderr)
    analise, sr_a = audio.carregar(entrada, sr=audio.SR_ANALISE, mono=True, **corte)
    print(f"  {len(analise) / sr_a / 60:.1f} min de áudio", file=sys.stderr)
    if len(analise) < sr_a:
        parser.error(f"menos de 1 s de áudio para processar; o arquivo tem "
                     f"{audio.duracao(entrada):.1f} s")
    refs = ler_referencias(parser, args, sr_a)

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

    separados = []
    if args.sobreposicao or args.sobreposicao_tudo:
        modo = "tudo" if args.sobreposicao_tudo else "auto"
        regioes = sobreposicao.regioes_candidatas(
            res.rotulos, res.margem, modo, args.percentil_ambiguo)
        total = sum(f - i for i, f in regioes) * 0.01
        print(f"Separando falas sobrepostas em {len(regioes)} trechos ({total:.0f} s)...",
              file=sys.stderr)
        if separador is None:
            separador = sobreposicao.SeparadorSepFormer(args.modelo_sobreposicao,
                                                        args.dispositivo)

        def progresso(i, n):
            print(f"\r  {i}/{n}", end="" if i < n else "\n", file=sys.stderr)

        separados = sobreposicao.separar_regioes(
            analise[:, 0], sr_a, regioes, separador, embedder, progresso)

    original, sr_o = audio.carregar(entrada, **corte)
    for k in alvos:
        escolhas = None
        if separados:
            # A referência, quando existe, descreve melhor o alvo que o centróide do grupo.
            vetor = res.ref if (k == res.alvo and res.ref is not None) else res.centroides[k]
            outros = np.delete(res.centroides, k, axis=0)
            escolhas = sobreposicao.escolher_fontes(separados, vetor, outros)
            aceitos = sum(j is not None for j in escolhas)
            print(f"  locutor {k}: voz recuperada em {aceitos} de {len(separados)} trechos "
                  "sobrepostos", file=sys.stderr)
        if args.saida and len(alvos) == 1:
            destino = Path(args.saida)
        else:
            sufixo = Path(args.saida).suffix if args.saida else ".wav"
            base = Path(args.saida).with_suffix("") if args.saida else entrada.with_suffix("")
            destino = Path(f"{base}_locutor{k}{sufixo}")
        faixa = extrair(original, res.rotulos, k, sr_o, args.modo, separados, escolhas, sr_a)
        audio.salvar(destino, faixa, sr_o)
        print(f"Salvo: {destino.resolve()}", file=sys.stderr)

        if args.segmentos:
            csv = destino.with_suffix(".csv")
            with open(csv, "w", encoding="utf-8") as f:
                f.write("inicio,fim,inicio_s,fim_s\n")
                for i, fim in res.segmentos(k):
                    f.write(f"{formatar_tempo(i)},{formatar_tempo(fim)},{i:.2f},{fim:.2f}\n")
            print(f"Salvo: {csv.resolve()}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
