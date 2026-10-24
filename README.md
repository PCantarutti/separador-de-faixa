# Separador de faixa

Isola a voz de **uma pessoa específica** em uma gravação com vários locutores — por
exemplo, extrair só a fala de um dos apresentadores de um podcast com duas pessoas.

## Como funciona

1. **Detecção de fala (VAD):** encontra os trechos com voz e descarta silêncios.
2. **Impressão vocal:** o áudio é cortado em janelas de 1,5 s e cada uma vira um vetor
   que descreve o timbre de quem fala (modelo [ECAPA-TDNN](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb)
   do SpeechBrain, treinado para reconhecer locutores).
3. **Agrupamento:** as janelas são agrupadas em N locutores (padrão 2) com k-means
   por similaridade de cosseno — isso é a *diarização* ("quem fala quando"). Com
   referência, o grupo da pessoa desejada parte da própria referência.
4. **Escolha do alvo:** se você fornece um trecho de referência com a voz da pessoa,
   o programa escolhe automaticamente o grupo mais parecido com ela.
5. **Falas sobrepostas (opcional, `-s`):** nos trechos em que as duas pessoas podem
   estar falando juntas, o [SepFormer](https://huggingface.co/speechbrain/sepformer-whamr16k)
   divide a mistura em duas vozes; a que tiver a impressão vocal da pessoa escolhida
   entra na faixa final.
6. **Saída:** gera um áudio em que só a pessoa escolhida aparece, mantendo a qualidade
   e a taxa de amostragem originais (com *fades* curtos para não haver cliques).

## Instalação

Requer Python 3.10+ e o [ffmpeg](https://ffmpeg.org/) instalado no PATH.

```bash
python -m venv .venv
source .venv/bin/activate        # no Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Na primeira execução o modelo ECAPA (~80 MB) é baixado para `modelos/ecapa`.

## Uso

### Com um trecho de referência (recomendado)

Recorte de 5 a 30 segundos em que **só a pessoa desejada** fala (pode ser do próprio
podcast) e passe com `-r`:

```bash
python -m separador podcast.mp3 -r voz_da_maria.wav -o maria.wav
```

Saída de exemplo:

```
Identificando quem fala quando...
  locutor 0:   1834.2 s de fala em 412 trechos  similaridade com a referência: 0.71  <- alvo
  locutor 1:   1620.9 s de fala em 398 trechos  similaridade com a referência: 0.12
Salvo: maria.wav
```

Pode passar várias referências (`-r a.wav -r b.wav`) para uma impressão vocal mais estável.

### Com um trecho da própria gravação como referência

Em vez de recortar um arquivo, informe o intervalo (início - fim) em que só a pessoa
desejada fala:

```bash
python -m separador podcast.mp3 -r 10:00 - 14:45 -o maria.wav
python -m separador podcast.mp3 -r 10:00-10:30 -r 52:10-52:40   # vários intervalos
```

Aceita `SS`, `MM:SS` ou `HH:MM:SS` (com decimais, ex.: `1:02:03.5`). O ideal é um
trecho de 10 a 60 s sem a outra pessoa falando; trechos maiores funcionam, mas quanto
mais a outra voz aparecer neles, pior fica a identificação.

### Separando falas sobrepostas

```bash
python -m separador podcast.mp3 -r 10:00 - 14:45 -s
```

Com `-s`, o SepFormer roda nas trocas de locutor (onde um costuma começar antes do
outro terminar) e nos 20% de trechos em que a identificação ficou mais ambígua. Para
recuperar também os "aham"/"sim" ditos no meio da fala do outro, use
`--sobreposicao-tudo`, que roda o SepFormer em toda a fala — bem mais lento, então
prefira com GPU (`--dispositivo cuda`).

O modelo (~110 MB) é baixado na primeira vez para `modelos/`. Outros modelos SepFormer
do SpeechBrain podem ser usados com `--modelo-sobreposicao`, por exemplo
`speechbrain/sepformer-wsj02mix` (8 kHz, voz limpa).

### Sem referência

```bash
python -m separador podcast.mp3 --todos
```

Gera `podcast_locutor0.wav` e `podcast_locutor1.wav`. Ouça e, nas próximas vezes,
use `--locutor 0` ou `--locutor 1` para gerar só o que interessa.

### Opções úteis

| Opção | Descrição |
|---|---|
| `-n 3` | número de pessoas na gravação (padrão 2) |
| `--trecho 20:00-25:00` | processa só esse pedaço (bom para testar rápido); os tempos de `-r` continuam contando do início do arquivo |
| `--modo concatenar` | junta só os trechos da pessoa (arquivo mais curto); o padrão `mutar` mantém a duração e silencia os outros, útil para editar em um DAW |
| `--segmentos` | salva também um `.csv` com os tempos de início/fim de cada fala |
| `--dispositivo cuda` | usa a GPU (bem mais rápido em episódios longos) |
| `--min-segmento 0.4` | trocas de locutor mais curtas que isso são ignoradas |
| `--margem-vad 12` | aumente se ruído de fundo estiver passando como fala; diminua se falas baixas forem cortadas |
| `--modelo mfcc` | extrator leve e 100% offline, porém bem menos preciso |
| `-s` / `--sobreposicao-tudo` | separa falas sobrepostas com SepFormer (veja acima) |
| `--percentil-ambiguo 20` | com `-s`, quanto da fala (os trechos mais ambíguos) vai para o SepFormer |

O formato de saída segue a extensão (`.wav`, `.flac`, `.mp3`, `.m4a`...).

## Limitações

- **Falas sobrepostas sem `-s`:** o trecho vai inteiro para quem predomina — a voz da
  outra pessoa aparece junto.
- **Com `-s`:** o modo padrão detecta bem as sobreposições nas trocas de locutor, mas
  pode deixar passar interjeições curtas no meio da fala do outro; para essas, use
  `--sobreposicao-tudo`. O SepFormer trabalha em 16 kHz e em mono, então nos trechos
  separados o som perde agudos acima de 8 kHz e a imagem estéreo, e pode ter leves
  artefatos.
- **Vozes muito parecidas** (mesmo gênero, timbre semelhante) podem gerar trocas
  pontuais; uma referência mais longa e limpa ajuda.
- Sem `--sobreposicao-tudo`, interjeições curtas ("aham", "sim") de menos de ~0,4 s
  tendem a ser absorvidas pela fala de quem está com a palavra.
- Coloque o arquivo de entrada **antes** das opções (`separador podcast.mp3 -r ...`).

## Testes

```bash
pip install pytest
python -m pytest
```

Os testes usam vozes sintéticas, o extrator `mfcc` e um SepFormer simulado, então
rodam offline.
