# Separador de faixa

Isola a voz de **uma pessoa específica** em uma gravação com vários locutores — por
exemplo, extrair só a fala de um dos apresentadores de um podcast com duas pessoas.

## Como funciona

1. **Detecção de fala (VAD):** encontra os trechos com voz e descarta silêncios.
2. **Impressão vocal:** o áudio é cortado em janelas de 1,5 s e cada uma vira um vetor
   que descreve o timbre de quem fala (modelo [ECAPA-TDNN](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb)
   do SpeechBrain, treinado para reconhecer locutores).
3. **Agrupamento:** as janelas são agrupadas em N locutores (padrão 2) — isso é a
   *diarização* ("quem fala quando").
4. **Escolha do alvo:** se você fornece um trecho de referência com a voz da pessoa,
   o programa escolhe automaticamente o grupo mais parecido com ela.
5. **Saída:** gera um áudio em que só a pessoa escolhida aparece, mantendo a qualidade
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
| `--modo concatenar` | junta só os trechos da pessoa (arquivo mais curto); o padrão `mutar` mantém a duração e silencia os outros, útil para editar em um DAW |
| `--segmentos` | salva também um `.csv` com os tempos de início/fim de cada fala |
| `--dispositivo cuda` | usa a GPU (bem mais rápido em episódios longos) |
| `--min-segmento 0.4` | trocas de locutor mais curtas que isso são ignoradas |
| `--margem-vad 12` | aumente se ruído de fundo estiver passando como fala; diminua se falas baixas forem cortadas |
| `--modelo mfcc` | extrator leve e 100% offline, porém bem menos preciso |

O formato de saída segue a extensão (`.wav`, `.flac`, `.mp3`, `.m4a`...).

## Limitações

- **Falas sobrepostas:** quando as duas pessoas falam ao mesmo tempo, o trecho vai
  inteiro para quem predomina — a voz da outra pessoa aparece junto. Para separar a
  sobreposição em si seria preciso um modelo de separação de fontes (ex.: SepFormer),
  o que pode ser adicionado como etapa extra.
- **Vozes muito parecidas** (mesmo gênero, timbre semelhante) podem gerar trocas
  pontuais; uma referência mais longa e limpa ajuda.
- Interjeições curtas ("aham", "sim") de menos de ~0,4 s tendem a ser absorvidas pela
  fala de quem está com a palavra.

## Testes

```bash
pip install pytest
python -m pytest
```

Os testes usam vozes sintéticas e o extrator `mfcc`, então rodam offline.
