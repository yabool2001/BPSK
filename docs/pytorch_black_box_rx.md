# PyTorch Black-Box RX

Nowy tor jest w dwóch plikach:

- `modules/neural_rx.py` - dataset, model 1D CNN, trening i inferencja bitowa na surowym I/Q.
- `bpsk_rx_nn.py` - CLI do treningu syntetycznego i scoringu plikow lub zywego bufora z Pluto/libiio.

## Zalozenie modelu

Model dostaje okno surowych probek I/Q o ksztalcie `2 x N` i zwraca:

- `p(bit=0 | okno)`
- `p(bit=1 | okno)`

Nie uzywa klasycznego lancucha RX jako warunku decyzyjnego. Okno jest traktowane jako czarna skrzynka.

## Trening syntetyczny

Przyklad szybkiego startu:

```bash
/home/yabool2001/python/BPSK/.venv/bin/python bpsk_rx_nn.py train-synth \
  --examples 4096 \
  --epochs 8 \
  --batch-size 8 \
  --window-size 131072 \
  --checkpoint logs/black_box_bit_net.pt
```

Dataset syntetyczny bierze istniejaca definicje ramki BPSK i naklada losowe:

- AWGN
- CFO
- rotacje fazy
- zmiane amplitudy
- DC offset
- prosty skew I/Q

## Scoring zapisanego capture

Jesli znasz payload wyslany przez TX i chcesz sprawdzic bit po bicie:

```bash
/home/yabool2001/python/BPSK/.venv/bin/python bpsk_rx_nn.py score-known-file \
  --input samples/rx_samples_00_issue40_002.0.si.npy \
  --payload 15,15,15,15 \
  --checkpoint logs/black_box_bit_net.pt \
  --window-size 131072
```

Jesli nie znasz payloadu i chcesz dostac przesuwany profil prawdopodobienstw:

```bash
/home/yabool2001/python/BPSK/.venv/bin/python bpsk_rx_nn.py score-file \
  --input samples/rx_samples_00_issue40_002.0.si.npy \
  --checkpoint logs/black_box_bit_net.pt \
  --window-size 131072 \
  --stride 4
```

## Scoring live z Pluto

```bash
/home/yabool2001/python/BPSK/.venv/bin/python bpsk_rx_nn.py score-live \
  --checkpoint logs/black_box_bit_net.pt \
  --window-size 131072 \
  --stride 4
```

## Uwaga praktyczna

`window-size=131072` odpowiada jednemu pelnemu buforowi z Pluto, ale jest kosztowne pamieciowo. Do debugowania szybciej jest zaczac od `1024`, `2048` lub `4096`, a dopiero potem przejsc na pelny bufor.