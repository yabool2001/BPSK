import iio
import numpy as np

# 1. Połączenie bezpośrednio przez libiio
ctx = iio.Context("ip:192.168.2.1")
dev = ctx.find_device("cf-ad9361-lpc") # Nazwa urządzenia wewnątrz Pluto

# 2. Konfiguracja kanałów i bufora
phy = ctx.find_device("ad9361-phy")
v0 = dev.find_channel("voltage0", False) # False = Input
v1 = dev.find_channel("voltage1", False)
v0.enabled = True
v1.enabled = True

# Tworzymy bufor (np. 307200 próbek)
buf = iio.Buffer(dev, 307200, False) # False = nie cykliczny

# 3. Pobranie danych
buf.refill()
raw_bytes = buf.read() # To są surowe bajty!

# 4. Konwersja bajtów na int16 (bez zamiany na float!)
# Pluto wysyła dane jako Little Endian int16
data_int16 = np.frombuffer(raw_bytes, dtype=np.int16)

print(f"Surowe int16: {data_int16[:10]}")
# Uwaga: Dane są tu przeplatane: [I, Q, I, Q, I, Q...]


# To jest to, co 'adi' robi pod spodem, ale bez narzutu
ctx = iio.Context("ip:192.168.2.1")
dev = ctx.find_device("cf-ad9361-lpc") # Nazwa urządzenia w Pluto
rx_chan_i = dev.find_channel("voltage0", False)
rx_chan_q = dev.find_channel("voltage1", False)

rx_chan_i.enabled = True
rx_chan_q.enabled = True

# Utworzenie bufora (np. 4096 próbek)
buf = iio.Buffer(dev, 4096, False) # False = brak cykliczności

# Pobranie danych
buf.refill()
raw_bytes = buf.read() # To są surowe bajty!

# Konwersja bajtów na int16
data_int16 = np.frombuffer(raw_bytes, dtype=np.int16)

# Jeśli chcesz int16 I/Q jako osobne tablice:
rx = rx_pluto.pluto_rx_ctx.rx()
i16 = rx.real.astype(np.int16)
q16 = rx.imag.astype(np.int16)

# Jeśli chcesz pary I/Q jako int16 w jednej tablicy:
rx = rx_pluto.pluto_rx_ctx.rx()
iq_i16 = np.column_stack((rx.real.astype(np.int16), rx.imag.astype(np.int16)))

# Jeśli chcesz tylko mniejszy float (oszczednosc pamieci):

rx = rx_pluto.pluto_rx_ctx.rx().astype(np.complex64)

# Szybka weryfikacja, ze to “raw” (wartosci calkowite):
rx = rx_pluto.pluto_rx_ctx.rx()
print(np.all(rx.real == rx.real.astype(np.int16)))
print(np.all(rx.imag == rx.imag.astype(np.int16)))


