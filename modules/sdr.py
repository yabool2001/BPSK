from pprint import pprint
import adi
import iio
import numpy as np
import plotly.express as px
import os
import time , tomllib

from dataclasses import dataclass, field
from modules import plot , filters
from numpy.typing import NDArray

script_filename = os.path.basename ( __file__ )

with open ( "settings.toml" , "rb" ) as settings_file :
    toml_settings = tomllib.load ( settings_file )

PLUTO_TX_SN = toml_settings["ADALM-Pluto"]["URI"]["SN_TX"]
PLUTO_RX_SN = toml_settings["ADALM-Pluto"]["URI"]["SN_RX"]
F_C = int ( toml_settings["ADALM-Pluto"][ "F_C" ] )    # Carrier frequency [Hz]
BW  = int ( toml_settings["ADALM-Pluto"][ "BW" ] )     # BandWidth [Hz]
#F_S = 521100     # Sampling frequency [Hz] >= 521e3 && <
F_S = int ( BW * 3 if ( BW * 3 ) >= 521100 and ( BW * 3 ) <= 61440000 else 521100 ) # Sampling frequency [Hz]
TX_GAIN = float ( toml_settings[ "ADALM-Pluto" ][ "TX_GAIN" ] )
RX_GAIN = int ( toml_settings[ "ADALM-Pluto" ][ "RX_GAIN" ] )
GAIN_CONTROL = toml_settings[ "ADALM-Pluto" ][ "GAIN_CONTROL" ]
SAMPLES_BUFFER_SIZE = int ( toml_settings[ "ADALM-Pluto" ][ "SAMPLES_BUFFER_SIZE" ] )
RX_OUTPUT_TYPE = toml_settings[ "ADALM-Pluto" ][ "RX_OUTPUT_TYPE" ]
PLUTO_DAC_SCALE = 16384  # precomputed value of 2**14 for slight performance gain. The PlutoSDR expects samples to be between -2^14 and +2^14, not -1 and +1 like some SDRs

f_c_tx0_readback = 0
f_c_rx0_readback = 0
f_s_tx0_readback = 0
f_s_rx0_readback = 0
bw_tx0_readback = 0
bw_rx0_readback = 0
tx0_gain_readback = 0
rx0_gain_readback = 0
rx0_gain_control_mode_readback = ""
rx0_samples_buffer_size_readback = 0

def init_pluto_v0_0_0 ( sn : str , tx_gain_float : float = TX_GAIN , gain_control_mode_chan0 : str = GAIN_CONTROL , rx_gain_chan0_int : int = RX_GAIN ) -> tuple[ iio.Context , iio.Buffer ] :
    '''Zarówno dla odbiornika (ADC), jak i nadajnika (DAC), kanały nazywają się tak samo:
        voltage0 output = True (kanał TX - Transmit)
        voltage0 output = False (kanał RX - Receive)
        voltage1 (kanał Q - Quadrature)
    Jedyną rzeczą, która je odróżnia w funkcji find_channel, jest flaga kierunku ( True dla wyjścia/TX , False dla wejścia/RX ).'''

    uri = get_uri ( sn )
    if uri is None:
        raise ValueError ( f"Error! ADALM-Pluto SN: {sn} is not connected. Check USB connection or IP settings.")
    ctx = iio.Context ( uri )
    rx_dev = ctx.find_device ( "cf-ad9361-lpc" )            # To jest "rura RX" (strumieniowanie)
    tx_dev = ctx.find_device ( "cf-ad9361-dds-core-lpc" )   # To jest "rura TX"
    phy = ctx.find_device ( "ad9361-phy" )                  # To jest "mózg" (ustawienia RF)
    
    # Ustawienie kanałów ohy
    f_c_rx0_readback = set_f_c_rx0 ( phy , F_C )
    f_c_tx0_readback = set_f_c_tx0 ( phy , F_C )
    f_s_rx0_readback = set_f_s_rx0 ( phy , F_S )
    f_s_tx0_readback = set_f_s_tx0 ( phy , F_S )
    
    # Ustawienie Sample Rate i Bandwidth. Ustawiamy to na kanale fizycznym (zazwyczaj voltage0 w PHY steruje całym chipem)
    rx_phy_chan = phy.find_channel ( "voltage0" , is_output = False ) # False = Input (RX)
    tx_phy_chan = phy.find_channel ( "voltage0" , is_output = True )  # True = Output (TX)
    
    # Sample rate (wpływa na oba tory zazwyczaj, ale ustawiamy na RX)
    rx_phy_chan.attrs[ "sampling_frequency" ].value = str ( int ( F_S ) )
    # Bandwidth
    rx_phy_chan.attrs[ "rf_bandwidth" ].value = str ( int ( BW ) )
    
    # RX Gain
    rx_phy_chan.attrs[ "gain_control_mode" ].value = gain_control_mode_chan0 
    # Jeśli manual, to ustawiamy wartość (zawsze jako string):
    if gain_control_mode_chan0 == "manual":
        rx_phy_chan.attrs[ "hardwaregain" ].value = str ( int ( rx_gain_chan0_int ) )

    # TX Gain (Tłumienie) - w AD9361 to zazwyczaj ujemne dB lub tłumienie
    tx_phy_chan.attrs[ "hardwaregain" ].value = str ( int ( tx_gain_float ) )

    # Konfiguracja Bufora i Strumieniowania (To odpowiada sdr.rx_buffer_size)
    # Musimy włączyć kanały I oraz Q w urządzeniu strumieniującym (nie PHY!)
    rx_v0 = rx_dev.find_channel ( "voltage0" , False )
    rx_v1 = rx_dev.find_channel ( "voltage1" , False )
    rx_v0.enabled = True
    rx_v1.enabled = True

    # Tworzenie bufora (sdr.rx_buffer_size = SAMPLES_BUFFER_SIZE)
    # Trzeci parametr False = Cyclic Buffer wyłączony
    return ctx , iio.Buffer ( rx_dev , SAMPLES_BUFFER_SIZE , False ) # False = nie cykliczny

def get_uri ( serial : str , type_preference : str = "usb" ) -> str | None :
    """ Zwraca URI kontekstu IIO dla danego numeru seryjnego.
    
    Arguments:
    - serial (str): numer seryjny urządzenia (pełny).
    - type_preference (str): "usb" lub "ip". Jeśli "ip", preferuje ip: ale wraca do usb: jeśli ip nie znaleziono.

    Returns:
    - str: URI w formacie usb:x.y.z lub ip:adres
    - None: jeśli nie znaleziono pasującego urządzenia """

    contexts = iio.scan_contexts ()

    ip_match = None
    usb_match = None

    for uri , description in contexts.items () :
        if serial in description:
            if uri.startswith ( "ip:" ) and type_preference == "ip" :
                ip_match = uri
            elif uri.startswith ( "usb:" ) :
                usb_match = uri

    if type_preference == "ip" and ip_match is not None :
        return ip_match or usb_match
    elif type_preference == "usb" and usb_match is not None :
        return usb_match

    return None

def set_f_c_rx0 ( phy : iio.Device , F_C : int ) -> int :
    """ Ustawia częstotliwość LO dla RX0 i zwraca odczytaną wartość po ustawieniu. """
    lo_rx0_channel = phy.find_channel ( toml_settings["ADALM-Pluto"]["channels"]["lo_rx0_channel_name"] , is_output = True ) # UWAGA: LO jest kanałem wyjściowym w PHY dlatego is_output = True, mimo że jest używany do odbioru (RX) - to jest specyfika AD9361
    lo_rx0_channel.attrs[ "frequency" ].value = str ( int ( F_C ) )
    if toml_settings["log"]["verbose_2"] : print ( f"{lo_rx0_channel.id=} {lo_rx0_channel.name=} {int ( lo_rx0_channel.attrs[ 'frequency' ].value )=:,} Hz" )
    return int ( lo_rx0_channel.attrs[ "frequency" ].value )

def set_f_c_tx0 ( phy : iio.Device , F_C : int ) -> int :
    """ Ustawia częstotliwość LO dla TX0 i zwraca odczytaną wartość po ustawieniu. """
    lo_tx0_channel = phy.find_channel ( toml_settings["ADALM-Pluto"]["channels"]["lo_tx0_channel_name"] , is_output = True )
    lo_tx0_channel.attrs[ "frequency" ].value = str ( int ( F_C ) )
    if toml_settings["log"]["verbose_2"] : print ( f"{lo_tx0_channel.id=} {lo_tx0_channel.name=} {int ( lo_tx0_channel.attrs[ 'frequency' ].value )=:,} Hz" )
    return int ( lo_tx0_channel.attrs[ "frequency" ].value )

def set_f_s_rx0 ( phy : iio.Device , F_C : int ) -> int :
    """ Ustawia częstotliwość samplowania dla RX0 i zwraca odczytaną wartość po ustawieniu. """
    rx0_channel = phy.find_channel ( toml_settings["ADALM-Pluto"]["channels"]["rx0tx0_channel_id"] , is_output = False )
    rx0_channel.attrs[ "sampling_frequency" ].value = str ( int ( F_C ) )
    if toml_settings["log"]["verbose_2"] : print ( f"{rx0_channel.id=} {rx0_channel.output=} {int ( rx0_channel.attrs[ 'sampling_frequency' ].value )=:,} Hz" )
    return int ( rx0_channel.attrs[ "sampling_frequency" ].value )

def set_f_s_tx0 ( phy : iio.Device , F_C : int ) -> int :
    """ Ustawia częstotliwość samplowania dla TX0 i zwraca odczytaną wartość po ustawieniu. """
    tx0_channel = phy.find_channel ( toml_settings["ADALM-Pluto"]["channels"]["rx0tx0_channel_id"] , is_output = True )
    tx0_channel.attrs[ "sampling_frequency" ].value = str ( int ( F_C ) )
    if toml_settings["log"]["verbose_2"] : print ( f"{tx0_channel.id=} {tx0_channel.output=} {int ( tx0_channel.attrs[ 'sampling_frequency' ].value )=:,} Hz" )
    return int ( tx0_channel.attrs[ "sampling_frequency" ].value )

def print_pluto_settings ( pluto_ctx : iio.Context ) :
    """ Wyświetlanie konfiguracji obiektu 'iio.Context'. """


    # 2. Znalezienie urządzenia PHY (tam siedzi konfiguracja RF)
    phy = pluto_ctx.find_device ( "ad9361-phy" )
    
    print(f"\n=== KONFIGURACJA SPRZĘTOWA: {phy.name} ===")
    
    # A. Atrybuty globalne urządzenia (np. tryb ENSM, kalibracje)
    print("\n[Ustawienia Globalne]")
    
    for attr in phy.attrs:
        try:
            value = phy.attrs[attr].value
        except OSError:
            value = "N/A (OSError)"
        print(f"  {attr}: {value}")

    # B. Atrybuty kanałów (Częstotliwości, Gain, Bandwidth)
    chan_set = "[Ustawienia Kanałów]\n\r"
    for chan in phy.channels:
        # Pomiń kanały, które nie są istotne (opcjonalnie)
        direction = "TX" if chan.output else "RX"
        chan_set += f"Kanał: {chan.id} ({direction})\n\r"
        
        for attr in chan.attrs:
            try:
                val = chan.attrs[attr].value
            except OSError:
                val = "N/A (OSError)"
            chan_set += f"    {attr}: {val}\n\r"

    print (f"{phy.channels[0].id} {phy.channels[0].name} : {phy.channels[0].attrs['frequency'].value} Hz of {phy.channels[0].attrs['frequency_available'].value} Hz")
    print (f"{phy.channels[4].id} {phy.channels[4].name} : {phy.channels[4].attrs['rf_bandwidth'].value} Hz of {phy.channels[4].attrs['rf_bandwidth_available'].value} Hz")
    print (f"{phy.channels[4].attrs['rf_bandwidth'].value} Hz")
    #print (f"{phy.channels[4].id} {phy.channels[4].name} : {phy.channels[4].attrs['rf_bandwidth'].value} Hz BW")
    #print (f"{phy.channels[0].id} {phy.channels[0].name} : {phy.channels[0].attrs['hardwaregain'].value} dB")
    
    print (f"{phy.channels[3].id} {phy.channels[3].name} : {phy.channels[3].attrs['frequency'].value} Hz")
    print (f"{phy.channels[4].attrs['frequency'].value} Hz")
    print (f"{phy.channels[4].attrs['rf_bandwidth'].value} Hz of {phy.channels[4].attrs['rf_bandwidth_available'].value} Hz")

    return
'''
    sdr = adi.Pluto ( uri )
    sdr.tx_lo = F_C
    sdr.rx_lo = F_C
    sdr.sample_rate = F_S
    sdr.rx_rf_bandwidth = BW
    sdr.rx_buffer_size = SAMPLES_BUFFER_SIZE
    sdr.tx_hardwaregain_chan0 = float ( tx_gain_float )
    sdr.gain_control_mode_chan0 = gain_control_mode_chan0
    sdr.rx_hardwaregain_chan0 = int ( rx_gain_chan0_int )
    sdr.rx_output_type = RX_OUTPUT_TYPE # "SI" gives samples in volts, "raw" gives integer values. SI is more intuitive for processing, but raw can be more efficient for high-throughput applications.
    sdr.tx_destroy_buffer ()
    sdr.tx_cyclic_buffer = False
    time.sleep ( 0.2 ) #delay after setting device parameters
    if toml_settings[ "log" ][ "verbose_0" ] : print ( f"{sn=} {F_C=} {BW=} {F_S=}" )
    if toml_settings[ "log" ][ "verbose_2" ] : help ( adi.Pluto.rx_output_type ) ; help ( adi.Pluto.gain_control_mode_chan0 ) ; help ( adi.Pluto.tx_lo ) ; help ( adi.Pluto.tx  )
    
    return sdr
'''
def init_pluto_v0_0_0_old ( sn : str , tx_gain_float : float = TX_GAIN , gain_control_mode_chan0 : str = GAIN_CONTROL , rx_gain_chan0_int : int = RX_GAIN ) -> adi.Pluto :
    
    uri = get_uri ( sn )
    if uri is None:
        raise ValueError ( f"Error! ADALM-Pluto SN: {sn} is not connected. Check USB connection or IP settings.")
    sdr = adi.Pluto ( uri )
    sdr.tx_lo = F_C
    sdr.rx_lo = F_C
    sdr.sample_rate = F_S
    sdr.rx_rf_bandwidth = BW
    sdr.rx_buffer_size = SAMPLES_BUFFER_SIZE
    sdr.tx_hardwaregain_chan0 = float ( tx_gain_float )
    sdr.gain_control_mode_chan0 = gain_control_mode_chan0
    sdr.rx_hardwaregain_chan0 = int ( rx_gain_chan0_int )
    sdr.rx_output_type = RX_OUTPUT_TYPE # "SI" gives samples in volts, "raw" gives integer values. SI is more intuitive for processing, but raw can be more efficient for high-throughput applications.
    sdr.tx_destroy_buffer ()
    sdr.tx_cyclic_buffer = False
    time.sleep ( 0.2 ) #delay after setting device parameters
    if toml_settings[ "log" ][ "verbose_0" ] : print ( f"{sn=} {F_C=} {BW=} {F_S=}" )
    if toml_settings[ "log" ][ "verbose_2" ] : help ( adi.Pluto.rx_output_type ) ; help ( adi.Pluto.gain_control_mode_chan0 ) ; help ( adi.Pluto.tx_lo ) ; help ( adi.Pluto.tx  )
    
    return sdr


'''

def scale_to_pluto_dac_v0_1_11 ( samples : NDArray[ np.complex128 ] , scale : float = 1.0 ) -> NDArray[ np.complex128 ] : # None, because In-place modification
    # In-place scales and clips of normalized samples to ADALM-Pluto DAC units (±PLUTO_DAC_SCALE)
    samples_scaled = samples * PLUTO_DAC_SCALE * scale
    #return np.clip ( samples_scaled, -PLUTO_DAC_SCALE, PLUTO_DAC_SCALE, out = samples_scaled )
    return samples_scaled


def validate_samples ( samples: np.ndarray , buffer_size ) :

    validation = True

    # Walidacja: typ danych
    if not isinstance(samples, np.ndarray):
        raise ValueError("❌ tx_samples is not a numpy array")
        validation = False

    if samples.dtype != np.complex128:
        raise ValueError(f"❌ tx_samples must be np.complex128, but got {samples.dtype}")
        validation = False

    # Walidacja: wymiar
    if samples.ndim != 1:
        raise ValueError("❌ tx_samples must be a 1D array")
        validation = False

    # Walidacja: zawartość
    if np.isnan(samples).any():
        raise ValueError("❌ tx_samples contains NaN values")
        validation = False

    if np.isinf(samples).any():
        raise ValueError("❌ tx_samples contains Inf values")
        validation = False
    
    if samples.size > buffer_size :
        raise ValueError("❌ tx_samples size is larger than sdr buffer size")
        validation = False

    return validation

def analyze_rx_signal ( samples ) :
    # Real vs Imag plot
    real = samples.real[:500]
    imag = samples.imag[:500]
    fig1 = px.line(y=[real, imag], title="Real vs Imag")
    fig1.update_traces(name='Real', selector=dict(name='0'))
    fig1.update_traces(name='Imag', selector=dict(name='1'))
    fig1.update_layout(showlegend=True, xaxis_showgrid=True, yaxis_showgrid=True)
    fig1.show()

    # Constellation plot
    fig2 = px.scatter(x=samples.real, y=samples.imag, opacity=0.3, title="Constellation")
    fig2.update_layout(yaxis=dict(scaleanchor="x", scaleratio=1))
    fig2.show()

    # Histogram amplitudy
    fig3 = px.histogram(x=np.abs(samples), nbins=100, title="Histogram amplitudy")
    fig3.show()
"""
def analyze_rx_signal_old ( samples ) :
    plt.plot(samples.real[:500])
    plt.plot(samples.imag[:500])
    plt.title("Real vs Imag")
    plt.grid()
    plt.scatter(samples.real, samples.imag, alpha=0.3)
    plt.axis('equal')
    plt.title("Constellation")
    plt.hist(np.abs(samples), bins=100)
    plt.title("Histogram amplitudy")
"""
'''
