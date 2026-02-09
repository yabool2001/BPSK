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

def init_pluto_v0_0_0 ( sn : str , tx_gain_float : float = TX_GAIN , gain_control_mode_chan0 : str = GAIN_CONTROL , rx_gain_chan0_int : int = RX_GAIN ) -> adi.Pluto :
    
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

def get_uri ( serial : str , type_preference : str = "usb" ) -> str | None :
    """
    Zwraca URI kontekstu IIO dla danego numeru seryjnego.
    
    Arguments:
    - serial (str): numer seryjny urządzenia (pełny).
    - type_preference (str): "usb" lub "ip". Jeśli "ip", preferuje ip: ale wraca do usb: jeśli ip nie znaleziono.

    Returns:
    - str: URI w formacie usb:x.y.z lub ip:adres
    - None: jeśli nie znaleziono pasującego urządzenia
    """
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