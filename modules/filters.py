import numpy as np
import time as t
import tomllib

from modules import modulation , plot
from numba import njit  # Dodane dla przyspieszenia obliczeń
from numpy.typing import NDArray
from pathlib import Path
from scipy.signal import lfilter , upfirdn , find_peaks

with open ( "settings.toml" , "rb" ) as settings_toml_file :
    toml_settings = tomllib.load ( settings_toml_file )

BETA = float ( toml_settings[ "rrc_filter" ][ "BETA" ] )
SPAN = int ( toml_settings[ "rrc_filter" ][ "SPAN" ] )
SPS  = int ( toml_settings[ "bpsk" ][ "SPS" ] )

def apply_tx_rrc_filter_v0_0_0 ( symbols: NDArray[ np.complex128 ] , upsample: bool = True , ) -> NDArray[ np.complex128 ] :
    """ Stosuje filtr Root Raised Cosine (RRC) z opcjonalnym upsamplingiem. Zawsze zwraca sygnał zespolony (complex128).
    Parametry:
        symbols: Sygnał wejściowy (real lub complex).
        beta: Współczynnik roll-off (0.0-1.0).
        sps: Próbek na symbol (samples per symbol).
        span: Długość filtra w symbolach.
        upsample: Czy wykonać upsampling (True) czy tylko filtrować (False).
    Zwraca:
        Przefiltrowany sygnał zespolony (complex128). """
    
    rrc_taps = rrc_filter_v0_0_0 ( beta = BETA , sps  = modulation.SPS , span = SPAN )
    
    if upsample:
        filtered = upfirdn ( rrc_taps , symbols , modulation.SPS )  # Auto-upsampling + filtracja
    else:
        filtered = lfilter ( rrc_taps , 1.0 , symbols )     # Tylko filtracja
    
    return ( filtered + 0j ) .astype ( np.complex128 )

def apply_rrc_rx_filter_v0_0_0 ( rx_samples: NDArray[ np.complex128 ] ) -> NDArray[ np.complex128 ] :
    """ Filtruje odebrane próbki z SDR filtrem RRC.
    Parametry:
        rx_samples: Odebrane próbki (complex128) z SDR.
    Zwraca:
        Przefiltrowane próbki (complex128). """

    # Generuj współczynniki filtra RRC
    rrc_taps = rrc_filter_v0_0_0 ( beta = BETA , sps  = SPS , span = SPAN )
    # Filtracja (uwaga: filtr musi być znormalizowany!)
    filtered = lfilter ( rrc_taps , 1.0 , rx_samples )
    
    return filtered.astype ( np.complex128 )  # Gwarancja complex128

@njit ( cache = True , fastmath = True )  # Kompilacja Just-In-Time z optymalizacjami
def rrc_filter_v0_0_0 ( beta , sps , span ) :
    """ DeepSeek -V3 R1 (Zoptymalizowana)
        Filtruje dane wejściowe za pomocą filtra RRC. """
    
    N = span * sps
    t = np.arange ( -N / 2 , N / 2 + 1 , dtype = np.float64 ) / sps

    # Obsługa beta = 0 (filtr sinc)
    if beta == 0 :
        h = np.sinc ( t )
        h = h / np.sqrt ( np.sum ( h ** 2 ) )  # Normalizacja
        return h
    
    h = np.zeros_like ( t )

    # Stałe pre-kalkulowane dla wydajności
    beta_pi = np.pi * beta
    inv_4beta = 1 / ( 4 * beta )
    sqrt2 = np.sqrt ( 2 )
    special_val = ( beta / sqrt2 ) * \
                ( ( 1 + 2 / np.pi ) * np.sin ( np.pi / ( 4 * beta ) ) + 
                ( 1 - 2 / np.pi ) * np.cos ( np.pi / ( 4 * beta ) ) )

    # Obliczenia z maskami (zoptymalizowane)
    for i in range ( len ( t ) ) :
        ti = t[i]
        if ti == 0.0 :
            h[i] = 1.0 - beta + ( 4 * beta / np.pi )
        elif np.abs ( np.abs ( ti ) - inv_4beta ) < 1e-10 :  # Tolerancja numeryczna
            h[i] = special_val
        else :
            numerator = np.sin ( np.pi * ti * ( 1 - beta ) ) + \
                      4 * beta * ti * np.cos ( np.pi * ti * ( 1 + beta ) )
            denominator = np.pi * ti * ( 1 - ( 4 * beta * ti ) ** 2 )
            h[i] = numerator / denominator

    # Bezpieczna obsługa NaN/Inf i normalizacja
    h[ np.isnan ( h ) | np.isinf ( h ) ] = 0
    h = h / np.sqrt ( np.sum ( h ** 2 ) )  # Normalizacja

    return h
