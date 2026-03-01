import numpy as np
import pandas as pd
import plotly.express as px
from scipy import signal
from typing import Optional

from modules import sdr
from numpy.typing import NDArray

def real_waveform_v0_0_0 ( signal_real : NDArray[ np.float64 ] , title : str = "Sygnał", marker_squares : bool = False, marker_peaks : Optional[ NDArray[ np.int_] ] = None ) -> None :
    """ Rozszerzona wersja funkcji real_waveform z dodatkowym parametrem marker_peaks. Jeśli marker_peaks zostanie przekazany (np.ndarray z indeksami), peaks zostaną zaznaczone trójkątami na wykresie.
    Parametry:
    - signal_real: NDArray[np.float64] (rzeczywisty)
    - title: tytuł wykresu
    - marker_squares: bool — czy rysować znaczniki (kwadraty) na wszystkich próbkach
    - marker_peaks: Optional[NDArray[np.int_]] — indeksy próbek, gdzie zaznaczyć trójkąty (rozmiar taki sam jak marker_squares). """

    if np.iscomplexobj ( signal_real ) :
        raise ValueError ( "Wejściowy sygnał musi być rzeczywisty NDArray[np.float64]" )

    df = pd.DataFrame ( {"index": np.arange ( len ( signal_real ) ) , "value" : signal_real} )

    if marker_squares :
        mode = 'lines+markers'
        marker_cfg = dict ( symbol = 'square' , size = 5 , color = 'rgba(0,0,0,0)' , line = dict ( color = 'blue' , width = 1 ) )
    else :
        mode = 'lines'
        marker_cfg = None

    fig = px.line ( df , x = "index" , y = "value" , title = f"{ title }" )
    fig.data = []  # usuń automatyczne ślady z px.line i dodaj własne z markerami
    fig.add_scatter ( x = df[ "index" ] , y = df[ "value" ] , mode = mode , name = "Wartość" , line = dict ( color = 'blue' ) , marker = marker_cfg )

    # Dodatek dla peaks
    if marker_peaks is not None:
        # Filtruj indeksy w zakresie
        valid_peaks = marker_peaks[ ( marker_peaks >= 0 ) & ( marker_peaks < len ( signal_real ) ) ]
        if len ( valid_peaks ) > 0 :
            peaks_values = signal_real[ valid_peaks ]
            # Trójkąty dla wartości
            fig.add_scatter ( x = valid_peaks , y = peaks_values , mode = 'markers' , name = "Peaks" , marker = dict ( symbol = 'triangle-up' , size = 10 , color = 'rgba(0,0,0,0)' , line = dict ( color = 'red' , width = 1 ) ) )

    fig.update_layout ( xaxis_title = "Numer próbki" , yaxis_title = "Amplituda" , xaxis = dict ( rangeslider_visible = True ) , legend = dict ( x = 0.01 , y = 0.99 ) , height = 500 )
    fig.show()

def complex_waveform_v0_0_0 ( signal_complex : NDArray[ np.complex128 ] , title : str = "Sygnał zespolony", marker_squares : bool = False , marker_peaks : Optional[ NDArray[ np.int_ ] ] = None ) -> None :
    """ Rozszerzona wersja funkcji complex_waveform z dodatkowym parametrem marker_peaks. Jeśli marker_peaks zostanie przekazany (np.ndarray z indeksami), peaks zostaną zaznaczone trójkątami na wykresie.
    Parametry:
    - signal_complex: NDArray[np.complex128] (zespolony)
    - title: tytuł wykresu
    - marker_squares: bool — czy rysować znaczniki (kwadraty) na wszystkich próbkach
    - marker_peaks: Optional[np.ndarray] — indeksy próbek, gdzie zaznaczyć trójkąty (rozmiar taki sam jak marker_squares). """
    
    if not np.iscomplexobj ( signal_complex ) :
        # Jeśli sygnał nie jest zespolony, sprawdź czy to nie surowe dane int16 z Pluto (przeplot I/Q) i dokonaj konwersji int16 -> complex128
        if signal_complex.dtype == np.int16 :
            signal_complex = signal_complex.astype ( np.float32 ).view ( np.complex64 ).astype ( np.complex128 )
        else :
            raise ValueError ( "Wejściowy sygnał musi być zespolony NDArray[np.complex128] lub surowy NDArray[np.int16] (I/Q interleaved)" )

    df = pd.DataFrame ( {"index" : np.arange ( len ( signal_complex ) ) , "real" : signal_complex.real , "imag" : signal_complex.imag} )

    if marker_squares :
        mode_real = 'lines+markers'
        mode_imag = 'lines+markers'
        marker_real_cfg = dict(symbol='square', size=5, color='rgba(0,0,0,0)', line=dict(color='blue', width=1))
        marker_imag_cfg = dict(symbol='square', size=5, color='rgba(0,0,0,0)', line=dict(color='orange', width=1))
    else :
        mode_real = 'lines'
        mode_imag = 'lines'
        marker_real_cfg = None
        marker_imag_cfg = None

    fig = px.line ( df , x = "index" , y = "real" , title = f"{ title }" )
    fig.data = []  # usuń automatyczne ślady z px.line i dodaj własne z markerami
    fig.add_scatter ( x = df[ "index" ], y = df[ "real" ], mode = mode_real , name = "I (real)" , line = dict ( color = 'blue' ) , marker = marker_real_cfg )
    fig.add_scatter ( x = df[ "index" ], y = df[ "imag" ], mode = mode_imag , name = "Q (imag)" , line = dict ( color = 'green' , dash = 'dash' ) , marker = marker_imag_cfg )

    # Dodatek dla peaks
    if marker_peaks is not None:
        # Filtruj indeksy w zakresie
        valid_peaks = marker_peaks[ ( marker_peaks >= 0 ) & ( marker_peaks < len ( signal_complex ) ) ]
        if len ( valid_peaks ) > 0 :
            peaks_real = signal_complex[ valid_peaks ].real
            peaks_imag = signal_complex[ valid_peaks ].imag
            # Trójkąty dla I (real)
            fig.add_scatter ( x = valid_peaks , y = peaks_real , mode = 'markers' , name = "Peaks I" , marker = dict ( symbol = 'triangle-up' , size = 10 , color = 'rgba(0,0,0,0)' , line = dict ( color = 'red' , width = 1 ) ) )
            # Trójkąty dla Q (imag)
            fig.add_scatter ( x = valid_peaks , y = peaks_imag , mode = 'markers' , name = "Peaks Q" , marker = dict ( symbol = 'triangle-up' , size = 10 , color = 'rgba(0,0,0,0)' , line = dict ( color = 'purple' , width = 1 ) ) )

    fig.update_layout ( xaxis_title = "Numer próbki" , yaxis_title = "Amplituda" , xaxis = dict ( rangeslider_visible = True ) , legend = dict ( x = 0.01 , y = 0.99 ) , height = 500 )
    fig.show ()

def plot_symbols_v0_0_0 ( symbols : NDArray[ np.complex128 ] , title : str = "Symbole" ) -> None :
    """
    Rysuje wykres symboli BPSK (np. z ADALM-Pluto) w postaci punktów połączonych przerywaną linią.

    Parametry:
    ----------
    symbols : NDArray[ np.complex128 ] Tablica symboli BPSK typu np.complex128
    title : str Tytuł wykresu.
    Zwraca: None
    """
    if not isinstance ( symbols , np.ndarray ):
        raise TypeError ( "Argument 'symbols' musi być typu NDArray[np.complex128]." )

    # Obsługa symboli zespolonych – bierzemy część rzeczywistą
    symbols_real = symbols.real if np.iscomplexobj ( symbols ) else symbols
    # Przygotowanie danych do wykresu
    df = pd.DataFrame ( { "symbol_index" : np.arange ( len ( symbols_real ) ) , "symbol" : symbols_real } )
    # Wykres punktowy
    fig = px.scatter ( df , x = "symbol_index" , y = "symbol" , title = f"{ title }" , labels = { "symbol" : "Wartość symbolu" , "symbol_index" : "Indeks symbolu" } )
    # Dodanie przerywanej linii łączącej punkty
    fig.add_scatter ( x = df[ "symbol_index" ] , y = df[ "symbol" ] , mode = 'lines+markers' , name = 'Symbole' , line = dict ( color = 'gray' , width = 1 , dash = 'dot' ) )
    # Konfiguracja osi i wyglądu
    fig.update_layout ( height = 500 , xaxis = dict ( rangeslider_visible = True ) , legend = dict ( x = 0.01 , y = 0.99 ) )
    # Oś Y dopasowana dynamicznie, ale możesz wymusić np. range=[-1.5, 1.5] jeśli chcesz sztywną skalę
    fig.show ()

def spectrum_occupancy_v0_0_0 ( samples , nperseg = 1024 , title: str = "Spectrum occupancy (PSD)" ) -> None :
    """
    Funkcja do wizualizacji zajętości widma (PSD) na podstawie próbek.
    Używa scipy.signal.welch do estymacji PSD, co jest efektywne dla sygnałów BPSK
    z dużym offsetem częstotliwości/fazy – pozwoli zobaczyć, czy sygnał jest centrowany
    wokół 0 Hz w baseband (po downconversion z rx_lo=2.9 GHz), i wykryć offsety.
    
    Parametry:
    - tsdr: Obiekt adi.Pluto z ustawionymi parametrami (sample_rate=3e6, rx_rf_bandwidth=1e6).
    - n_samples: Liczba próbek do pobrania (domyślnie rx_buffer_size=32768).
    - nperseg: Długość segmentu dla welch (trade-off: rozdzielczość vs. wariancja; mniejsza dla szybszego obliczenia).
    
    Zwraca: Interaktywny wykres PSD w dB vs. częstotliwość (w Hz, centrowana wokół rx_lo).
    """
    
    # Estymacja PSD z scipy (welch dla uśredniania, hanning window dla BPSK)
    f_s = sdr.F_S  # 3e6 Hz

    # Dynamiczne dostosowanie nperseg i noverlap
    len_samples = len ( samples )
    if nperseg > len_samples:
        nperseg = len_samples  # Automatyczna redukcja jak w scipy
    noverlap = min ( nperseg // 2 , nperseg - 1 )  # Zapewnij noverlap < nperseg
    f, Pxx = signal.welch ( samples , fs = f_s , window = 'hann' , nperseg = nperseg , noverlap = noverlap , detrend = 'constant' , scaling = 'density' )
    
    # Przesunięcie częstotliwości o rx_lo (2.9 GHz) dla wizualizacji pełnego widma RF
    f_c = sdr.F_C
    f_rf = f + f_c  # Centrowanie wokół 2.9 GHz (uwzględnia offsety)
    
    # Normalizacja do dB (dla lepszej wizualizacji zajętości widma)
    Pxx_db = 10 * np.log10 ( Pxx + 1e-12 )  # Unikanie log(0)
    
    # DataFrame do wizualizacji z pandas i plotly
    df = pd.DataFrame ( { 'Częstotliwość [Hz]': f_rf , 'PSD [dB/Hz]': Pxx_db } )
    
    # Wykres interaktywny – idealny do analizy offsetu fazy/częstotliwości w BPSK
    fig = px.line ( df , x = 'Częstotliwość [Hz]' , y = 'PSD [dB/Hz]' , title = title )
    fig.update_layout ( xaxis_title = 'Częstotliwość [Hz]' , yaxis_title = 'Moc spektralna [dB/Hz]' ,
                      xaxis_range = [ f_c - f_s / 2 , f_c + f_s / 2 ] )  # Zakres wokół lo ± fs/2
    fig.show ()
    
    # Opcjonalnie: Zintegruj z numba dla szybszego przetwarzania dużych buforów, jeśli potrzeba
    # (np. @jit na custom PSD, ale welch jest wystarczająco szybki dla N=32768)