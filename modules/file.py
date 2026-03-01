import csv
import numpy as np
import pandas as pd
import plotly.express as px

from numpy.typing import NDArray
from pathlib import Path

def open_complex_samples_from_csv ( filename : str ) -> NDArray[ np.complex128 ] :
    data = np.loadtxt ( filename , delimiter = ',' , skiprows = 1 )  # pomija nagłówek
    samples = data[ : , 0 ] + 1j * data[ : , 1 ]
    return samples.astype ( np.complex128 )

def open_complex_samples_from_npf ( filename : str ) -> NDArray[ np.complex128 ] :
    try :
        samples = np.load ( filename ).astype ( np.complex128 , copy = False )
    except OSError as exc :
        print ( f"Nie można załadować sampli z pliku {filename}: {exc}" )
        raise
    return samples

def save_complex_samples_2_npf ( filename : str , samples : NDArray[ np.complex128 ] | NDArray[ np.int16 ] ) -> None :
    """ Save complex samples to a .npf file with type validation. """
    
    if not isinstance ( samples , np.ndarray ) :
        raise TypeError ( "Samples must be NDArray" )
    
    if samples.dtype == np.int16 :
        # Konwersja int16 -> complex128 dla spójności zapisu (chyba że zależy nam na raw)
        # Tutaj zakładam, że chcemy zapisać jako complex128, aby pliki były wymienne
        samples = samples.astype ( np.float32 ).view ( np.complex64 ).astype ( np.complex128 )
    
    if samples.dtype != np.complex128 :
        raise TypeError ( f"Samples must be NDArray[np.complex128] or NDArray[np.int16]. Got {samples.dtype}" )

    target_path = Path ( filename )
    try :
        np.save ( target_path , samples )
    except OSError as exc :
        print ( f"Nie udało się zapisać {filename}: {exc}" )
        raise

def save_log_thread ( filename : str , content : str ) -> None :
    ''' Zapis próbek do pliku CSV w osobnym wątku aby nie blokować głównego wątku odbioru próbek'''
    with open ( filename , "a" ) as wrt_file :
        wrt_file.write ( content )


'''
def open_and_init_symbols_csv ( filename ) :
    csv_file = open ( filename , mode = "w" , newline = '' )
    csv_writer = csv.writer ( csv_file )
    csv_writer.writerow ( [ "symbol" ] )
    return csv_file , csv_writer

def open_and_init_complex_samples_csv ( filename ) :
    csv_file = open ( filename , mode = "w" , newline = '' )
    csv_writer = csv.writer ( csv_file )
    csv_writer.writerow ( [ "real" , "imag" ] )
    return csv_file , csv_writer

def open_and_write_symbols_2_csv ( name , symbols ) :
    csv_file , csv_writer = open_and_init_symbols_csv ( name )
    for symbol in symbols :
        csv_writer.writerow ( [ symbol ] )
    return csv_file , csv_writer

def open_and_write_samples_2_csv ( name , samples ) :
    csv_file , csv_writer = open_and_init_complex_samples_csv ( name )
    for sample in samples :
        csv_writer.writerow ( [ sample.real , sample.imag ] )
    return csv_file , csv_writer

def save_complex_samples_2_csv ( filename , samples : NDArray[ np.complex128 ] | NDArray[ np.int16 ] ) :
    if samples.dtype == np.int16 :
        real = samples[ 0 : : 2 ]
        imag = samples[ 1 : : 2 ]
        length = min ( len ( real ) , len ( imag ) ) # na wypadek nieparzystej liczby
        real = real [ : length ]
        imag = imag [ : length ]
    elif np.iscomplexobj ( samples ) :
        real = samples.real
        imag = samples.imag
    else :
        raise TypeError ( "Unsupported sample type. Expected complex128 or int16 interleaved." )

    with open ( filename , mode = 'w' , newline = '') as file :
        writer = csv.writer ( file )
        writer.writerow ( [ 'real' , 'imag' ] )  # nagłówki kolumn
        for r, i in zip ( real , imag ) :
            writer.writerow ( [ r , i ] )
def write_samples_2_csv ( filename , samples : NDArray[ np.complex128 ] ) :
    with open ( filename , mode = 'w' , newline = '') as file :
        writer = csv.writer ( file )
        writer.writerow ( [ 'real' , 'imag' ] )  # nagłówki kolumn
        for sample in samples :
            writer.writerow ( [ sample.real , sample.imag ] )

def save_samples_2_npf ( filename : str , samples : NDArray[ np.complex128 ] ) -> None :
    target_path = Path ( filename )
    try :
        np.save ( target_path , samples )
    except OSError as exc :
        print ( f"Nie udało się zapisać {filename}: {exc}" )
        raise

def save_float64_samples_2_npf ( filename : str , samples : NDArray[ np.float64 ] ) -> None :
    """
    Save float64 samples to a .npf file with type validation.
    """
    if not isinstance ( samples , np.ndarray ) or samples.dtype != np.float64 :
        raise TypeError ( "Samples must be NDArray[np.float64]" )
    target_path = Path ( filename )
    try :
        np.save ( target_path , samples )
    except OSError as exc :
        print ( f"Nie udało się zapisać {filename}: {exc}" )
        raise

def open_real_float64_samples_from_npf ( filename : str ) -> NDArray[ np.float64 ] :
    try :
        samples = np.load ( filename ).astype ( np.float64 , copy = False )
    except OSError as exc :
        print ( f"Nie załadować sampli z pliku {filename}: {exc}" )
        raise
    return samples

def append_symbols_2_csv ( csv_writer , symbols ) :
    for symbol in symbols :
        csv_writer.writerow ( [ symbol ] )
        
def append_samples_2_csv ( csv_writer , samples ) :
    for sample in samples :
        csv_writer.writerow ( [ sample.real , sample.imag ] )

def flush_data ( csv_file ) :
    csv_file.flush ()

def close_csv ( csv_file ) :
    csv_file.close ()

def flush_data_and_close_csv ( csv_file ) :
    flush_data ( csv_file )
    close_csv ( csv_file )

def plot_samples ( filename ) :
    # Wczytanie danych i wyświetlenie wykresu w Plotly
    print ( "Rysuję wykres..." )
    df = pd.read_csv ( filename )
    # Zbuduj sygnał zespolony (opcjonalnie, jeśli chcesz jako 1D)
    # signal = df["real"].values + 1j * df["imag"].values
    # Przygotuj dane do wykresu
    df["index"] = df.index
    # Wykres Plotly Express – wersja liniowa z filtrowanym sygnałem
    fig = px.line ( df , x = "index" , y = "real" , title = f"Sygnał {filename} BPSK raw: I i Q" )
    fig.add_scatter ( x = df[ "index" ] , y=df[ "imag" ] , mode = "lines" , name = "Q (imag filtrowane)" , line = dict ( dash = "dash" ) )
    fig.update_layout (
        xaxis_title = "Numer próbki" ,
        yaxis_title = "Amplituda" ,
        xaxis = dict ( rangeslider_visible = True ) ,
        legend = dict ( x = 0.01 , y = 0.99 ) ,
        height = 500
    )
    fig.show ()

def plot_symbols ( filename ) :
    # Wczytanie danych
    print("Rysuję wykres symboli BPSK...")
    df = pd.read_csv(filename)
    
    # Dodanie indeksu symboli
    df["symbol_index"] = df.index
    
    # Tworzenie wykresu
    fig = px.scatter(df, x="symbol_index", y="symbol", 
                     title=f"Symbole BPSK z pliku {filename}",
                     labels={"symbol": "Wartość symbolu", "symbol_index": "Indeks symbolu"})
    
    # Dodanie linii łączącej symbole dla lepszej wizualizacji
    fig.add_scatter(x=df["symbol_index"], y=df["symbol"], 
                    mode='lines+markers', name='Symbole BPSK',
                    line=dict(color='gray', width=1, dash='dot'))
    
    # Konfiguracja wyglądu wykresu
    fig.update_layout(
        height=500,
        xaxis=dict(rangeslider_visible=True),
        legend=dict(x=0.01, y=0.99)
    )
    
    # Ustawienie stałej skali Y dla lepszej wizualizacji BPSK
    fig.update_yaxes(range=[-1.5, 1.5])
    
    fig.show()

    '''