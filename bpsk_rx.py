'''
Sekwencja uruchomienia skryptu:
cd ~/python/temp/
source .venv/bin/activate
python tx.py

ssh do fedora na Surface 9 Pro: ssh yabool2001@192.168.1.60
Invalid rx_output_type: invalid. Must be raw or SI
'''

import numpy as np
import os , threading , tomllib , sys
import time as t

from modules import file, packet , sdr
from numpy import real
from numpy.typing import NDArray
from pathlib import Path

script_filename = os.path.basename ( __file__ )
Path ( "np.samples" ).mkdir ( parents = True , exist_ok = True )

# Wczytaj plik TOML z konfiguracją
with open ( "settings.toml" , "rb" ) as settings_file :
    toml_settings = tomllib.load ( settings_file )

if len ( sys.argv ) > 1 :
    gain_control_mode_chan0 = sys.argv[ 1 ]
    if len ( sys.argv ) > 2 :
        rx_gain_chan0_int = int ( sys.argv[ 2 ] )
    else :
        rx_gain_chan0_int = int ( toml_settings["ADALM-Pluto"][ "RX_GAIN" ] )
else :
    gain_control_mode_chan0 = toml_settings["ADALM-Pluto"][ "GAIN_CONTROL" ]
    rx_gain_chan0_int = int ( toml_settings["ADALM-Pluto"][ "RX_GAIN" ] )

samples_filename = "samples/rx_samples_00_issue40_002.0.si.npy"

wrt_filename_npy = "samples/rx_samples_0.0.0.npy"
wrt_filename_csv = "samples.csv/rx_samples_last.csv"
wrt_filename_log = "logs/rx_perf_log.csv"

with open ( wrt_filename_log , "w" ) as wrt_file :
    wrt_file.write ( "time,log_name\n" )
    wrt_file.write ( packet.log_packet )

received_bytes : NDArray[ np.uint8 ] = np.array ( [] , dtype = np.uint8 )
previous_samples_leftovers : NDArray[ np.complex128 ] = np.array ( [] , dtype = np.complex128 )

real = True
debug = False
plt = True
wrt = False

rx_pluto = packet.RxPluto_v0_0_0 ( sn = sdr.PLUTO_RX_SN , gain_control_mode_chan0 = gain_control_mode_chan0 , rx_gain_chan0_int = rx_gain_chan0_int ) if real else packet.RxPluto_v0_0_0 ()
print ( f"\n{ rx_pluto= }" )

while ( len ( received_bytes ) < 100000 and real ) or ( not real and received_bytes.size == 0 ) :
    
    if real :
        rx_pluto_samples = packet.RxSamples_v0_0_0 ( pluto_rx_ctx = rx_pluto.pluto_rx_ctx )
        rx_pluto_samples.rx ( previous_samples_leftovers = previous_samples_leftovers )
    else :
        rx_pluto_samples = packet.RxSamples_v0_0_0 ()
        rx_pluto_samples.rx ( samples_filename = samples_filename )    
    if debug :
        if rx_pluto_samples.has_amp_greater_than_ths : rx_pluto_samples.plot_complex_samples ( title = f"{ script_filename } { rx_pluto_samples.has_amp_greater_than_ths= }" )
    
    rx_pluto_samples.plot_complex_samples ( title = f"{ script_filename } {rx_pluto_samples.samples.size=}" )
    print ( f"{ script_filename } {rx_pluto_samples.samples.dtype=}, {rx_pluto=}" )
    print ( f"{ script_filename } {rx_pluto_samples.samples[:10]=}" )
    break

    rx_pluto_samples.detect_frames ( deep = False )
    
    if rx_pluto_samples.frames.has_leftovers :
        previous_samples_leftovers = rx_pluto_samples.samples_leftovers

    if rx_pluto_samples.frames.sync_sequence_peaks.size > 0 :
        if plt : rx_pluto_samples.plot_complex_samples_filtered ( title = f"{ script_filename } {rx_pluto_samples.frames.sync_sequence_peaks.size=}" , peaks = rx_pluto_samples.frames.sync_sequence_peaks )
        if wrt and real:
            rx_pluto_samples.save_complex_samples_2_npf ( wrt_filename_npy )

    if rx_pluto_samples.frames.samples_payloads_bytes.size > 0 :
        received_bytes = np.concatenate ( [ received_bytes , rx_pluto_samples.frames.samples_payloads_bytes ] )
        print ( f"{ rx_pluto_samples.frames.samples_payloads_bytes[0]= }, { rx_pluto_samples.frames.samples_payloads_bytes.size= } { received_bytes.size= }" )
        if debug : rx_pluto_samples.analyze ()
    
    if packet.log_packet != "" :
        # To jest najlepszy i najprostszy wybór dla aplikacji SDR działającej w pętli.
        # Pozwala na płynny odbiór próbek bez dławienia się przy zapisie na dysk.
        # Ryzyko, że "wątek zginie" przy zamykaniu programu jest minimalne w porównaniu do korzyści z płynności działania,
        # a systemowy bufor pliku i tak zazwyczaj zdąży się opróżnić.
        log_thread = threading.Thread ( target = file.save_log_thread , args = ( wrt_filename_log , packet.log_packet ) , daemon = True  )
        log_thread.start ()
        packet.log_packet = ""

    if not real :
        break
