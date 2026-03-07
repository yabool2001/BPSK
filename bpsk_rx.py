'''
Sekwencja uruchomienia skryptu:
cd ~/python/temp/
source .venv/bin/activate
python3 bpsk_v0.1.16-tx.py 4 10 -10.0

ssh do fedora na Surface 9 Pro: ssh yabool2001@192.168.1.60
Invalid rx_output_type: invalid. Must be raw or SI

RX_GAIN = 71                        # receive gain
GAIN_CONTROL = "slow_attack"        # gain control mode

Chcę wykorzystać zainstalowany pytorch, żeby potraktować cały łańcuch odbiorczy jako "czarną skrzynkę".
Wrzucić do głębokiej sieci neuronowej przesuwną ramkę surowych, zaszumionych próbek I/Q (np. 131072 próbek prosto z libiio),
a na wyjściu żądać prawdopodobieństwa, że ukryty tam bit to 0 lub 1.

'''

import numpy as np
import os , threading , tomllib , sys
import torch
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

torch.cuda.is_available ()

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
previous_samples_leftovers : NDArray[ np.int16 ] = np.array ( [] , dtype = np.int16 )

samples : list [ packet.RxSamples_v0_0_0 ] = []

real = True
debug = False
plt = True
wrt = False

counter = 0

rx_pluto = packet.RxPluto_v0_0_0 ( sn = sdr.PLUTO_RX_SN , gain_control_mode_chan0 = gain_control_mode_chan0 , rx_gain_chan0_int = rx_gain_chan0_int ) if real else packet.RxPluto_v0_0_0 ()
#sdr.print_pluto_settings ( rx_pluto.pluto_rx_ctx )


while ( len ( received_bytes ) < 100000 and real ) or ( not real and received_bytes.size == 0 ) :
    
    if real :
        rx_pluto_samples = packet.RxSamples_v0_0_0 ( pluto_rx_buf = rx_pluto.pluto_rx_buf )
        rx_pluto_samples.rx ( previous_samples_leftovers = previous_samples_leftovers )
    else :
        rx_pluto_samples = packet.RxSamples_v0_0_0 ()
        rx_pluto_samples.rx ( samples_filename = samples_filename )    
    
    if real :
        previous_samples_leftovers = rx_pluto_samples.leftovers
        if counter > 1 :
            break
    

    #rx_pluto_samples.detect_frames ( deep = False )
    if rx_pluto_samples.has_amp_greater_than_ths :
        samples.append ( rx_pluto_samples )
        counter += 1

counter = 0
while len ( samples ) > counter :
    if plt : samples[counter].plot_complex_samples ( title = f"{ script_filename } {samples[counter].raw.size=}" )
    if wrt and real : samples[counter].save_complex_samples_2_npf ( wrt_filename_npy )
    counter += 1