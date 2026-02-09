
'''
Sekwencja uruchomienia skryptu:
cd ~/python/temp/
source .venv/bin/activate
python bpsk_v0.1.16-tx.py

Przykłady komend wysyłających pakiety UDP do skryptu:
printf "\x34" | nc -u 192.168.1.60 10001
echo -n "34" | nc -u 192.168.1.50 10001
echo -n 4 | nc -u 192.168.1.50 10001

echo -n -e '\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0A\x0B\x0C\x0D\x0E\x0F\x10' | nc -u 192.168.1.60 10001

'''

import curses # Moduł wbudowany w Python do obsługi terminala (obsługa klawiatury)
import socket
import select
import numpy as np
import os
import sys
import time as t
import tomllib

from modules import packet , payload_test_data as ptd , sdr

# Wczytaj plik TOML z konfiguracją
with open ( "settings.toml" , "rb" ) as settings_file :
    toml_settings = tomllib.load ( settings_file )

if len ( sys.argv ) > 1 :
    n_o_bytes_uint16 = np.uint16 ( int ( sys.argv[ 1 ] ) )
    n_o_repeats_uint32 = np.uint32 ( int ( sys.argv[ 2 ] ) )
    tx_gain_float = float ( sys.argv[ 3 ] )
else :
    n_o_bytes_uint16 = np.uint16 ( 4 )
    n_o_repeats_uint32 = np.uint32 ( 10 )
    tx_gain_float = float ( toml_settings["ADALM-Pluto"][ "TX_GAIN" ] )

np.set_printoptions ( threshold = np.inf , linewidth = np.inf )

script_filename = os.path.basename ( __file__ )

plt = False
tx_pluto = packet.TxPluto_v0_1_16 ( sn = sdr.PLUTO_TX_SN, tx_gain_float = tx_gain_float )
print ( f"\n{ script_filename= } { tx_pluto= }" )
tx_pluto.samples.create_samples4pluto ( payload_bytes = ptd.generate_payload_i_bytes_dec_15 ( n_o_bytes_uint16 ) )
if plt :
    tx_pluto.samples.plot_symbols ( f"{ script_filename } " )
    tx_pluto.samples.plot_complex_samples_filtered ( f"{ script_filename } filtered samples" )
    tx_pluto.samples.plot_complex_samples4pluto ( f"{ script_filename } samples4pluto" )
    tx_pluto.samples.plot_samples_spectrum ( f"{ script_filename } samples4pluto" )

stdscr = curses.initscr ()
curses.noecho ()
stdscr.keypad ( True )
print ( "Naciśnij:" )
print ( " - 't' aby wysłać pakiet jednorazowo" )
print ( " - 'c' aby rozpocząć transmisję cykliczną" )
print ( " - 's' aby zatrzymać transmisję cykliczną" )

# Setup UDP Socket
udp_sock = socket.socket ( socket.AF_INET , socket.SOCK_DGRAM )

# Automatyczne wykrywanie adresu IP z sieci 192.168.1.x
local_ip = "0.0.0.0" # Domyślny fallback
try:
    # Tworzymy tymczasowy socket żeby sprawdzić routing do sieci 192.168.1.x
    # Łączymy się z przykładowym adresem (np. bramą) w tej sieci aby system wskazał właściwy interfejs
    temp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    temp_sock.connect(("192.168.1.1", 1)) 
    local_ip = temp_sock.getsockname()[0]
    temp_sock.close()
except Exception:
    pass

print(f"Binding UDP to {local_ip}:10001")
udp_sock.bind ( ( local_ip , 10001 ) )
udp_sock.setblocking ( False )
stdscr.nodelay ( True )

try :
    while True :
        try :
            payload_udp = udp_sock.recv ( 1500 )
            print ( f"\n\r[UDP] Received { len(payload_udp) } bytes: { payload_udp }" )
            tx_pluto.samples.create_samples4pluto ( payload_bytes = payload_udp )
            tx_pluto.samples.tx ()
            tx_pluto.samples.create_samples4pluto ( payload_bytes = ptd.generate_payload_i_bytes_dec_15 ( n_o_bytes_uint16 ) )
        except BlockingIOError :
            pass
        except Exception :
            pass
        try :
            key = stdscr.getkey ()
        except curses.error :
            key = ''
        if key ==  'o' :
            t.sleep ( 1 )  # anty-dubler
            print ( f"\n\r[o] Please, press '1' to send packet once." )
        elif key == 'c' :
            t.sleep ( 1 ) # anty-dubler
            tx_pluto.samples.tx_cyclic ()
            print ( f"\n\r[c] Tx cyclic started for { tx_pluto.samples.payload_bytes[0]= } { tx_pluto.samples.payload_bytes.size= }." )
            print ( f"\n\r { tx_pluto.pluto_tx_ctx.tx_cyclic_buffer= }" )
        elif key == 's' :
            t.sleep ( 1 ) # anty-dubler
            tx_pluto.samples.stop_tx_cyclic ()
            print ( "\n\r[s] Tx cyclic stopped" )
            print ( f"\n\r { tx_pluto.pluto_tx_ctx.tx_cyclic_buffer= }" )
        elif key == 't' :
            t.sleep ( 1 ) # anty-dubler
            print ( f"\n\r{ tx_pluto.pluto_tx_ctx.tx_cyclic_buffer= }" )
            tx_pluto.samples.tx_incremeant_payload_and_repeat ( n_o_bytes = n_o_bytes_uint16 , n_o_repeats = n_o_repeats_uint32 )
            print ( f"\n\r[t] Tx.size = { n_o_bytes_uint16 } bytes with zeros payload created, incremented & repeated { n_o_repeats_uint32 } times." )
            tx_pluto.samples.create_samples4pluto ( payload_bytes = ptd.generate_payload_i_bytes_dec_15 ( n_o_bytes_uint16 ) )
        elif key > '0' and key <= '9' : # advanced test mode
            t.sleep ( 1 )  # anty-dubler
            i = np.uint32 ( key )
            if i % 2 == 0 :
                i = i * 5
                print ( f"\n\rNotice: Odd number multiplied by 5." )
            tx_pluto.samples.tx ( repeat = i )
            print ( f"\n\r{ tx_pluto.samples.payload_bytes[0]= } { tx_pluto.samples.payload_bytes.size= } sent { i } time(s)." )
        elif key == '\x1b' :  # ESCAPE
            tx_pluto.samples.stop_tx_cyclic ()
            break
        t.sleep ( 0.05 )  # odciążenie CPU
finally :
    curses.echo ()
    stdscr.keypad ( False )
    curses.endwin ()
