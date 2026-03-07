from __future__ import annotations

import math
import tomllib
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F

from numpy.random import Generator
from numpy.typing import NDArray
from scipy.signal import upfirdn
from torch import Tensor, nn
from torch.utils.data import Dataset


DEFAULT_WINDOW_SIZE = 131072
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available () else "cpu"


with open ( "settings.toml" , "rb" ) as settings_file :
    SETTINGS = tomllib.load ( settings_file )


SPS = int ( SETTINGS[ "bpsk" ][ "SPS" ] )
BETA = float ( SETTINGS[ "rrc_filter" ][ "BETA" ] )
SPAN = int ( SETTINGS[ "rrc_filter" ][ "SPAN" ] )
BARKER13_BITS = np.array ( SETTINGS[ "BARKER13_BITS" ] , dtype = np.uint8 )
PACKET_LEN_LEN_BITS = 11


def bytes2bits ( data : NDArray[ np.uint8 ] ) -> NDArray[ np.uint8 ] :
    return np.unpackbits ( np.asarray ( data , dtype = np.uint8 ) ).astype ( np.uint8 , copy = False )


def dec2bits ( dec : int , num_bits : int ) -> NDArray[ np.uint8 ] :
    return np.array ( [ ( dec >> i ) & 1 for i in range ( num_bits - 1 , -1 , -1 ) ] , dtype = np.uint8 )


def pad_bits2bytes ( bits : NDArray[ np.uint8 ] ) -> NDArray[ np.uint8 ] :
    pad = ( -len ( bits ) ) % 8
    if pad :
        bits = np.concatenate ( [ bits , np.zeros ( pad , dtype = np.uint8 ) ] )
    return np.packbits ( bits )


def create_crc32_bytes ( payload_bytes : NDArray[ np.uint8 ] ) -> NDArray[ np.uint8 ] :
    crc32 = zlib.crc32 ( payload_bytes.tobytes () )
    return np.frombuffer ( crc32.to_bytes ( 4 , "big" ) , dtype = np.uint8 )


def rrc_filter_taps () -> NDArray[ np.float64 ] :
    n_symbols = SPAN * SPS
    t = np.arange ( -n_symbols / 2 , n_symbols / 2 + 1 , dtype = np.float64 ) / SPS

    if BETA == 0.0 :
        taps = np.sinc ( t )
        return taps / np.sqrt ( np.sum ( taps ** 2 ) )

    taps = np.zeros_like ( t )
    special_val = ( BETA / np.sqrt ( 2.0 ) ) * (
        ( 1.0 + 2.0 / np.pi ) * np.sin ( np.pi / ( 4.0 * BETA ) )
        + ( 1.0 - 2.0 / np.pi ) * np.cos ( np.pi / ( 4.0 * BETA ) )
    )
    inv_4beta = 1.0 / ( 4.0 * BETA )

    for idx , t_sample in enumerate ( t ) :
        if t_sample == 0.0 :
            taps[ idx ] = 1.0 - BETA + ( 4.0 * BETA / np.pi )
        elif np.abs ( np.abs ( t_sample ) - inv_4beta ) < 1e-10 :
            taps[ idx ] = special_val
        else :
            numerator = np.sin ( np.pi * t_sample * ( 1.0 - BETA ) ) + 4.0 * BETA * t_sample * np.cos ( np.pi * t_sample * ( 1.0 + BETA ) )
            denominator = np.pi * t_sample * ( 1.0 - ( 4.0 * BETA * t_sample ) ** 2 )
            taps[ idx ] = numerator / denominator

    taps = taps / np.sqrt ( np.sum ( taps ** 2 ) )
    return taps


def create_bpsk_symbols ( bits : NDArray[ np.uint8 ] ) -> NDArray[ np.complex64 ] :
    return ( bits.astype ( np.float32 ) * 2.0 - 1.0 + 0j ).astype ( np.complex64 )


def apply_tx_rrc_filter ( symbols : NDArray[ np.complex64 ] ) -> NDArray[ np.complex64 ] :
    return upfirdn ( rrc_filter_taps () , symbols , up = SPS ).astype ( np.complex64 )


def build_tx_frame_bytes ( payload_bytes : Sequence[ int ] | NDArray[ np.uint8 ] ) -> NDArray[ np.uint8 ] :
    payload_arr = np.asarray ( payload_bytes , dtype = np.uint8 )
    packet_crc = create_crc32_bytes ( payload_arr )
    packet_bytes = np.concatenate ( [ payload_arr , packet_crc ] )
    packet_len_bits = dec2bits ( len ( packet_bytes ) , PACKET_LEN_LEN_BITS )
    header_bytes = pad_bits2bytes ( np.concatenate ( [ BARKER13_BITS , packet_len_bits ] ) )
    header_crc = create_crc32_bytes ( header_bytes )
    return np.concatenate ( [ header_bytes , header_crc , packet_bytes ] )


def coerce_samples_to_complex64 ( samples : NDArray[ np.int16 ] | NDArray[ np.complexfloating ] ) -> NDArray[ np.complex64 ] :
    arr = np.asarray ( samples )
    if np.iscomplexobj ( arr ) :
        return arr.astype ( np.complex64 , copy = False )
    if arr.dtype != np.int16 :
        raise TypeError ( f"Unsupported dtype {arr.dtype}. Expected int16 interleaved I/Q or complex samples." )
    if arr.size % 2 != 0 :
        raise ValueError ( "Interleaved I/Q samples must contain an even number of int16 values." )
    i_samples = arr[ 0 : : 2 ].astype ( np.float32 )
    q_samples = arr[ 1 : : 2 ].astype ( np.float32 )
    return ( i_samples + 1j * q_samples ).astype ( np.complex64 )


def normalize_complex_samples ( samples : NDArray[ np.complex64 ] ) -> NDArray[ np.complex64 ] :
    centered = samples - np.mean ( samples )
    rms = np.sqrt ( np.mean ( np.abs ( centered ) ** 2 ) + 1e-12 )
    return ( centered / rms ).astype ( np.complex64 , copy = False )


def complex_samples_to_tensor ( samples : NDArray[ np.complex64 ] ) -> Tensor :
    return torch.from_numpy ( np.stack ( [ samples.real , samples.imag ] , axis = 0 ).astype ( np.float32 , copy = False ) )


def centered_complex_window ( samples : NDArray[ np.complex64 ] , center_idx : int , window_size : int ) -> NDArray[ np.complex64 ] :
    half_window = window_size // 2
    start_idx = center_idx - half_window
    end_idx = start_idx + window_size
    if start_idx >= 0 and end_idx <= samples.size :
        return samples[ start_idx : end_idx ]

    window = np.zeros ( window_size , dtype = np.complex64 )
    src_start = max ( start_idx , 0 )
    src_end = min ( end_idx , samples.size )
    dst_start = src_start - start_idx
    dst_end = dst_start + ( src_end - src_start )
    if src_end > src_start :
        window[ dst_start : dst_end ] = samples[ src_start : src_end ]
    return window


def build_tx_frame_bits ( payload_bytes : Sequence[ int ] | NDArray[ np.uint8 ] ) -> NDArray[ np.uint8 ] :
    return bytes2bits ( build_tx_frame_bytes ( payload_bytes ) )


def build_tx_frame_reference ( payload_bytes : Sequence[ int ] | NDArray[ np.uint8 ] ) -> NDArray[ np.complex64 ] :
    return apply_tx_rrc_filter ( create_bpsk_symbols ( build_tx_frame_bits ( payload_bytes ) ) )


def build_bit_center_indices ( n_bits : int , frame_start_idx : int ) -> NDArray[ np.int64 ] :
    group_delay = SPAN * SPS // 2
    return frame_start_idx + group_delay + np.arange ( n_bits , dtype = np.int64 ) * SPS


def estimate_frame_start ( capture_samples : NDArray[ np.complex64 ] , reference_samples : NDArray[ np.complex64 ] ) -> int :
    if capture_samples.size < reference_samples.size :
        raise ValueError ( "Capture must be at least as long as the frame reference." )
    corr = np.abs ( np.correlate ( capture_samples , np.conj ( reference_samples ) , mode = "valid" ) )
    return int ( np.argmax ( corr ) )


def apply_channel_impairments ( samples : NDArray[ np.complex64 ] , rng : Generator ) -> NDArray[ np.complex64 ] :
    n = np.arange ( samples.size , dtype = np.float32 )
    amplitude = rng.uniform ( 0.25 , 1.75 )
    phase = rng.uniform ( -np.pi , np.pi )
    freq_offset = rng.uniform ( -0.12 , 0.12 ) / SPS
    dc_i = rng.uniform ( -0.05 , 0.05 )
    dc_q = rng.uniform ( -0.05 , 0.05 )
    iq_skew = rng.uniform ( 0.85 , 1.15 )
    noise_std = rng.uniform ( 0.01 , 0.30 )

    rotated = samples * amplitude * np.exp ( 1j * ( phase + 2.0 * np.pi * freq_offset * n ) )
    distorted = rotated.real.astype ( np.float32 ) + 1j * ( rotated.imag.astype ( np.float32 ) * iq_skew )
    distorted = distorted + ( dc_i + 1j * dc_q )
    noise = rng.normal ( 0.0 , noise_std , size = samples.size ).astype ( np.float32 )
    noise = noise + 1j * rng.normal ( 0.0 , noise_std , size = samples.size ).astype ( np.float32 )
    return normalize_complex_samples ( ( distorted + noise ).astype ( np.complex64 ) )


def capture_score_centers ( n_complex_samples : int , window_size : int , stride : int ) -> NDArray[ np.int64 ] :
    first_center = window_size // 2
    last_center = max ( first_center , n_complex_samples - ( window_size - first_center ) )
    return np.arange ( first_center , last_center + 1 , stride , dtype = np.int64 )


@dataclass ( slots = True )
class CaptureAlignment :
    frame_start_idx : int
    centers : NDArray[ np.int64 ]
    bits : NDArray[ np.uint8 ]


def align_capture_to_known_payload (
    raw_samples : NDArray[ np.int16 ] | NDArray[ np.complexfloating ] ,
    payload_bytes : Sequence[ int ] | NDArray[ np.uint8 ] ,
    frame_start_idx : int | None = None ,
) -> tuple[ NDArray[ np.complex64 ] , CaptureAlignment ] :
    complex_samples = normalize_complex_samples ( coerce_samples_to_complex64 ( raw_samples ) )
    frame_bits = build_tx_frame_bits ( payload_bytes )
    if frame_start_idx is None :
        frame_start_idx = estimate_frame_start ( complex_samples , build_tx_frame_reference ( payload_bytes ) )
    centers = build_bit_center_indices ( len ( frame_bits ) , frame_start_idx )
    return complex_samples , CaptureAlignment ( frame_start_idx = frame_start_idx , centers = centers , bits = frame_bits )


class CaptureBitDataset ( Dataset[ tuple[ Tensor , Tensor ] ] ) :

    def __init__ (
        self ,
        captures : Sequence[ NDArray[ np.complex64 ] ] ,
        labels : Sequence[ NDArray[ np.uint8 ] ] ,
        centers : Sequence[ NDArray[ np.int64 ] ] ,
        window_size : int = DEFAULT_WINDOW_SIZE ,
    ) -> None :
        if len ( captures ) != len ( labels ) or len ( captures ) != len ( centers ) :
            raise ValueError ( "captures, labels and centers must have identical lengths." )
        self.captures = list ( captures )
        self.labels = list ( labels )
        self.centers = list ( centers )
        self.window_size = int ( window_size )
        self.index : list[ tuple[ int , int ] ] = []
        for capture_idx , bits in enumerate ( self.labels ) :
            for bit_idx in range ( len ( bits ) ) :
                self.index.append ( ( capture_idx , bit_idx ) )

    def __len__ ( self ) -> int :
        return len ( self.index )

    def __getitem__ ( self , idx : int ) -> tuple[ Tensor , Tensor ] :
        capture_idx , bit_idx = self.index[ idx ]
        samples = self.captures[ capture_idx ]
        center = int ( self.centers[ capture_idx ][ bit_idx ] )
        bit = int ( self.labels[ capture_idx ][ bit_idx ] )
        window = centered_complex_window ( samples , center , self.window_size )
        return complex_samples_to_tensor ( window ) , torch.tensor ( bit , dtype = torch.long )

    @classmethod
    def from_capture (
        cls ,
        raw_samples : NDArray[ np.int16 ] | NDArray[ np.complexfloating ] ,
        payload_bytes : Sequence[ int ] | NDArray[ np.uint8 ] ,
        window_size : int = DEFAULT_WINDOW_SIZE ,
        frame_start_idx : int | None = None ,
    ) -> CaptureBitDataset :
        complex_samples , alignment = align_capture_to_known_payload ( raw_samples = raw_samples , payload_bytes = payload_bytes , frame_start_idx = frame_start_idx )
        return cls (
            captures = [ complex_samples ] ,
            labels = [ alignment.bits ] ,
            centers = [ alignment.centers ] ,
            window_size = window_size ,
        )


class SyntheticBitDataset ( Dataset[ tuple[ Tensor , Tensor ] ] ) :

    def __init__ (
        self ,
        examples : int ,
        window_size : int = DEFAULT_WINDOW_SIZE ,
        payload_nbytes : int = 32 ,
        seed : int = 7 ,
    ) -> None :
        self.examples = int ( examples )
        self.window_size = int ( window_size )
        self.payload_nbytes = int ( payload_nbytes )
        self.rng = np.random.default_rng ( seed )

    def __len__ ( self ) -> int :
        return self.examples

    def __getitem__ ( self , idx : int ) -> tuple[ Tensor , Tensor ] :
        del idx
        payload = self.rng.integers ( 0 , 256 , size = self.payload_nbytes , dtype = np.uint8 )
        frame_bits = build_tx_frame_bits ( payload )
        tx_samples = build_tx_frame_reference ( payload )
        channel_samples = apply_channel_impairments ( tx_samples , self.rng )
        bit_idx = int ( self.rng.integers ( 0 , frame_bits.size ) )
        center_idx = int ( build_bit_center_indices ( frame_bits.size , 0 )[ bit_idx ] )
        window = centered_complex_window ( channel_samples , center_idx , self.window_size )
        return complex_samples_to_tensor ( window ) , torch.tensor ( int ( frame_bits[ bit_idx ] ) , dtype = torch.long )


class ConvBlock ( nn.Module ) :

    def __init__ ( self , in_channels : int , out_channels : int , kernel_size : int , stride : int = 1 ) -> None :
        super () .__init__ ()
        padding = kernel_size // 2
        self.block = nn.Sequential (
            nn.Conv1d ( in_channels , out_channels , kernel_size = kernel_size , stride = stride , padding = padding , bias = False ) ,
            nn.BatchNorm1d ( out_channels ) ,
            nn.GELU () ,
            nn.Conv1d ( out_channels , out_channels , kernel_size = kernel_size , padding = padding , bias = False ) ,
            nn.BatchNorm1d ( out_channels ) ,
        )
        self.skip = nn.Identity () if in_channels == out_channels and stride == 1 else nn.Sequential (
            nn.Conv1d ( in_channels , out_channels , kernel_size = 1 , stride = stride , bias = False ) ,
            nn.BatchNorm1d ( out_channels ) ,
        )
        self.act = nn.GELU ()

    def forward ( self , x : Tensor ) -> Tensor :
        return self.act ( self.block ( x ) + self.skip ( x ) )


class BlackBoxBitNet ( nn.Module ) :

    def __init__ ( self , base_channels : int = 32 , dropout : float = 0.15 ) -> None :
        super () .__init__ ()
        self.features = nn.Sequential (
            ConvBlock ( 2 , base_channels , kernel_size = 9 , stride = 2 ) ,
            ConvBlock ( base_channels , base_channels * 2 , kernel_size = 9 , stride = 2 ) ,
            ConvBlock ( base_channels * 2 , base_channels * 4 , kernel_size = 7 , stride = 2 ) ,
            ConvBlock ( base_channels * 4 , base_channels * 4 , kernel_size = 5 , stride = 2 ) ,
            nn.AdaptiveAvgPool1d ( 1 ) ,
        )
        self.classifier = nn.Sequential (
            nn.Flatten () ,
            nn.Dropout ( p = dropout ) ,
            nn.Linear ( base_channels * 4 , base_channels * 2 ) ,
            nn.GELU () ,
            nn.Dropout ( p = dropout ) ,
            nn.Linear ( base_channels * 2 , 2 ) ,
        )

    def forward ( self , x : Tensor ) -> Tensor :
        return self.classifier ( self.features ( x ) )


@dataclass ( slots = True )
class EpochStats :
    loss : float
    accuracy : float


def run_epoch (
    model : nn.Module ,
    loader ,
    optimizer : torch.optim.Optimizer | None ,
    device : str = DEFAULT_DEVICE ,
) -> EpochStats :
    is_training = optimizer is not None
    model.train ( is_training )
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for features , labels in loader :
        features = features.to ( device )
        labels = labels.to ( device )
        with torch.set_grad_enabled ( is_training ) :
            logits = model ( features )
            loss = F.cross_entropy ( logits , labels )
            if is_training :
                optimizer.zero_grad ( set_to_none = True )
                loss.backward ()
                optimizer.step ()
        total_loss += float ( loss.item () ) * features.size ( 0 )
        total_correct += int ( ( logits.argmax ( dim = 1 ) == labels ).sum ().item () )
        total_examples += int ( features.size ( 0 ) )

    if total_examples == 0 :
        return EpochStats ( loss = math.nan , accuracy = math.nan )
    return EpochStats ( loss = total_loss / total_examples , accuracy = total_correct / total_examples )


def predict_windows (
    model : nn.Module ,
    samples : NDArray[ np.int16 ] | NDArray[ np.complexfloating ] ,
    centers : Sequence[ int ] ,
    window_size : int = DEFAULT_WINDOW_SIZE ,
    batch_size : int = 16 ,
    device : str = DEFAULT_DEVICE ,
) -> NDArray[ np.float32 ] :
    model.eval ()
    complex_samples = normalize_complex_samples ( coerce_samples_to_complex64 ( samples ) )
    probabilities : list[ NDArray[ np.float32 ] ] = []

    with torch.no_grad () :
        for start_idx in range ( 0 , len ( centers ) , batch_size ) :
            batch_centers = centers[ start_idx : start_idx + batch_size ]
            batch = torch.stack ( [ complex_samples_to_tensor ( centered_complex_window ( complex_samples , int ( center ) , window_size ) ) for center in batch_centers ] )
            logits = model ( batch.to ( device ) )
            probs = torch.softmax ( logits , dim = 1 ).cpu ().numpy ().astype ( np.float32 , copy = False )
            probabilities.append ( probs )

    if not probabilities :
        return np.empty ( ( 0 , 2 ) , dtype = np.float32 )
    return np.concatenate ( probabilities , axis = 0 )


def predict_capture_stride (
    model : nn.Module ,
    samples : NDArray[ np.int16 ] | NDArray[ np.complexfloating ] ,
    window_size : int = DEFAULT_WINDOW_SIZE ,
    stride : int = 4 ,
    batch_size : int = 16 ,
    device : str = DEFAULT_DEVICE ,
) -> tuple[ NDArray[ np.int64 ] , NDArray[ np.float32 ] ] :
    complex_samples = coerce_samples_to_complex64 ( samples )
    centers = capture_score_centers ( complex_samples.size , window_size = window_size , stride = stride )
    probs = predict_windows ( model = model , samples = complex_samples , centers = centers , window_size = window_size , batch_size = batch_size , device = device )
    return centers , probs


def save_checkpoint ( model : nn.Module , checkpoint_path : str | Path ) -> None :
    target = Path ( checkpoint_path )
    target.parent.mkdir ( parents = True , exist_ok = True )
    torch.save ( model.state_dict () , target )


def load_checkpoint ( checkpoint_path : str | Path , device : str = DEFAULT_DEVICE ) -> BlackBoxBitNet :
    model = BlackBoxBitNet ()
    state = torch.load ( Path ( checkpoint_path ) , map_location = device )
    model.load_state_dict ( state )
    model.to ( device )
    model.eval ()
    return model