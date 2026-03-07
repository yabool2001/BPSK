import argparse
import tomllib

import numpy as np
import torch

from torch.utils.data import DataLoader, random_split

from modules import packet, sdr
from modules.neural_rx import (
    BlackBoxBitNet,
    CaptureBitDataset,
    DEFAULT_DEVICE,
    DEFAULT_WINDOW_SIZE,
    SyntheticBitDataset,
    load_checkpoint,
    predict_capture_stride,
    predict_windows,
    run_epoch,
    save_checkpoint,
)


with open ( "settings.toml" , "rb" ) as settings_file :
    SETTINGS = tomllib.load ( settings_file )

SPS = int ( SETTINGS[ "bpsk" ][ "SPS" ] )


def parse_payload_bytes ( payload_text : str ) -> np.ndarray :
    cleaned = payload_text.strip ()
    if not cleaned :
        raise ValueError ( "Payload cannot be empty." )
    if "," in cleaned :
        values = [ int ( item.strip () , 0 ) for item in cleaned.split ( "," ) if item.strip () ]
        return np.asarray ( values , dtype = np.uint8 )
    if cleaned.lower ().startswith ( "hex:" ) :
        hex_text = cleaned[ 4 : ].replace ( " " , "" )
        return np.frombuffer ( bytes.fromhex ( hex_text ) , dtype = np.uint8 )
    return np.frombuffer ( cleaned.encode ( "utf-8" ) , dtype = np.uint8 )


def train_synthetic ( args : argparse.Namespace ) -> None :
    dataset = SyntheticBitDataset (
        examples = args.examples ,
        window_size = args.window_size ,
        payload_nbytes = args.payload_nbytes ,
        seed = args.seed ,
    )
    val_len = max ( 1 , int ( len ( dataset ) * args.val_fraction ) )
    train_len = len ( dataset ) - val_len
    train_dataset , val_dataset = random_split ( dataset , [ train_len , val_len ] , generator = torch.Generator ().manual_seed ( args.seed ) )

    train_loader = DataLoader ( train_dataset , batch_size = args.batch_size , shuffle = True , num_workers = 0 , pin_memory = args.device == "cuda" )
    val_loader = DataLoader ( val_dataset , batch_size = args.batch_size , shuffle = False , num_workers = 0 , pin_memory = args.device == "cuda" )

    model = BlackBoxBitNet ().to ( args.device )
    optimizer = torch.optim.AdamW ( model.parameters () , lr = args.lr , weight_decay = 1e-4 )

    best_val_acc = -1.0
    for epoch_idx in range ( 1 , args.epochs + 1 ) :
        train_stats = run_epoch ( model = model , loader = train_loader , optimizer = optimizer , device = args.device )
        val_stats = run_epoch ( model = model , loader = val_loader , optimizer = None , device = args.device )
        print (
            f"epoch={epoch_idx} train_loss={train_stats.loss:.5f} train_acc={train_stats.accuracy:.4f} "
            f"val_loss={val_stats.loss:.5f} val_acc={val_stats.accuracy:.4f}"
        )
        if val_stats.accuracy > best_val_acc :
            best_val_acc = val_stats.accuracy
            save_checkpoint ( model , args.checkpoint )
            print ( f"saved checkpoint to {args.checkpoint}" )


def score_known_payload_file ( args : argparse.Namespace ) -> None :
    raw_samples = np.load ( args.input )
    payload_bytes = parse_payload_bytes ( args.payload )
    model = load_checkpoint ( args.checkpoint , device = args.device )
    dataset = CaptureBitDataset.from_capture (
        raw_samples = raw_samples ,
        payload_bytes = payload_bytes ,
        window_size = args.window_size ,
    )
    centers = dataset.centers[ 0 ]
    bits = dataset.labels[ 0 ]
    probs = predict_windows (
        model = model ,
        samples = dataset.captures[ 0 ] ,
        centers = centers ,
        window_size = args.window_size ,
        batch_size = args.batch_size ,
        device = args.device ,
    )
    predicted = probs.argmax ( axis = 1 )
    accuracy = float ( np.mean ( predicted == bits ) )
    print ( f"bits={bits.size} accuracy={accuracy:.4f}" )
    print ( "bit_idx,center,p0,p1,pred,true" )
    for bit_idx , center in enumerate ( centers ) :
        print ( f"{bit_idx},{int ( center )},{probs[ bit_idx , 0 ]:.6f},{probs[ bit_idx , 1 ]:.6f},{int ( predicted[ bit_idx ] )},{int ( bits[ bit_idx ] )}" )


def score_stride_file ( args : argparse.Namespace ) -> None :
    raw_samples = np.load ( args.input )
    model = load_checkpoint ( args.checkpoint , device = args.device )
    centers , probs = predict_capture_stride (
        model = model ,
        samples = raw_samples ,
        window_size = args.window_size ,
        stride = args.stride ,
        batch_size = args.batch_size ,
        device = args.device ,
    )
    print ( "center,p0,p1,pred" )
    for center , prob in zip ( centers , probs ) :
        print ( f"{int ( center )},{prob[ 0 ]:.6f},{prob[ 1 ]:.6f},{int ( np.argmax ( prob ) )}" )


def score_stride_live ( args : argparse.Namespace ) -> None :
    rx_pluto = packet.RxPluto_v0_0_0 ( sn = sdr.PLUTO_RX_SN , gain_control_mode_chan0 = args.gain_control , rx_gain_chan0_int = args.rx_gain )
    rx_samples = packet.RxSamples_v0_0_0 ( pluto_rx_buf = rx_pluto.pluto_rx_buf )
    rx_samples.rx ()
    model = load_checkpoint ( args.checkpoint , device = args.device )
    centers , probs = predict_capture_stride (
        model = model ,
        samples = rx_samples.raw ,
        window_size = args.window_size ,
        stride = args.stride ,
        batch_size = args.batch_size ,
        device = args.device ,
    )
    print ( f"captured_int16={rx_samples.raw.size} captured_complex={rx_samples.raw.size // 2}" )
    print ( "center,p0,p1,pred" )
    for center , prob in zip ( centers , probs ) :
        print ( f"{int ( center )},{prob[ 0 ]:.6f},{prob[ 1 ]:.6f},{int ( np.argmax ( prob ) )}" )


def build_parser () -> argparse.ArgumentParser :
    parser = argparse.ArgumentParser ( description = "PyTorch black-box BPSK RX bit classifier" )
    parser.add_argument ( "--device" , default = DEFAULT_DEVICE , choices = [ "cpu" , "cuda" ] )
    subparsers = parser.add_subparsers ( dest = "command" , required = True )

    train_parser = subparsers.add_parser ( "train-synth" , help = "Train on synthetic BPSK frames with channel impairments" )
    train_parser.add_argument ( "--examples" , type = int , default = 4096 )
    train_parser.add_argument ( "--epochs" , type = int , default = 8 )
    train_parser.add_argument ( "--batch-size" , type = int , default = 8 )
    train_parser.add_argument ( "--lr" , type = float , default = 1e-3 )
    train_parser.add_argument ( "--payload-nbytes" , type = int , default = 32 )
    train_parser.add_argument ( "--window-size" , type = int , default = DEFAULT_WINDOW_SIZE )
    train_parser.add_argument ( "--val-fraction" , type = float , default = 0.1 )
    train_parser.add_argument ( "--seed" , type = int , default = 7 )
    train_parser.add_argument ( "--checkpoint" , default = "logs/black_box_bit_net.pt" )
    train_parser.set_defaults ( handler = train_synthetic )

    known_parser = subparsers.add_parser ( "score-known-file" , help = "Score a capture when payload bytes are known" )
    known_parser.add_argument ( "--input" , required = True )
    known_parser.add_argument ( "--payload" , required = True , help = "Comma-separated ints, UTF-8 text, or hex:0011aa" )
    known_parser.add_argument ( "--checkpoint" , default = "logs/black_box_bit_net.pt" )
    known_parser.add_argument ( "--window-size" , type = int , default = DEFAULT_WINDOW_SIZE )
    known_parser.add_argument ( "--batch-size" , type = int , default = 8 )
    known_parser.set_defaults ( handler = score_known_payload_file )

    file_parser = subparsers.add_parser ( "score-file" , help = "Slide over a saved capture and emit bit probabilities" )
    file_parser.add_argument ( "--input" , required = True )
    file_parser.add_argument ( "--checkpoint" , default = "logs/black_box_bit_net.pt" )
    file_parser.add_argument ( "--window-size" , type = int , default = DEFAULT_WINDOW_SIZE )
    file_parser.add_argument ( "--stride" , type = int , default = SPS )
    file_parser.add_argument ( "--batch-size" , type = int , default = 8 )
    file_parser.set_defaults ( handler = score_stride_file )

    live_parser = subparsers.add_parser ( "score-live" , help = "Read one libiio buffer from Pluto and emit sliding probabilities" )
    live_parser.add_argument ( "--checkpoint" , default = "logs/black_box_bit_net.pt" )
    live_parser.add_argument ( "--window-size" , type = int , default = DEFAULT_WINDOW_SIZE )
    live_parser.add_argument ( "--stride" , type = int , default = SPS )
    live_parser.add_argument ( "--batch-size" , type = int , default = 8 )
    live_parser.add_argument ( "--gain-control" , default = sdr.GAIN_CONTROL_MODE_CH0 )
    live_parser.add_argument ( "--rx-gain" , type = int , default = sdr.RX_GAIN_CH0 )
    live_parser.set_defaults ( handler = score_stride_live )

    return parser


def main () -> None :
    parser = build_parser ()
    args = parser.parse_args ()
    args.handler ( args )


if __name__ == "__main__" :
    main ()