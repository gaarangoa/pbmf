import argparse
import sys
import yaml

from PBMF.bin.train import train
from PBMF.bin.show_config import show_config
from PBMF.bin.pruning import pruning
from PBMF.bin.predict import predict 
from PBMF.bin.distil import distil
from PBMF.bin.report import report

from samecode.logger.logger import logger


import tensorflow as tf
# tf.config.run_functions_eagerly(True)
# tf.data.experimental.enable_debug_mode()

# tf.config.threading.set_intra_op_parallelism_threads(1)
# tf.config.threading.set_inter_op_parallelism_threads(1)


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

logg = logger(name='PBMF')
logg.propagate = False

def pipeline(kwargs):
    args = yaml.safe_load(
        open(kwargs.config, 'r')
    )

    if args['pipeline']['train']:
        logg.info('Training PBMF ...')
        train(kwargs)

    if args['pipeline']['predict']:
        logg.info('Predicting PBMF ...')
        predict(kwargs)

    if args['pipeline']['pruning']:
        logg.info('Pruning PBMF ...')
        pruning(kwargs)

    if args['pipeline']['distil']:
        logg.info('Distilation PBMF ...')
        distil(kwargs)

    if args['pipeline']['report']:
        logg.info('Report PBMF ...')
        report(kwargs)

def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    # use deeparg section
    reads = subparsers.add_parser("train", help="Execute PBMF")
    reads.add_argument('-c', '--config', required=True, help='Config file [*.yaml]')
    reads.add_argument('--force', required=False, default=False, action='store_true', help='force rewrite content inside output directory')
    reads.set_defaults(func=train)

    reads = subparsers.add_parser("predict", help="Predict")
    reads.add_argument('-c', '--config', required=True, help='Config file [*.yaml]')
    reads.set_defaults(func=predict)

    prun = subparsers.add_parser('pruning', help='pruning algorithm')
    prun.add_argument('-c', '--config', required=True, help='Config file [*.yaml]')
    prun.set_defaults(func=pruning)

    dst = subparsers.add_parser('distil', help='perform distilation')
    dst.add_argument('-c', '--config', required=True, help='Config file [*.yaml]')
    dst.set_defaults(func=distil)

    rpt = subparsers.add_parser('report', help='perform distilation')
    rpt.add_argument('-c', '--config', required=True, help='Config file [*.yaml]')
    rpt.set_defaults(func=report)

    pipl = subparsers.add_parser("pipeline", help="Execute PBMF")
    pipl.add_argument('-c', '--config', required=True, help='Config file [*.yaml]')
    pipl.add_argument('--force', required=False, default=False, action='store_true', help='force rewrite content inside output directory')
    pipl.set_defaults(func=pipeline)

    args = parser.parse_args()
    parser.parse_args()
    args.func(args)

if __name__ == '__main__':
    main()