import sys
import os
import logging
import copy
import torch
from utils import factory
from utils.data_manager import DataManager
from utils.toolkit import count_parameters


def train(args):
    seed_list = copy.deepcopy(args['seed'])
    device = copy.deepcopy(args['device'])

    for seed in seed_list:
        args['seed'] = seed
        args['device'] = device
        _train(args)


def _train(args):

    init_cls = 0 if args ["init_cls"] == args["increment"] else args["init_cls"]        

    _set_random()
    _set_device(args)
    print_args(args)

    data_manager = DataManager(args['dataset'], args['shuffle'], args['seed'], args['init_cls'], args['increment'])

    model = factory.get_model(args['model_name'], args)

    cnn_curve, nme_curve = {'top1': [], 'top5': []}, {'top1': [], 'top5': []}
    for task in range(data_manager.nb_tasks):
        print('All params: {}'.format(count_parameters(model._network)))
        print('Trainable params: {}'.format(count_parameters(model._network, True)))
        model.incremental_train(data_manager)
        cnn_accy, nme_accy = model.eval_task()
        model.after_task()

        if nme_accy is not None and cnn_accy is not None:
            print('CNN: {}'.format(cnn_accy['grouped']))
            print('NME: {}'.format(nme_accy['grouped']))

            cnn_curve['top1'].append(cnn_accy['top1'])
            cnn_curve['top5'].append(cnn_accy['top5'])

            nme_curve['top1'].append(nme_accy['top1'])
            nme_curve['top5'].append(nme_accy['top5'])

            print('CNN top1 curve: {}'.format(cnn_curve['top1']))
            print('CNN top5 curve: {}'.format(cnn_curve['top5']))
            print('NME top1 curve: {}'.format(nme_curve['top1']))
            print('NME top5 curve: {}\n'.format(nme_curve['top5']))
        elif nme_accy is None:
            print('No NME accuracy.')
            print('CNN: {}'.format(cnn_accy['grouped']))

            cnn_curve['top1'].append(cnn_accy['top1'])
            cnn_curve['top5'].append(cnn_accy['top5'])

            print('CNN top1 curve: {}'.format(cnn_curve['top1']))
            print('CNN top5 curve: {}\n'.format(cnn_curve['top5']))
        else:
            print('No CNN accuracy.')
            print('NME: {}'.format(nme_accy['grouped']))

            nme_curve['top1'].append(nme_accy['top1'])
            nme_curve['top5'].append(nme_accy['top5'])

            print('NME top1 curve: {}'.format(nme_curve['top1']))
            print('NME top5 curve: {}\n'.format(nme_curve['top5']))




def _set_device(args):
    requested = args.get('device', [])
    devices = []

    # Normalize to a list for uniform handling
    if not isinstance(requested, (list, tuple)):
        requested = [requested]

    # Build a list of valid CUDA devices if available
    if torch.cuda.is_available():
        cuda_count = torch.cuda.device_count()
        for d in requested:
            if d == -1 or str(d).lower() in ("cpu", "none"):
                devices.append(torch.device('cpu'))
            else:
                try:
                    idx = int(d)
                    if 0 <= idx < cuda_count:
                        devices.append(torch.device(f'cuda:{idx}'))
                except Exception:
                    # Ignore invalid entries
                    pass
        # Fallback to a single GPU if none validated
        if len([dv for dv in devices if dv.type == 'cuda']) == 0:
            devices = [torch.device('cuda:0')]
    else:
        # CUDA not available: honor explicit CPU if provided, otherwise CPU
        for d in requested:
            if d == -1 or str(d).lower() in ("cpu", "none"):
                devices.append(torch.device('cpu'))
        if not devices:
            devices = [torch.device('cpu')]

    args['device'] = devices


def _set_random():
    torch.manual_seed(1)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1)
        torch.cuda.manual_seed_all(1)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def print_args(args):
    for key, value in args.items():
        print('{}: {}'.format(key, value))
