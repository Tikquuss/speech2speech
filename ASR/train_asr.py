import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
# from torch import nn
from tqdm import tqdm

from .config import device, print_freq, vocab_size, sos_id, eos_id
from .data_gen1 import AiShellDataset, pad_collate
from .transformer.decoder import Decoder
from .transformer.encoder import Encoder
from .transformer.loss import cal_performance
from .transformer.optimizer import TransformerOptimizer
from .transformer.transformer import Transformer
from .utils1 import parse_args, save_checkpoint, AverageMeter, get_logger

### avg_cer
import argparse
import pickle
from .config import pickle_file, device, input_dim, LFR_m, LFR_n, sos_id, eos_id
from .data_gen1 import build_LFR_features
from .utils1 import extract_feature
from .xer import cer_function

## valid/test
from .train1 import train, valid

def init_training(params, parser, log = False):
    a = parse_args(parser, parse_arg = False)
    args = a.parse_args()
    
    trainer, evaluator = {}, {}
    
    torch.manual_seed(7)
    np.random.seed(7)
    checkpoint  = args.checkpoint
    start_epoch = 0
    best_loss   = float('inf')
    writer      = SummaryWriter()
    epochs_since_improvement = 0
    
    trainer["args"]                     = args
    trainer["writer"]                   = writer
    trainer["best_loss"]                = best_loss
    trainer["epochs_since_improvement"] = epochs_since_improvement 

    # Initialize / load checkpoint
    if checkpoint is None:
        # model
        encoder = Encoder(args.d_input * args.LFR_m, args.n_layers_enc, args.n_head,
                          args.d_k, args.d_v, args.d_model, args.d_inner,
                          dropout=args.dropout, pe_maxlen=args.pe_maxlen)
        decoder = Decoder(sos_id, eos_id, vocab_size,
                          args.d_word_vec, args.n_layers_dec, args.n_head,
                          args.d_k, args.d_v, args.d_model, args.d_inner,
                          dropout=args.dropout,
                          tgt_emb_prj_weight_sharing=args.tgt_emb_prj_weight_sharing,
                          pe_maxlen=args.pe_maxlen)
        model = Transformer(encoder, decoder)
        # print(model)
        # model = nn.DataParallel(model)

        # optimizer
        optimizer = TransformerOptimizer(
            torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-09))

    else:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        epochs_since_improvement = checkpoint['epochs_since_improvement']
        model = checkpoint['model']
        optimizer = checkpoint['optimizer']

    trainer["epoch"] = start_epoch
    
    trainer["model"] = model
    trainer["optimizer"] = optimizer

    logger = get_logger()
    trainer["logger"] = logger

    # Move to GPU, if available
    model = model.to(device)

    # Custom dataloaders
    train_dataset = AiShellDataset(args, 'train')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=pad_collate,
                                               pin_memory=True, shuffle=True, num_workers=args.num_workers)
    trainer["train_loader"] = train_loader

    valid_dataset = AiShellDataset(args, 'dev')
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, collate_fn=pad_collate,
                                               pin_memory=True, shuffle=False, num_workers=args.num_workers)
    evaluator["valid_loader"] = valid_loader

    test_dataset = AiShellDataset(args, 'test')
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=pad_collate,
                                               pin_memory=True, shuffle=False, num_workers=args.num_workers)

    evaluator["test_loader"] = test_loader

    return  trainer, evaluator

def one_epoch(args, trainer):
    # One epoch's training
    train_loss = train(
                    train_loader = trainer["train_loader"], 
                    model = trainer["model"], 
                    optimizer = trainer["optimizer"], 
                    epoch = args.epoch, 
                    logger = trainer["logger"],
                    args = trainer["args"] 
                    )
    trainer["writer"].add_scalar('model/train_loss', train_loss, args.epoch)

    lr = trainer["optimizer"].lr
    print('\nLearning rate: {}'.format(lr))
    trainer["writer"].add_scalar('model/learning_rate', lr, args.epoch)
    step_num = trainer["optimizer"].step_num
    print('Step num: {}\n'.format(step_num))
    trainer["epoch"] = args.epoch

def run_eval(params, trainer, evaluator, run_test = False):
    
    scores = {}
    # One epoch's validation
    valid_loss = valid(
                    valid_loader = evaluator["valid_loader"], 
                    model = trainer["model"], 
                    logger = trainer["logger"],
                    args = trainer["args"] 
                    )
    trainer["writer"].add_scalar('model/valid_loss', valid_loss, params.epoch)
    scores["valid_loss"] = valid_loss

    with open(pickle_file, 'rb') as file:
        data = pickle.load(file)
    char_list = data['IVOCAB']
    
    valid_avg_cer = avg_cer(trainer["args"], model = trainer["model"], samples = data['dev'], char_list = char_list)
    scores["valid_avg_cer"] = valid_avg_cer

    if run_test :
        test_loss = valid(
                    valid_loader = evaluator["valid_loader"], 
                    model = trainer["model"], 
                    logger = trainer["logger"]
                    )
        trainer["writer"].add_scalar('model/test_loss', test_loss, params.epoch)
        scores["test_loss"] = test_loss

        test_avg_cer = avg_cer(trainer["args"], model = trainer["model"], samples = data['test'], char_list = char_list)
        scores["test_avg_cer"] = test_avg_cer
    

    return scores

def end_of_epoch(params, trainer, scores):
    # Check if there was an improvement
    valid_loss = scores["valid_loss"]
    is_best = valid_loss < trainer["best_loss"]
    trainer["best_loss"] = min(valid_loss, trainer["best_loss"])
    if not is_best:
        trainer["epochs_since_improvement"] += 1
        print("\nEpochs since last improvement: %d\n" % (trainer["epochs_since_improvement"],))
    else:
        trainer["epochs_since_improvement"] = 0

    # Save checkpoint
    save_checkpoint(
                    trainer["epoch"], 
                    trainer["epochs_since_improvement"], 
                    trainer["model"], 
                    trainer["optimizer"], 
                    trainer["best_loss"], 
                    is_best
                  )
    print(scores)

def avg_cer(args, model, samples, char_list, log = False, tqdm = False):
    
    num_samples = len(samples)

    total_cer = 0
    
    #iterator = range(num_samples)
    for i in (tqdm(range(num_samples)) if tqdm else range(num_samples)):
        sample = samples[i]
        wave = sample['wave']
        trn = sample['trn']

        feature = extract_feature(input_file=wave, feature='fbank', dim=input_dim, cmvn=True)
        feature = build_LFR_features(feature, m=LFR_m, n=LFR_n)
        # feature = np.expand_dims(feature, axis=0)
        input = torch.from_numpy(feature).to(device)
        input_length = [input[0].shape[0]]
        input_length = torch.LongTensor(input_length).to(device)
        with torch.no_grad():
            nbest_hyps = model.recognize(input, input_length, char_list, args)

        hyp_list = []
        for hyp in nbest_hyps:
            out = hyp['yseq']
            try :
                out = [char_list[idx] for idx in out if idx not in (sos_id, eos_id)]
            except KeyError as ke:
                # todo :
                print(ke)
                keys = char_list.keys()
                out = [char_list[idx] for idx in out if idx not in (sos_id, eos_id) and idx in keys]
            
            out = ''.join(out)
            hyp_list.append(out)

        if log :
            print(hyp_list)
        
        try :
            gt = [char_list[idx] for idx in trn if idx not in (sos_id, eos_id)]
        except KeyError as ke:
            # todo :
            print(ke)
            keys = char_list.keys()
            gt = [char_list[idx] for idx in trn if idx not in (sos_id, eos_id) and idx in keys]
            
        gt = ''.join(gt)
        gt_list = [gt]

        if log :
            print(gt_list)

        cer = cer_function(gt_list, hyp_list)
        total_cer += cer

    avg_cer = total_cer / num_samples
    
    return avg_cer
