from tensorboardX import SummaryWriter
import torchvision.utils as vutils
import os
from tqdm import tqdm

from .preprocess import get_dataset, DataLoader, collate_fn_transformer
from .network import *
from .config import lr as hplr, batch_size, image_step, save_step, checkpoint_path

def adjust_learning_rate(optimizer, step_num, warmup_step=4000):
    lr = hplr * warmup_step**0.5 * min(step_num * warmup_step**-1.5, step_num**-0.5)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def init_training(params, log = True):
    trainer, evaluator = {}, {}

    dataset = get_dataset()
    global_step = 0
    trainer["dataset"]     = dataset
    trainer["global_step"] = global_step
    
    m = nn.DataParallel(Model().cuda())
    m.train()
    
    optimizer = t.optim.Adam(m.parameters(), lr=hplr)
    trainer["model"] = m
    trainer["optimizer"] = optimizer

    pos_weight = t.FloatTensor([5.]).cuda()
    writer = SummaryWriter()
    trainer["pos_weight"] = pos_weight
    trainer["writer"] = writer

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_transformer, drop_last=True, num_workers=16)
    trainer["dataloader"] = dataloader

    return trainer, evaluator

def one_epoch(params, trainer):

    pbar = tqdm(trainer["dataloader"])
    
    for i, data in enumerate(pbar):
        pbar.set_description("Processing at epoch %d"%params.epoch)
        trainer["global_step"] += 1
        if trainer["global_step"] < 400000:
            adjust_learning_rate(trainer["optimizer"], trainer["global_step"])
                
        character, mel, mel_input, pos_text, pos_mel, _ = data
        
        stop_tokens = t.abs(pos_mel.ne(0).type(t.float) - 1)
        
        character = character.cuda()
        mel = mel.cuda()
        mel_input = mel_input.cuda()
        pos_text = pos_text.cuda()
        pos_mel = pos_mel.cuda()
        
        mel_pred, postnet_pred, attn_probs, stop_preds, attns_enc, attns_dec = trainer["model"].forward(character, mel_input, pos_text, pos_mel)

        mel_loss = nn.L1Loss()(mel_pred, mel)
        post_mel_loss = nn.L1Loss()(postnet_pred, mel)
            
        loss = mel_loss + post_mel_loss
            
        trainer["writer"].add_scalars('training_loss',{
                'mel_loss':mel_loss,
                'post_mel_loss':post_mel_loss,

            }, trainer["global_step"])
                
        trainer["writer"].add_scalars('alphas',{
                'encoder_alpha':trainer["model"].module.encoder.alpha.data,
               'decoder_alpha':trainer["model"].module.decoder.alpha.data,
            }, trainer["global_step"])
            
            
        if trainer["global_step"] % image_step == 1:
            
            for i, prob in enumerate(attn_probs):
                
                num_h = prob.size(0)
                for j in range(4):
            
                    x = vutils.make_grid(prob[j*16] * 255)
                    trainer["writer"].add_image('Attention_%d_0'%trainer["global_step"], x, i*4+j)
                
            for i, prob in enumerate(attns_enc):
                num_h = prob.size(0)
                    
                for j in range(4):
                
                    x = vutils.make_grid(prob[j*16] * 255)
                    trainer["writer"].add_image('Attention_enc_%d_0'%trainer["global_step"], x, i*4+j)
            
            for i, prob in enumerate(attns_dec):

                num_h = prob.size(0)
                for j in range(4):
                
                    x = vutils.make_grid(prob[j*16] * 255)
                    trainer["writer"].add_image('Attention_dec_%d_0'%trainer["global_step"], x, i*4+j)
                
        trainer["optimizer"].zero_grad()
        # Calculate gradients
        loss.backward()
            
        nn.utils.clip_grad_norm_(trainer["model"].parameters(), 1.)
            
        # Update weights
        trainer["optimizer"].step()
        
        if trainer["global_step"] % save_step == 0:
            t.save({'model':trainer["model"].state_dict(),
                    'optimizer':trainer["optimizer"].state_dict()},
                    os.path.join(checkpoint_path,'checkpoint_transformer_%d.pth.tar' % trainer["global_step"]))

                        
def run_eval(params, trainer, evaluator):
    scores = {}
    return scores

def end_of_epoch(params, trainer, scores):
    pass
        
