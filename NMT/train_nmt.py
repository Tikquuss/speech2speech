import json
import random
import argparse 

import copy
import gc


from .XLM.src.slurm import init_signal_handler, init_distributed_mode
from .XLM.src.data.loader import check_data_params, load_data
from .XLM.src.utils import bool_flag, initialize_exp, set_sampling_probs, shuf_order
from .XLM.src.model import check_model_params, build_model
from .XLM.src.model.memory import HashingMemory
from .XLM.src.trainer import SingleTrainer, EncDecTrainer
from .XLM.src.evaluation.evaluator import SingleEvaluator, EncDecEvaluator


def init_training(params, log = True):
    # initialize the multi-GPU / multi-node training
    init_distributed_mode(params)
    
    if log :
        # initialize the experiment
        meta_params = copy.deepcopy(params).meta_params
        params.meta_params = "..." # to long to be log
        logger = initialize_exp(params)
        params.meta_params = meta_params

    # initialize SLURM signal handler for time limit / pre-emption
    init_signal_handler()

    # load data
    data = load_data(params)
       
    # build model
    p = params.meta_params[data['key']]
    if params.encoder_only:
        model = build_model(params = p, dico = data['dico'])
    else:
        encoder, decoder = build_model(params = p, dico = data['dico'])
     
    # build trainer, reload potential checkpoints / build evaluator
    params.n_words = p.n_words
    params.bos_index = p.bos_index
    params.eos_index = p.eos_index
    params.pad_index = p.pad_index
    params.unk_index = p.unk_index
    params.mask_index = p.mask_index
    if params.encoder_only:
        trainer = SingleTrainer(model, data, params)
        evaluator = SingleEvaluator(trainer, data, params)
    else:
        trainer = EncDecTrainer(encoder, decoder, data, params)
        evaluator = EncDecEvaluator(trainer, data, params)

    return trainer, evaluator


def one_epoch(params, trainer):
    if not params.meta_learning :
        trainer.n_sentences = 0
        while trainer.n_sentences < trainer.epoch_size :
            # CLM steps
            for lang1, lang2 in shuf_order(params.clm_steps, params):
                trainer.clm_step(lang1, lang2, params.lambda_clm)
                
            # MLM steps (also includes TLM if lang2 is not None)
            for lang1, lang2 in shuf_order(params.mlm_steps, params):
                trainer.mlm_step(lang1, lang2, params.lambda_mlm)

            # parallel classification steps
            for lang1, lang2 in shuf_order(params.pc_steps, params):
                trainer.pc_step(lang1, lang2, params.lambda_pc)

            # denoising auto-encoder steps
            for lang in shuf_order(params.ae_steps):
                trainer.mt_step(lang, lang, params.lambda_ae)

            # machine translation steps
            for lang1, lang2 in shuf_order(params.mt_steps, params):
                trainer.mt_step(lang1, lang2, params.lambda_mt)

            # back-translation steps
            for lang1, lang2, lang3 in shuf_order(params.bt_steps):
                trainer.bt_step(lang1, lang2, lang3, params.lambda_bt)

            trainer.iter()
    else :
     
        trainer.n_sentences = {}
        """
        Here we build language lists for each of our meta-taks. Indeed, for two language lists l1 and l2, 
        the objective will be done with l1[i] and l2[i] respectively, this for each index i of the two lists. 
        """
        lang1_dic, lang2_dic, lang3_dic = {}, {}, {}
        """
        In the case of meta-learning, we have a (meta-)data dictionary for each (meta-)task, 
        so the keys are the languages conserved by the task. 
        """
        data_keys_dic = {}
                 
        # equivalent to "for task in list of task" in the original algorithm,  except here we prepare all the tasks beforehand.
        for lgs in params.meta_params.keys() :
            trainer.n_sentences[lgs] = 0
                
            # CLM
            try :
                lang1_dic['clm_step']
            except KeyError :
                lang1_dic['clm_step'], lang2_dic['clm_step'], data_keys_dic['clm_step'] = [], [], []
            for lang1, lang2 in shuf_order(params.meta_params[lgs].clm_steps, params):
                lang1_dic['clm_step'].append(lang1)
                lang2_dic['clm_step'].append(lang2)
                data_keys_dic['clm_step'].append(lgs)
                    
            # MLM  
            try :
                lang1_dic['mlm_step']
            except KeyError :
                lang1_dic['mlm_step'], lang2_dic['mlm_step'], data_keys_dic['mlm_step'] = [], [], []
            for lang1, lang2 in shuf_order(params.meta_params[lgs].mlm_steps, params):
                lang1_dic['mlm_step'].append(lang1)
                lang2_dic['mlm_step'].append(lang2)
                data_keys_dic['mlm_step'].append(lgs)
                           
                # parallel classification
            try :
                lang1_dic['pc_step']
            except KeyError :
                lang1_dic['pc_step'], lang2_dic['pc_step'], data_keys_dic['pc_step'] = [], [], []
            for lang1, lang2 in shuf_order(params.meta_params[lgs].pc_steps, params):
                lang1_dic['pc_step'].append(lang1)
                lang2_dic['pc_step'].append(lang2)
                data_keys_dic['pc_step'].append(lgs)
                    
            # denoising auto-encoder
            try :
                lang1_dic['ae_step']
            except KeyError :
                lang1_dic['ae_step'], data_keys_dic['ae_step'] = [], []
            for lang1 in shuf_order(params.meta_params[lgs].ae_steps):
                lang1_dic['ae_step'].append(lang1)
                data_keys_dic['ae_step'].append(lgs)
                 
            # machine translation 
            try :
                lang1_dic['mt_step']
            except KeyError :
                lang1_dic['mt_step'], lang2_dic['mt_step'], data_keys_dic['mt_step'] = [], [], []
            for lang1, lang2 in shuf_order(params.meta_params[lgs].mt_steps, params):
                lang1_dic['mt_step'].append(lang1)
                lang2_dic['mt_step'].append(lang2)
                data_keys_dic['mt_step'].append(lgs)
                   
            # back-translation
            try :
                lang1_dic['bt_step']
            except KeyError :
                lang1_dic['bt_step'], lang2_dic['bt_step'], lang3_dic['bt_step'], data_keys_dic['bt_step'] = [], [], [], []
            for lang1, lang2, lang3 in shuf_order(params.meta_params[lgs].bt_steps):
                lang1_dic['bt_step'].append(lang1)
                lang2_dic['bt_step'].append(lang2) 
                lang3_dic['bt_step'].append(lang3)
                data_keys_dic['bt_step'].append(lgs)
                        
        flag = True
                
        # equivalent to "while not done do" in the original algorithm
        while flag :
                        
            # CLM steps
            #print("clm_step", flag)
            a = trainer.clm_step(lang1_dic['clm_step'] , lang2_dic['clm_step'], params.lambda_clm, data_keys_dic['clm_step'])
                        
            #print("mlm_step", flag)
            # MLM steps (also includes TLM if lang2 is not None) 
            b = trainer.mlm_step(lang1_dic['mlm_step'] , lang2_dic['mlm_step'], params.lambda_mlm, data_keys_dic['mlm_step']) 
                       
            # parallel classification steps
            c = trainer.pc_step(lang1_dic['pc_step'] , lang2_dic['pc_step'], params.lambda_pc, data_keys_dic['pc_step']) 
                        
            if isinstance(trainer, EncDecTrainer) :
               
                # denoising auto-encoder steps
                d = trainer.mt_step(lang1_dic['ae_step'] , lang1_dic['ae_step'], params.lambda_ae, data_keys_dic['ae_step']) 
                        
                # machine translation steps    
                e = trainer.mt_step(lang1_dic['mt_step'] , lang2_dic['mt_step'], params.lambda_mt, data_keys_dic['mt_step']) 

                # back-translation steps
                f = trainer.bt_step(lang1_dic['bt_step'] , lang2_dic['bt_step'], lang3_dic['bt_step'], params.lambda_bt, data_keys_dic['bt_step'])    
                        
                # do things better
                if (not a) and (not b) and (not c) and (not d) and (not e) and (not f) :
                    flag = False # End of epoch
                else :
                    flag = True
            else :
                # do things better
                if (not a) and (not b) and (not c) :
                    flag = False # End of epoch
                else :
                    flag = True
                        
            trainer.iter()  
                        
def run_eval(params, trainer, evaluator, eval_only=False):
    # evaluate perplexity
    scores = evaluator.run_all_evals(trainer)
        
    # print / JSON log
    if not params.meta_learning :
        for k, v in scores.items():
            logger.info("%s -> %.6f" % (k, v))
    else :
        for lgs in params.meta_params.keys() :
            logger.info("============ task : %s " % lgs)
            for k, v in scores[lgs].items():
                if k != "epoch":
                    logger.info("%s -> %.6f" % (k, v))
        logger.info("============ all")
        for k, v in scores.items():
            if not (k in (list(params.meta_params.keys())+['epoch'])) :
                logger.info("%s -> %.6f" % (k, v))
                
    if not eval_only :
        if params.is_master:
            logger.info("__log__:%s" % json.dumps(scores))
    else :
        logger.info("__log__:%s" % json.dumps(scores))

    return scores

def end_of_epoch(params, trainer, scores):
    # end of epoch
    trainer.save_best_model(scores)
    trainer.save_periodic()
    trainer.end_epoch(scores)
        