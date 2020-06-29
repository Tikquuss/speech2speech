import random
import gc

## NMT
import NMT.train_nmt as train_nmt
from NMT.XLM.src.utils import bool_flag
## ASR
import ASR.train_asr as train_asr
## ASR
import TTS.train_tts as train_tts

import argparse

def get_parser():
    parser = argparse.ArgumentParser(description="Language transfer")
    parser.add_argument("--tasks", type=str, default="nmt-asr-tts", help="")
    parser.add_argument("--max_epoch", type=int, default=100, help="Maximum epoch size")
    parser.add_argument("--eval_only", type=bool_flag, default=False, help="Only run evaluations")
    return parser

def main(params, parser):

    trainers, evaluators, training_func  = {}, {}, {}
    
    if "nmt" in params.tasks :
        trainers["train_mt"],  evaluators["train_mt"]  = train_nmt.init_training(params, parser = parser)
        training_func.update({train_nmt : [trainers["train_mt"], evaluators["train_mt"]]})
        pass 
    
    if "asr" in params.tasks :
        trainers["train_asr"], evaluators["train_asr"] = train_asr.init_training(params, parser = parser)
        training_func.update({train_asr : [trainers["train_asr"], evaluators["train_asr"]]})

    if "tts" in params.tasks :
        trainers["train_tts"], evaluators["train_tts"] = train_tts.init_training(params, parser = parser)
        training_func.update({train_tts : [trainers["train_tts"], evaluators["train_tts"]]})

    params.epoch = 0 # update
    if params.eval_only:
        for module, func in training_func.items() :
            module.run_eval(params, func[0], func[1])
        exit()
    
    # models training
    #for epoch in range(start_epoch, args.epochs):
    for epoch in range(params.max_epoch):

        #logger.info("============ Starting epoch %i ... ============" % trainer["train_mt"].epoch)

        keys = list(training_func.keys())
        keys = random.sample(keys, len(keys)) # shuffle
        
        for module, func in {key : training_func[key] for key in keys}.items() :
            module.one_epoch(params, func[0])

        #logger.info("============ End of epoch %i ============" % trainer["train_mt"].epoch)

        params.epoch = epoch # update
        scores_list = []
        for module, func in training_func.items() :
            scores_list.append(module.run_eval(params, func[0], func[1]))

        for (module, func), scores in zip(training_func.items(), scores_list)  :
            module.end_of_epoch(params, func[0], scores)

        #logger.info("============ garbage collector collecting %d ..." % gc.collect())

if __name__ == '__main__':
    # generate parser / parse parameters
    parser = get_parser()
    params = parser.parse_args()
    
    # check parameters
    assert params.tasks != ""
    params.tasks = params.tasks.split("-")
    assert all([task in ["nmt", "asr", "tts"] for task in params.tasks])
    
    # run experiment
    main(params, parser)