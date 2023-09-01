import json
import logging
import random
import time
import copy
from argparse import ArgumentParser
from itertools import chain
from pprint import pformat
import copy
import numpy as np

import torch
import torch.nn.functional as F

from transformers import *
from VideoGPT2 import *
from train7_dop1_nohist_novideoloss import SPECIAL_TOKENS, SPECIAL_TOKENS_DICT
from dataset_test import get_dataset_test as get_dataset 
from dataset_test import build_input_from_segments

def top_filtering(logits, top_k=0, top_p=0.0, threshold=-float('Inf'), filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k, top-p (nucleus) and/or threshold filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k: <=0: no filtering, >0: keep only top k tokens with highest probability.
            top_p: <=0.0: no filtering, >0.0: keep only a subset S of candidates, where S is the smallest subset
                whose total probability mass is greater than or equal to the threshold top_p.
                In practice, we select the highest probability tokens whose cumulative probability mass exceeds
                the threshold top_p.
            threshold: a minimal threshold to keep logits
    """
    assert logits.dim() == 1  # Only work for batch size 1 for now - could update but it would obfuscate a bit the code
    top_k = min(top_k, logits.size(-1))
    if top_k > 0:
        # Remove all tokens with a probability less than the last token in the top-k tokens
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        # Compute cumulative probabilities of sorted tokens
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probabilities = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probabilities > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # Back to unsorted indices and set them to -infinity
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value

    indices_to_remove = logits < threshold
    logits[indices_to_remove] = filter_value

    return logits

def sample_sequence(caption, history, tokenizer, model, args, current_output=None, video=None):
    special_tokens_ids = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)
    if current_output is None:
        current_output = []
    for i in range(args.max_length):
        instance, sequence = build_input_from_segments(caption, history, current_output, tokenizer, with_eos=False, drop_caption=1)

        input_ids = torch.tensor(instance["input_ids"], device=args.device).unsqueeze(0)
        token_type_ids = torch.tensor(instance["token_type_ids"], device=args.device).unsqueeze(0)
        input_embs = model.transformer.wte(input_ids)
        if video is not None:
            input_embs = torch.cat([model.video_ff(video), input_embs], dim=1)
            token_type_ids = torch.cat([torch.ones((1, video.size(1))).long().to(args.device) * tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[-2]), token_type_ids], dim=1)

        logits = model(input_embs, token_type_ids=token_type_ids)
        if "gpt2" == args.model:
            logits = logits[0]
        logits = logits[0, -1, :] / args.temperature
        logits = top_filtering(logits, top_k=args.top_k, top_p=args.top_p)
        probs = F.softmax(logits, dim=-1)

        prev = torch.topk(probs, 1)[1] if args.no_sample else torch.multinomial(probs, 1)
        if i < args.min_length and prev.item() in special_tokens_ids:
            while prev.item() in special_tokens_ids:
                prev = torch.multinomial(probs, num_samples=1)

        if prev.item() in special_tokens_ids:
            break
        current_output.append(prev.item())

    return current_output

def beam_search(caption, history, tokenizer, model, args, current_output=None, video=None):
    special_tokens_ids = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)
    if current_output is None:
        current_output = []
    hyplist = [([], 0., current_output)]
    best_state = None
    comp_hyplist = []
    
    for i in range(args.max_length):
        new_hyplist = []
        argmin = 0
        for out, lp, st in hyplist:
            instance, sequence = build_input_from_segments(caption, history, st, tokenizer, with_eos=False, drop_caption=1)

            input_ids = torch.tensor(instance["input_ids"], device=args.device).unsqueeze(0)
            token_type_ids = torch.tensor(instance["token_type_ids"], device=args.device).unsqueeze(0)
            input_embs = model.transformer.wte(input_ids)
            if video is not None:
                input_embs = torch.cat([model.video_ff(video), input_embs], dim=1)
                token_type_ids = torch.cat([torch.ones((1, video.size(1))).long().to(args.device) * tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[-3]), token_type_ids], dim=1)

            logits = model(input_embs, token_type_ids=token_type_ids)
            if "gpt2" == args.model:
                logits = logits[0]
            logp = F.log_softmax(logits, dim=-1)[:, -1, :]
            lp_vec = logp.cpu().data.numpy() + lp
            lp_vec = np.squeeze(lp_vec)
            if i >= args.min_length:
                new_lp = lp_vec[tokenizer.eos_token_id] + args.penalty * (len(out) + 1)
                comp_hyplist.append((out, new_lp))
                if best_state is None or best_state < new_lp:
                    best_state = new_lp
            count = 1
            for o in np.argsort(lp_vec)[::-1]:
                if o == tokenizer.unk_token_id or o == tokenizer.eos_token_id:
                    continue
                new_lp = lp_vec[o]
                if len(new_hyplist) == args.beam_size:
                    if new_hyplist[argmin][1] < new_lp:
                        new_st = copy.deepcopy(st)
                        new_st.append(int(o))
                        new_hyplist[argmin] = (out + [o], new_lp, new_st)
                        argmin = min(enumerate(new_hyplist), key=lambda h: h[1][1])[0]
                    else:
                        break
                else:
                    new_st = copy.deepcopy(st)
                    new_st.append(int(o))
                    new_hyplist.append((out + [o], new_lp, new_st))
                    if len(new_hyplist) == args.beam_size:
                        argmin = min(enumerate(new_hyplist), key=lambda h: h[1][1])[0]
                count += 1
        hyplist = new_hyplist 
    if len(comp_hyplist) > 0:
        maxhyps = sorted(comp_hyplist, key=lambda h: -h[1])[:1]
        return maxhyps
    else:
        return [([], 0)]

def greedy_decode(caption, history, tokenizer, model, args, current_output=None, video=None):
    special_tokens_ids = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)
    ys = []
    
    for i in range(args.max_length):
        instance, sequence = build_input_from_segments(caption, history, ys, tokenizer, with_eos=False, drop_caption=1)

        input_ids = torch.tensor(instance["input_ids"], device=args.device).unsqueeze(0)
        token_type_ids = torch.tensor(instance["token_type_ids"], device=args.device).unsqueeze(0)
        input_embs = model.transformer.wte(input_ids)
        if video is not None:
            input_embs = torch.cat([model.video_ff(video), input_embs], dim=1)
            token_type_ids = torch.cat([torch.ones((1, video.size(1))).long().to(args.device) * tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[-2]), token_type_ids], dim=1)

        logits = model(input_embs, token_type_ids=token_type_ids)
        if "gpt2" == args.model:
            logits = logits[0][0]
        logits = logits.cpu().data.numpy()
        next_word = np.argsort(logits[-1])[-1]
        if next_word == special_tokens_ids[1]:
            break
        ys.append(next_word)
    return ys    


# Evaluation routine
def generate_response(model, data, dataset, args, ref_data=None):
    result_dialogs = []
    model.eval()
    with torch.no_grad():
        qa_id = 0
        for idx, dialog in enumerate(data['dialogs']):
            vid = dialog['image_id']
            out_dialog = dialog['dialog'][-1:]
            pred_dialog = {'image_id': vid,
                           'dialog': copy.deepcopy(out_dialog)}
            result_dialogs.append(pred_dialog)

            vgg = np.load("../../visvideo_data/vggish/"+vid+".npy")
            i3d_flow = np.load("../../visvideo_data/i3d_flow/"+vid+".npy")
            i3d_rgb = np.load("../../visvideo_data/i3d_rgb/"+vid+".npy")
        
            sample_i3d_flow = i3d_flow[range(1, i3d_flow.shape[0], 1)]
            sample_i3d_rgb = i3d_rgb[range(1, i3d_rgb.shape[0], 1)]

            vgg = torch.from_numpy(vgg).float().to(args.device)
            i3d_flow = torch.from_numpy(sample_i3d_flow).float().to(args.device)
            i3d_rgb = torch.from_numpy(sample_i3d_rgb).float().to(args.device)
            min_length = min([i3d_flow.size(0), i3d_rgb.size(0), vgg.size(0)])
            i3d = torch.cat([i3d_flow[:min_length], i3d_rgb[:min_length], vgg[:min_length]], dim=1).unsqueeze(0)

            for t, qa in enumerate(out_dialog):
                logging.info('%d %s_%d' % (qa_id, vid, t))
                logging.info('QS: ' + qa['question'])
                # prepare input data
                start_time = time.time()
                qa_id += 1
                
                if args.beam_search:
                   #hypstr = greedy_decode(dataset[idx]["caption"], dataset[idx]["history"], tokenizer, model, args, video=i3d)
                   hypstr = beam_search(dataset[idx]["caption"], dataset[idx]["history"], tokenizer, model, args, video=i3d)
                   hypstr = hypstr[0][0]
                else:
                   hypstr=sample_sequence(dataset[idx]["caption"], dataset[idx]["history"], tokenizer, model, args, video=i3d)
                hypstr=tokenizer.decode(hypstr, skip_special_tokens=True)
                logging.info('HYP: ' + hypstr)
                pred_dialog['dialog'][t]['answer'] = hypstr
                logging.info('ElapsedTime: %f' % (time.time() - start_time))
                logging.info('-----------------------')


    return {'dialogs': result_dialogs}


##################################
# main
if __name__ =="__main__":
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt2", help="Model type (gpt or gpt2)")
    parser.add_argument("--model_checkpoint", type=str, default="log_without_caption_with_valid/", help="Path, url or short name of the model")
    parser.add_argument("--max_history", type=int, default=3, help="Number of previous utterances to keep in history")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")

    parser.add_argument("--no_sample", action='store_true', help="Set to use greedy decoding instead of sampling")
    parser.add_argument("--beam_search", action='store_true', help="Set to use beam search instead of sampling")
    parser.add_argument("--beam_size", type=int, default=5, help="Beam size")
    parser.add_argument("--max_length", type=int, default=20, help="Maximum length of the output utterances")
    parser.add_argument("--min_length", type=int, default=1, help="Minimum length of the output utterances")
    parser.add_argument("--penalty", type=float, default=0.3, help="elngth penalty")
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    parser.add_argument("--temperature", type=int, default=0.7, help="Sampling softmax temperature")
    parser.add_argument("--top_k", type=int, default=0, help="Filter top-k tokens before sampling (<=0: no filtering)")
    parser.add_argument("--top_p", type=float, default=0.9, help="Nucleus filtering (top-p) before sampling (<=0.0: no filtering)")
    parser.add_argument("--test_set", type=str, default="data/test_set4DSTC8-AVSD.json")
    parser.add_argument("--lbl_test_set", type=str, default="data/lbl_undisclosedonly_test_set4DSTC7-AVSD.json")
    parser.add_argument("--output", type=str, default="result.json")
    parser.add_argument("--model_pth", type=str, default="checkpoint_mymodel_4.pth")
    args = parser.parse_args()
    for arg in vars(args):
        print("{}={}".format(arg, getattr(args, arg)))

    logging.basicConfig(level=logging.INFO,
            format='%(asctime)s %(levelname)s: %(message)s')
 
    logging.info('Loading model params from ' + args.model_checkpoint)
    
    tokenizer_class = GPT2Tokenizer if "gpt2" == args.model else OpenAIGPTTokenizer
    tokenizer = tokenizer_class.from_pretrained(args.model_checkpoint)
    tokenizer.add_special_tokens(SPECIAL_TOKENS_DICT)
    model_class = VideoGPT2LMHeadModel if "gpt2" == args.model else OpenAIGPTLMHeadModel
    model_config = GPT2Config.from_pretrained(args.model_checkpoint)
    model = model_class.from_pretrained(args.model_checkpoint+args.model_pth, config=model_config)
    model.to(args.device)
    model.eval()



    logging.info('Loading test data from ' + args.test_set)
    test_data = json.load(open(args.test_set,'r'))
    test_dataset = get_dataset(tokenizer, args.test_set, undisclosed_only=True, n_history=args.max_history)
    # generate sentences
    logging.info('-----------------------generate--------------------------')
    start_time = time.time()
    result = generate_response(model, test_data, test_dataset, args)
    logging.info('----------------')
    logging.info('wall time = %f' % (time.time() - start_time))
    if args.output:
        logging.info('writing results to ' + args.output)
        json.dump(result, open(args.output, 'w'), indent=4)
    logging.info('done')
