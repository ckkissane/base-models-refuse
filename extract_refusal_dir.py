import torch
import functools
import gc
import argparse

from transformer_lens import HookedTransformer, utils
from transformers import AutoTokenizer, AutoModelForCausalLM

from utils import get_harmful_instructions, get_harmless_instructions, tokenize_instructions, GEMMA_CHAT_TEMPLATE, GEMMA_BASE_TEMPLATE, QWEN_CHAT_TEMPLATE, LLAMA_CHAT_TEMPLATE, LLAMA_BASE_TEMPLATE

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--model_name', type=str, default='gemma-2-9b-it',
        help='Name of model to extract refusal direction from')
    parser.add_argument(
        '--n_inst_train', type=int, default=32,
        help='Number of contrast pairs to compute refusal dir')
    
    args = parser.parse_args()
    torch.set_grad_enabled(False)
    
    harmful_inst_train, harmful_inst_test = get_harmful_instructions()
    harmless_inst_train, harmless_inst_test = get_harmless_instructions()
    
    if args.model_name.startswith('gemma-2-9b'):
        model = HookedTransformer.from_pretrained_no_processing(
            args.model_name,
            device='cuda:0',
            default_padding_side='left',
            dtype=torch.float16,
        )

        model.tokenizer.padding_side = 'left'

        pos = -1
        layer = 23
        
        if args.model_name == 'gemma-2-9b-it':
            template = GEMMA_CHAT_TEMPLATE
        elif args.model_name == 'gemma-2-9b':
            template = GEMMA_BASE_TEMPLATE
        else:
            raise ValueError(f'Unsupported model: {args.model_name}')
        
    elif args.model_name.startswith('qwen1.5-0.5b'):
        model = HookedTransformer.from_pretrained(
            args.model_name,
            default_padding_side='left',
        )

        model.tokenizer.padding_side = 'left'
        model.tokenizer.add_special_tokens({'pad_token': '[PAD]'})  # add padding token to tokenizer

        pos = -1
        layer = 13
    
        template = QWEN_CHAT_TEMPLATE
    
    elif args.model_name == 'vicuna-7b-v1.1':
        tokenizer = AutoTokenizer.from_pretrained("lmsys/vicuna-7b-v1.1")
        hf_model = AutoModelForCausalLM.from_pretrained("lmsys/vicuna-7b-v1.1", torch_dtype=torch.float16)

        model = HookedTransformer.from_pretrained(
            "llama-7b",
            hf_model=hf_model,
            device="cpu",
            fold_ln=False,
            center_writing_weights=False,
            center_unembed=False,
            tokenizer=tokenizer,
            dtype=torch.float16,
            default_padding_side='left',
        )
        model.tokenizer.padding_side = 'left'
        model = model.cuda()

        pos = -1
        layer = 14

        template = LLAMA_CHAT_TEMPLATE

    elif args.model_name == 'llama-7b':
        tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-7b")
        hf_model = AutoModelForCausalLM.from_pretrained("huggyllama/llama-7b", torch_dtype=torch.float16)

        model = HookedTransformer.from_pretrained(
            "llama-7b",
            hf_model=hf_model,
            device="cpu",
            fold_ln=False,
            center_writing_weights=False,
            center_unembed=False,
            tokenizer=tokenizer,
            dtype=torch.float16,
            default_padding_side='left',
        )
        model.tokenizer.padding_side = 'left'
        model = model.cuda()

        pos = -1
        layer = 14
        
        template = LLAMA_BASE_TEMPLATE
        
    else:
        raise ValueError(f'Unsupported model: {args.model_name}')
        
        
    tokenize_instructions_fn = functools.partial(tokenize_instructions, tokenizer=model.tokenizer, template=template)

    print(f"Using {args.n_inst_train} pairs to compute refusal direction")
    harmful_toks = tokenize_instructions_fn(instructions=harmful_inst_train[:args.n_inst_train])
    harmless_toks = tokenize_instructions_fn(instructions=harmless_inst_train[:args.n_inst_train])

    harmful_logits, harmful_cache = model.run_with_cache(harmful_toks, names_filter=utils.get_act_name('resid_pre', layer))
    harmless_logits, harmless_cache = model.run_with_cache(harmless_toks, names_filter=utils.get_act_name('resid_pre', layer))

    harmful_mean_act = harmful_cache['resid_pre', layer][:, pos, :].mean(dim=0)
    harmless_mean_act = harmless_cache['resid_pre', layer][:, pos, :].mean(dim=0)

    refusal_dir = harmful_mean_act - harmless_mean_act
    refusal_dir = refusal_dir / refusal_dir.norm()

    # clean up memory
    del harmful_cache, harmless_cache, harmful_logits, harmless_logits
    gc.collect(); torch.cuda.empty_cache()

    save_file = f'refusal_directions/refusal_dir_{args.model_name}.pt'
    torch.save(refusal_dir, save_file)
    print(f'Saved refusal direction to {save_file}')