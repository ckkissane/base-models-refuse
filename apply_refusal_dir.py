import torch
import functools
from colorama import Fore
import argparse
import textwrap
from jaxtyping import Float
import einops
from torch import Tensor

from transformer_lens import HookedTransformer, utils
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformer_lens.hook_points import HookPoint

from utils import get_harmful_instructions, get_harmless_instructions, tokenize_instructions, get_generations, get_refusal_scores, save_records_to_json, GEMMA_CHAT_TEMPLATE, GEMMA_BASE_TEMPLATE, QWEN_CHAT_TEMPLATE, LLAMA_CHAT_TEMPLATE, LLAMA_BASE_TEMPLATE

def act_add_hook(
    activation: Float[Tensor, "... d_act"],
    hook: HookPoint,
    direction: Float[Tensor, "d_act"],
    steering_coef: int
):
    activation += steering_coef * direction
    return activation

def direction_ablation_hook(
    activation: Float[Tensor, "... d_act"],
    hook: HookPoint,
    direction: Float[Tensor, "d_act"]
):
    direction = direction / (direction.norm(dim=-1, keepdim=True) + 1e-8)
    proj = einops.einsum(activation, direction.view(-1, 1), '... d_act, d_act single -> ... single') * direction
    return activation - proj

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--model_name', type=str, default='gemma-2-9b-it',
        help='Name of model to extract refusal direction from')
    parser.add_argument(
        '--n_inst_test', type=int, default=100,
        help='Number of contrast pairs to compute refusal dir')
    parser.add_argument(
        '--max_tokens_generated', type=int, default=16,
        help='Maximum number of tokens to generate from model')
    parser.add_argument(
        '--batch_size', type=int, default=4,
        help='Batch size when generating completions')
    parser.add_argument(
        '--intervention_type', type=str, default='addition',
        help='Type of intervention to apply. (addition or ablation)')
    parser.add_argument(
        '--instruction_type', type=str, default='harmful',
        help='Type of instructions to use. (harmful or harmless)')
    parser.add_argument(
        '--print_completions', action='store_true',
        help='Whether to print the baseline and intervention completions')
    
    args = parser.parse_args()
    torch.set_grad_enabled(False)
    
    assert args.intervention_type in ['addition', 'ablation']
    assert args.instruction_type in ['harmful', 'harmless']
    
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

        layer = 23
        
        if args.model_name == 'gemma-2-9b-it':
            template = GEMMA_CHAT_TEMPLATE
        elif args.model_name == 'gemma-2-9b':
            template = GEMMA_BASE_TEMPLATE
        else:
            raise ValueError(f'Unsupported model: {args.model_name}')
    
        chat_model_name = 'gemma-2-9b-it'
        base_model_name = 'gemma-2-9b'
        
        steering_coef = 20.0 if args.instruction_type == 'harmful' else 42.0
        chat_steering_coef, base_steering_coef = steering_coef, steering_coef
        
    elif args.model_name.startswith('qwen1.5-0.5b'):
        model = HookedTransformer.from_pretrained(
            args.model_name,
            default_padding_side='left',
        )

        model.tokenizer.padding_side = 'left'
        model.tokenizer.add_special_tokens({'pad_token': '[PAD]'})  # add padding token to tokenizer

        layer = 13

        template = QWEN_CHAT_TEMPLATE

        chat_model_name = 'qwen1.5-0.5b-chat'
        base_model_name = 'qwen1.5-0.5b'
        
        steering_coef = 2.0
        chat_steering_coef, base_steering_coef = steering_coef, steering_coef
    
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

        layer = 14

        template = LLAMA_CHAT_TEMPLATE

        chat_model_name = 'vicuna-7b-v1.1'
        base_model_name = 'llama-7b'

        assert args.intervention_type == 'ablation', 'Only ablation is supported for vicuna-7b-v1.1'

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

        layer = 14
    
        template = LLAMA_BASE_TEMPLATE
        
        chat_model_name = 'vicuna-7b-v1.1'
        base_model_name = 'llama-7b'
        
        chat_steering_coef = 8.0 if args.instruction_type == 'harmless' else 4.0
        base_steering_coef = 12.0
        
    else:
        raise ValueError(f'Unsupported model: {args.model_name}')
        
        
    tokenize_instructions_fn = functools.partial(tokenize_instructions, tokenizer=model.tokenizer, template=template)

    chat_refusal_dir = torch.load(f'refusal_directions/refusal_dir_{chat_model_name}.pt')

    base_refusal_dir = torch.load(f'refusal_directions/refusal_dir_{base_model_name}.pt')

    if args.intervention_type == 'addition':
        intervention_layers = [layer]
    else: # ablation
        intervention_layers = list(range(model.cfg.n_layers))
    
    if args.instruction_type == 'harmless':
        inst_test = harmless_inst_test
    else: # harmful
        inst_test = harmful_inst_test

    print("Generating completions (no intervention)")
    baseline_generations = get_generations(model, inst_test[:args.n_inst_test], tokenize_instructions_fn, fwd_hooks=[], max_tokens_generated=args.max_tokens_generated, batch_size=args.batch_size)

    if args.intervention_type == 'addition':
        chat_hook_fn = functools.partial(act_add_hook, direction=chat_refusal_dir, steering_coef=chat_steering_coef)
    else: # ablation
        chat_hook_fn = functools.partial(direction_ablation_hook, direction=chat_refusal_dir)
    
    print(f"Generating completions with chat refusal vector {args.intervention_type}")
    chat_fwd_hooks = [(utils.get_act_name(act_name, l), chat_hook_fn) for l in intervention_layers for act_name in ['resid_pre', 'resid_mid', 'resid_post']]
    chat_intervention_generations = get_generations(model, inst_test[:args.n_inst_test], tokenize_instructions_fn, fwd_hooks=chat_fwd_hooks, max_tokens_generated=args.max_tokens_generated, batch_size=args.batch_size)

    if args.intervention_type == 'addition':
        base_hook_fn = functools.partial(act_add_hook, direction=base_refusal_dir, steering_coef=base_steering_coef)
    else: # ablation
        base_hook_fn = functools.partial(direction_ablation_hook, direction=base_refusal_dir)
    
    print(f"Generating completions with base refusal vector {args.intervention_type}")
    base_fwd_hooks = [(utils.get_act_name(act_name, l), base_hook_fn) for l in intervention_layers for act_name in ['resid_pre', 'resid_mid', 'resid_post']]
    base_intervention_generations = get_generations(model, inst_test[:args.n_inst_test], tokenize_instructions_fn, fwd_hooks=base_fwd_hooks, max_tokens_generated=args.max_tokens_generated, batch_size=args.batch_size)

    if args.print_completions:
        for i in range(args.n_inst_test):
            print(f"INSTRUCTION {i}: {repr(inst_test[i])}")
            print(Fore.GREEN + f"BASELINE COMPLETION:")
            print(textwrap.fill(repr(baseline_generations[i]), width=100, initial_indent='\t', subsequent_indent='\t'))
            print(Fore.RED + f"INTERVENTION COMPLETION:")
            print(textwrap.fill(repr(chat_intervention_generations[i]), width=100, initial_indent='\t', subsequent_indent='\t'))
            print(Fore.BLUE + f"BASE INTERVENTION COMPLETION:")
            print(textwrap.fill(repr(base_intervention_generations[i]), width=100, initial_indent='\t', subsequent_indent='\t'))
            print(Fore.RESET)

    print("Baseline refusal score", get_refusal_scores(baseline_generations))
    print("Chat Intervention refusal score", get_refusal_scores(chat_intervention_generations))
    print("Base Intervention refusal score", get_refusal_scores(base_intervention_generations))
    
    records = {
        "model_name": args.model_name,
        "intervention": args.intervention_type,
        "dataset": args.instruction_type,
        "baseline_refusal_score": get_refusal_scores(baseline_generations),
        "chat_intervention_refusal_score": get_refusal_scores(chat_intervention_generations),
        "base_intervention_refusal_score": get_refusal_scores(base_intervention_generations),
        "prompt_template": template
    }
    
    results_file = f"results/refusal_scores.json"
    save_records_to_json(records, results_file)
        