

import warnings
warnings.filterwarnings("ignore")
import argparse
from transformers import AutoTokenizer
from data_utils import load_datasets
from metrics import *
from train_utils import train_and_evaluate


def run(args):
    #### Prepare datasets
    datasets = load_datasets(args.dataset, args.few_shot, args.llm)

    tokenizer = AutoTokenizer.from_pretrained(args.from_pretrained)

    if args.model_type == 'task_prefix' and args.llm:
        def tokenize_function(examples, is_train=True):
            model_inputs = tokenizer([examples['task_prefix'][0] + text for text in examples['input']], max_length=args.max_input_length, truncation=True)
            expl_model_inputs = tokenizer([examples['explain_prefix'][0] + text for text in examples['input']], max_length=args.max_input_length, truncation=True)

            model_inputs['expl_input_ids'] = expl_model_inputs['input_ids']
            model_inputs['expl_attention_mask'] = expl_model_inputs['attention_mask']

            with tokenizer.as_target_tokenizer():
                label_output_encodings = tokenizer(examples['label'], max_length=256, truncation=True)
                if is_train:
                    rationale_output_encodings = tokenizer(examples['rationale'], max_length=256, truncation=True)

            model_inputs['labels'] = label_output_encodings['input_ids']
            if is_train:
                model_inputs['aux_labels'] = rationale_output_encodings['input_ids']

            return model_inputs

    elif args.model_type == 'standard':
        def tokenize_function(examples):
            model_inputs = tokenizer(
                examples['input'],
                max_length=args.max_input_length,
                truncation=True)
            with tokenizer.as_target_tokenizer():
                label_output_encodings = tokenizer(examples['label'], max_length=256, truncation=True)
            model_inputs['labels'] = label_output_encodings['input_ids']

            return model_inputs
    else:
        raise ValueError


    if not args.llm:
        tokenized_datasets = datasets.map(
            tokenize_function,
            remove_columns=['idx', 'input', 'label', 'task_prefix', 'explain_prefix', 'icl'],
            batched=True
        )
        tokenized_train_data = tokenized_datasets['train']
        tokenized_test_data = tokenized_datasets['test']
    else:
        tokenized_train_data = datasets['train'].map(
            lambda examples: tokenize_function(examples, is_train=True),
            remove_columns=['idx','input', 'label', 'rationale', 'task_prefix', 'explain_prefix', 'icl'],
            batched=True)
        tokenized_test_data = datasets['test'].map(
            lambda examples: tokenize_function(examples, is_train=False),
            remove_columns=['idx', 'input', 'label', 'task_prefix', 'explain_prefix', 'icl'],
            batched=True)

    if args.model_type == 'standard':
        if args.dataset in ['MedNLI']:
            compute_metrics = compute_accuracy_text_aux(tokenizer)
        elif args.dataset in ['NCBI', 'BC5CDR']:
            compute_metrics = compute_prf_text_aux(tokenizer,NER=True)
        else:
            compute_metrics = compute_prf_text_aux(tokenizer)
    else:
        if args.dataset in ['MedNLI']:
            compute_metrics = compute_accuracy_text(tokenizer)
        elif args.dataset in ['NCBI', 'BC5CDR']:
            compute_metrics = compute_prf_text(tokenizer,NER=True)
        else:
            compute_metrics = compute_prf_text(tokenizer)


    train_and_evaluate(args, args.seed, tokenizer, tokenized_train_data, tokenized_test_data, compute_metrics)





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--alpha', type=float, default=0.6)
    parser.add_argument('--max_steps', type=int, default=10000)
    parser.add_argument('--eval_steps', type=int, default=250)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--optimizer_name', type=str, default='AdamW')
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--from_pretrained', type=str, default='ClinicalT5-base')
    parser.add_argument('--max_input_length', type=int, default=1024)
    parser.add_argument('--gen_max_len', type=int, default=256)
    parser.add_argument('--few_shot', type=int, default=0, choices=[0, 16, 32, 64])
    parser.add_argument('--grad_steps', type=int, default=1)
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--parallelize', action='store_true')
    parser.add_argument('--model_type', type=str, default='task_prefix', choices=['task_prefix','standard'])#
    parser.add_argument('--llm', action='store_true')
    parser.add_argument('--mixture_training', action='store_true')
    parser.add_argument('--bf16', action='store_true')
    parser.add_argument('--no_log', action='store_true')
    parser.add_argument('--output_rationale', action='store_true')

    args = parser.parse_args()

    for arg, value in sorted(vars(args).items()):
        print(f'{arg}: {value}')

    run(args)



