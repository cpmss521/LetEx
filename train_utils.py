


import os
import shutil
import logging
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import T5ForConditionalGeneration
from transformers import DataCollatorForSeq2Seq
from transformers.trainer_utils import set_seed
from model_utils import TaskPrefixDataCollator, TaskPrefixTrainer


def get_config_dir(args):
    return f'{args.dataset}/{args.from_pretrained.split("/")[-1]}/{args.model_type}/{args.alpha}/{args.max_input_length}/{args.grad_steps * args.batch_size}/{args.optimizer_name}/{args.lr}'


def train_and_evaluate(args, seed, tokenizer, tokenized_train_data, tokenized_test_data, compute_metrics):
    set_seed(seed)

    model = T5ForConditionalGeneration.from_pretrained(args.from_pretrained, from_flax=True)

    if args.parallelize:
        model.parallelize()

    config_dir = get_config_dir(args)
    output_dir = f'ckpts/{config_dir}/{seed}'  # for model ckpts
    logging_dir = f'logs/{config_dir}/{seed}'  # for training logs

    if args.no_log:
        logging_strategy = 'no'
        logging_dir = None
    else:
        logging_strategy = 'steps'

    # clear output dir if already exists
    if os.path.exists(output_dir):
        logging.info('Found existing ckpt directory. Deleted the old directory for the latest run.')
        shutil.rmtree(output_dir)

    training_args = Seq2SeqTrainingArguments(
        output_dir,
        remove_unused_columns=False,
        evaluation_strategy='steps',
        eval_steps=args.eval_steps,
        save_strategy='no', ### (save) steps / not save(no)
        save_steps=args.eval_steps,
        logging_dir=logging_dir,
        logging_strategy=logging_strategy,##
        logging_steps=args.eval_steps,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        gradient_accumulation_steps=args.grad_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        predict_with_generate=True,
        seed=seed,
        local_rank=args.local_rank,
        bf16=args.bf16,
        generation_max_length=args.gen_max_len,
        prediction_loss_only=False,
    )

    if args.model_type == 'task_prefix':
        data_collator = TaskPrefixDataCollator(tokenizer=tokenizer, model=model)
    elif args.model_type == 'standard':
        data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    else:
        raise ValueError

    trainer_kwargs = {
        'alpha': args.alpha,
        'output_rationale': args.output_rationale,
        'model': model,
        'args': training_args,
        'train_dataset': tokenized_train_data,
        'eval_dataset': {'test': tokenized_test_data, },
        'data_collator': data_collator,
        'tokenizer': tokenizer,
        'compute_metrics': compute_metrics,
    }

    if args.model_type == 'task_prefix':
        trainer = TaskPrefixTrainer(**trainer_kwargs)

    elif args.model_type == 'standard':
        trainer_kwargs.pop('alpha')
        trainer_kwargs.pop('output_rationale')
        trainer = Seq2SeqTrainer(**trainer_kwargs)
    else:
        raise ValueError

    trainer.train()