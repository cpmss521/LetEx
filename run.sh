#!/bin/bash


T5_model_path="../ClinicalT5-base"
data_names=("NCBI" "BC5CDR" "AIMed" "HPRD50" "i2b2_2010" "MedNLI")
few_shots=(16 32 64)


###------------------------------------------------###
## COT training of few-shot lerning
for data_name in "${data_names[@]}"
do
    for few_shot in "${few_shots[@]}"
    do
        echo "Processing dataset: ${data_name} under single task cot fine-tuning of ${few_shot} few-shot learning"
        torchrun --nproc_per_node 2 run.py --from_pretrained ${T5_model_path} --dataset ${data_name} --model_type task_prefix --batch_size 8 --lr 5e-5 --gen_max_len 256 --max_steps 5000 --few_shot ${few_shot} --llm > ./results/logs/${data_name}_${few_shot}_task_prefix.txt 2>&1
    done
done

echo "finish and exit the script."
exit 1


###------------------------------------------------###
## COT training of full supervision
for data_name in "${data_names[@]}"
do
    echo "Processing dataset: ${data_name} under single task cot training of full supervision"
    python run.py --from_pretrained ${T5_model_path} --dataset ${data_name} --model_type task_prefix --batch_size 8 --lr 5e-5 --gen_max_len 256 --llm > ./results/logs/${data_name}_task_prefix.txt 2>&1
done









###------------------------------------------------###
## Standard fine-tuning of few-shot learning
for data_name in "${data_names[@]}"
do
    for few_shot in "${few_shots[@]}"
    do
        echo "Processing dataset: ${data_name} under Standard fine-tuning of ${few_shot} few-shot learning"
        python run.py --from_pretrained ${T5_model_path} --dataset ${data_name} --model_type standard --batch_size 8 --lr 5e-5 --max_input_length 256 --max_steps 5000 --few_shot ${few_shot} > ./results/logs/${data_name}_${few_shot}_standard.txt 2>&1
    done
done

###------------------------------------------------###
## Standard fine-tuning of full supervision
for data_name in "${data_names[@]}"
do
    echo "Processing dataset: ${data_name} under Standard fine-tuning of full supervision"
    python run.py --from_pretrained ${T5_model_path} --dataset ${data_name} --model_type standard --batch_size 8 --lr 5e-5 --max_input_length 256 > ./results/logs/${data_name}_standard.txt 2>&1
done

