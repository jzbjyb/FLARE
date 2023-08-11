#!/usr/bin/env bash
set -e

debug=false

source keys.sh
num_keys=${#keys[@]}

dataset=$1
config_file=$2

config_filename=$(basename -- "${config_file}")
config_filename="${config_filename%.*}"

debug_batch_size=1
batch_size=8
model=text-davinci-003
temperature=0

output=output/${dataset}/${model}/${config_filename}.jsonl
echo 'output to:' $output

prompt_type=""
if [[ ${dataset} == '2wikihop' ]]; then
    input="--input data/2wikimultihopqa"
    engine=elasticsearch
    index_name=wikipedia_dpr
    fewshot=8
    max_num_examples=500
    max_generation_len=256
elif [[ ${dataset} == 'strategyqa' ]]; then
    input="--input data/strategyqa/dev_beir"
    engine=elasticsearch
    index_name=wikipedia_dpr
    fewshot=6
    max_num_examples=229
    max_generation_len=256
elif [[ ${dataset} == 'asqa' ]]; then
    prompt_type="--prompt_type general_hint_in_output"
    input="--input data/asqa/ASQA.json"
    engine=elasticsearch
    index_name=wikipedia_dpr
    fewshot=8
    max_num_examples=500
    max_generation_len=256
elif [[ ${dataset} == 'asqa_hint' ]]; then
    prompt_type="--prompt_type general_hint_in_input"
    dataset=asqa
    input="--input data/asqa/ASQA.json"
    engine=elasticsearch
    index_name=wikipedia_dpr
    fewshot=8
    max_num_examples=500
    max_generation_len=256
elif [[ ${dataset} == 'wikiasp' ]]; then
    input="--input data/wikiasp"
    engine=bing
    index_name=wikiasp
    fewshot=4
    max_num_examples=500
    max_generation_len=512
else
    exit
fi

# query api
if [[ ${debug} == "true" ]]; then
    python -m src.openai_api \
        --model ${model} \
        --dataset ${dataset} ${input} ${prompt_type} \
        --config_file ${config_file} \
        --fewshot ${fewshot} \
        --search_engine ${engine} \
        --index_name ${index_name} \
        --max_num_examples 100 \
        --max_generation_len ${max_generation_len} \
        --batch_size ${debug_batch_size} \
        --output test.jsonl \
        --num_shards 1 \
        --shard_id 0 \
        --openai_keys ${test_key} \
        --debug
    exit
fi

function join_by {
  local d=${1-} f=${2-}
  if shift 2; then
    printf %s "$f" "${@/#/$d}"
  fi
}

joined_keys=$(join_by " " "${keys[@]:0:${num_keys}}")

python -m src.openai_api \
    --model ${model} \
    --dataset ${dataset} ${input} ${prompt_type} \
    --config_file ${config_file} \
    --fewshot ${fewshot} \
    --search_engine ${engine} \
    --index_name ${index_name} \
    --max_num_examples ${max_num_examples} \
    --max_generation_len ${max_generation_len} \
    --temperature ${temperature} \
    --batch_size ${batch_size} \
    --output ${output} \
    --num_shards 1 \
    --shard_id 0 \
    --openai_keys ${joined_keys} \
