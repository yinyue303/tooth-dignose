import os
import torch
import gc
from transformers import Qwen2_5_VLForConditionalGeneration,AutoProcessor,AutoModel,AutoTokenizer
from qwen_vl_utils import process_vision_info
from peft import PeftModel
import pandas as pd
import torch.nn.functional as F
from nltk.tokenize import wordpunct_tokenize
from nltk.translate.bleu_score import sentence_bleu,SmoothingFunction
from nltk.translate.meteor_score import meteor_score

if __package__:
    from .diagnose_eval_utils import (
        calculate_grouped_metrics,
        format_metric_summary,
        parse_image_paths,
        prompt_for_language,
        validate_language_values,
        validate_required_columns,
    )
else:
    from diagnose_eval_utils import (
        calculate_grouped_metrics,
        format_metric_summary,
        parse_image_paths,
        prompt_for_language,
        validate_language_values,
        validate_required_columns,
    )

lora_adapter='rank16_lr5e-4_epoch3'
dataset='diagnose'
batch_size=2
prompt='2'
model_size='3'
model_dir='../model/'
max_new_tokens=4096
embedding_batch_size =1
embedding_max_length=8192


script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)

gemma_model_path =os.path.join(
    project_dir,
    'model',
    'gemma-2b',
)
llama_model_path=os.path.join(
    project_dir,
    'model',
    'Llama-3.2-1B',
)
base_model_path = os.path.join(
    project_dir,
    'model',
    f'Qwen2.5-VL-{model_size}B-Instruct'
)
model=Qwen2_5_VLForConditionalGeneration.from_pretrained(
    base_model_path,
    dtype="auto",
    device_map="auto",
    local_files_only=True
)
processor=AutoProcessor.from_pretrained(base_model_path,local_files_only=True)
processor.tokenizer.padding_side='left'
if(lora_adapter):
    lora_weights_path=f'saves/pretrain-cls-SFT_{model_size}B_{lora_adapter}/'
    model=PeftModel.from_pretrained(model,lora_weights_path)
lora=[]
for name,param in model.named_parameters():
    if'lora' in name:lora.append(name)
print('LoRA layers:',len(lora))

def chat(messages):
    texts=[processor.apply_chat_template(msg,tokenize=False,add_generation_prompt=True) for msg in messages]
    image_inputs,video_inputs=process_vision_info(messages)
    inputs=processor(text=texts,images=image_inputs,videos=video_inputs,padding=True,return_tensors='pt')
    inputs=inputs.to('cuda')
    generated_ids=model.generate(**inputs,max_new_tokens=max_new_tokens)
    generated_ids_trimed=[out_ids[len(in_ids):] for in_ids,out_ids in zip(inputs.input_ids,generated_ids)]
    output_texts=processor.batch_decode(generated_ids_trimed,skip_special_tokens=True,clean_up_tokenization_spaces=False)
    return output_texts
print('Base Model:',base_model_path)
print('LORA:',lora_weights_path if lora_adapter else'NA')
bleu_smoothing = SmoothingFunction().method1


def score_bleu(reference_tokens, prediction_tokens):
    return sentence_bleu(
        [reference_tokens],
        prediction_tokens,
        weights=(0.25, 0.25, 0.25, 0.25),
        smoothing_function=bleu_smoothing,
    )


def score_meteor(reference_tokens, prediction_tokens):
    return meteor_score([reference_tokens], prediction_tokens)


def encode_reports(model_path, texts):
    """
    使用因果语言模型最后一层隐藏状态生成报告嵌入。

    池化方法：
    1. 排除 padding token；
    2. 对有效 token 做平均池化；
    3. 对句向量进行 L2 归一化。
    """
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    embedding_model = AutoModel.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map="auto",
        local_files_only=True,
    )
    embedding_model.eval()

    model_device = next(embedding_model.parameters()).device
    all_embeddings = []

    for batch_start in range(
        0,
        len(texts),
        embedding_batch_size,
    ):
        batch_texts = texts[
            batch_start:batch_start + embedding_batch_size
        ]

        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=embedding_max_length,
            return_tensors="pt",
        )

        inputs = {
            name: tensor.to(model_device)
            for name, tensor in inputs.items()
        }

        with torch.inference_mode():
            outputs = embedding_model(
                **inputs,
                return_dict=True,
            )

        hidden_states = outputs.last_hidden_state

        attention_mask = inputs["attention_mask"].unsqueeze(-1)
        attention_mask = attention_mask.to(hidden_states.dtype)

        embeddings = (
            hidden_states * attention_mask
        ).sum(dim=1)

        embeddings = embeddings / attention_mask.sum(
            dim=1
        ).clamp(min=1e-9)

        embeddings = F.normalize(
            embeddings.float(),
            p=2,
            dim=1,
        )

        all_embeddings.append(embeddings.cpu())

        del inputs
        del outputs
        del hidden_states
        del embeddings

    all_embeddings = torch.cat(all_embeddings, dim=0)

    del embedding_model
    del tokenizer

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return all_embeddings
def calculate_model_cosine(
    model_path,
    references,
    predictions,
):
    """
    使用一个模型计算每个 reference/prediction 对的余弦相似度。
    """
    all_texts = references + predictions

    all_embeddings = encode_reports(
        model_path,
        all_texts,
    )

    sample_count = len(references)

    reference_embeddings = all_embeddings[:sample_count]
    prediction_embeddings = all_embeddings[sample_count:]

    # encode_reports 已经做过 L2 归一化，
    # 因此向量点积就等于余弦相似度。
    cosine_scores = (
        reference_embeddings * prediction_embeddings
    ).sum(dim=1)

    return cosine_scores.numpy()



df_file=os.path.join(
    project_dir,
    'dataset',
    'excel',
    'test-diagnostic.xlsx'
)
result_file = os.path.join(
    project_dir,
    'results',
    f'{dataset}_{model_size}B_L{lora_adapter}_P{prompt}_HPC.txt'
)
df=pd.read_excel(df_file)
validate_required_columns(df.columns)
df['language'] = validate_language_values(df['language'].tolist())
df['q'] = df['q'].fillna('')
df['a'] = df['a'].fillna('')
df['images']=df['images'].apply(parse_image_paths)

sys_prompt_zh = None
sys_prompt_en = None
if(prompt):
    prompt_file_zh=os.path.join(
        project_dir,
        'prompt',
        f'{dataset}--{prompt}.txt'
    )
    prompt_file_en=os.path.join(
        project_dir,
        'prompt',
        f'{dataset}--{prompt}-en.txt'
    )
    with open(prompt_file_zh,'r',encoding='utf8')as f:
        sys_prompt_zh=f.read()
    with open(prompt_file_en,'r',encoding='utf8')as f:
        sys_prompt_en=f.read()
print('DF File:',df_file)
print('Result File',result_file)

ds=[]
for batch_start in range(0,len(df),batch_size):
    batch=df[batch_start:batch_start+batch_size]
    batch_messages=[]
    batch_rows=[]

    for i,row in batch.iterrows():
        row=row.to_dict()
        content=[]
        images=row['images']
        for image in images:
            image_path=os.path.join(
                project_dir,
                'dataset',
                str(image).lstrip('/\\')
            )
            img_content={'type':'image','image':image_path}
            content.append(img_content)
        content.append({'type': 'text', 'text': str(row['q'])})
        messages = [{"role": "user", "content": content}]

        if(prompt):
            sys_prompt = prompt_for_language(
                row['language'],
                sys_prompt_zh,
                sys_prompt_en,
            )
            messages.insert(0,{'role':'system','content':sys_prompt})
        batch_messages.append(messages)
        batch_rows.append(row)
    responses=chat(batch_messages)

    for idx,row in enumerate(batch_rows):
        new_row={
            'id':row['id'],
            'true':row['a'],
            'pred':responses[idx],
            'language':row['language'],
        }
        ds.append(new_row)
        for k,v in new_row.items():
            if k != 'language':
                print(f'{k},{v}')
    pd.DataFrame(ds)[['id','true','pred']].to_csv(
        result_file,
        sep='\t',
        index=None,
    )

# ==============================
# 释放报告生成模型占用的显存
# ==============================
del model
del processor

gc.collect()

if torch.cuda.is_available():
    torch.cuda.empty_cache()

# ==============================
# 整理生成结果
# ==============================
result_df = pd.DataFrame(ds)

references = result_df["true"].astype(str).tolist()
predictions = result_df["pred"].astype(str).tolist()
languages = result_df["language"].tolist()

# ==============================
# Gemma-2B 余弦相似度
# ==============================
print("正在使用 Gemma-2B 计算报告嵌入……")

gemma_cosine_scores = calculate_model_cosine(
    gemma_model_path,
    references,
    predictions,
)

# Gemma 会在 calculate_model_cosine 返回前释放，
# 然后再加载 Llama，避免同时占用显存。

# # ==============================
# # Llama-3.2-1B 余弦相似度
# # ==============================
# print("正在使用 Llama-3.2-1B 计算报告嵌入……")

# llama_cosine_scores = calculate_model_cosine(
#     llama_model_path,
#     references,
#     predictions,
# )

metrics = calculate_grouped_metrics(
    references,
    predictions,
    languages,
    gemma_cosine_scores,
    score_bleu,
    score_meteor,
    wordpunct_tokenize,
)

result_df[['id','true','pred']].to_csv(
    result_file,
    sep="\t",
    index=False,
)

print("\n最终评估结果：")
print(format_metric_summary(metrics))
print("逐样本结果：", result_file)
print("DONE!")
