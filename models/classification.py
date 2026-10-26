import torch,os
from transformers import Qwen2_5_VLForConditionalGeneration,AutoProcessor
from qwen_vl_utils import process_vision_info
from peft import PeftModel
import pandas as pd 
import numpy as np
from sklearn.metrics import classification_report
from result_metrics import append_metrics, calculate_classification_metrics
lora_adapter=''
dataset='cls'
batch_size=2
prompt='1'
model_size='7'
model_dir='../model/'
max_new_tokens=20
CLASS_NAMES = [
    "Malocclusion",
    "Caries",
    "Pulpitis",
    "Tooth Loss",
    "Gingivitis",
    "Tooth Structure Loss",
]
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)

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
# LoRA 可选加载
lora_weights_path = None
lora=[]
if lora_adapter:
    lora_weights_path = os.path.join(
        project_dir,
        "saves",
        lora_adapter,
    )

    adapter_config_path = os.path.join(
        lora_weights_path,
        "adapter_config.json",
    )

    if not os.path.isfile(adapter_config_path):
        raise FileNotFoundError(
            f"LoRA 目录无效或缺少 adapter_config.json："
            f"{lora_weights_path}"
        )

    model = PeftModel.from_pretrained(
        model,
        lora_weights_path,
        local_files_only=True,
    )

    lora_layers = [
        name
        for name, _ in model.named_parameters()
        if "lora" in name.lower()
    ]

    print("已加载 LoRA：", lora_weights_path)
    print("LoRA layers：", len(lora_layers))
else:
    print("未指定 LoRA，直接使用基础模型。")

model.eval()
for name,param in model.named_parameters():
    if'lora' in name:lora.append(name)
print('LoRA layers:',len(lora))

def chat(messages):
    texts = [
        processor.apply_chat_template(
            message,
            tokenize=False,
            add_generation_prompt=True,
        )
        for message in messages
    ]

    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=texts,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    tokenizer = processor.tokenizer
    eos_token_id = tokenizer.eos_token_id
    pad_token_id = tokenizer.pad_token_id

    # 将六个合法类别转换成 token 序列，并在结尾添加 EOS。
    allowed_sequences = [
        tokenizer.encode(
            class_name,
            add_special_tokens=False,
        ) + [eos_token_id]
        for class_name in CLASS_NAMES
    ]

    # 因为输入经过 padding，所以所有样本的提示词张量长度相同。
    prompt_length = inputs.input_ids.shape[1]

    def prefix_allowed_tokens_fn(batch_id, input_ids):
        generated_tokens = input_ids[prompt_length:].tolist()
        allowed_next_tokens = []

        for sequence in allowed_sequences:
            generated_length = len(generated_tokens)

            if (
                generated_length < len(sequence)
                and sequence[:generated_length] == generated_tokens
            ):
                allowed_next_tokens.append(
                    sequence[generated_length]
                )

        if not allowed_next_tokens:
            return [eos_token_id]

        return sorted(set(allowed_next_tokens))

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max(
                len(sequence)
                for sequence in allowed_sequences
            ),
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
        )

    generated_ids_trimmed = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(
            inputs.input_ids,
            generated_ids,
        )
    ]

    output_texts = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    output_texts = [
        output_text.strip()
        for output_text in output_texts
    ]

    # 防止异常输出悄悄进入评估文件。
    for output_text in output_texts:
        if output_text not in CLASS_NAMES:
            raise RuntimeError(
                f"模型产生了非法分类结果：{output_text!r}"
            )

    return output_texts
print('Base Model:',base_model_path)
print('LORA:',lora_weights_path if lora_adapter else'NA')

df_file=os.path.join(
    project_dir,
    'dataset',
    'excel',
    'test-cls.xlsx'
)
result_file=result_file = os.path.join(
    project_dir,
    'results',
    f'{dataset}_{model_size}B_L{lora_adapter}_P{prompt}_HPC.txt'
)
df=pd.read_excel(df_file).fillna('')

df['images']=df['images'].apply(lambda x:eval(x))
if(prompt):
    prompt_file=os.path.join(
        project_dir,
        'prompt',
        f'{dataset}--{prompt}.txt'
    )
    with open(prompt_file,'r',encoding='utf8')as f: sys_prompt=f.read()
print('DF File:',df_file)
print('Result File',result_file)
print(sys_prompt)

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
        content.append({'type': 'text', 'text': row['q']})
        messages = [{"role": "user", "content": content}]

        if(prompt): messages.insert(0,{'role':'system','content':sys_prompt})
        batch_messages.append(messages)
        batch_rows.append(row)
    responses=chat(batch_messages)

    for idx,row in enumerate(batch_rows):
        new_row={'id':row['id'],'true':row['a'],'pred':responses[idx]}
        ds.append(new_row)
        for k,v in new_row.items():print(f'{k},{v}')
    pd.DataFrame(ds).to_csv(result_file,sep='\t',index=None)
result_df=pd.DataFrame(ds)
y_true=result_df['true']
y_pred=result_df['pred'].apply(
    lambda x:x.strip() if isinstance(x,str) else x
    )
metrics=calculate_classification_metrics(y_true,y_pred)
for metric_name,metric_value in metrics.items():
    print(f'{metric_name}:',metric_value)
append_metrics(result_file,metrics)

print(classification_report(
    y_true,
    y_pred,
    zero_division=0
))
confusion = pd.crosstab(
    result_df["true"],
    result_df["pred"],
    rownames=["真实类别"],
    colnames=["预测类别"],
    margins=True,
)

print(confusion)

normalized_confusion = pd.crosstab(
    result_df["true"],
    result_df["pred"],
    rownames=["真实类别"],
    colnames=["预测类别"],
    normalize="index",
)

print(normalized_confusion.round(3))
del model
print('DONE!')







