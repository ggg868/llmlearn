import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import json
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import DataCollatorForCompletionOnlyLM, SFTTrainer

#dataset = load_dataset('json', data_files="./IEMOCAP_train_data.json", split='train')
#print(dataset[0])
#dataset = dataset.filter(lambda x: x.get('speaker') is not None)

#加载并扁平化数据集
with open("./IEMOCAP_train_data.json", "r") as f:
    raw_data = json.load(f)

# raw_data是一个列表，每个元素是一个对话的列表
all_utterances = []
for conversation in raw_data:
    all_utterances.extend(conversation)

dataset = Dataset.from_list(all_utterances)
#dataset = dataset.filter(lambda x: x.get('label') is not None)

#加载预训练模型
model_name = "facebook/opt-350m"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)


def formatting_prompts_func(example):
    return (
        f"### Speaker: {example['speaker']}\n"
        f"### Utterance: {example['text']}\n"
        f"### Emotion of the Utterance: {example['label']}"
    )

response_template = "### Emotion of the Utterance:"

collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

# 检查token的labels值
tokenized = tokenizer(
    formatting_prompts_func(dataset[0]),
    return_tensors="pt"
)
collated = collator([{
    "input_ids": tokenized["input_ids"][0],
    "attention_mask": tokenized["attention_mask"][0]
}])

print("\n样本标签张量:")
print(collated["labels"][0])
print(formatting_prompts_func(dataset[0]))

trainer = SFTTrainer(
    model,
    train_dataset=dataset,
    formatting_func=formatting_prompts_func,
    data_collator=collator,
)

trainer.train()