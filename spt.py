from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import DataCollatorForCompletionOnlyLM, SFTTrainer

dataset = load_dataset(r"G:\data\IEMOCAP_train_data.json",split="train")

#加载预训练模型
model_name = "facebook/opt-350m"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example['text'])):
        text = f"### Utterance: {example['text'][i]}\n ### Emotion of the Utterance: {example['label'][i]}"
        output_texts.append(text)
    return output_texts

response_template = " ### Emotion of the Utterance:"

collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer = tokenizer)

trainer = SFTTrainer(
    model,
    train_dataset=dataset,
    formatting_func=formatting_prompts_func,
    data_collator=collator,
)

trainer.train()