import torch 
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import torch 
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path

def generate_answer(question, model_path="../models/final/checkpoint-3", max_length=80):
    # Convert Windows path to POSIX (forward-slash) format
    model_path = Path(model_path).resolve().as_posix()

    tokenizer = AutoTokenizer.from_pretrained("../models/final", local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained("../models/final", local_files_only=True)
    model.eval()

    input_text = f"Q: {question} A:"
    inputs = tokenizer(input_text, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=max_length,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = generated_text.split("A:")[-1].strip()
    print("[DEBUG] Raw generated text:", generated_text)
    print("[DEBUG] Extracted answer:", answer)
    if not answer:
        return generated_text
    return answer

if __name__ == "__main__":
    print("Enter 'exit' to quit.")
    while True:
        question = input("\nEnter your programming question: ")
        if question.lower() == "exit":
            break
        answer = generate_answer(question)
        print(f"\nModel answer:\n{answer}")
