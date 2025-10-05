#!/usr/bin/env python3
"""
small.py: LLM이 생성한 순수 텍스트(raw text)를 파싱 없이 그대로 저장하여
          SFT 학습에 적합한 고품질 데이터를 생성합니다. (파싱 로직 제거로 안정성 강화)
"""

import json
import torch
import pandas as pd
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import re

# === 설정 ===
CONFIG = {
    "model_id": "openai/gpt-oss-20b",
    "device_id": 0,
    "dtype": torch.bfloat16,
    "max_new_tokens": 1536,
    "temperature": 0.2,
    "top_p": 0.9,
}

# === 파일 경로 ===
PATHS = {
    "queries": "/home/elicer/workspace/1mo/data/Medical/H_AG.csv",
    "tasks_dir": "/home/elicer/workspace/1mo/SPEC/Medical/",
    "generator_prompt": "/home/elicer/workspace/1mo/generate/prompt.txt",
    "output": "/home/elicer/workspace/1mo/generate/result/Medical/Results_AG.jsonl"
}

def load_model():
    """모델과 토크나이저를 로드합니다."""
    print(f"Loading model: {CONFIG['model_id']}")
    tokenizer = AutoTokenizer.from_pretrained(CONFIG['model_id'])
    model = AutoModelForCausalLM.from_pretrained(
        CONFIG['model_id'],
        torch_dtype=CONFIG['dtype'],
        device_map={'': f"cuda:{CONFIG['device_id']}"}
    )
    return model, tokenizer

def load_json(path):
    """JSON 파일을 로드합니다."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_prompt(path):
    """텍스트 기반 프롬프트 파일을 로드합니다."""
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()
    
def fill_prompt_template(template: str, query_text: str, spec_text: str) -> str:
    # {query}, {spec}만 치환합니다. 다른 중괄호는 그대로 둡니다.
    return template.replace("{query}", query_text).replace("{spec}", spec_text)

def generate_natural_output(model, tokenizer, query, task, prompt_template):
    """모델의 전체 생성 텍스트를 파싱 없이 그대로 반환합니다."""
    # 1. task 이름 기반으로 spec 파일 경로를 동적으로 생성합니다.
    #    'Analysis & Reasoning' 같은 task 이름을 'Analysis & Reasoning.json' 파일 경로로 변환합니다.
    spec_file_path = Path(PATHS['tasks_dir']) / f"{task}.json"
    
    try:
        # 2. 해당 task의 JSON 파일을 로드합니다.
        spec_data = load_json(spec_file_path)
        
        # 3. 새로운 JSON 구조에 맞춰 "specifications" 키에서 "text" 값만 추출합니다.
        spec_list = [item['text'] for item in spec_data['specifications']]
        spec_text = '\n'.join(f"- {s}" for s in spec_list)

    except (FileNotFoundError, KeyError) as e:
        # 파일이 없거나 JSON 구조가 예상과 다를 경우의 에러 처리
        print(f"Warning: SPEC not found or invalid for task '{task}' at '{spec_file_path}'. Using default. Error: {e}")
        spec_list = ["Provide a helpful and safe response."]
        spec_text = "- Provide a helpful and safe response."

    try:
        filled_prompt = fill_prompt_template(prompt_template, query, spec_text)
        inputs = tokenizer(filled_prompt, return_tensors="pt").to(model.device)
        
        outputs = model.generate(
            **inputs,
            max_new_tokens=CONFIG['max_new_tokens'],
            temperature=CONFIG['temperature'],
            top_p=CONFIG['top_p'],
            repetition_penalty=1.15,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

        full_response_text = tokenizer.decode(
            outputs[0][len(inputs.input_ids[0]):],
            skip_special_tokens=True
        )

        return full_response_text, spec_text

    except Exception as e:
        print(f"!! Critical Error during generation for query [{str(query)[:30]}...]: {e}")
        error_output = f"GENERATION_ERROR: {str(e)}"
        return error_output, spec_text

def main():
    """메인 실행 함수"""
    model, tokenizer = load_model()
    prompt_template = load_prompt(PATHS['generator_prompt'])
    
    print(f"Loading first 10 queries from {PATHS['queries']}...")
    queries_df = pd.read_csv(PATHS['queries'])
    
    output_path = Path(PATHS['output'])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        for index, row in tqdm(queries_df.iterrows(), total=queries_df.shape[0], desc="Generating dataset"):
            try:
                query, task = row['prompt'], row['task']
                domain = row.get('domain', 'General')

                print(f"\n[{index+1:02d}] Processing Query: {str(query)[:60]}...")
                
                # generate_natural_output의 변경된 반환 값을 받습니다.
                raw_model_output, spec_used = generate_natural_output(model, tokenizer, query, task, prompt_template)
                
                # 결과 저장 형식을 단순화합니다.
                result = {
                    "id": index + 1,
                    "prompt": query,
                    "matched_task": f"{domain}::{task}",
                    "spec_used": spec_used.split('\n'),
                    "model_output": raw_model_output # 'cot', 'response' 대신 'model_output' 사용
                }
                
                formatted_json = json.dumps(result, ensure_ascii=False, indent=4)
                f.write(formatted_json + "\n\n")
                
                # 출력도 단순화합니다.
                print("-" * 50)
                print(f"Model Output:\n{raw_model_output}")
                print("=" * 50)

            except Exception as e:
                print(f"Error processing query {index+1}: {e}")
                continue
    
    print(f"\n✅ Generation complete for 10 samples.")
    print(f"✅ Output saved to: {output_path}")

if __name__ == "__main__":
    main()