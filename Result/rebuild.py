import json
import re

def process_multiline_json(input_path, output_path):
    """
    여러 줄로 구성된 JSON 객체들이 포함된 파일을 읽어 처리합니다.
    'spec_used'를 제거하고 'model_output'을 'cot'과 'response'로 분리하여
    표준 JSONL 형식으로 저장합니다.

    Args:
        input_path (str): 원본 파일 경로.
        output_path (str): 결과를 저장할 새로운 JSONL 파일 경로.
    """
    try:
        # 1. 파일 전체 내용을 하나의 문자열로 읽어옵니다.
        with open(input_path, 'r', encoding='utf-8') as f:
            file_content = f.read()

        # 2. 정규표현식을 사용하여 '{...}' 패턴의 모든 JSON 객체 문자열을 찾습니다.
        # re.DOTALL 옵션은 '.'이 줄바꿈 문자도 포함하도록 하여 여러 줄에 걸친 객체를 찾게 해줍니다.
        json_object_strings = re.findall(r'\{.*?\}', file_content, re.DOTALL)
        
        if not json_object_strings:
            print("Warning: 파일에서 유효한 JSON 객체를 찾지 못했습니다.")
            return

        processed_records = []
        for json_str in json_object_strings:
            try:
                # 3. 찾은 각 문자열을 JSON 객체로 파싱합니다.
                data = json.loads(json_str)

                # 4. 요청하신 대로 데이터를 변환합니다.
                # 'spec_used'와 'model_output' 제외
                new_data = {k: v for k, v in data.items() if k not in ['spec_used', 'model_output']}
                
                # 'model_output'을 'cot'과 'response'로 분리
                model_output = data.get('model_output', '')
                
                if 'assistantfinal' in model_output:
                    parts = model_output.split('assistantfinal', 1)
                    cot = parts[0].strip()
                    response = parts[1].strip()
                else:
                    cot = model_output.strip()
                    response = ""
                
                new_data['cot'] = cot
                new_data['response'] = response
                
                processed_records.append(new_data)

            except json.JSONDecodeError:
                print(f"Warning: JSON으로 변환하지 못한 블록을 건너뜁니다: {json_str[:100]}...")
                continue
        
        # 5. 변환된 데이터를 표준 JSONL 형식(한 줄에 한 객체)으로 저장합니다.
        with open(output_path, 'w', encoding='utf-8') as outfile:
            for record in processed_records:
                outfile.write(json.dumps(record, ensure_ascii=False) + '\n')
        
        print(f"✅ 파일 처리가 완료되었습니다. 결과가 '{output_path}' 파일에 저장되었습니다.")

    except FileNotFoundError:
        print(f"오류: '{input_path}' 파일을 찾을 수 없습니다. 파일 경로를 확인해주세요.")
    except Exception as e:
        print(f"예상치 못한 오류가 발생했습니다: {e}")

if __name__ == '__main__':
    # 입력 파일과 출력 파일 경로를 설정합니다.
    # 사용하시는 파일 이름으로 변경하세요.
    input_file1 = '/home/elicer/workspace/1mo/generate/result/Results_JB.jsonl' 
    output_file1 = '/home/elicer/workspace/1mo/SFT_dataset/General/JailBreak_General.jsonl'
    
    input_file2 = '/home/elicer/workspace/1mo/generate/result/Results_AG.jsonl' 
    output_file2 = '/home/elicer/workspace/1mo/SFT_dataset/General/Aegis_General.jsonl'
    
    input_file3 = '/home/elicer/workspace/1mo/generate/result/Results_FR.jsonl' 
    output_file3 = '/home/elicer/workspace/1mo/SFT_dataset/General/FalseReject_General.jsonl'

    # 함수를 실행합니다.
    process_multiline_json(input_file1, output_file1)
    process_multiline_json(input_file2, output_file2)
    process_multiline_json(input_file3, output_file3)