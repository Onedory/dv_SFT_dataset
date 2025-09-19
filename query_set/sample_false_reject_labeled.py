#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FalseReject 균등 샘플링 + 도메인/태스크 자동 라벨링
- Dataset: AmazonScience/FalseReject (HF Datasets)
- 샘플링: category_text 기준 균등 랜덤 샘플링으로 총 N개(기본 1500)
- 출력 CSV: id, prompt, category, domain, task, cot_response, instruct_response
- 도메인/태스크 라벨 규칙: 사용자가 제공한 정의에 맞춘 키워드/패턴 점수화 기반

사용법:
pip install datasets pandas
python sample_false_reject_labeled.py --out ./false_reject_1500.csv --size 1500 --seed 42

라이선스 유의:
- AmazonScience/FalseReject는 비상업(CC BY-NC 4.0) 성격입니다. 연구/비영리 목적과 출처 표기를 준수하세요.
"""

import argparse
import re
import random
from typing import Dict, List, Tuple, Any
import pandas as pd
from datasets import load_dataset


# ---------------------------
# 1) 규칙: Domain / Task 키워드 패턴
#   - prompt + category_text를 결합해 스코어링 후 최대 득점 라벨 선택
#   - 다국어 키워드(영/한) 혼합, word-boundary 우선 / 일부는 부분일치
# ---------------------------

DOMAIN_RULES: Dict[str, Dict[str, Any]] = {
    # 1. Legal & Regulatory
    "Legal & Regulatory": {
        "keywords": [
            r"\b(contract|clause|agreement|nda|terms|policy|privacy policy)\b",
            r"\b(law|statute|regulation|regulatory|compliance|gdpr|ccpa)\b",
            r"\b(case law|precedent|litigation|lawsuit|court|tort)\b",
            r"(계약|조항|약관|정책|개인정보 처리방침|법|법률|규정|규제|준수|판례|소송|법원|GDPR|CCPA)"
        ],
        "weight": 1.0,
    },
    # 2. Healthcare & Medicine
    "Healthcare & Medicine": {
        "keywords": [
            r"\b(patient|diagnosis|symptom|treatment|therapy|clinical|clinic|dose|dosage|drug|medication|side effect)s?\b",
            r"\b(disease|condition|ICD|CPR|first aid|triage)\b",
            r"(환자|진단|증상|치료|치료법|임상|용량|약물|약|부작용|질환|CPR|응급처치|트리아지)"
        ],
        "weight": 1.0,
    },
    # 3. Finance & Banking
    "Finance & Banking": {
        "keywords": [
            r"\b(finance|financial|bank|banking|accounting|portfolio|investment|invest|valuation)\b",
            r"\b(loan|mortgage|interest rate|apr|bond|stock|equity|etf|derivative|risk|var|p&l|balance sheet|income statement|cash flow)\b",
            r"(금융|은행|회계|투자|가치평가|대출|모기지|이자율|채권|주식|ETF|파생상품|위험|손익|재무제표|대차대조표|손익계산서|현금흐름)"
        ],
        "weight": 1.0,
    },
    # 4. Telecommunications
    "Telecommunications": {
        "keywords": [
            r"\b(telecom|telecommunication|networking|network|routing|switching|protocol|throughput|latency|qos)\b",
            r"\b(5g|4g|lte|nr|gnodeb|enodeb|ims|sip|volte|ims\s*core)\b",
            r"\b(mpls|bgp|ospf|sdn|nfv|voip|wi-?fi|802\.11|tcp/ip)\b",
            r"(통신|텔레콤|네트워크|라우팅|스위칭|프로토콜|지연|대역폭|품질|5G|LTE|NR|기지국|IMS|SIP|VoLTE|MPLS|BGP|OSPF|SDN|VoIP|와이파이|802\.11|TCP/IP)"
        ],
        "weight": 1.0,
    },
    # 5. General (기본값)
    "General": {"keywords": [], "weight": 1.0},
}

TASK_RULES: Dict[str, Dict[str, Any]] = {
    # 1. Information Processing
    "Information Processing": {
        "keywords": [
            r"\b(extract|retrieve|pull|gather|aggregate|integrate|merge|dedupe|normalize)\b",
            r"\b(summar(y|ise|ize)|outline|compare|table|tabulate|structure|organize|convert)\b",
            r"(추출|수집|통합|병합|정규화|요약|개요|비교|표|정리|구조화|변환)"
        ],
        "weight": 1.0,
    },
    # 2. Communication
    "Communication": {
        "keywords": [
            r"\b(email|reply|respond|message|write back|draft a reply|tone|polite)\b",
            r"\b(translate|paraphrase|rephrase|summarize for me|make it concise)\b",
            r"(이메일|답장|회신|메시지|말투|정중|번역|패러프레이즈|의역|요약해줘)"
        ],
        "weight": 1.0,
    },
    # 3. Classification & Categorization
    "Classification & Categorization": {
        "keywords": [
            r"\b(classify|categorize|label|tag|detect (topic|intent)|assign category)\b",
            r"(분류|카테고리화|레이블링|태깅|주제 감지|의도 감지|카테고리 지정)"
        ],
        "weight": 1.0,
    },
    # 4. Text Generation
    "Text Generation": {
        "keywords": [
            r"\b(write|draft|compose|generate (text|story|article|blog|poem)|script)\b",
            r"(글을|문장을|스토리|기사|블로그|시|대본).*(?:써|작성|생성)",
            r"(작성|초안|생성)\b"
        ],
        "weight": 1.0,
    },
    # 5. Image Generation
    "Image Generation": {
        "keywords": [
            r"\b(generate|create|draw|make)\b.*\b(image|picture|logo|icon|art|illustration|graphic)\b",
            r"(이미지|그림|로고|아이콘|아트|일러스트|그래픽).*(만들|그려|생성)"
        ],
        "weight": 1.0,
    },
    # 6. Code Generation
    "Code Generation": {
        "keywords": [
            r"\b(code|function|script|program|algorithm|regex|sql|query)\b",
            r"\b(python|java|c\+\+|c#|go|rust|javascript|typescript|bash|shell|powershell|sql)\b",
            r"(코드|함수|스크립트|프로그램|알고리즘|정규식|SQL|쿼리|파이썬|자바|씨\+\+|고|러스트|자바스크립트|타입스크립트|배시|셸|파워셸)"
        ],
        "weight": 1.0,
    },
    # 7. Analysis & Reasoning
    "Analysis & Reasoning": {
        "keywords": [
            r"\b(analy[sz]e|reason|prove|derive|calculate|compute|estimate|infer|logic)\b",
            r"(분석|추론|증명|도출|계산|추정|논리)"
        ],
        "weight": 1.0,
    },
    # 8. Agent Actions
    "Agent Actions": {
        "keywords": [
            r"\b(call|use|invoke)\b.*\b(api|tool|function)\b",
            r"\b(browse|search the web|scrape|automate|schedule|send email)\b",
            r"(API|툴|도구|웹검색|크롤|자동화|일정 추가|이메일 보내)"
        ],
        "weight": 1.0,
    },
    # 9. Decision Support
    "Decision Support": {
        "keywords": [
            r"\b(recommend|which (is|one) (is )?better|choose|select|pick|prioritize|trade[- ]off|pros and cons)\b",
            r"\b(risk|evaluation|assessment|forecast|predict|estimate outcome)\b",
            r"(추천|무엇이 더 좋|선택|고르|우선순위|트레이드오프|장단점|리스크|평가|예측)"
        ],
        "weight": 1.0,
    },
    # 10. Multimodal Tasks
    "Multimodal Tasks": {
        "keywords": [
            r"\b(image|audio|video|speech|caption|transcribe|ocr|describe the picture)\b",
            r"(이미지|오디오|비디오|음성|캡션|자막|전사|OCR|사진 묘사)"
        ],
        "weight": 1.0,
    },
}


def compile_regex_list(patterns: List[str]) -> List[re.Pattern]:
    regs: List[re.Pattern] = []
    for p in patterns:
        regs.append(re.compile(p, re.IGNORECASE))
    return regs


COMPILED_DOMAIN_RULES = {
    name: [compile_regex_list(cfg["keywords"]), cfg["weight"]]
    for name, cfg in DOMAIN_RULES.items()
}
COMPILED_TASK_RULES = {
    name: [compile_regex_list(cfg["keywords"]), cfg["weight"]]
    for name, cfg in TASK_RULES.items()
}


def score_labels(text: str, compiled_rules: Dict[str, Any], default_label: str) -> str:
    scores: Dict[str, float] = {}
    for label, (regex_list, weight) in compiled_rules.items():
        if not regex_list:  # e.g., General
            continue
        s = 0
        for rgx in regex_list:
            if rgx.search(text):
                s += 1
        if s > 0:
            scores[label] = s * weight

    if not scores:
        return default_label

    # 1) 가장 높은 점수, 2) 동률이면 알파벳/한글 순으로 안정 선택
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]


# ---------------------------
# 2) 카테고리 균등 샘플링
# ---------------------------

def stratified_sample_by_category(df: pd.DataFrame, total_n: int, seed: int) -> pd.DataFrame:
    rng = random.Random(seed)
    counts = df["category_text"].value_counts().to_dict()
    cats = sorted(counts.keys())
    K = len(cats)

    base_n = total_n // K
    target = {c: min(counts[c], base_n) for c in cats}

    current_total = sum(target.values())
    remainder = total_n - current_total

    # 라운드로빈으로 잔여 분배
    while remainder > 0:
        candidates = [c for c in cats if counts[c] > target[c]]
        if not candidates:
            break
        rng.shuffle(candidates)
        for c in candidates:
            if remainder <= 0:
                break
            if target[c] < counts[c]:
                target[c] += 1
                remainder -= 1

    # 실제 샘플링
    frames = []
    for i, c in enumerate(cats):
        n = target[c]
        sub = df[df["category_text"] == c]
        take = sub.sample(n=n, random_state=seed + i, replace=False)
        frames.append(take)

    out = pd.concat(frames, ignore_index=True)
    if len(out) > total_n:  # 안전장치
        out = out.sample(n=total_n, random_state=seed)
    return out.reset_index(drop=True)


# ---------------------------
# 3) CoT 필드 정규화
# ---------------------------

def normalize_cot(x: Any) -> str:
    """
    FalseReject의 cot_response가 dict인 경우가 있어 평탄화합니다.
    - 예상 키: reasoning_content, solution
    """
    if isinstance(x, dict):
        rc = x.get("reasoning_content", "")
        sol = x.get("solution", "")
        joined = "\n\n".join([s for s in [rc, sol] if s])
        return joined.strip()
    return (x or "").strip()


# ---------------------------
# 4) 메인
# ---------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, required=True, help="출력 CSV 경로")
    parser.add_argument("--size", type=int, default=1500, help="샘플 개수(기본 1500)")
    parser.add_argument("--seed", type=int, default=42, help="랜덤 시드")
    args = parser.parse_args()

    print("[1/6] Loading dataset: AmazonScience/FalseReject ...")
    ds = load_dataset("AmazonScience/FalseReject", split="train")
    df = ds.to_pandas()

    required = ["prompt", "category_text", "instruct_response", "cot_response"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column in dataset: {col}")

    print("[2/6] Normalizing CoT field ...")
    df["cot_response_str"] = df["cot_response"].apply(normalize_cot)

    print("[3/6] Stratified sampling by category_text ...")
    sampled = stratified_sample_by_category(df, total_n=args.size, seed=args.seed)

    print("[4/6] Preparing text for labeling ...")
    # 라벨링 신뢰도 향상을 위해 prompt + category_text 결합
    label_texts = (sampled["prompt"].fillna("") + " || " + sampled["category_text"].fillna("")).str.lower()

    print("[5/6] Assigning domain/task labels ...")
    domains: List[str] = []
    tasks: List[str] = []
    for txt in label_texts.tolist():
        domain = score_labels(txt, COMPILED_DOMAIN_RULES, default_label="General")
        task = score_labels(txt, COMPILED_TASK_RULES, default_label="Information Processing")
        domains.append(domain)
        tasks.append(task)

    print("[6/6] Building and saving CSV ...")
    out_df = pd.DataFrame({
        "id": range(1, len(sampled) + 1),
        "prompt": sampled["prompt"].fillna("").astype(str),
        "category": sampled["category_text"].fillna("").astype(str),
        "domain": domains,
        "task": tasks,
        "cot_response": sampled["cot_response_str"].fillna("").astype(str),
        "instruct_response": sampled["instruct_response"].fillna("").astype(str),
    })

    out_df.to_csv(args.out, index=False, encoding="utf-8-sig")

    # 간단 리포트
    print("\n=== Sampling summary (by category) ===")
    print(out_df["category"].value_counts().sort_index().to_string())
    print("\n=== Domain distribution ===")
    print(out_df["domain"].value_counts().to_string())
    print("\n=== Task distribution ===")
    print(out_df["task"].value_counts().to_string())
    print(f"\nSaved: {args.out} (rows={len(out_df)})")


if __name__ == "__main__":
    main()
