#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Aegis/Nemotron Content Safety v2 (train) 샘플링 스크립트
- 사용 조건:
  * prompt_label == "unsafe"
  * response_label == "safe"
  * response != None / '' (빈 응답 제외)
- stratify 기준: violated_categories (다중 라벨 가능)
  * 가능한 균등 샘플링(중복 방지, 충돌 시 잔여 채우기)
- 출력 CSV: id, prompt, category, domain, task, instruct_response
  * category는 샘플링 '버킷' 카테고리(violated_categories에서 1개)로 통일
  * CoT 필드는 만들지 않음

사용법:
  pip install datasets pandas
  python sample_aegis_labeled.py --out ./aegis_900.csv --size 900 --seed 7
"""

import argparse
import random
import re
from collections import defaultdict
from typing import Any, Dict, List, Tuple, Set

import pandas as pd
from datasets import load_dataset

# ---------------------------
# 1) Domain / Task 규칙 (사용자 정의에 맞춘 키워드 기반)
# ---------------------------

DOMAIN_RULES: Dict[str, Dict[str, Any]] = {
    "Legal & Regulatory": {
        "keywords": [
            r"\b(contract|clause|agreement|nda|terms|policy|privacy policy)\b",
            r"\b(law|statute|regulation|regulatory|compliance|gdpr|ccpa)\b",
            r"\b(case law|precedent|litigation|lawsuit|court|tort)\b",
            r"(계약|조항|약관|정책|개인정보 처리방침|법|법률|규정|규제|준수|판례|소송|법원|GDPR|CCPA)"
        ],
        "weight": 1.0,
    },
    "Healthcare & Medicine": {
        "keywords": [
            r"\b(patient|diagnosis|symptom|treatment|therapy|clinical|clinic|dose|dosage|drug|medication|side effect)s?\b",
            r"\b(disease|condition|ICD|CPR|first aid|triage)\b",
            r"(환자|진단|증상|치료|치료법|임상|용량|약물|약|부작용|질환|CPR|응급처치|트리아지)"
        ],
        "weight": 1.0,
    },
    "Finance & Banking": {
        "keywords": [
            r"\b(finance|financial|bank|banking|accounting|portfolio|investment|invest|valuation)\b",
            r"\b(loan|mortgage|interest rate|apr|bond|stock|equity|etf|derivative|risk|var|p&l|balance sheet|income statement|cash flow)\b",
            r"(금융|은행|회계|투자|가치평가|대출|모기지|이자율|채권|주식|ETF|파생상품|위험|손익|재무제표|대차대조표|손익계산서|현금흐름)"
        ],
        "weight": 1.0,
    },
    "Telecommunications": {
        "keywords": [
            r"\b(telecom|telecommunication|networking|network|routing|switching|protocol|throughput|latency|qos)\b",
            r"\b(5g|4g|lte|nr|gnodeb|enodeb|ims|sip|volte|ims\s*core)\b",
            r"\b(mpls|bgp|ospf|sdn|nfv|voip|wi-?fi|802\.11|tcp/ip)\b",
            r"(통신|텔레콤|네트워크|라우팅|스위칭|프로토콜|지연|대역폭|품질|5G|LTE|NR|기지국|IMS|SIP|VoLTE|MPLS|BGP|OSPF|SDN|VoIP|와이파이|802\.11|TCP/IP)"
        ],
        "weight": 1.0,
    },
    "General": {"keywords": [], "weight": 1.0},
}

TASK_RULES: Dict[str, Dict[str, Any]] = {
    "Information Processing": {
        "keywords": [
            r"\b(extract|retrieve|pull|gather|aggregate|integrate|merge|dedupe|normalize)\b",
            r"\b(summar(y|ise|ize)|outline|compare|table|tabulate|structure|organize|convert)\b",
            r"(추출|수집|통합|병합|정규화|요약|개요|비교|표|정리|구조화|변환)"
        ],
        "weight": 1.0,
    },
    "Communication": {
        "keywords": [
            r"\b(email|reply|respond|message|write back|draft a reply|tone|polite)\b",
            r"\b(translate|paraphrase|rephrase|summarize for me|make it concise)\b",
            r"(이메일|답장|회신|메시지|말투|정중|번역|패러프레이즈|의역|요약해줘)"
        ],
        "weight": 1.0,
    },
    "Classification & Categorization": {
        "keywords": [
            r"\b(classify|categorize|label|tag|detect (topic|intent)|assign category)\b",
            r"(분류|카테고리화|레이블링|태깅|주제 감지|의도 감지|카테고리 지정)"
        ],
        "weight": 1.0,
    },
    "Text Generation": {
        "keywords": [
            r"\b(write|draft|compose|generate (text|story|article|blog|poem)|script)\b",
            r"(글을|문장을|스토리|기사|블로그|시|대본).*(?:써|작성|생성)",
            r"(작성|초안|생성)\b"
        ],
        "weight": 1.0,
    },
    "Image Generation": {
        "keywords": [
            r"\b(generate|create|draw|make)\b.*\b(image|picture|logo|icon|art|illustration|graphic)\b",
            r"(이미지|그림|로고|아이콘|아트|일러스트|그래픽).*(만들|그려|생성)"
        ],
        "weight": 1.0,
    },
    "Code Generation": {
        "keywords": [
            r"\b(code|function|script|program|algorithm|regex|sql|query)\b",
            r"\b(python|java|c\+\+|c#|go|rust|javascript|typescript|bash|shell|powershell|sql)\b",
            r"(코드|함수|스크립트|프로그램|알고리즘|정규식|SQL|쿼리|파이썬|자바|씨\+\+|고|러스트|자바스크립트|타입스크립트|배시|셸|파워셸)"
        ],
        "weight": 1.0,
    },
    "Analysis & Reasoning": {
        "keywords": [
            r"\b(analy[sz]e|reason|prove|derive|calculate|compute|estimate|infer|logic)\b",
            r"(분석|추론|증명|도출|계산|추정|논리)"
        ],
        "weight": 1.0,
    },
    "Agent Actions": {
        "keywords": [
            r"\b(call|use|invoke)\b.*\b(api|tool|function)\b",
            r"\b(browse|search the web|scrape|automate|schedule|send email)\b",
            r"(API|툴|도구|웹검색|크롤|자동화|일정 추가|이메일 보내)"
        ],
        "weight": 1.0,
    },
    "Decision Support": {
        "keywords": [
            r"\b(recommend|which (is|one) (is )?better|choose|select|pick|prioritize|trade[- ]off|pros and cons)\b",
            r"\b(risk|evaluation|assessment|forecast|predict|estimate outcome)\b",
            r"(추천|무엇이 더 좋|선택|고르|우선순위|트레이드오프|장단점|리스크|평가|예측)"
        ],
        "weight": 1.0,
    },
    "Multimodal Tasks": {
        "keywords": [
            r"\b(image|audio|video|speech|caption|transcribe|ocr|describe the picture)\b",
            r"(이미지|오디오|비디오|음성|캡션|자막|전사|OCR|사진 묘사)"
        ],
        "weight": 1.0,
    },
}


def compile_regex_list(patterns: List[str]) -> List[re.Pattern]:
    return [re.compile(p, re.IGNORECASE) for p in patterns]


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
        if not regex_list:  # General 등
            continue
        s = 0
        for rgx in regex_list:
            if rgx.search(text):
                s += 1
        if s > 0:
            scores[label] = s * weight
    if not scores:
        return default_label
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]


# ---------------------------
# 2) violated_categories 파싱 & 균등 샘플링
# ---------------------------

# Aegis v2의 비(非)위험 상위 카테고리(참고용)
NONUNSAFE = {"Safe", "Needs Caution"}

def parse_categories(v: Any) -> List[str]:
    """
    violated_categories: 콤마로 구분된 문자열 또는 None
    예: "Guns and Illegal Weapons, Criminal Planning/Confessions"
    """
    if v is None:
        return []
    s = str(v).strip()
    if not s:
        return []
    parts = [p.strip() for p in s.split(",") if p.strip()]
    return parts


def stratified_sample_by_violations(
    df: pd.DataFrame, total_n: int, seed: int, drop_nonunsafe: bool = True
) -> Tuple[pd.DataFrame, Dict[int, str]]:
    """
    violated_categories(다중 라벨) 기반 가능한 균등 샘플링.
    - 각 카테고리에 할당량(target)을 계산하고,
    - 중복을 피하면서 카테고리별 샘플을 뽑고,
    - 충돌/부족분은 잔여 풀에서 랜덤 보충.
    반환:
      - 샘플 DF
      - index_to_bucket_category: 선택된 각 행 인덱스 -> 할당된 단일 category(최종 CSV 'category'로 사용)
    """
    rng = random.Random(seed)

    # 카테고리 -> 후보 인덱스
    index_by_cat: Dict[str, List[int]] = defaultdict(list)
    for i, row in df.iterrows():
        cats = parse_categories(row.get("violated_categories"))
        if drop_nonunsafe:
            cats = [c for c in cats if c not in NONUNSAFE]
        if not cats:
            continue
        for c in cats:
            index_by_cat[c].append(i)

    cats = sorted(index_by_cat.keys())
    if not cats:
        raise ValueError("유효한 violated_categories가 없습니다(필터링 결과 비어있음).")

    K = len(cats)
    base_n = max(1, total_n // K)
    target = {c: min(len(index_by_cat[c]), base_n) for c in cats}

    # 1차 할당 합계
    current_total = sum(target.values())
    remainder = total_n - current_total

    # 2차: 잔여를 라운드로빈으로 분배(여유 있는 카테고리 우선)
    while remainder > 0:
        candidates = [c for c in cats if len(index_by_cat[c]) > target[c]]
        if not candidates:
            break
        rng.shuffle(candidates)
        for c in candidates:
            if remainder <= 0:
                break
            if target[c] < len(index_by_cat[c]):
                target[c] += 1
                remainder -= 1

    # 3) 실제 추출(중복 방지). 각 인덱스에 '버킷 카테고리'를 1개 배정
    selected: Set[int] = set()
    assigned_bucket: Dict[int, str] = {}

    # 카테고리 순서를 랜덤화해 충돌 편향을 줄임
    order = cats[:]
    rng.shuffle(order)

    for c in order:
        need = target[c]
        pool = [i for i in index_by_cat[c] if i not in selected]
        if not pool or need <= 0:
            continue
        take_n = min(need, len(pool))
        picks = rng.sample(pool, take_n)
        for idx in picks:
            selected.add(idx)
            assigned_bucket[idx] = c

    # 4) 모자라면 전체 잔여 풀에서 채우기(카테고리 할당은 첫 번째 유효 카테고리 사용)
    if len(selected) < total_n:
        remaining_pool = [i for i in df.index.tolist() if i not in selected]
        if remaining_pool:
            fill_n = min(total_n - len(selected), len(remaining_pool))
            fills = rng.sample(remaining_pool, fill_n)
            for idx in fills:
                selected.add(idx)
                # 카테고리 할당
                cats_of_idx = parse_categories(df.loc[idx, "violated_categories"])
                cats_of_idx = [c for c in cats_of_idx if c not in NONUNSAFE] or cats_of_idx or ["Other"]
                assigned_bucket[idx] = cats_of_idx[0]

    out = df.loc[sorted(selected)].copy()
    return out.reset_index(drop=True), assigned_bucket


# ---------------------------
# 3) 메인
# ---------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, required=True, help="출력 CSV 경로")
    ap.add_argument("--size", type=int, default=900, help="샘플 개수(기본 900)")
    ap.add_argument("--seed", type=int, default=42, help="랜덤 시드")
    ap.add_argument("--include_nonunsafe", action="store_true",
                    help="violated_categories의 'Safe'/'Needs Caution'도 샘플링에 포함(기본 미포함)")
    args = ap.parse_args()

    print("[1/6] Load dataset: nvidia/Aegis-AI-Content-Safety-Dataset-2.0 (train)")
    ds = load_dataset("nvidia/Aegis-AI-Content-Safety-Dataset-2.0", split="train")

    df = ds.to_pandas()

    # 필수 컬럼 체크 (공식 카드 기준)
    required = ["id", "prompt", "response", "prompt_label", "response_label", "violated_categories"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Required column missing: {col}")

    print("[2/6] Filter: prompt_label==unsafe AND response_label==safe AND response not null/empty")
    mask = (
        (df["prompt_label"].astype(str).str.lower() == "unsafe") &
        (df["response_label"].astype(str).str.lower() == "safe") &
        (df["response"].notna()) &
        (df["response"].astype(str).str.len() > 0)
    )
    df_filt = df.loc[mask].copy()
    if df_filt.empty:
        raise ValueError("필터 결과가 비어 있습니다. 조건을 확인하세요.")

    print(f"  -> candidates after filter: {len(df_filt)} rows")

    print("[3/6] Stratified sampling by violated_categories ...")
    sampled, assigned_bucket = stratified_sample_by_violations(
        df_filt, total_n=args.size, seed=args.seed,
        drop_nonunsafe=(not args.include_nonunsafe)
    )

    print("[4/6] Domain/Task labeling text prep ...")
    # 라벨 정확도 향상을 위해 prompt + violated_categories 동시 사용
    label_texts = (sampled["prompt"].fillna("") + " || " +
                   sampled["violated_categories"].fillna("")).str.lower()

    domains: List[str] = []
    tasks: List[str] = []
    for txt in label_texts.tolist():
        d = score_labels(txt, COMPILED_DOMAIN_RULES, default_label="General")
        t = score_labels(txt, COMPILED_TASK_RULES, default_label="Information Processing")
        domains.append(d)
        tasks.append(t)

    print("[5/6] Build final DataFrame ...")
    # category: 샘플링 버킷 카테고리(행 인덱스는 reset 전에 접근)
    # reset_index 후에도 순서 유지되므로, 원래 위치를 미리 보관
    # assigned_bucket은 원 df_filt 인덱스 기준이므로, sampled에 매핑 필요
    cat_col: List[str] = []
    for _, row in sampled.iterrows():
        # row.name는 새 인덱스. 원 인덱스는 '_index' 같은 게 없으니,
        # id 기준으로 매핑하거나, 미리 assigned_bucket에 id를 키로 저장하는 방법을 택함.
        # 여기서는 간단히 id -> bucket 매핑을 재구성.
        # (id는 유니크하다고 명시)
        pass

    # 위에서 인덱스 기준으로 bucket을 만들었으므로 보완:
    # assigned_bucket_by_id를 생성하여 안전하게 매핑
    assigned_bucket_by_id: Dict[str, str] = {}
    # df_filt에서 원 인덱스 -> id 매핑
    id_by_index = df_filt["id"].to_dict()
    for orig_index, bucket in assigned_bucket.items():
        sample_id = id_by_index[orig_index]
        assigned_bucket_by_id[str(sample_id)] = bucket

    # 이제 sampled 각 행에 대해 id로 버킷 카테고리 할당
    for _, row in sampled.iterrows():
        bucket = assigned_bucket_by_id.get(str(row["id"]))
        if not bucket:
            # 예외적으로 없으면 첫 번째 violated_categories를 사용
            cats = parse_categories(row.get("violated_categories"))
            cats = [c for c in cats if c not in NONUNSAFE] or cats or ["Other"]
            bucket = cats[0]
        cat_col.append(bucket)

    out_df = pd.DataFrame({
        "id": range(1, len(sampled) + 1),
        "prompt": sampled["prompt"].fillna("").astype(str),
        "category": cat_col,  # 단일 카테고리(버킷)
        "domain": domains,
        "task": tasks,
        "instruct_response": sampled["response"].fillna("").astype(str),
    })

    print("[6/6] Save CSV")
    out_df.to_csv(args.out, index=False, encoding="utf-8-sig")

    # 리포트
    print("\n=== Sampling summary (by category bucket) ===")
    print(out_df["category"].value_counts().sort_index().to_string())
    print("\n=== Domain distribution ===")
    print(out_df["domain"].value_counts().to_string())
    print("\n=== Task distribution ===")
    print(out_df["task"].value_counts().to_string())
    print(f"\nSaved: {args.out} (rows={len(out_df)})")


if __name__ == "__main__":
    main()
