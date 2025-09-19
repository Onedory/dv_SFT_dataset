#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
JailBreakV-28k 균등 샘플링 + 도메인/태스크 자동 라벨링
- Dataset: JailbreakV-28K/JailBreakV-28k (HF Datasets)
- 서브셋: 기본 'JailBreakV_28K' (옵션 'RedTeam_2K')
- 샘플링: policy 기준 균등 랜덤 샘플링 총 N개
- 텍스트 공격만 사용 옵션: --text_only (기본 True; Logic/Template/Persuade만 허용)
- 출력 CSV: id, prompt, category, domain, task
- 주의: Jailbreak에는 답변/CoT가 없으므로 포함하지 않음

사용법:
  pip install datasets pandas
  python sample_jailbreak_labeled.py --out ./jailbreak_600.csv --size 600 --seed 42
  python sample_jailbreak_labeled.py --out ./jb_redteam_600.csv --subset RedTeam_2K
"""

import argparse
import random
import re
from typing import Dict, List, Any
import pandas as pd
from datasets import load_dataset


# ---------------------------
# 1) Domain / Task 규칙 (영어 키워드 기반)
# ---------------------------

DOMAIN_RULES: Dict[str, Dict[str, Any]] = {
    "Legal & Regulatory": {
        "keywords": [
            r"\b(contract|clause|agreement|nda|terms|policy|privacy policy)\b",
            r"\b(law|statute|regulation|regulatory|compliance|gdpr|ccpa)\b",
            r"\b(case law|precedent|litigation|lawsuit|court|tort)\b",
        ],
        "weight": 1.0,
    },
    "Healthcare & Medicine": {
        "keywords": [
            r"\b(patient|diagnosis|symptom|treatment|therapy|clinical|dose|drug|medication|side effect)s?\b",
            r"\b(disease|condition|ICD|CPR|first aid|triage)\b",
        ],
        "weight": 1.0,
    },
    "Finance & Banking": {
        "keywords": [
            r"\b(finance|financial|bank|banking|accounting|portfolio|investment|invest|valuation)\b",
            r"\b(loan|mortgage|interest rate|apr|bond|stock|equity|etf|derivative|risk|balance sheet|income statement|cash flow)\b",
        ],
        "weight": 1.0,
    },
    "Telecommunications": {
        "keywords": [
            r"\b(telecom|telecommunication|networking|routing|switching|protocol|throughput|latency|qos)\b",
            r"\b(5g|4g|lte|nr|gnodeb|enodeb|ims|sip|volte|mpls|bgp|ospf|sdn|voip|wi-?fi|802\.11|tcp/ip)\b",
        ],
        "weight": 1.0,
    },
    "General": {"keywords": [], "weight": 1.0},
}

TASK_RULES: Dict[str, Dict[str, Any]] = {
    "Information Processing": {
        "keywords": [
            r"\b(extract|retrieve|gather|aggregate|integrate|merge|normalize)\b",
            r"\b(summar(y|ise|ize)|outline|compare|table|tabulate|structure|organize|convert)\b",
        ],
        "weight": 1.0,
    },
    "Communication": {
        "keywords": [
            r"\b(email|reply|respond|message|tone|polite)\b",
            r"\b(translate|paraphrase|rephrase|summarize)\b",
        ],
        "weight": 1.0,
    },
    "Classification & Categorization": {
        "keywords": [
            r"\b(classify|categorize|label|tag|detect (topic|intent)|assign category)\b",
        ],
        "weight": 1.0,
    },
    "Text Generation": {
        "keywords": [
            r"\b(write|draft|compose|generate (text|story|article|blog|poem)|script)\b",
        ],
        "weight": 1.0,
    },
    "Image Generation": {
        "keywords": [
            r"\b(generate|create|draw|make)\b.*\b(image|picture|logo|icon|art|illustration|graphic)\b",
        ],
        "weight": 1.0,
    },
    "Code Generation": {
        "keywords": [
            r"\b(code|function|script|program|algorithm|regex|sql|query)\b",
            r"\b(python|java|c\+\+|c#|go|rust|javascript|typescript|bash|shell|powershell|sql)\b",
        ],
        "weight": 1.0,
    },
    "Analysis & Reasoning": {
        "keywords": [
            r"\b(analy[sz]e|reason|prove|derive|calculate|compute|estimate|infer|logic)\b",
        ],
        "weight": 1.0,
    },
    "Agent Actions": {
        "keywords": [
            r"\b(call|use|invoke)\b.*\b(api|tool|function)\b",
            r"\b(browse|search the web|scrape|automate|schedule|send email)\b",
        ],
        "weight": 1.0,
    },
    "Decision Support": {
        "keywords": [
            r"\b(recommend|which (is|one) (is )?better|choose|select|pick|prioritize|trade[- ]off|pros and cons)\b",
            r"\b(risk|evaluation|assessment|forecast|predict|estimate outcome)\b",
        ],
        "weight": 1.0,
    },
    "Multimodal Tasks": {
        "keywords": [
            r"\b(image|audio|video|speech|caption|transcribe|ocr|describe the picture)\b",
        ],
        "weight": 1.0,
    },
}


def _compile(patterns: List[str]) -> List[re.Pattern]:
    return [re.compile(p, re.IGNORECASE) for p in patterns]


COMPILED_DOMAIN = {k: (_compile(v["keywords"]), v["weight"]) for k, v in DOMAIN_RULES.items()}
COMPILED_TASK = {k: (_compile(v["keywords"]), v["weight"]) for k, v in TASK_RULES.items()}


def score_label(text: str, compiled_rules: Dict[str, Any], default_label: str) -> str:
    scores: Dict[str, float] = {}
    for label, (regs, w) in compiled_rules.items():
        if not regs:
            continue
        s = sum(1 for rgx in regs if rgx.search(text))
        if s:
            scores[label] = s * w
    if not scores:
        return default_label
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]


# ---------------------------
# 2) stratified sampling by "policy"
# ---------------------------

def stratified_by_policy(df: pd.DataFrame, total_n: int, seed: int) -> pd.DataFrame:
    rng = random.Random(seed)
    counts = df["policy"].value_counts().to_dict()
    policies = sorted(counts.keys())
    k = len(policies)
    base = total_n // k
    take = {p: min(counts[p], base) for p in policies}

    # remainder round-robin
    rem = total_n - sum(take.values())
    while rem > 0:
        cands = [p for p in policies if counts[p] > take[p]]
        if not cands:
            break
        rng.shuffle(cands)
        for p in cands:
            if rem <= 0:
                break
            if take[p] < counts[p]:
                take[p] += 1
                rem -= 1

    # sample per policy
    frames = []
    for i, p in enumerate(policies):
        n = take[p]
        sub = df[df["policy"] == p]
        if n > 0:
            frames.append(sub.sample(n=n, random_state=seed + i, replace=False))
    out = pd.concat(frames, ignore_index=True) if frames else df.head(0)
    if len(out) > total_n:
        out = out.sample(n=total_n, random_state=seed)
    return out.reset_index(drop=True)


# ---------------------------
# 3) Loading & Normalization
# ---------------------------

TEXT_FORMATS = {"Logic", "Template", "Persuade"}  # text-only LLM transfer
IMAGE_FORMATS = {"FigStep", "QueryRelevant"}      # image-based MLLM attacks

def load_and_normalize(subset: str, text_only: bool) -> pd.DataFrame:
    """
    subset: 'JailBreakV_28K' or 'RedTeam_2K'
    normalize to columns: prompt, policy, format (opt), image_path (opt)
    """
    if subset not in {"JailBreakV_28K", "RedTeam_2K"}:
        raise ValueError("--subset must be JailBreakV_28K or RedTeam_2K")

    ds = load_dataset("JailbreakV-28K/JailBreakV-28k", subset)
    df = ds[subset].to_pandas()

    # Normalize prompt/policy fields
    if subset == "JailBreakV_28K":
        if "jailbreak_query" not in df.columns or "policy" not in df.columns:
            raise ValueError("Unexpected schema for JailBreakV_28K.")
        norm = pd.DataFrame({
            "prompt": df["jailbreak_query"].astype(str),
            "policy": df["policy"].astype(str),
            "format": df.get("format", pd.Series([""] * len(df))).astype(str),
            "image_path": df.get("image_path", pd.Series([""] * len(df))).astype(str),
        })
        if text_only:
            norm = norm[norm["format"].isin(TEXT_FORMATS)].copy()
    else:  # RedTeam_2K
        if "question" not in df.columns or "policy" not in df.columns:
            raise ValueError("Unexpected schema for RedTeam_2K.")
        norm = pd.DataFrame({
            "prompt": df["question"].astype(str),
            "policy": df["policy"].astype(str),
            "format": pd.Series(["RedTeam"] * len(df)),
            "image_path": pd.Series([""] * len(df)),
        })
        # RedTeam_2K는 텍스트만

    # 기본 정리
    norm = norm[(norm["prompt"].str.len() > 0) & (norm["policy"].str.len() > 0)].copy()
    norm.reset_index(drop=True, inplace=True)
    return norm


# ---------------------------
# 4) Main
# ---------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, required=True, help="출력 CSV 경로")
    ap.add_argument("--size", type=int, default=600, help="샘플 개수(기본 600)")
    ap.add_argument("--seed", type=int, default=42, help="랜덤 시드")
    ap.add_argument("--subset", choices=["JailBreakV_28K", "RedTeam_2K"], default="JailBreakV_28K",
                    help="사용할 서브셋 선택 (기본 JailBreakV_28K)")
    ap.add_argument("--text_only", action="store_true", default=True,
                    help="텍스트 기반 공격만 사용(Logic/Template/Persuade). 해제하려면 --no-text_only 사용")
    ap.add_argument("--no-text_only", action="store_false", dest="text_only")
    args = ap.parse_args()

    print(f"[1/5] Loading subset: {args.subset} ...")
    df = load_and_normalize(args.subset, text_only=args.text_only)
    if df.empty:
        raise ValueError("No rows after filtering. Try --no-text_only or switch subset.")

    print(f"[2/5] Stratified sampling by 'policy' ...")
    sampled = stratified_by_policy(df, total_n=args.size, seed=args.seed)

    print(f"[3/5] Domain/Task labeling ...")
    # 라벨 정확도를 위해 prompt + policy 동시 사용
    texts = (sampled["prompt"].fillna("") + " || " + sampled["policy"].fillna("")).str.lower()
    domains, tasks = [], []
    for txt in texts.tolist():
        d = score_label(txt, COMPILED_DOMAIN, default_label="General")
        t = score_label(txt, COMPILED_TASK, default_label="Information Processing")
        domains.append(d)
        tasks.append(t)

    print(f"[4/5] Building final DataFrame ...")
    out_df = pd.DataFrame({
        "id": range(1, len(sampled) + 1),
        "prompt": sampled["prompt"].astype(str),
        "category": sampled["policy"].astype(str),  # category=policy
        "domain": domains,
        "task": tasks,
    })

    print(f"[5/5] Saving CSV -> {args.out}")
    out_df.to_csv(args.out, index=False, encoding="utf-8-sig")

    # 리포트
    print("\n=== Sampling summary (by policy) ===")
    print(out_df["category"].value_counts().sort_index().to_string())
    print("\n=== Domain distribution ===")
    print(out_df["domain"].value_counts().to_string())
    print("\n=== Task distribution ===")
    print(out_df["task"].value_counts().to_string())
    print(f"\nSaved: {args.out} (rows={len(out_df)})")


if __name__ == "__main__":
    main()
