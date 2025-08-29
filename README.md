# LLM4CobotTS
LLM4CobotTS is a framework for analyzing collaborative robot (cobot) time-series data using Large Language Models (LLMs). This project explores natural language–driven interpretation, anomaly detection, and task-specific insights from multi-sensor robotic data streams.


### - Developers 👩‍💻

- **Ye-Seul Park**  
  ✉️ [yeseuly777@gmail.com](mailto:yeseuly777@gmail.com)

- **Minseo Choi**  
  ✉️ [minseo000624@gmail.com](mailto:minseo000624@gmail.com)

---

## 🔧 Environment Variables

| Name | Default | Description |
|---|---:|---|
| `OPENAI_API_KEY` | — | OpenAI API Key (필수) |
| `OPENAI_MODEL` | `gpt-4o-mini` | LangChain `ChatOpenAI`에 사용할 모델 |
| `OPENAI_CHAT_MODEL` | `gpt-4o` | OpenAI Python SDK(저수준)에서 사용할 모델 |
| `OPENAI_TEMPERATURE` | `0` | (LangChain) 응답 창의성 |
| `OPENAI_TOP_P` | `1.0` | (LangChain) nucleus sampling |
| `OPENAI_MAX_TOKENS` | `2048` | (LangChain) 최대 토큰 수 |

---
## 📦 modules/preprocessor.py — Time-series Preprocessing

### Public API
| Function | Purpose | Key Params | Returns | Notes |
|---|---|---|---|---|
| `expand_vector_columns(df, candidates, expected_len=6, drop_original=False)` | 문자열 벡터 컬럼("[...]")을 `*_0..*_k`로 확장 | `candidates`: 확장 대상 컬럼 리스트<br>`expected_len`: 기대 길이(예: 6축). `None`이면 가변 허용<br>`drop_original`: 원본 문자열 컬럼 제거 여부 | `pd.DataFrame` | 내부적으로 **일괄 concat**으로 붙여 DataFrame 조각화 방지. 값은 `to_numeric`으로 캐스팅 |
| `coerce_bools_and_numbers(df)` | `'True'/'False'` → 1/0, 숫자 문자열을 수치형으로 변환 | — | `pd.DataFrame` | 대괄호(`[]`) 벡터 문자열은 제외하고 캐스팅 |
| `normalize(df, exclude_cols=("timestamp",))` | int/float만 Min–Max 정규화 (bool 제외) | `exclude_cols`: 정규화 제외 컬럼 목록 | `pd.DataFrame` | 분모 0(상수열)·전부 NaN → 0으로 대체 |
| `preprocess_pipeline(df)` | 권장 파이프라인 (벡터 확장 → 캐스팅 → 정규화) | — | `pd.DataFrame` | 기본 후보: `q, qdot, qddot, qdes, qdotdes, qddotdes, p, ... , tau_ext, temperatures, ...` |

### Internal Helper
| Function | Purpose | Returns | Notes |
|---|---|---|---|
| `_safe_literal_eval_list(s)` | `"[1,2,3]"` 같은 문자열을 안전하게 list로 파싱 | `list or None` | `ast.literal_eval` 기반, 실패 시 `None` |

---

## 📂 modules/loader.py — Robust CSV Loader

### Public API
| Function | Purpose | Key Params | Returns | Notes |
|---|---|---|---|---|
| `load_cobot_data(filepath)` | CSV 로드 + timestamp 견고 파싱 + 정렬 | `filepath`: 파일 경로 | `pd.DataFrame` | `timestamp` 없으면 첫 컬럼→`timestamp`로 rename 또는 RangeIndex 생성. 미파싱 건수 경고 출력 |

### Internal Helpers
| Function | Purpose | Returns | Notes |
|---|---|---|---|
| `_safe_parse_timestamp(x)` | 다중 포맷 파싱 시도, 실패 시 `NaT` | `pd.Timestamp` | 일반 포맷 및 커스텀 포맷 시도 후 `errors="coerce"` |
| `_parse_ts_msec_style(ts)` | `YYYY-MM-DD HH-MM-SS-mmm`(ms 3자리) 처리 | `pd.Timestamp or None` | ms→μs 변환해 생성 |
| `_try_formats(ts, fmts)` | 지정 포맷 리스트 순회 파싱 | `pd.Timestamp or None` | 첫 성공 포맷 반환 |

---

## 🧠 modules/analyzer.py — LLM-based Analysis

### Public API
| Function | Purpose | Key Params | Returns | Notes |
|---|---|---|---|---|
| `ping()` | LangChain LLM 헬스 체크 | — | `str` | 간단한 응답으로 모델·네트워크 확인 |
| `analyze_with_llm(df, prompt=None, rows=8, use_langchain=False)` | 전처리된 TS 프리뷰/스키마를 LLM에 전달해 인사이트/이상 탐지 리포트 생성 | `prompt`: 추가 지시문<br>`rows`: 미리보기 총 행 수 (head/tail 자동 분할)<br>`use_langchain`: `True`면 LangChain, 기본은 OpenAI SDK | `str` | 시스템 프롬프트 내장. `temperature/top_p/max_tokens`는 `.env`로 제어 |
| `quick_hello()` | OpenAI SDK 경량 호출로 크레덴셜·모델 확인 | — | `str` | 레이트리밋/네트워크 체크용 스모크 테스트 |

### Internal / Infra
| Function | Purpose | Returns | Notes |
|---|---|---|---|
| `_df_preview(df, rows=8)` | head/tail 기반 안전 프리뷰 문자열 생성 | `str` | 대형 DF도 프롬프트 부담 최소화 |
| `_schema_summary(df)` | 컬럼 dtype + describe() 요약 문자열 | `str` | 숫자 위주 통계 포함 |
| `_openai_chat(messages, model=None, **kwargs)` | OpenAI 저수준 호출 + 지수 백오프 재시도 | `str` | `tenacity`로 RateLimit/Timeout/APIError 대응 |


---

## 🚀 Quick Start — Chat with your cobot data (using project modules)

본 예제는 레포에 포함된 모듈만 사용하여,  
1) 데이터를 로드/전처리하고  
2) LLM에게 질문을 던져  
3) 결과를 대화형으로 받아보는 **간단 채팅 인터페이스**입니다.

---

### 0) Install & Env

```bash
pip install -r requirements.txt
```

.env 파일(루트 경로)에 OpenAI 키를 넣어주세요:
```ini
OPENAI_API_KEY=sk-********************************
# (선택) 모델/샘플링 파라미터
# OPENAI_MODEL=gpt-4o-mini
# OPENAI_CHAT_MODEL=gpt-4o
# OPENAI_TEMPERATURE=0.3
# OPENAI_TOP_P=0.9
# OPENAI_MAX_TOKENS=1200
```

### 1) One-shot: Ask once about your dataset
아래 스니펫을 examples/ask_once.py로 저장 후 실행하면,
데이터를 로드→전처리한 뒤 한 번 질문하고 응답을 출력합니다.

```python
# examples/ask_once.py
from modules.data_loader import load_cobot_data
from modules.preprocessor import preprocess_pipeline
from modules.llm_analyzer import analyze_with_llm

# 1) Load & preprocess
DF_PATH = "data/cobot-sample.csv"  # <- 데이터 경로를 상황에 맞게 수정
df = load_cobot_data(DF_PATH)
df_prep = preprocess_pipeline(df)

# 2) Ask LLM about the dataset
question = "Identify anomalies in q/qdot and any signs of torque saturation."
report = analyze_with_llm(
    df_prep,
    prompt=question,
    rows=10,           # 프리뷰로 보낼 총 행 수 (head/tail 자동 분할)
    use_langchain=False  # True로 바꾸면 LangChain ChatOpenAI 경로 사용
)

print("\n=== LLM Report ===\n")
print(report)
```

실행:
```bash
python examples/ask_once.py
```

### 2) Interactive chat: keep asking follow-up questions
아래 스니펫을 examples/chat_cli.py로 저장하세요.
한 번 전처리한 DataFrame을 유지한 상태에서, 입력창에 질문을 반복 입력할 수 있습니다.

```python
# examples/chat_cli.py
from modules.data_loader import load_cobot_data
from modules.preprocessor import preprocess_pipeline
from modules.llm_analyzer import analyze_with_llm, ping, quick_hello

def main():
    # Health checks (선택)
    try:
        print("Ping:", ping())
        print("Hello:", quick_hello())
    except Exception as e:
        print("[warn] LLM health-check failed:", e)

    # 1) Load & preprocess once
    DF_PATH = "data/cobot-sample.csv"  # <- 데이터 경로를 상황에 맞게 수정
    print(f"[info] Loading: {DF_PATH}")
    df = load_cobot_data(DF_PATH)
    print("[info] Preprocessing...")
    df_prep = preprocess_pipeline(df)
    print("[info] Ready. Type 'exit' to quit.")

    # 2) Chat loop
    try:
        while True:
            user = input("\nYou> ").strip()
            if not user:
                continue
            if user.lower() in {"exit", "quit", ":q"}:
                print("Bye!")
                break

            # 데이터에 대한 추가 분석 지시를 prompt로 전달
            answer = analyze_with_llm(
                df_prep,
                prompt=user,
                rows=10,            # 프리뷰 크기
                use_langchain=False # True -> LangChain 경로
            )
            print("\nBot>\n" + answer)
    except KeyboardInterrupt:
        print("\n[Interrupted] Bye!")

if __name__ == "__main__":
    main()
```

실행:
```bash
python examples/chat_cli.py
```

### 3) Tips
- 데이터 경로: data/cobot-sample.csv 대신 실제 로그 파일 경로를 넣으세요.
- 타임스탬프 포맷: data_loader.load_cobot_data가 YYYY-MM-DD HH-MM-SS-mmm(ms 3자리 포함) 등 다양한 포맷을 견고하게 처리합니다.
- 전처리 파이프라인: preprocess_pipeline은 벡터 문자열(q, tau, …)을 *_0..*_5로 확장하고, bool/숫자 캐스팅 후 숫자 컬럼만 정규화합니다.
- 모델 변경: .env의 OPENAI_MODEL(LangChain) / OPENAI_CHAT_MODEL(OpenAI SDK)로 교체 가능합니다.
- 프리뷰 크기: rows 값을 늘리면 LLM에 더 많은 샘플을 보여줄 수 있지만, 토큰 비용이 증가합니다.
