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
