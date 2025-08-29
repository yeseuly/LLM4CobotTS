import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# examples/ask_once.py
from modules.loader import load_cobot_data
from modules.preprocessor import preprocess_pipeline
from modules.analyzer import analyze_with_llm
def main():
    # 1) Load & preprocess
    DF_PATH = str(ROOT)+"/data/indy_sample.csv" 
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

if __name__ == "__main__":
    main()