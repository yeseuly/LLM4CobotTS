import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# examples/chat_cli.py
from modules.loader import load_cobot_data
from modules.preprocessor import preprocess_pipeline
from modules.analyzer import analyze_with_llm, ping, quick_hello

def main():
    try:
        print("Ping:", ping())
        print("Hello:", quick_hello())
    except Exception as e:
        print("[warn] LLM health-check failed:", e)

    # 1) Load & preprocess once
    DF_PATH = str(ROOT)+"/data/indy_sample.csv" 
    print(f"[info] Loading: {DF_PATH}")
    df = load_cobot_data(DF_PATH)
    print("[info] Preprocessing...")
    df_prep = preprocess_pipeline(df)
    print("[info] Ready. Type 'exit' to quit.")

    # 2) Chat loop (with sample data: indy_sample.csv)
    try:
        while True:
            user = input("\nYou> ").strip()
            if not user:
                continue
            if user.lower() in {"exit", "quit", ":q"}:
                print("Bye!")
                break

            answer = analyze_with_llm(
                df_prep,
                prompt=user,
                rows=10,            
                use_langchain=False 
            )
            print("\nBot>\n" + answer)
    except KeyboardInterrupt:
        print("\n[Interrupted] Bye!")

if __name__ == "__main__":
    main()
