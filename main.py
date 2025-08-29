from modules.loader import load_cobot_data
from modules.preprocessor import preprocess_pipeline
from modules.analyzer import analyze_with_llm, ping, quick_hello

if __name__ == "__main__":
    print("ME:", ping())
    print("LLM:", quick_hello())

    df = load_cobot_data("data/indy_sample.csv")
    df_prep = preprocess_pipeline(df)

    question = "Find several anomalous points in this data: w.r.t joints (q, qdot) and torques (tau)."
    
    print("ME: "+question)
    print("LLM: ")
    result = analyze_with_llm(
        df_prep,
        prompt=question, 
        rows=8,
        use_langchain=False,
    )
    print("\n=== LLM Analysis Result ===\n", result)
