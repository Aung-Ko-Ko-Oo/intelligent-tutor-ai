import torch
from src.model import HybridNLPDKT
from src.parser import BurmeseSemanticParser


def run_demo():
    print("=== Hybrid NLP-DKT Model Demo ===")

    # 1. Sample Training Data for Burmese Semantic Parser
    sample_texts = [
        "x တန်ဖိုးရှာရန် ၅ ကို အပေါင်းပြောင်းရမည်",  # Misconception / Incorrect
        (
            "ဖက်ရှင်အညီအမျှ ညီမျှခြင်း၏ ညာဘက်သို့ ၅ ကို နှုတ်ပေးရမည်"
        ),  # Correct reasoning
        (
            "မသိကိန်းကို သီးသန့်ခွဲထုတ်ရန် ၅ ဖြင့် စားပါမည်"
        ),  # Correct reasoning
        "ကိန်းဂဏန်း နှစ်ခုကို ပေါင်းလိုက်လျှင် ရလဒ်ထွက်မည်",  # Incorrect
    ]
    sample_labels = [0, 1, 1, 0]

    # Initialize and fit parser
    parser = BurmeseSemanticParser(n_gram_range=(2, 4))
    parser.fit(sample_texts, sample_labels)

    # 2. Extract features from a test open-ended response
    test_explanation = "x အပေါင်း ၅ ကို ၁၂ မှ နှုတ်ပါမည်"
    v_text, c_t = parser.extract_features(test_explanation)

    print(f"\nInput Explanation: '{test_explanation}'")
    print(f"Extracted Text Feature Vector Dimension: {v_text.shape[0]}")
    print(f"Predicted Semantic Correctness Scalar (c_t): {c_t.item():.4f}")

    # 3. Instantiate Hybrid DKT Model
    num_skills = 5
    text_dim = v_text.shape[0]
    model = HybridNLPDKT(num_skills=num_skills, text_dim=text_dim)

    # Simulate a sequence of 3 student interactions
    seq_len = 3
    batch_v_text = (
        torch.stack([v_text] * seq_len).unsqueeze(0)
    )  # Shape: [1, 3, text_dim]
    batch_s_t = torch.tensor([[0, 2, 1]])  # Skill IDs for time-steps t=1..3
    batch_c_t = (
        torch.tensor([0.2, 0.85, 0.9]).unsqueeze(-1).unsqueeze(0)
    )  # Shape: [1, 3, 1]

    # Model Forward Pass
    model.eval()
    with torch.no_grad():
        y_pred, _ = model(batch_v_text, batch_s_t, batch_c_t)

    print("\nPredicted Mastery Trajectory over time-steps:")
    for t in range(seq_len):
        print(f"Time-step t={t+1}:")
        print(
            f"  Skill target evaluated: {batch_s_t[0, t].item()} | Semantic"
            f" Correctness: {batch_c_t[0, t, 0].item():.2f}"
        )
        print(
            f"  Predicted mastery across all skills (t+1):"
            f" {y_pred[0, t].numpy().round(3)}"
        )


if __name__ == "__main__":
    run_demo()