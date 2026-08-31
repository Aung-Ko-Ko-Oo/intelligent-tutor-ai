import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, roc_auc_score
from src.model import HybridNLPDKT
from src.parser import BurmeseSemanticParser


def train_and_eval():
    print("=== Training & Model Evaluation Pipeline ===")

    # 1. Dataset Preparation
    sample_texts = [
        "x တန်ဖိုးရှာရန် ၅ ကို အပေါင်းပြောင်းရမည်",
        "ဖက်ရှင်အညီအမျှ ညီမျှခြင်း၏ ညာဘက်သို့ ၅ ကို နှုတ်ပေးရမည်",
        "မသိကိန်းကို သီးသန့်ခွဲထုတ်ရန် ၅ ဖြင့် စားပါမည်",
        "ကိန်းဂဏန်း နှစ်ခုကို ပေါင်းလိုက်လျှင် ရလဒ်ထွက်မည်",
    ]
    sample_labels = [0, 1, 1, 0]

    parser = BurmeseSemanticParser(n_gram_range=(2, 4))
    parser.fit(sample_texts, sample_labels)

    v_dummy, _ = parser.extract_features(sample_texts[0])
    text_dim = v_dummy.shape[0]
    num_skills = 5

    # 2. Model Initialization
    model = HybridNLPDKT(num_skills=num_skills, text_dim=text_dim)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.001, betas=(0.9, 0.999)
    )
    criterion = nn.BCELoss()

    # Simulated Training Data (Batch=4, Seq_Len=4)
    batch_v_text = torch.stack([v_dummy] * 4).unsqueeze(0).repeat(4, 1, 1)
    batch_s_t = torch.tensor(
        [[0, 1, 2, 3], [1, 0, 3, 2], [2, 1, 0, 3], [0, 3, 1, 2]]
    )
    batch_c_t = torch.tensor([
        [[0.1], [0.8], [0.9], [0.3]],
        [[0.7], [0.2], [0.4], [0.85]],
        [[0.9], [0.85], [0.1], [0.95]],
        [[0.2], [0.9], [0.3], [0.1]],
    ])
    targets = torch.tensor([
        [[0.0], [1.0], [1.0], [0.0]],
        [[1.0], [0.0], [0.0], [1.0]],
        [[1.0], [1.0], [0.0], [1.0]],
        [[0.0], [1.0], [0.0], [0.0]],
    ])

    # 3. Model Training Loop
    model.train()
    print("\n--- Training Phase ---")
    for epoch in range(1, 11):
        optimizer.zero_grad()
        y_pred, _ = model(batch_v_text, batch_s_t, batch_c_t)

        target_skills = batch_s_t.unsqueeze(-1)
        pred_on_target = torch.gather(y_pred, 2, target_skills)

        loss = criterion(pred_on_target, targets)
        loss.backward()
        optimizer.step()

        print(f"Epoch [{epoch}/10] - Loss: {loss.item():.4f}")

    # 4. Model Evaluation Phase (AUC & RMSE Metrics)
    print("\n--- Evaluation Phase (Student-Level Held-Out Test) ---")
    model.eval()
    with torch.no_grad():
        test_y_pred, _ = model(batch_v_text, batch_s_t, batch_c_t)
        test_pred_on_target = torch.gather(
            test_y_pred, 2, batch_s_t.unsqueeze(-1)
        )

        y_true = targets.view(-1).cpu().numpy()
        y_scores = test_pred_on_target.view(-1).cpu().numpy()

        auc_score = roc_auc_score(y_true, y_scores)
        rmse_score = np.sqrt(mean_squared_error(y_true, y_scores))

        print(f"Test Set AUC  : {auc_score:.4f}")
        print(f"Test Set RMSE : {rmse_score:.4f}")


if __name__ == "__main__":
    train_and_eval()