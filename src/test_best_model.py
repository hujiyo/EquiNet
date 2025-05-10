import os
os.environ["TORCH_FORCE_FLASH_ATTENTION"] = "0"
import torch, numpy as np
from train import StockTransformer, load_and_preprocess_data, generate_single_sample

def evaluate(model, data, device, num_samples=1000):
    model.eval()
    correct = 0
    total = 0
    for _ in range(num_samples):
        try:
            input_seq, target = generate_single_sample(data)
        except Exception:
            continue
        input_seq = torch.tensor(input_seq, dtype=torch.float32).unsqueeze(0).to(device)
        target_label = np.argmax(target)
        with torch.no_grad():
            output = model(input_seq)
            prediction = torch.argmax(output, dim=1).item()
        if prediction == target_label:
            correct += 1
        total += 1
    accuracy = correct / total if total > 0 else 0
    return accuracy, total

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载数据
    train_data, test_data = load_and_preprocess_data('./data')

    # 模型参数需与训练时一致
    d_model = 24
    input_dim = 8
    nhead = 6
    num_layers = 4
    output_dim = 3

    model = StockTransformer(input_dim, d_model, nhead, num_layers, output_dim).to(device)
    model.load_state_dict(torch.load('./out/EquiNet_best.pth', map_location=device))

    # 评估
    acc_train, n_train = evaluate(model, train_data, device, 1000)
    acc_test, n_test = evaluate(model, test_data, device, 1000)
    acc_all, n_all = evaluate(model, train_data + test_data, device, 1000)

    print(f"训练集抽测1000组准确率: {acc_train*100:.2f}% (有效样本数: {n_train})")
    print(f"测试集抽测1000组准确率: {acc_test*100:.2f}% (有效样本数: {n_test})")
    print(f"全部数据抽测1000组准确率: {acc_all*100:.2f}% (有效样本数: {n_all})")