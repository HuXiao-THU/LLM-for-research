import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from Example_01_llm_api import call_llm
import re
from tqdm import tqdm

siliconflow_api_key = "Your API key here"
siliconflow_url = "https://api.siliconflow.cn/v1"

original_model = "Qwen/Qwen2.5-7B-Instruct"
finetuned_model = "Your fine-tuned model"

temperature = 1     # 尝试不同的temperature试试？

def evaluate_single_example(example, api_key, model):
    """评估单个样本"""
    try:
        # 从示例中提取系统提示词、用户提示和真实答案
        system_prompt = None
        user_prompt = None
        actual_response = None
        
        # 遍历消息列表，根据角色提取对应内容
        for msg in example["messages"]:
            role = msg["role"]
            content = msg["content"]
            
            if role == "system":
                system_prompt = content
            elif role == "user":
                user_prompt = content
            elif role == "assistant":
                actual_response = content
        
        # 调用API
        response, prompt_tokens, completion_tokens = call_llm(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            api_key=api_key,
            base_url=siliconflow_url,
            model=model,
            temperature=temperature,
            top_p=1,
            max_tokens=4096
        )
        # 对于云服务商的模型需要控制访问频率，每次请求后等待，确保不要太快访问
        time.sleep(0.15)
        
        # 使用正则表达式提取JSON格式的回答
        pattern = r"""\{"truck_risk": "(high|low)"\}"""
        pred_match = re.search(pattern, response)
        actual_match = re.search(pattern, actual_response)
        
        if pred_match and actual_match:
            prediction = pred_match.group(1).lower()
            actual = actual_match.group(1).lower()
            return prediction, actual, prompt_tokens, completion_tokens
        else:
            print(f"无法解析响应格式: \n预测: {response}\n实际: {actual_response}")
            return None, None, prompt_tokens, completion_tokens
            
    except Exception as e:
        print(f"评估样本时出错: {str(e)}")
        return None, None, 0, 0

def evaluate_model(model):
    """评估模型性能"""
    total_prompt_tokens = 0
    total_completion_tokens = 0

    # 设置环境
    api_key = siliconflow_api_key
    
    # 加载测试数据
    with open('FT_data/FT_data_test.jsonl', 'r') as f:
        test_data = [json.loads(line) for line in f]
    
    # 使用线程池进行并行处理
    predictions = []
    actuals = []
    error_counts = 0
    total_examples = len(test_data)
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        for example in test_data:
            future = executor.submit(
                evaluate_single_example,
                example,
                api_key,
                model
            )
            futures.append(future)
        
        # 使用tqdm显示进度
        for future in tqdm(futures, total=total_examples, desc="评估进度"):
            prediction, actual, prompt_tokens, completion_tokens = future.result()
            if prediction is None:
                error_counts += 1
            else:
                predictions.append(prediction)
                actuals.append(actual)

            total_prompt_tokens += prompt_tokens
            total_completion_tokens += completion_tokens
    
    # 计算准确率
    # 计算二分类指标
    correct_predictions = 0
    true_high = 0  # 真实为high的数量
    pred_high = 0  # 预测为high的数量
    true_high_pred_high = 0  # 真实为high且预测也为high的数量
    
    for pred, actual in zip(predictions, actuals):
        if pred == actual:
            correct_predictions += 1
        
        if actual == "high":
            true_high += 1
        
        if pred == "high":
            pred_high += 1
        
        if actual == "high" and pred == "high":
            true_high_pred_high += 1
    
    # 计算指标
    accuracy = correct_predictions / (total_examples - error_counts) if (total_examples - error_counts) > 0 else 0
    precision = true_high_pred_high / pred_high if pred_high > 0 else 0
    recall = true_high_pred_high / true_high if true_high > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\n评估结果:")
    print(f"总样本数: {total_examples}")
    print(f"有效预测数: {total_examples - error_counts}")
    print(f"正确预测数: {correct_predictions}")

    print(f"准确率 (Accuracy): {accuracy:.2%}")
    print(f"精确率 (Precision): {precision:.2%}")
    print(f"召回率 (Recall): {recall:.2%}")
    print(f"F1分数: {f1:.2%}")
    
    print(f"总输入token: {total_prompt_tokens}")
    print(f"总输出token: {total_completion_tokens}")

if __name__ == "__main__":
    print(f"评估原始模型: {original_model}")
    evaluate_model(original_model)

    print("--------------------------------")

    print(f"评估微调模型: {finetuned_model}")
    evaluate_model(finetuned_model)