from openai import OpenAI  

siliconflow_api_key = "Your API key here"
siliconflow_url = "https://api.siliconflow.cn/v1"
deepseek_model = "deepseek-ai/DeepSeek-V3"

def call_llm(system_prompt, user_prompt, api_key=None, base_url="http://localhost:8000/v1", model="test", temperature=1, top_p=1, max_tokens=4096):
    """
    调用LLM API
    :param system_prompt: 系统提示词
    :param user_prompt: 用户提示词
    :param api_key: API密钥
    :param model: 模型名称
    :param temperature: 温度
    :param top_p: top_p
    :param max_tokens: 最大token数
    """
    
    # client = OpenAI(api_key="0", base_url="http://localhost:8000/v1") # 本地部署模型的url 
    client = OpenAI(api_key=api_key, base_url=base_url)
    response = client.chat.completions.create(  
        model=model,  
        messages=[  
            {"role": "system", "content": system_prompt},  
            {"role": "user", "content": user_prompt}  
        ], 
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens
    )

    response_content = response.choices[0].message.content  # 回答的内容
    prompt_tokens = response.usage.prompt_tokens     # 提示词的token数，用于计费
    response_tokens = response.usage.completion_tokens  # 回答的token数，用于计费

    return response_content, prompt_tokens, response_tokens

if __name__ == '__main__':
    system_prompt = """
    You are a helpful assistant named "LEAD helper".
    """

    user_prompt = """
    Hello! What's your name? 
    """

    response, prompt_tokens, completion_tokens = call_llm(
        system_prompt, 
        user_prompt, 
        api_key=siliconflow_api_key, 
        base_url=siliconflow_url,
        model=deepseek_model,
        temperature=0.7,
        top_p=1,
        max_tokens=4096
        )
    
    print(response)
    print(f"prompt_tokens: {prompt_tokens}, completion_tokens: {completion_tokens}")