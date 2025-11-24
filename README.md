# LLM-for-research
playground of LLM for research

## 安装依赖

### 方法一：使用pip安装
```bash
pip install -r requirements.txt
```

### 方法二：使用conda安装
```bash
conda install --file requirements.txt
```

## 内容介绍
### LLM的调用和测试
- `Example_01_llm_api.py` 提供了调用硅基流动API和本地模型API的框架
- `Example_02_evaluate_llm.py` 提供了对硅基流动上微调后的LLM的预测能力评估的框架
- `FT_data_siliconflow` 目录下提供了用于硅基流动上微调和测试模型的样例数据

### 利用cursor加速科研中的数据分析和科研绘图
- `Example_03_data_analysis.ipynb` 提供了数据分析的提示词模板

### 一个简单的智能体实例
- `Example_04_travel_agent.ipynb` 提供了调用硅基流动API实现一个出行规划智能体的示例

### llamafactory本地部署和微调的示例文件
- `llamafactory_configs` 目录下提供了使用llamafactory启动api服务和训练功能的示例配置文件
- `FT_data_llamafactory` 目录下提供了使用llamafactory微调LLM的示例数据

## llamafactory安装
先新建一个conda虚拟环境

```bash
conda create -n llamafactory python=3.10
conda activate llamafactory

git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics]" --no-build-isolation
```
