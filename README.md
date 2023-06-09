# Ziya-LLaMA-13B-deployment
## 大模型部署实战（一）——Ziya-LLaMA-13B
Ziya-LLaMA-13B是IDEA基于LLaMa的130亿参数的大规模预训练模型，具备翻译，编程，文本分类，信息抽取，摘要，文案生成，常识问答和数学计算等能力。目前姜子牙通用大模型已完成大规模预训练、多任务有监督微调和人类反馈学习三阶段的训练过程。本文主要用于Ziya-LLaMA-13B的本地部署。
在线体验地址：
博主自己部署的地址：[http://www.yourmetaverse.cn:39000/](http://www.yourmetaverse.cn:39000/)
官方部署的地址：[https://modelscope.cn/studios/Fengshenbang/Ziya_LLaMA_13B_v1_online/summary](https://modelscope.cn/studios/Fengshenbang/Ziya_LLaMA_13B_v1_online/summary)

## 1. 部署准备
### 1.1 硬件环境
显卡最低显存为54GB，可以为一张A100（80GB）或者两张A100（40GB）。
### 1.2 python环境

```bash
cpm_kernels==1.0.11
gradio==3.34.0
huggingface_hub==0.15.1
Pillow==9.5.0
torch==1.12.1+cu113
tqdm==4.64.1
transformers==4.29.2
accelerate==0.17.1
```
### 1.3 模型下载
- **（1）llama-13B原始参数下载**
原始参数可以通过下面网址下载。
[https://docs.google.com/forms/d/e/1FAIpQLSfqNECQnMkycAp2jP4Z9TFX0cGR4uf7b_fBxjY_OjhJILlKGA/viewform](https://docs.google.com/forms/d/e/1FAIpQLSfqNECQnMkycAp2jP4Z9TFX0cGR4uf7b_fBxjY_OjhJILlKGA/viewform)
- **（2）Ziya-LLaMA-13B-v1.1 delta参数下载**
delta参数进入下面网页下载。
[https://huggingface.co/IDEA-CCNL/Ziya-LLaMA-13B-v1.1/tree/main](https://huggingface.co/IDEA-CCNL/Ziya-LLaMA-13B-v1/tree/main)

- **（3）相关代码下载**
进入下面网页下载llama权重转hf权重脚本。
[https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/convert_llama_weights_to_hf.py](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/convert_llama_weights_to_hf.py)
进入下面网页下载llama权重转Ziya-LLaMA-13B权重脚本。
[https://github.com/IDEA-CCNL/Fengshenbang-LM/blob/main/fengshen/utils/apply_delta.py](https://github.com/IDEA-CCNL/Fengshenbang-LM/blob/main/fengshen/utils/apply_delta.py)

## 2. 模型部署
首先说明一下项目的文件系统目录

```bash
-llama-13B
-llama-13B-convert
-ziya_v1.1_delta
-ziya_v1.1
-apply_delta.py
-convert_llama_weights_to_hf.py
-launch.py
-utils.py
```
其中llama-13B为llama原始参数存放的目录，llama-13B-convert为转换成huggingface形式的参数存放的目录，ziya_v1.1_delta为huggingface上的权重文件，ziya_v1.1为最终转换后的权重文件。launch.py为本地化部署文件，详见后续章节，utils.py为官方给的文件，直接从[https://modelscope.cn/studios/Fengshenbang/Ziya_LLaMA_13B_v1_online/files](https://modelscope.cn/studios/Fengshenbang/Ziya_LLaMA_13B_v1_online/files)下载即可。
### 2.1 llama-13B权重转换
首先第一步需要将llama-13B的原始权重转换成huggingface的权重形式，使用convert_llama_weights_to_hf.py脚本进行转换，转换代码如下：

```bash
python convert_llama_weights_to_hf.py --input_dir $你的llama-13B路径 --model_size 13B --output_dir $你的llama-13B模型转换后的路径
```
### 2.2 结合基础的llama权重和Ziya-LLaMA-13B delta权重得到Ziya-LLaMA-13B权重
使用如下代码得到最终的Ziya-LLaMA-13B权重。
```bash
python -m apply_delta --base $你的llama-13B模型转换后的路径 --target $你的最终权重存储路径 --delta $你下载的Ziya-LLaMA-13B权重路径
```
### 2.3 模型部署
使用下面这段代码进行模型部署。该代码修改自[https://modelscope.cn/studios/Fengshenbang/Ziya_LLaMA_13B_v1_online/files](https://modelscope.cn/studios/Fengshenbang/Ziya_LLaMA_13B_v1_online/files)
在官方代码的基础上增加了一些参数和优化内容。

```python
import gradio as gr
import os
import gc
import torch


from transformers import AutoTokenizer
#指定环境的GPU，我的环境是2张A100（40GB）显卡，于是我设置了两张卡，也可以一张80GB的A100
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"
#这个utils文件直接下载官方给的文件即可
from utils import SteamGenerationMixin


class MindBot(object):
    def __init__(self):
    	#这个model_path为你本地的模型路径
        model_path = './ziya_v1.1'
        self.model = SteamGenerationMixin.from_pretrained(model_path, device_map='auto').half()
        self.model.eval()
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    
    def build_prompt(self, instruction, history, human='<human>', bot='<bot>'):
        pmt = ''
        if len(history) > 0:
            for line in history:
                pmt += f'{human}: {line[0].strip()}\n{bot}: {line[1]}\n'
        pmt += f'{human}: {instruction.strip()}\n{bot}: \n'
        return pmt
    
    def interaction(
        self,
        instruction,
        history,
        max_new_tokens,
        temperature,
        top_p,
        max_memory=1024
    ):
               
        prompt = self.build_prompt(instruction, history)
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        if input_ids.shape[1] > max_memory:
            input_ids = input_ids[:, -max_memory:]
            
        prompt_len = input_ids.shape[1]
        # stream generation method
        try:
            tmp = history.copy()
            output = ''
            with torch.no_grad():
                for generation_output in self.model.stream_generate(
                    input_ids.cuda(),
                    max_new_tokens=max_new_tokens, 
                    do_sample=True,
                    top_p=top_p, 
                    temperature=temperature, 
                    repetition_penalty=1., 
                    eos_token_id=2, 
                    bos_token_id=1, 
                    pad_token_id=0
                ):
                    s = generation_output[0][prompt_len:]
                    output = self.tokenizer.decode(s, skip_special_tokens=True)
                    # output = output.replace('\n', '<br>')
                    output = output.replace('\n', '\n\n')
                    tmp.append((instruction, output))
                    yield  '', tmp
                    tmp.pop()
                    # gc.collect()
                    # torch.cuda.empty_cache()
                history.append((instruction, output))
                print('input -----> \n', prompt)
                print('output -------> \n', output)
                print('history: ======> \n', history)
        except torch.cuda.OutOfMemoryError:
            gc.collect()
            torch.cuda.empty_cache()
            self.model.empty_cache()
            history.append((instruction, "【显存不足，请清理历史信息后再重试】"))
        return "", history
        
    def chat(self):
        
        with gr.Blocks(title='IDEA MindBot', css=".bgcolor {color: white !important; background: #FFA500 !important;}") as demo:
            with gr.Row():
                gr.Column(scale=0.25)
                with gr.Column(scale=0.5):
                    gr.Markdown("<center><h1>IDEA Ziya</h1></center>")
                    gr.Markdown("<center>姜子牙通用大模型V1.1是基于LLaMa的130亿参数的大规模预训练模型，具备翻译，编程，文本分类，信息抽取，摘要，文案生成，常识问答和数学计算等能力。目前姜子牙通用大模型已完成大规模预训练、多任务有监督微调和人类反馈学习三阶段的训练过程。</center>")
                gr.Column(scale=0.25)
            with gr.Row():
                gr.Column(scale=0.25)
                with gr.Column(scale=0.5):
                    chatbot = gr.Chatbot(label='Ziya').style(height=500)
                    msg = gr.Textbox(label="Input")
                # gr.Column(scale=0.25)
                with gr.Column(scale=0.25):
                    max_new_tokens = gr.Slider(0, 2048, value=1024, step=1.0, label="Max_new_tokens", interactive=True)
                    top_p = gr.Slider(0, 1, value=0.85, step=0.01, label="Top P", interactive=True)
                    temperature = gr.Slider(0, 1, value=0.8, step=0.01, label="Temperature", interactive=True)
            with gr.Row():
                gr.Column(scale=0.25)
                with gr.Column(scale=0.25):
                    clear = gr.Button("Clear")
                with gr.Column(scale=0.25):
                    submit = gr.Button("Submit")
                gr.Column(scale=0.25)
                
            msg.submit(self.interaction, [msg, chatbot,max_new_tokens,top_p,temperature], [msg, chatbot])
            clear.click(lambda: None, None, chatbot, queue=False)
            submit.click(self.interaction, [msg, chatbot,max_new_tokens,top_p,temperature], [msg, chatbot])
        return demo.queue(concurrency_count=10).launch(share=False,server_name="127.0.0.1", server_port=7886)
        

if __name__ == '__main__':
    mind_bot = MindBot()
    mind_bot.chat()
```
该代码中需要修改的地方已经在代码块上标出。
