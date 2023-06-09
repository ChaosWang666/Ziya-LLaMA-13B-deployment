import gradio as gr
import os
import gc
import torch


from transformers import AutoTokenizer
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"
from utils import SteamGenerationMixin


class MindBot(object):
    def __init__(self):
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