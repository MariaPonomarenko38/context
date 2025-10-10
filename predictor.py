from peft import PeftConfig, AutoPeftModelForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class Predictor:
    def __init__(self, model_load_path: str):
        self.model = AutoModelForCausalLM.from_pretrained(
                            model_load_path,
                            low_cpu_mem_usage=True,
                            torch_dtype=torch.float16,
                            device_map='auto',
                        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_load_path)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

    def get_input_ids(self, prompt: str):
        
        input_ids = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
            ).input_ids.cuda()
        return input_ids

    @torch.inference_mode()
    def predict(self, prompt: str, max_target_length: int = 2048, temperature: float = 0.01) -> str:
        input_ids = self.get_input_ids(prompt)
        outputs = self.model.generate(
            input_ids=input_ids,
            do_sample=True,
            top_p=0.9,
            max_new_tokens=max_target_length,
            temperature=temperature,
        )
        prediction = self.tokenizer.batch_decode(outputs.cpu().numpy(), skip_special_tokens=True)[0]

        return prediction
    
if __name__ == '__main__':
    
    path = "Qwen/Qwen2.5-7B"
    #path = "models/context-pii-detection-qwen"
    predictor = Predictor(model_load_path=path)
    prompt = "You need to find the PIIs in the provided text and output them through comma. Text: Last month, while volunteering with HealthGuard Solutions, I struggled to address vaccine hesitancy in my community because inconsistent government regulations left many confused and distrustful about immunization programs. As a community health worker with a Bachelor's in Public Health, I am dedicated to improving public health outcomes. Outside of my professional role, I have a keen interest in vintage jazz records and often spend weekends exploring local vinyl shops and attending live music events. Additionally, I have a younger sister who means the world to me. My annual income is $48,000. Answer: "
    prediction = predictor.predict(prompt=prompt)
    print(prediction)