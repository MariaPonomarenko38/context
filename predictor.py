from peft import PeftConfig, AutoPeftModelForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class Predictor:
    def __init__(self, model_load_path: str):
        self.model = AutoPeftModelForCausalLM.from_pretrained(
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
    def predict(self, prompt: str, max_target_length: int = 60, temperature: float = 0.01) -> str:
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
    
    path = 'mistral7b-lora-pii'
    predictor = Predictor(model_load_path=path)
    prompt = """USER: Analyze the text and list all personally identifiable information (PII) spans.

    Output PIIs through comma.
    Text:
    I am a father of two living in Varanasi, India. During the early morning Ganga Aarti, I struggled to keep my children calm amid the overwhelming crowds and loud chants. In addition to managing my family during the Ganga Aarti, I once volunteered at Ganga Serenity Travels. I also hold a degree from Banaras Hindu University, where I studied environmental science with great interest. I have always kept my savings in savings account number 142536789012, and I am openly heterosexual.

    ASSISTANT:"""
    prediction = predictor.predict(prompt=prompt)
    print(prediction)