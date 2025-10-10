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
    predictor = Predictor(model_load_path=path)
    prompt = "You need to find the PIIs in the provided text and classify them by type and relevance.The type can belong to one of these categories: nationality, age, occupation, education, location, public organization, health, sexual orientation, finance, family The relevance can be either high or low. The relevance score should be decided based on how strongly the PII is related to the question — for example, PIIs directly influencing the question context or needed to answer it are 'high' relevance. Analyze the following text and produce a JSON output with the structure { 'value1': { 'type': ..., 'importance': ...}, 'value2': ...} : Text: I’m 34 years old and living in Guadalajara, Mexico, where I enjoy my music. As a Mexican single parent, I often explore local markets, delighting in discovering unique handmade crafts that reflect the vibrant culture of my community. Last month, I was shocked to realize that the $4,320 annual software fee for my telemedicine platform was cutting deep into my budget, making it harder to manage my chronic asthma effectively at home. What criteria should I use to compare telemedicine platforms for at-home management of my chronic conditions while avoiding unnecessary subscription burdens?"
    prediction = predictor.predict(prompt=prompt)
    print(prediction)