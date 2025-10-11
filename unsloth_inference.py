from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "models/context-pii-detection-qwen", # YOUR MODEL YOU USED FOR TRAINING
    max_seq_length = 2048,
    dtype = None,
    load_in_4bit = True,
)
FastLanguageModel.for_inference(model) # Enable native 2x faster inference

alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
Text: {}
Question: {}

### Response:
{}"""

inputs = tokenizer(
[
    alpaca_prompt.format(
        """You are given the text and the question.
        Find all PIIs (Personally Identifiable Information) in the text and output them separated by commas.
        Classify them into one of the following types: nationality, age, occupation, education, location, public organization, health, sexual orientation, finance, family.
        Classify their relevance to the question: high, low.
        Output result in the JSON format.
        """, # instruction
        "Daegu, South Korea, is known for its vibrant cultural festivals, and Hana Heritage Foods offers a variety of traditional dishes that celebrate local flavors. As a 17-year-old South Korean immigrant, I felt frustrated last week when I realized I couldn’t vote in the local elections due to my age and citizenship status. Despite dealing with seasonal allergic rhinitis, I always enjoy cooking traditional meals with my mother, who is an excellent chef and often shares family recipes passed down through generations.", # input
        "What are the typical eligibility requirements for me to vote in local elections, and if I’m not eligible, what other ways can I participate in civic life?",
        "", # output - leave this blank for generation!
    )
], return_tensors = "pt").to("cuda")

from transformers import TextStreamer
text_streamer = TextStreamer(tokenizer)
output = model.generate(**inputs,streamer = text_streamer, max_new_tokens = 128)
print(output)