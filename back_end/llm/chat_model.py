# import torch
# import sentencepiece
# from langchain_huggingface import ChatHuggingFace

# from transformers import (
#     pipeline,
#     AutoModelForCausalLM,
#     AutoTokenizer,
#     BitsAndBytesConfig
# )

# from langchain_huggingface import HuggingFacePipeline


# class LLM:
#     def __init__(self, model_id: str = 'meta-llama/Llama-3.2-3B', device: str = 'cuda'):
#         self.bnb_config = BitsAndBytesConfig(
#             load_in_4bit=True,
#             bnb_4bit_quant_type="nf4",
#             bnb_4bit_compute_dtype=torch.bfloat16,
#             bnb_4bit_use_double_quant=True
#         )

#         self.tokenizer = AutoTokenizer.from_pretrained(model_id)
#         self.model = AutoModelForCausalLM.from_pretrained(
#             model_id,
#             trust_remote_code=True,
#             quantization_config=self.bnb_config,
#             token="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"
#         )

#         self.llm = ChatHuggingFace(
#            model_id = model_id,
#             llm=HuggingFacePipeline(
#                 pipeline=pipeline(
#                     task='text-generation',
#                     model=self.model,
#                     tokenizer=self.tokenizer,
#                     max_new_tokens=512,
#                 )
#             ),
#             tokenizer=self.tokenizer,
#         )


#     def get_info(self):
#         print(f"""model_info:\n\t'model_id': {self.model}\n\t'quantized': True""")

#     def get_llm(self) -> ChatHuggingFace:
#         return self.llm
# -------------------------------------------------------

import torch
import sentencepiece
from langchain_huggingface import ChatHuggingFace

from transformers import (
    pipeline,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from langchain_huggingface import HuggingFacePipeline
from peft import LoraConfig, get_peft_model


class LLM:
    def __init__(self, model_id: str = 'meta-llama/Llama-3.2-3B', device: str = 'cuda'):
        # Quantization Configuration
        self.bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True
        )

        # Load Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        # Load Quantized Model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            quantization_config=self.bnb_config,
            token="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"
        )


        # Apply LoRA Fine-Tuning
        self.lora_config = LoraConfig(
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["qkv_proj"]
        )
        self.model = get_peft_model(self.model, self.lora_config)

        # Create Pipeline for Inference
        self.llm = ChatHuggingFace(
           model_id=model_id,
            llm=HuggingFacePipeline(
                pipeline=pipeline(
                    task='text-generation',
                    model=self.model,
                    tokenizer=self.tokenizer,
                    max_new_tokens=256,  # Reduced for faster generation
                    temperature=0.7,     # Controls randomness
                    top_p=0.9,
                    do_sample = True,          # Nucleus sampling
                    top_k=50,            # Limits the token pool
                    repetition_penalty=1.2  # Reduces repetitive phrases
                )
            ),
            tokenizer=self.tokenizer,
        )

    def get_info(self):
        print(f"""model_info:\n\t'model_id': {self.model}\n\t'quantized': True""")

    def get_llm(self) -> ChatHuggingFace:
        return self.llm
# -------------------------------------------------------

# import torch
# import sentencepiece
# from langchain_huggingface import ChatHuggingFace

# from transformers import (
#     pipeline,
#     AutoModelForCausalLM,
#     AutoTokenizer,
#     BitsAndBytesConfig
# )
# from langchain_huggingface import HuggingFacePipeline
# from peft import LoraConfig, get_peft_model
# from torch.nn.utils.prune import random_unstructured, remove

# class LLM:
#     def __init__(self, model_id: str = 'meta-llama/Llama-3.2-3B', device: str = 'cuda'):
#         # Quantization Configuration
#         self.bnb_config = BitsAndBytesConfig(
#             load_in_4bit=True,
#             bnb_4bit_quant_type="nf4",
#             bnb_4bit_compute_dtype=torch.bfloat16,
#             bnb_4bit_use_double_quant=True
#         )

#         # Load Tokenizer
#         self.tokenizer = AutoTokenizer.from_pretrained(model_id)

#         # Load Quantized Model
#         self.model = AutoModelForCausalLM.from_pretrained(
#             model_id,
#             trust_remote_code=True,
#             quantization_config=self.bnb_config,
#             token="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"
#         )

#                 # Apply Pruning
#         for name, module in self.model.named_modules():
#             if hasattr(module, 'weight') and isinstance(module.weight, torch.Tensor):
#                 # Dequantize weight if needed
#                 if module.weight.dtype != torch.float32:
#                     module.weight.data = module.weight.data.to(torch.float32)
#                 # Prune 30% of weights randomly
#                 random_unstructured(module, name='weight', amount=0.3)
#                 # Optionally remove pruning reparameterization
#                 remove(module, 'weight')
#                 # Requantize weight if necessary
#                 if self.bnb_config.load_in_4bit:
#                     module.weight.data = module.weight.data.to(torch.bfloat16)


#         # Apply LoRA Fine-Tuning
#         self.lora_config = LoraConfig(
#             r=8,
#             lora_alpha=32,
#             lora_dropout=0.1,
#             bias="none",
#             task_type="CAUSAL_LM",
#             target_modules=["qkv_proj"]
#         )
#         self.model = get_peft_model(self.model, self.lora_config)

#         # Create Pipeline for Inference
#         self.llm = ChatHuggingFace(
#             model_id=model_id,
#             llm=HuggingFacePipeline(
#                 pipeline=pipeline(
#                     task='text-generation',
#                     model=self.model,
#                     tokenizer=self.tokenizer,
#                     max_new_tokens=256,
#                     temperature=0.7,
#                     top_p=0.9,
#                     do_sample=True,
#                     top_k=50,
#                     repetition_penalty=1.2
#                 )
#             ),
#             tokenizer=self.tokenizer,
#         )


#     def get_llm(self) -> ChatHuggingFace:
#         return self.llm
