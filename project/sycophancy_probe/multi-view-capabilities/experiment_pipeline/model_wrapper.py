import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
from typing import List, Dict, Any, Optional
import os


class ModelWrapper:
    def __init__(
        self,
        model_name: str,
        device: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None,
        trust_remote_code: bool = False,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
    ):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.torch_dtype = torch_dtype or torch.float16
        self.hook_handles = []
        
        # Handle HuggingFace authentication for gated models
        hf_token = os.getenv("HF_TOKEN")
        if hf_token:
            try:
                login(token=hf_token, add_to_git_credential=False)
            except Exception as e:
                print(f"Warning: HuggingFace login failed: {e}")
        else:
            print("No hugging face token found, so will proceed without logging in")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=trust_remote_code,
                padding_side="left"
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
        #     self.model = AutoModelForCausalLM.from_pretrained(
        #         model_name,
        #         torch_dtype=self.torch_dtype,
        #         device_map="auto" if self.device == "cuda" else None,
        #         trust_remote_code=trust_remote_code,
        #         load_in_8bit=load_in_8bit,
        #         load_in_4bit=load_in_4bit,
        #     )
            
        #     if not (load_in_8bit or load_in_4bit) and self.device != "cuda":
        #         self.model = self.model.to(self.device)
                
        except Exception as e:
            raise RuntimeError(f"Failed to load model {model_name}: {str(e)}")
        
    def add_hook(self, hook_fn, target_layer_idx): 
        '''register hook at desired layer
        target layer should be the index of the layer'''
        target_layer = self.model.model.layers[target_layer_idx]
        hook_handle = target_layer.register_forward_hook(hook_fn)
        self.hook_handles.append(hook_handle)
        return hook_handle
    
    def remove_hook(self, hook_handle):
        '''remove hook with given hook_handle'''
        hook_handle.remove()
        self.hook_handles.remove(hook_handle)

    def clear_hooks(self):
        '''clear all existing model hooks'''
        for h in self.hook_handles:
            h.remove()
        self.hook_handles.clear()

    def apply_chat_template(
        self,
        messages: List[Dict[str, str]],
    ) -> str:
        if hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(
                messages, 
                tokenize = False,
                add_generation_prompt = True
            )
        else:
            return self._format_chat_manually(messages)
    
    def prompt_with_chat_template(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        do_sample: bool = False, # default behavior is greedy decoding 
        top_p: float = 0.9,
        return_full_text: bool = False, 
        clear_hooks_after = True, # whether or not hooks should be cleared after generation
        **generation_kwargs
    ) -> str:

        try:
            if hasattr(self.tokenizer, "apply_chat_template"):
                formatted_prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            else:
                formatted_prompt = self._format_chat_manually(messages)
            
            inputs = self.tokenizer(
                formatted_prompt,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=do_sample,
                    top_p=top_p,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    **generation_kwargs
                )
            
            if clear_hooks_after:
                self.clear_hooks()

            
            if return_full_text:
                response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            else:
                new_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
                response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            
            return response.strip()
            
        except Exception as e:
            raise RuntimeError(f"Generation failed: {str(e)}")
        
    
    def _format_chat_manually(self, messages: List[Dict[str, str]]) -> str:
        formatted = ""
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            
            if role == "system":
                formatted += f"System: {content}\n\n"
            elif role == "user":
                formatted += f"User: {content}\n\n"
            elif role == "assistant":
                formatted += f"Assistant: {content}\n\n"
        
        formatted += "Assistant: "
        return formatted
    
    def get_model_info(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "device": self.device,
            "torch_dtype": str(self.torch_dtype),
            "num_parameters": sum(p.numel() for p in self.model.parameters()),
            "trainable_parameters": sum(p.numel() for p in self.model.parameters() if p.requires_grad),
        }
    
    def __repr__(self) -> str:
        return f"ModelWrapper(model_name='{self.model_name}', device='{self.device}')"