from src.utils.utility import my_utils
from src.utils.enums_class import iit_layer_type_enum
import re
import gc
import torch
import numpy as np
import torch.nn.functional as F


class llm_representation: 

    def extract_representation(self, text: str, model, tokenizer, layer_type: iit_layer_type_enum) -> tuple[np.ndarray, float, float]:
        """Return hidden states from the given model (no gradients)."""
        # model.eval()  
        with torch.no_grad():
            inputs = tokenizer(text, return_tensors='pt').to(model.device)
            outputs = model(**inputs, labels=inputs["input_ids"], output_hidden_states=True)
            hidden = torch.cat(outputs.hidden_states, dim=0).detach().cpu().float().numpy()
            loss = outputs.loss.cpu()
            logits = outputs.logits.squeeze(0).detach().float()
            entropy = my_utils.calculate_entropy(logits)
            
            del outputs, inputs, logits
            torch.cuda.empty_cache()
            
            if iit_layer_type_enum.SOME == layer_type: 
                sampled_layers = np.linspace(1, hidden.shape[0]-1, num=12)
                sampled_layers = [int(x) for x in np.round(sampled_layers).tolist()]
                layer_2_3 = int(round((hidden.shape[0]-1) * 2 / 3))
                if layer_2_3 not in sampled_layers:
                    sampled_layers.append(layer_2_3)
                sampled_layers = sorted(set(sampled_layers))
                filtered_hidden = hidden[sampled_layers, :, :].copy()
                del hidden
                return filtered_hidden, loss.item(), entropy
            
            elif iit_layer_type_enum.ALL == layer_type: 
                return hidden, loss.item(), entropy

            elif iit_layer_type_enum.LAST == layer_type: 
                filtered_hidden = hidden[-1, :, :].copy()
                del hidden
                return filtered_hidden, loss.item(), entropy

        return None, None, None

    def extract_representation_last_layer(self, text, model, tokenizer):
        """Return hidden states from the given model (no gradients)."""
        # model.eval()  
        # with torch.no_grad():
        inputs = tokenizer(text, return_tensors='pt').to(model.device)
        outputs = model(**inputs, labels=inputs["input_ids"], output_hidden_states=True)
        hidden = torch.cat(outputs.hidden_states, dim=0).detach().cpu().float().numpy()
        loss = outputs.loss.cpu()
        
        del outputs, inputs
        torch.cuda.empty_cache()
        
        # sample layers (you can tune this for speed)
        return np.mean(hidden[hidden.shape[0]-1], axis=0), loss

    def calculate_whitebox_confidence(self, text: str, model, tokenizer) -> tuple[float, float, float]:
        device = next(model.parameters()).device

        inputs = tokenizer(text, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        input_ids = inputs["input_ids"]

        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)

            logits = outputs.logits  # (batch, seq_len, vocab)
            loss = outputs.loss

        # shift for next-token prediction
        shift_logits = logits[:, :-1, :]
        shift_labels = input_ids[:, 1:]

        log_probs = F.log_softmax(shift_logits, dim=-1)
        probs = torch.exp(log_probs)

        token_probs = []
        token_entropy = []

        for t in range(shift_labels.shape[1]):
            true_token_id = shift_labels[0, t].item()

            prob = probs[0, t, true_token_id].item()
            token_probs.append(prob)

            # entropy = -sum p log p
            entropy = -(probs[0, t] * log_probs[0, t]).sum().item()
            token_entropy.append(entropy)

        sum_probs = sum(token_probs) 
        avg_probs = sum_probs / len(token_probs)
        avg_entropy = sum(token_entropy) / len(token_entropy)
            
        del outputs, inputs, input_ids, logits
        torch.cuda.empty_cache()
       
        return loss.item(), sum_probs, avg_probs, avg_entropy

    def calculate_entropy(self, text, model, tokenizer):
        with torch.no_grad():
            inputs = tokenizer(text, return_tensors='pt').to(model.device)
            outputs = model(**inputs, labels=inputs["input_ids"])
            logits = outputs.logits.squeeze(0).detach().float()
            entropy = my_utils.calculate_entropy(logits)
            loss = my_utils.tensor_tostring(outputs.loss.cpu())
            perplexity = my_utils.calculate_perplexity(outputs.loss.cpu())
            

        del outputs, inputs, logits
        torch.cuda.empty_cache()
        gc.collect()
        return entropy, loss, perplexity

    def compute_conditional_loss(self, model, tokenizer, context, target):
        # Combine context + target
        full_text = context + target

        # Tokenize
        inputs = tokenizer(full_text, return_tensors="pt").to(model.device)
        input_ids = inputs["input_ids"]

        # Create labels
        labels = input_ids.clone()

        # Mask out context tokens (we only compute loss on target)
        context_ids = tokenizer(context, return_tensors="pt")["input_ids"]
        context_length = context_ids.shape[1]

        labels[:, :context_length] = -100  # ignore context

        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss

        return loss
    
    def clean_prompt_for_phi(self, raw_prompt: str) -> str:
        """
        1) Remove the <|im_start|>system ... <|im_end|> block.
        2) Extract the last 'User:' segment content.
        3) Drop boilerplate ('A conversation between ...').
        4) Remove any leftover <think>...</think> from the prompt (if present).
        5) Trim whitespace.
        """
        SYSTEM_BLOCK_RE = re.compile(
            r"<\|im_start\|>\s*system\b.*?<\|im_end\|>\s*", re.DOTALL | re.IGNORECASE
        )
        GENERIC_BLOCK_RE = re.compile(
            r"<\|im_start\|>.*?<\|im_end\|>\s*", re.DOTALL | re.IGNORECASE
        )

        p = SYSTEM_BLOCK_RE.sub("", raw_prompt)

        m = list(re.finditer(
            r"User:\s*(.*?)(?=\n<\|im_start\|>|\nAssistant:|$)",
            p,
            flags=re.DOTALL | re.IGNORECASE
        ))
        if m:
            text = m[-1].group(1)
        else:
            
            text = GENERIC_BLOCK_RE.sub("", p)

        # Remove boilerplate lines that sometimes appear before/around "User:"
        text = re.sub(
            r"^A conversation between.*?User:\s*",
            "",
            text,
            flags=re.DOTALL | re.IGNORECASE
        )

        # Remove any <think>...</think> that may be embedded in the prompt text
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)

        return text.strip()


