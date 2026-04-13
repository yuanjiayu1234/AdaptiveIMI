import torch


class SamplingMixin:
    def sampling(self, logits, do_sample=False, temperature=0.6, top_p=0.95, top_k=20):
        if not do_sample:
            output_ids = logits.argmax(dim=-1)  # [bsz, 1], torch.int64
        else:
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1, dtype=torch.float32)  # [bsz, 1, vocab_size]
            probs = probs.squeeze(1) # [bsz, vocab_size]
            if top_k != 0:
                output_ids = flashinfer.sampling.top_k_top_p_sampling_from_probs(probs, top_p=top_p, top_k=top_k)
            else:
                output_ids = flashinfer.sampling.top_p_sampling_from_probs(probs, top_p=top_p)
            output_ids = output_ids.unsqueeze(1) # [bsz, 1], torch.int32

        return output_ids
    def _collect_stop_token_ids(self):
        tokenizer = getattr(self, "tokenizer", None)
        if tokenizer is None:
            return []
        stop_token_ids = []
        eos_token_id = tokenizer.eos_token_id
        if isinstance(eos_token_id, (list, tuple)):
            for token_id in eos_token_id:
                if token_id is not None:
                    stop_token_ids.append(token_id)
        elif eos_token_id is not None:
            stop_token_ids.append(eos_token_id)
        for token in ("<|eot_id|>", "<|end_of_turn|>", "<|eom_id|>"):
            token_id = tokenizer.convert_tokens_to_ids(token)
            if token_id is None:
                continue
            if tokenizer.unk_token_id is not None and token_id == tokenizer.unk_token_id:
                continue
            if token_id not in stop_token_ids:
                stop_token_ids.append(token_id)
        return stop_token_ids
