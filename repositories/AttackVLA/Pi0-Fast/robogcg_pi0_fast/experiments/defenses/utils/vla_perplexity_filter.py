import torch
import numpy as np

"""
Adapted from:
  https://github.com/neelsjain/baseline-defenses/blob/main/perplexity_filter.py
as cited in

@misc{jain2023baselinedefensesadversarialattacks,
  title={Baseline Defenses for Adversarial Attacks Against Aligned Language Models},
  author={Neel Jain and Avi Schwarzschild and Yuxin Wen and Gowthami Somepalli
          and John Kirchenbauer and Ping-yeh Chiang and Micah Goldblum
          and Aniruddha Saha and Jonas Geiping and Tom Goldstein},
  year={2023},
  eprint={2309.00614},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2309.00614},
}

Modified to work with OpenVLA (or other multimodal models) without manually
concatenating image tokens. The model's forward() handles image tokens internally.
"""

class VLAPerplexityFilter:
    """
    Filter sequences based on perplexity (negative log-likelihood of the text portion).

    Parameters
    ----------
    model : OpenVLAForActionPrediction (or a similar HF model)
        The multimodal model used for perplexity calculation.
    processor : AutoProcessor
        The tokenizer/processor for text & image. Must produce "input_ids" for the text
        and "pixel_values" for the image if the model uses them. The model’s forward()
        typically handles image tokens internally (by marking them as -100 in the labels).
    threshold : float
        Threshold for average NLL. If a prompt’s average NLL <= threshold, it "passes."
    window_size : int
        Window size for the optional filter_window approach.
    device : str
        Typically "cuda" or "cpu."
    """
    def __init__(self, model, processor, threshold, window_size=10, device="cuda"):
        self.model = model.to(device)
        self.processor = processor
        self.threshold = threshold
        self.window_size = window_size
        self.cn_loss = torch.nn.CrossEntropyLoss(reduction="none")

    def get_log_perplexity(self, prompt, img):
        """
        Get a single scalar cross-entropy loss (i.e. perplexity) for one prompt+image
        by letting the model’s forward() do all label processing.

        The model automatically inserts image tokens as needed and sets them to -100,
        so the returned loss is over text tokens only.

        Returns
        -------
        float
            The mean cross-entropy across the text tokens.
        """
        inputs = self.processor(prompt, img, return_tensors="pt")
        inputs = inputs.to(self.model.device, dtype=torch.bfloat16)

        with torch.no_grad():
            outputs = self.model(**inputs, labels=inputs["input_ids"])
            # outputs.loss is a scalar averaged over the text tokens
            loss_val = outputs.loss.item()

        return loss_val

    def get_max_log_perplexity_of_goals(self, prompts, imgs):
        """
        Compute the maximum cross-entropy among multiple (prompt, image) pairs.

        Returns
        -------
        float : The highest cross-entropy loss among the given prompts/images.
        """
        all_loss = []
        for prompt, img in zip(prompts, imgs):
            inputs = self.processor(prompt, img, return_tensors="pt")
            inputs = inputs.to(self.model.device, dtype=torch.bfloat16)

            with torch.no_grad():
                out = self.model(**inputs, labels=inputs["input_ids"])
                all_loss.append(out.loss.item())

        return max(all_loss)

    def get_max_win_log_ppl_of_goals(self, prompts, imgs):
        """
        Placeholder for a "window-based" approach across multiple prompts.
        Currently just the max cross-entropy, but you can adapt it to do actual
        window slicing if needed.
        """
        all_loss = []
        for prompt, img in zip(prompts, imgs):
            inputs = self.processor(prompt, img, return_tensors="pt")
            inputs = inputs.to(self.model.device, dtype=torch.bfloat16)

            with torch.no_grad():
                out = self.model(**inputs, labels=inputs["input_ids"])
                all_loss.append(out.loss.item())

        return max(all_loss)

    def get_log_prob(self, prompt, img):
        """
        Return the per-token negative log-likelihood (NLL) for the text portion only.

        Implementation:
          1) We encode prompt+image to get input_ids (shape [1, text_len]) + pixel_values.
          2) We run model(**inputs) WITHOUT ignoring or rewriting labels ourselves. Instead,
             we do NOT pass "labels=..." so the model won't automatically do the cross-entropy.
          3) We get logits with shape [1, text_len + image_tokens, vocab_size].
          4) We measure cross-entropy only on the text positions (standard shift approach).
             => shift_logits = logits[:, :text_len-1, :]
             => shift_labels = input_ids[:, 1:]
          5) Return the 1D NLL for each of those text tokens.

        Returns
        -------
        torch.Tensor : shape [ text_len - 1 ] with the NLL for each text token (excl. first).
        """
        # 1) Encode text+image
        inputs = self.processor(prompt, img, return_tensors="pt")
        inputs = inputs.to(self.model.device, dtype=torch.bfloat16)
        input_ids = inputs["input_ids"]  # shape [1, text_len]

        # 2) Forward pass without "labels=", so we get raw logits. The model might
        #    internally insert image tokens, but won't average a final loss automatically.
        with torch.no_grad():
            out = self.model(**inputs)
            logits = out.logits  # shape [1, text_len + image_tokens, vocab_size]

        # 3) Restrict ourselves to the text portion for perplexity analysis.
        #    text_len = input_ids.shape[1]
        #    So we shift the logits up to text_len - 1 tokens:
        text_len = input_ids.shape[1]
        # logits[:, :text_len, :]. 
        # Then SHIFT: comparing logits[:, 0..text_len-2] to target_ids[:, 1..text_len-1].
        # Actually we'll do:
        shift_logits = logits[:, :text_len-1, :].contiguous()  # shape [1, text_len-1, vocab_size]
        shift_labels = input_ids[:, 1:].contiguous()           # shape [1, text_len-1]

        # 4) Flatten & compute cross-entropy
        log_probs = self.cn_loss(
            shift_logits.view(-1, shift_logits.size(-1)),  # e.g. [ (text_len-1), vocab_size ]
            shift_labels.view(-1)                          # e.g. [ (text_len-1) ]
        )
        # => shape [ (text_len-1) ], which we can interpret as per-token NLL.

        return log_probs

    def filter(self, prompts, imgs):
        """
        Evaluate the mean NLL (negative log-likelihood) across each prompt’s text tokens
        and compare it to self.threshold. If mean_NLL <= threshold => pass.

        Returns
        -------
        filtered_log_ppl : list[float]
            The average NLL for each prompt’s text tokens.
        passed_filter : list[bool]
            True/False depending on whether average NLL <= threshold.
        """
        filtered_log_ppl = []
        passed_filter = []

        for prompt, img in zip(prompts, imgs):
            # Get per-token NLL
            log_probs = self.get_log_prob(prompt, img)
            avg_nll = log_probs.mean().item()

            filtered_log_ppl.append(avg_nll)
            passed_filter.append(avg_nll <= self.threshold)

        return filtered_log_ppl, passed_filter

    def filter_window(self, prompts, imgs, reverse=False):
        """
        A "windowed" version of filtering that slices the text tokens into smaller chunks.

        If reverse=False, chunk from the beginning forward.
        If reverse=True, chunk from the end backward.

        Returns
        -------
        filtered_log_ppl_by_window : list[list[float]]
            Outer list for each prompt, inner list for window means.
        passed_filter_by_window : list[list[bool]]
            Outer list for each prompt, booleans for each window passing threshold.
        passed : list[bool]
            Overall pass/fail for each prompt (True if all windows pass).
        """
        filtered_log_ppl_by_window = []
        passed_filter_by_window = []
        passed = []

        for prompt, img in zip(prompts, imgs):
            # 1) Get the per-token NLL for just the text portion
            log_probs = self.get_log_prob(prompt, img)
            NLL_by_token = log_probs

            window_scores = []
            window_pass = []

            i = 0
            while i < len(NLL_by_token):
                if not reverse:
                    window = NLL_by_token[i : i + self.window_size]
                else:
                    # If reversing from the end
                    if i == 0:
                        window = NLL_by_token[-self.window_size:]
                    elif -(-i - self.window_size) > len(NLL_by_token) and i != 0:
                        window = NLL_by_token[:-i]
                    else:
                        window = NLL_by_token[-i - self.window_size : -i]
                i += self.window_size

                mean_nll = window.mean().item()
                window_scores.append(mean_nll)
                window_pass.append(mean_nll <= self.threshold)

            filtered_log_ppl_by_window.append(window_scores)
            passed_filter_by_window.append(window_pass)
            passed.append(all(window_pass))

        return filtered_log_ppl_by_window, passed_filter_by_window, passed