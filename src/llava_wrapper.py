import inspect
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration


@dataclass
class PreparedInput:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor
    prompt_ids: torch.Tensor
    image_positions: torch.Tensor
    image_features: torch.Tensor
    pixel_values: torch.Tensor
    inputs_embeds_full: torch.Tensor


class LlavaRetransWrapper:
    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        cache_dir: Optional[str] = None,
        local_files_only: bool = False,
        device_map: Optional[str] = None,
    ):
        self.device = device
        self.dtype = dtype
        self.cache_dir = cache_dir
        self.local_files_only = local_files_only
        self.device_map = device_map

        self.processor = AutoProcessor.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
        model_kwargs = {
            "cache_dir": cache_dir,
            "local_files_only": local_files_only,
            "torch_dtype": dtype,
            "low_cpu_mem_usage": True,
        }
        if device_map:
            model_kwargs["device_map"] = device_map

        self.model = LlavaForConditionalGeneration.from_pretrained(model_name, **model_kwargs)
        if not device_map:
            self.model = self.model.to(device)
        self.model.eval()

        for p in self.model.parameters():
            p.requires_grad_(False)

        self.image_token_id = self.model.config.image_token_index
        self.image_token = getattr(self.processor, "image_token", "<image>")
        self.input_device = self.model.get_input_embeddings().weight.device
        self.vision_device = self._module_device(getattr(self.model, "vision_tower", self.model))

    def _module_device(self, module: torch.nn.Module) -> torch.device:
        for parameter in module.parameters(recurse=True):
            if parameter.device.type != "meta":
                return parameter.device
        return self.input_device

    def _language_model(self) -> torch.nn.Module:
        if hasattr(self.model, "language_model"):
            return self.model.language_model
        if hasattr(self.model, "model") and hasattr(self.model.model, "language_model"):
            return self.model.model.language_model
        raise AttributeError("Could not locate the wrapped language model.")

    def _language_layers(self):
        language_model = self._language_model()
        if hasattr(language_model, "model") and hasattr(language_model.model, "layers"):
            return language_model.model.layers
        if hasattr(language_model, "layers"):
            return language_model.layers
        raise AttributeError("Could not locate language model layers.")

    def build_prompt(self, question: str) -> str:
        return f"USER: {self.image_token}\n{question}\nASSISTANT:"

    def build_full_text(self, question: str, answer: str) -> str:
        return f"USER: {self.image_token}\n{question}\nASSISTANT: {answer}"

    def _processor_to_device(self, batch):
        return {
            key: value.to(self.device) if hasattr(value, "to") else value
            for key, value in batch.items()
        }

    def _tokenize_and_expand_image_tokens(
        self,
        text: str,
        num_image_tokens: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.processor.tokenizer(text, return_tensors="pt")
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]

        expanded_ids = []
        expanded_masks = []
        for row_ids, row_mask in zip(input_ids, attention_mask):
            positions = (row_ids == self.image_token_id).nonzero(as_tuple=False).squeeze(-1)
            if positions.numel() != 1:
                raise RuntimeError(
                    f"Expected exactly one image token in prompt, got {positions.numel()}."
                )
            pos = int(positions.item())
            repeated = row_ids.new_full((num_image_tokens,), self.image_token_id)
            repeated_mask = row_mask.new_ones((num_image_tokens,))
            expanded_ids.append(torch.cat([row_ids[:pos], repeated, row_ids[pos + 1 :]], dim=0))
            expanded_masks.append(
                torch.cat([row_mask[:pos], repeated_mask, row_mask[pos + 1 :]], dim=0)
            )

        return torch.stack(expanded_ids, dim=0), torch.stack(expanded_masks, dim=0)

    def _get_pixel_values(self, image: Image.Image) -> torch.Tensor:
        image_inputs = self.processor.image_processor(images=image, return_tensors="pt")
        pixel_values = image_inputs["pixel_values"].to(self.vision_device, dtype=self.dtype)
        return pixel_values

    def _get_image_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        kwargs = {"pixel_values": pixel_values}
        signature = inspect.signature(self.model.get_image_features)
        if "vision_feature_layer" in signature.parameters:
            kwargs["vision_feature_layer"] = getattr(self.model.config, "vision_feature_layer", None)
        if "vision_feature_select_strategy" in signature.parameters:
            kwargs["vision_feature_select_strategy"] = getattr(
                self.model.config,
                "vision_feature_select_strategy",
                None,
            )
        if "image_sizes" in signature.parameters:
            kwargs["image_sizes"] = None

        image_features = self.model.get_image_features(**kwargs)
        if hasattr(image_features, "pooler_output"):
            image_features = image_features.pooler_output
        if isinstance(image_features, tuple):
            image_features = image_features[0]
        if isinstance(image_features, list):
            image_features = image_features[0]
        image_features = image_features.to(dtype=self.dtype)
        if image_features.dim() == 2:
            image_features = image_features.unsqueeze(0)
        return image_features.to(self.input_device)

    @torch.no_grad()
    def prepare_inputs(self, image_path: str, question: str, answer: str) -> PreparedInput:
        image = Image.open(image_path).convert("RGB")

        pixel_values = self._get_pixel_values(image)
        image_features = self._get_image_features(pixel_values)
        num_image_tokens = image_features.shape[1]

        prompt = self.build_prompt(question)
        full_text = self.build_full_text(question, answer)

        prompt_ids, _ = self._tokenize_and_expand_image_tokens(prompt, num_image_tokens)
        input_ids, attention_mask = self._tokenize_and_expand_image_tokens(
            full_text,
            num_image_tokens,
        )
        prompt_ids = prompt_ids.to(self.input_device)
        input_ids = input_ids.to(self.input_device)
        attention_mask = attention_mask.to(self.input_device)

        prompt_len = prompt_ids.shape[1]
        image_positions = (input_ids[0] == self.image_token_id).nonzero(as_tuple=False).squeeze(-1)

        if image_positions.numel() != num_image_tokens:
            raise RuntimeError(
                f"Image placeholder count mismatch. input has {image_positions.numel()} "
                f"image token positions, but get_image_features returns {num_image_tokens} tokens."
            )

        labels = input_ids.clone()
        labels[:, :prompt_len] = -100
        labels[:, image_positions] = -100
        labels[attention_mask == 0] = -100

        if labels.ne(-100).sum().item() == 0:
            raise RuntimeError("No answer tokens left after label masking.")

        inputs_embeds_full = self.input_ids_to_embeds(input_ids)
        inputs_embeds_full[:, image_positions, :] = image_features[0]

        return PreparedInput(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            prompt_ids=prompt_ids,
            image_positions=image_positions,
            image_features=image_features,
            pixel_values=pixel_values,
            inputs_embeds_full=inputs_embeds_full,
        )

    def input_ids_to_embeds(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Replace image-token ids before embedding lookup to avoid out-of-range ids.
        These positions are overwritten by image features immediately.
        """
        safe_ids = input_ids.to(self.input_device).clone()
        image_mask = safe_ids == self.image_token_id

        pad_id = self.processor.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.processor.tokenizer.eos_token_id

        safe_ids[image_mask] = pad_id
        return self.model.get_input_embeddings()(safe_ids).to(dtype=self.dtype)

    def make_embeds_with_image_features(
        self,
        prepared: PreparedInput,
        image_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        image_features: [1, N, D]
        """
        embeds = prepared.inputs_embeds_full.clone()
        embeds[:, prepared.image_positions, :] = image_features[0]
        return embeds

    @torch.no_grad()
    def debug_prepared_input(self, prepared: PreparedInput) -> dict[str, float | bool | str]:
        """
        Print and return label-mask / image-replacement sanity checks.
        Intended for smoke tests, not the training path.
        """
        prompt_ids = prepared.prompt_ids[0].to(prepared.input_ids.device)
        full_prefix = prepared.input_ids[0, : prompt_ids.numel()]
        prefix_equal = torch.equal(prompt_ids, full_prefix)

        print("prefix equal:", prefix_equal)
        if not prefix_equal:
            print("prompt decoded:")
            print(self.processor.tokenizer.decode(prompt_ids.detach().cpu()))
            print("full prefix decoded:")
            print(self.processor.tokenizer.decode(full_prefix.detach().cpu()))

        valid_label_pos = (prepared.labels[0] != -100).nonzero(as_tuple=False).squeeze(-1)
        first_valid_pos = int(valid_label_pos[0].item())
        first_ids = prepared.labels[0, valid_label_pos[:10]]
        first_supervised = self.processor.tokenizer.decode(first_ids.detach().cpu())

        print("first valid label position:", first_valid_pos)
        print("first supervised tokens:")
        print(first_supervised)

        embeds = self.make_embeds_with_image_features(prepared, prepared.image_features)
        replacement_diff = (
            embeds[0, prepared.image_positions, :] - prepared.image_features[0]
        ).abs().max()
        replacement_diff_value = float(replacement_diff.item())
        print("image replacement max diff:", replacement_diff_value)

        return {
            "prefix_equal": prefix_equal,
            "first_valid_label_position": first_valid_pos,
            "first_supervised_tokens": first_supervised,
            "image_replacement_max_diff": replacement_diff_value,
        }

    @torch.no_grad()
    def debug_compare_native_and_manual(self, prepared: PreparedInput) -> dict[str, float]:
        """
        Compare the manual inputs_embeds path against the native pixel_values path.
        A large diff means the manual visual-token replacement path is not equivalent.
        """
        manual_loss = self.compute_loss_from_image_features(prepared, prepared.image_features)
        native_outputs = self.model(
            input_ids=prepared.input_ids,
            pixel_values=prepared.pixel_values,
            attention_mask=prepared.attention_mask.to(prepared.input_ids.device),
            labels=prepared.labels.to(prepared.input_ids.device),
            use_cache=False,
            return_dict=True,
        )

        native_loss = getattr(native_outputs, "loss", None)
        if native_loss is None:
            native_loss = self._loss_from_logits(native_outputs.logits, prepared.labels)[0]
        native_loss = native_loss.detach()
        abs_diff = abs(float(manual_loss.item()) - float(native_loss.item()))

        print("manual full loss:", float(manual_loss.item()))
        print("native full loss:", float(native_loss.item()))
        print("abs diff:", abs_diff)

        return {
            "manual_full_loss": float(manual_loss.item()),
            "native_full_loss": float(native_loss.item()),
            "abs_diff": abs_diff,
        }

    @torch.no_grad()
    def compute_loss_from_image_features(
        self,
        prepared: PreparedInput,
        image_features: torch.Tensor,
    ) -> torch.Tensor:
        embeds = self.make_embeds_with_image_features(prepared, image_features)

        outputs = self.model(
            inputs_embeds=embeds,
            attention_mask=prepared.attention_mask.to(embeds.device),
            use_cache=False,
            return_dict=True,
        )
        return self._loss_from_logits(outputs.logits, prepared.labels).detach()[0]

    def _loss_from_logits(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        labels = labels.to(logits.device)
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        loss_per_token = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
            reduction="none",
        ).view(shift_labels.shape)

        valid = shift_labels.ne(-100)
        return loss_per_token.sum(dim=1) / valid.sum(dim=1).clamp_min(1)

    def _answer_shift_logits(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        labels = labels.to(logits.device)
        valid = labels[0, 1:].ne(-100)
        if valid.sum().item() == 0:
            raise RuntimeError("No answer-token positions available for teacher KL.")
        return logits[:, :-1, :][:, valid, :].contiguous()

    def _teacher_kl_from_logits(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        teacher_log_probs: torch.Tensor,
    ) -> torch.Tensor:
        """
        D_KL(p_clean || p_current) averaged over supervised answer-token positions.
        teacher_log_probs: [T, vocab], from the clean/full-image forward pass.
        """
        answer_logits = self._answer_shift_logits(logits, labels).float()
        current_log_probs = torch.log_softmax(answer_logits, dim=-1)
        teacher_log_probs = teacher_log_probs.to(
            device=current_log_probs.device,
            dtype=torch.float32,
        )
        if teacher_log_probs.shape != current_log_probs.shape[1:]:
            raise RuntimeError(
                "Teacher distribution shape mismatch: "
                f"teacher={tuple(teacher_log_probs.shape)}, "
                f"current={tuple(current_log_probs.shape[1:])}"
            )

        teacher_probs = teacher_log_probs.exp()
        kl_per_token = (teacher_probs.unsqueeze(0) * (
            teacher_log_probs.unsqueeze(0) - current_log_probs
        )).sum(dim=-1)
        return kl_per_token.mean(dim=1)

    @torch.no_grad()
    def compute_teacher_log_probs_from_image_features(
        self,
        prepared: PreparedInput,
        image_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Return clean teacher next-token log-probabilities at answer-token positions.
        shape: [num_answer_tokens, vocab]
        """
        embeds = self.make_embeds_with_image_features(prepared, image_features)
        outputs = self.model(
            inputs_embeds=embeds,
            attention_mask=prepared.attention_mask.to(embeds.device),
            use_cache=False,
            return_dict=True,
        )
        answer_logits = self._answer_shift_logits(outputs.logits, prepared.labels)
        return torch.log_softmax(answer_logits[0].float(), dim=-1).detach()

    @torch.no_grad()
    def compute_teacher_kl_from_image_features(
        self,
        prepared: PreparedInput,
        teacher_log_probs: torch.Tensor,
        image_features: torch.Tensor,
    ) -> torch.Tensor:
        embeds = self.make_embeds_with_image_features(prepared, image_features)
        outputs = self.model(
            inputs_embeds=embeds,
            attention_mask=prepared.attention_mask.to(embeds.device),
            use_cache=False,
            return_dict=True,
        )
        return self._teacher_kl_from_logits(
            outputs.logits,
            prepared.labels,
            teacher_log_probs,
        ).detach()[0]

    @torch.no_grad()
    def compute_loss_batch_from_embeds(
        self,
        embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        embeds: [B, S, D]
        labels: [B, S]
        Returns per-sample CE loss, not averaged across batch.
        """
        outputs = self.model(
            inputs_embeds=embeds,
            attention_mask=attention_mask.to(embeds.device),
            use_cache=False,
            return_dict=True,
        )

        return self._loss_from_logits(outputs.logits, labels).detach()

    @torch.no_grad()
    def compute_teacher_kl_batch_from_embeds(
        self,
        embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        teacher_log_probs: torch.Tensor,
    ) -> torch.Tensor:
        """
        embeds: [B, S, D]
        Returns per-sample D_KL(p_clean || p_current), averaged over answer tokens.
        """
        outputs = self.model(
            inputs_embeds=embeds,
            attention_mask=attention_mask.to(embeds.device),
            use_cache=False,
            return_dict=True,
        )
        return self._teacher_kl_from_logits(
            outputs.logits,
            labels,
            teacher_log_probs,
        ).detach()

    @torch.no_grad()
    def get_layer_visual_hidden(
        self,
        prepared: PreparedInput,
        image_features: torch.Tensor,
        layer_idx: int,
    ) -> torch.Tensor:
        """
        Returns visual hidden states at language layer layer_idx.
        shape: [N, D]
        """
        cache = {}

        def hook_fn(module, inputs, outputs):
            hidden = outputs[0] if isinstance(outputs, tuple) else outputs
            cache["hidden"] = hidden.detach()

        layer = self._language_layers()[layer_idx]
        handle = layer.register_forward_hook(hook_fn)

        embeds = self.make_embeds_with_image_features(prepared, image_features)
        try:
            _ = self.model(
                inputs_embeds=embeds,
                attention_mask=prepared.attention_mask.to(embeds.device),
                use_cache=False,
                return_dict=True,
            )
        finally:
            handle.remove()

        if "hidden" not in cache:
            raise RuntimeError(f"Layer hook did not capture hidden state at layer {layer_idx}.")

        hidden = cache["hidden"]
        visual_hidden = hidden[0, prepared.image_positions.to(hidden.device), :]
        return visual_hidden.detach()

    @torch.no_grad()
    def build_candidate_restored_embeds(
        self,
        prepared: PreparedInput,
        corrupted_features: torch.Tensor,
        full_features: torch.Tensor,
        candidate_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        For each candidate i, build one input embedding where only token i is restored.
        Return embeds: [C, S, D]
        """
        candidate_indices = candidate_indices.to(corrupted_features.device)
        count = candidate_indices.numel()
        base_features = corrupted_features.repeat(count, 1, 1)

        row_ids = torch.arange(count, device=candidate_indices.device)
        base_features[row_ids, candidate_indices, :] = full_features[0, candidate_indices, :]

        base_embeds = prepared.inputs_embeds_full.repeat(count, 1, 1)
        base_embeds[:, prepared.image_positions, :] = base_features
        return base_embeds
