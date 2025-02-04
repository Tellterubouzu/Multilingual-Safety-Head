# from transformers import (
#     LlamaModel,
#     LlamaForCausalLM,
#     MistralModel,
#     MistralForCausalLM
# )

# from transformers.models.llama.modeling_llama import (
#     LlamaAttention,
#     LlamaDecoderLayer
# )

# from transformers.models.llama.configuration_llama import LlamaConfig

# from transformers.modeling_outputs import (
#     BaseModelOutputWithPast,
#     CausalLMOutputWithPast,
# )

# from transformers import Cache, DynamicCache

# from transformers.utils import (
#     add_start_docstrings_to_model_forward,
#     replace_return_docstrings,
# )
# from transformers.utils import logging

# import math
# import torch
# from torch import nn
# from torch.nn import CrossEntropyLoss
# import torch.nn.functional as F
# from typing import List, Optional, Tuple, Union

# LLAMA_INPUTS_DOCSTRING = r"""
#     Args:
#         input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
#             Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
#             it.

#             Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
#             [`PreTrainedTokenizer.__call__`] for details.

#             [What are input IDs?](../glossary#input-ids)
#         attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
#             Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

#             - 1 for tokens that are **not masked**,
#             - 0 for tokens that are **masked**.

#             [What are attention masks?](../glossary#attention-mask)

#             Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
#             [`PreTrainedTokenizer.__call__`] for details.

#             If `past_key_values` is used, optionally only the last `input_ids` have to be input (see
#             `past_key_values`).

#             If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
#             and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
#             information on the default strategy.

#             - 1 indicates the head is **not masked**,
#             - 0 indicates the head is **masked**.
#         position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
#             Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
#             config.n_positions - 1]`.

#             [What are position IDs?](../glossary#position-ids)
#         past_key_values (`Cache` or `tuple(tuple(torch.FloatTensor))`, *optional*):
#             Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
#             blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
#             returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

#             Two formats are allowed:
#             - a [`~cache_utils.Cache`] instance;
#             - Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
#             shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`). This is also known as the legacy
#             cache format.

#             The model will output the same cache format that is fed as input. If no `past_key_values` are passed, the
#             legacy cache format will be returned.

#             If `past_key_values` are used, the user can optionally input only the last `input_ids` (those that don't
#             have their past key value states given to this model) of shape `(batch_size, 1)` instead of all `input_ids`
#             of shape `(batch_size, sequence_length)`.
#         inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
#             Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
#             is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
#             model's internal embedding lookup matrix.
#         use_cache (`bool`, *optional*):
#             If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
#             `past_key_values`).
#         output_attentions (`bool`, *optional*):
#             Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
#             tensors for more detail.
#         output_hidden_states (`bool`, *optional*):
#             Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
#             more detail.
#         return_dict (`bool`, *optional*):
#             Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
#         cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
#             Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
#             this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
#             the complete sequence length.
# """
# logger = logging.get_logger(__name__)
# _CONFIG_FOR_DOC = "LlamaConfig"


# def rotate_half(x):
#     """Rotates half the hidden dims of the input."""
#     x1 = x[..., : x.shape[-1] // 2]
#     x2 = x[..., x.shape[-1] // 2:]
#     return torch.cat((-x2, x1), dim=-1)


# def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
#     """Applies Rotary Position Embedding to the query and key tensors.

#     Args:
#         q (`torch.Tensor`): The query tensor.
#         k (`torch.Tensor`): The key tensor.
#         cos (`torch.Tensor`): The cosine part of the rotary embedding.
#         sin (`torch.Tensor`): The sine part of the rotary embedding.
#         position_ids (`torch.Tensor`, *optional*):
#             Deprecated and unused.
#         unsqueeze_dim (`int`, *optional*, defaults to 1):
#             The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
#             sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
#             that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
#             k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
#             cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
#             the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
#     Returns:
#         `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
#     """
#     cos = cos.unsqueeze(unsqueeze_dim)
#     sin = sin.unsqueeze(unsqueeze_dim)
#     q_embed = (q * cos) + (rotate_half(q) * sin)
#     k_embed = (k * cos) + (rotate_half(k) * sin)
#     return q_embed, k_embed


# def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
#     """
#     This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
#     num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
#     """
#     batch, num_key_value_heads, slen, head_dim = hidden_states.shape
#     if n_rep == 1:
#         return hidden_states
#     hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
#     return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


# class CustomLlamaAttention(LlamaAttention):

#     def forward(
#             self,
#             hidden_states: torch.Tensor,
#             attention_mask: Optional[torch.Tensor] = None,
#             position_ids: Optional[torch.LongTensor] = None,
#             past_key_value: Optional[Cache] = None,
#             output_attentions: bool = False,
#             use_cache: bool = False,
#             cache_position: Optional[torch.LongTensor] = None,
#             head_mask: Optional[dict] = None,
#             mask_type: Optional[str] = None,
#             scale_factor: Optional[float] = None,
#             mask_para: Optional[bool] = None,
#             head_dim: Optional[int] = None,
#             **kwargs,
#     ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
#         """
#         head_mask is a custom parameter, which masks specific head.
#         For example:
#         head_mask = {
#             (3, 5): ['q', 'k', 'v'], # mask q, k and v for head 5 in layer 3
#             (25, 0): ['q'], # only mask q for head 0 in layer 25
#         }
#         mask_type is custom parameter, which control mask type when mask specific head.
#         We provide scale_mask and mean_mask. and scale mask is multiplied by 0 by default.
#         """
#         if mask_type not in ['scale_mask', 'mean_mask'] and head_mask is not None:
#             raise ValueError('please provide mask in ["scale_mask", "mean_mask"]')
#         elif mask_type == 'scale_mask' and scale_factor is None:
#             scale_factor = 0.0

#         bsz, q_len, _ = hidden_states.size()

#         if head_mask is not None and mask_para is True:
#             for head_info, qkv_list in head_mask.items():
#                 if head_info[0] == self.layer_idx:
#                     head_idx = head_info[1]
#                     for qkv in qkv_list:
#                         assert head_dim is not None, "head_dim is None!"
#                         start_index = head_idx * head_dim
#                         end_index = start_index + head_dim
#                         if qkv == "q":
#                             if mask_type == 'scale_mask':
#                                 self.q_proj.weight.data[start_index:end_index, :] *= scale_factor
#                             elif mask_type == 'mean_mask':
#                                 self.q_proj.weight.data[start_index:end_index, :] = self.q_proj.weight.data.view(self.num_heads, self.head_dim, -1).mean(dim=0,keepdim=False)
#                         elif qkv == 'k':
#                             if mask_type == 'scale_mask':
#                                 self.k_proj.weight.data[start_index:end_index, :] *= scale_factor
#                             elif mask_type == 'mean_mask':
#                                 self.k_proj.weight.data[start_index:end_index, :] = self.k_proj.weight.data.view(self.num_heads, self.head_dim, -1).mean(dim=0,keepdim=False)
#                         elif qkv == 'v':
#                             if mask_type == 'scale_mask':
#                                 self.v_proj.weight.data[start_index:end_index, :] *= scale_factor
#                             elif mask_type == 'mean_mask':
#                                 self.v_proj.weight.data[start_index:end_index, :] = self.v_proj.weight.data.view(self.num_heads, self.head_dim, -1).mean(dim=0,keepdim=False)

#         if self.config.pretraining_tp > 1:
#             key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
#             query_slices = self.q_proj.weight.split(
#                 (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
#             )
#             key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
#             value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

#             query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
#             query_states = torch.cat(query_states, dim=-1)

#             key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
#             key_states = torch.cat(key_states, dim=-1)

#             value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
#             value_states = torch.cat(value_states, dim=-1)

#         else:
#             query_states = self.q_proj(hidden_states)
#             key_states = self.k_proj(hidden_states)
#             value_states = self.v_proj(hidden_states)

#         query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
#         key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
#         value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

#         if head_mask is not None and (mask_para is False or mask_para is None):
#             for head_info, qkv_list in head_mask.items():
#                 if head_info[0] == self.layer_idx:
#                     head_idx = head_info[1]
#                     for qkv in qkv_list:
#                         if qkv == "q":
#                             if mask_type == 'scale_mask':
#                                 query_states[:, head_idx, :, :] *= scale_factor
#                             elif mask_type == 'mean_mask':
#                                 query_states[:, head_idx, :, :] = query_states[:, :, :, :].mean(dim=1,keepdim=False)
#                         elif qkv == 'k':
#                             if mask_type == 'scale_mask':
#                                 key_states[:, head_idx, :, :] *= scale_factor
#                             elif mask_type == 'mean_mask':
#                                 key_states[:, head_idx, :, :] = key_states[:, :, :, :].mean(dim=1,keepdim=False)

#         cos, sin = self.rotary_emb(value_states, position_ids)
#         query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

#         if past_key_value is not None:
#             # sin and cos are specific to RoPE models; cache_position needed for the static cache
#             cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
#             key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

#         key_states = repeat_kv(key_states, self.num_key_value_groups)
#         value_states = repeat_kv(value_states, self.num_key_value_groups)

#         attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

#         if attention_mask is not None:  # no matter the length, we just slice it
#             causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
#             attn_weights = attn_weights + causal_mask

#         # upcast attention to fp32
#         attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
#         attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
#         attn_output = torch.matmul(attn_weights, value_states)
#         if head_mask is not None and (mask_para is False or mask_para is None):
#             for head_info, qkv_list in head_mask.items():
#                 if head_info[0] == self.layer_idx:
#                     head_idx = head_info[1]
#                     for qkv in qkv_list:
#                         if qkv == 'v':
#                             if mask_type == 'scale_mask':
#                                 attn_output[:, head_idx, :, :] *= scale_factor
#                             elif mask_type == 'mean_mask':
#                                 attn_output[:, head_idx, :, :] = attn_output[:, :, :, :].mean(dim=1,keepdim=False)

#         if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
#             raise ValueError(
#                 f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
#                 f" {attn_output.size()}"
#             )

#         attn_output = attn_output.transpose(1, 2).contiguous()
#         attn_output = attn_output.reshape(bsz, q_len, -1)

#         if self.config.pretraining_tp > 1:
#             attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
#             o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
#             attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
#         else:
#             attn_output = self.o_proj(attn_output)

#         if not output_attentions:
#             attn_weights = None

#         return attn_output, attn_weights, past_key_value


# class CustomLlamaDecoderLayer(LlamaDecoderLayer):
#     def __init__(self, config, layer_idx: int):
#         super().__init__(config, layer_idx)
#         self.self_attn = CustomLlamaAttention(config=config, layer_idx=layer_idx)

#     def forward(
#             self,
#             hidden_states: torch.Tensor,
#             attention_mask: Optional[torch.Tensor] = None,
#             position_ids: Optional[torch.LongTensor] = None,
#             past_key_value: Optional[Cache] = None,
#             output_attentions: Optional[bool] = False,
#             use_cache: Optional[bool] = False,
#             cache_position: Optional[torch.LongTensor] = None,
#             head_mask: Optional[dict] = None,
#             mask_type: Optional[str] = None,
#             scale_factor: Optional[float] = None,
#             mask_para: Optional[bool] = None,
#             head_dim: Optional[int] = None,
#             **kwargs,
#     ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
#         """
#         Args:
#             hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
#             attention_mask (`torch.FloatTensor`, *optional*):
#                 attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
#                 query_sequence_length, key_sequence_length)` if default attention is used.
#             output_attentions (`bool`, *optional*):
#                 Whether or not to return the attentions tensors of all attention layers. See `attentions` under
#                 returned tensors for more detail.
#             use_cache (`bool`, *optional*):
#                 If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
#                 (see `past_key_values`).
#             past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
#             cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
#                 Indices depicting the position of the input sequence tokens in the sequence
#             kwargs (`dict`, *optional*):
#                 Arbitrary kwargs to be ignored, used for FSDP and other methods that injects code
#                 into the model
#         """
#         residual = hidden_states

#         hidden_states = self.input_layernorm(hidden_states)

#         # Self Attention
#         hidden_states, self_attn_weights, present_key_value = self.self_attn(
#             hidden_states=hidden_states,
#             attention_mask=attention_mask,
#             position_ids=position_ids,
#             past_key_value=past_key_value,
#             output_attentions=output_attentions,
#             use_cache=use_cache,
#             cache_position=cache_position,
#             head_mask=head_mask,
#             mask_type=mask_type,
#             scale_factor=scale_factor,
#             mask_para=mask_para,
#             head_dim=head_dim,
#             **kwargs,
#         )
#         hidden_states = residual + hidden_states

#         # Fully Connected
#         residual = hidden_states
#         hidden_states = self.post_attention_layernorm(hidden_states)
#         hidden_states = self.mlp(hidden_states)
#         hidden_states = residual + hidden_states

#         outputs = (hidden_states,)

#         if output_attentions:
#             outputs += (self_attn_weights,)

#         if use_cache:
#             outputs += (present_key_value,)

#         return outputs


# class CustomLlamaModel(LlamaModel):
#     def __init__(self, config: LlamaConfig):
#         super().__init__(config)
#         self.layers = nn.ModuleList(
#             [CustomLlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
#         )

#     @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
#     def forward(
#             self,
#             input_ids: torch.LongTensor = None,
#             attention_mask: Optional[torch.Tensor] = None,
#             position_ids: Optional[torch.LongTensor] = None,
#             past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
#             inputs_embeds: Optional[torch.FloatTensor] = None,
#             use_cache: Optional[bool] = None,
#             output_attentions: Optional[bool] = None,
#             output_hidden_states: Optional[bool] = None,
#             return_dict: Optional[bool] = None,
#             cache_position: Optional[torch.LongTensor] = None,
#             head_mask: Optional[dict] = None,
#             mask_type: Optional[str] = None,
#             scale_factor: Optional[float] = None,
#             mask_para: Optional[bool] = None,
#             head_dim: Optional[int] = None,
#     ) -> Union[Tuple, BaseModelOutputWithPast]:
#         output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
#         output_hidden_states = (
#             output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
#         )
#         use_cache = use_cache if use_cache is not None else self.config.use_cache
#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict

#         if (input_ids is None) ^ (inputs_embeds is not None):
#             raise ValueError(
#                 "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
#             )

#         if self.gradient_checkpointing and self.training and use_cache:
#             logger.warning_once(
#                 "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
#             )
#             use_cache = False

#         if inputs_embeds is None:
#             inputs_embeds = self.embed_tokens(input_ids)

#         return_legacy_cache = False
#         if use_cache and not isinstance(past_key_values, Cache):  # kept for BC (non `Cache` `past_key_values` inputs)
#             return_legacy_cache = True
#             past_key_values = DynamicCache.from_legacy_cache(past_key_values)

#         if cache_position is None:
#             past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
#             cache_position = torch.arange(
#                 past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
#             )
#         if position_ids is None:
#             position_ids = cache_position.unsqueeze(0)

#         causal_mask = self._update_causal_mask(
#             attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
#         )

#         # embed positions
#         hidden_states = inputs_embeds

#         # decoder layers
#         all_hidden_states = () if output_hidden_states else None
#         all_self_attns = () if output_attentions else None
#         next_decoder_cache = None

#         for decoder_layer in self.layers:
#             if output_hidden_states:
#                 all_hidden_states += (hidden_states,)

#             if self.gradient_checkpointing and self.training:
#                 layer_outputs = self._gradient_checkpointing_func(
#                     decoder_layer.__call__,
#                     hidden_states,
#                     causal_mask,
#                     position_ids,
#                     past_key_values,
#                     output_attentions,
#                     use_cache,
#                     cache_position,
#                 )
#             else:
#                 layer_outputs = decoder_layer(
#                     hidden_states,
#                     attention_mask=causal_mask,
#                     position_ids=position_ids,
#                     past_key_value=past_key_values,
#                     output_attentions=output_attentions,
#                     use_cache=use_cache,
#                     cache_position=cache_position,
#                     head_mask=head_mask,
#                     mask_type=mask_type,
#                     scale_factor=scale_factor,
#                     mask_para=mask_para,
#                     head_dim=head_dim,
#                 )

#             hidden_states = layer_outputs[0]

#             if use_cache:
#                 next_decoder_cache = layer_outputs[2 if output_attentions else 1]

#             if output_attentions:
#                 all_self_attns += (layer_outputs[1],)

#         hidden_states = self.norm(hidden_states)

#         # add hidden states from the last decoder layer
#         if output_hidden_states:
#             all_hidden_states += (hidden_states,)

#         next_cache = next_decoder_cache if use_cache else None
#         if return_legacy_cache:
#             next_cache = next_cache.to_legacy_cache()

#         if not return_dict:
#             return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
#         return BaseModelOutputWithPast(
#             last_hidden_state=hidden_states,
#             past_key_values=next_cache,
#             hidden_states=all_hidden_states,
#             attentions=all_self_attns,
#         )


# class CustomLlamaModelForCausalLM(LlamaForCausalLM):
#     def __init__(self, config: LlamaConfig):
#         super().__init__(config)
#         self.model = CustomLlamaModel(config)

#     @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
#     @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
#     def forward(
#             self,
#             input_ids: torch.LongTensor = None,
#             attention_mask: Optional[torch.Tensor] = None,
#             position_ids: Optional[torch.LongTensor] = None,
#             past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
#             inputs_embeds: Optional[torch.FloatTensor] = None,
#             labels: Optional[torch.LongTensor] = None,
#             use_cache: Optional[bool] = None,
#             output_attentions: Optional[bool] = None,
#             output_hidden_states: Optional[bool] = None,
#             return_dict: Optional[bool] = None,
#             cache_position: Optional[torch.LongTensor] = None,
#             head_mask: Optional[dict] = None,
#             mask_type: Optional[str] = None,
#             scale_factor: Optional[float] = None,
#             mask_para: Optional[bool] = None,
#             head_dim: Optional[int] = None,
#     ) -> Union[Tuple, CausalLMOutputWithPast]:
#         r"""
#         Args:
#             labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
#                 Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
#                 config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
#                 (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

#         Returns:

#         Example:

#         ```python
#         >>> from transformers import AutoTokenizer, LlamaForCausalLM

#         >>> model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
#         >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

#         >>> prompt = "Hey, are you conscious? Can you talk to me?"
#         >>> inputs = tokenizer(prompt, return_tensors="pt")

#         >>> # Generate
#         >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
#         >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
#         "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
#         ```"""
#         output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
#         output_hidden_states = (
#             output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
#         )
#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict

#         # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
#         outputs = self.model(
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#             position_ids=position_ids,
#             past_key_values=past_key_values,
#             inputs_embeds=inputs_embeds,
#             use_cache=use_cache,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
#             cache_position=cache_position,
#             head_mask=head_mask,
#             mask_type=mask_type,
#             scale_factor=scale_factor,
#             mask_para=mask_para,
#             head_dim=head_dim,
#         )

#         hidden_states = outputs[0]
#         if self.config.pretraining_tp > 1:
#             lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
#             logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
#             logits = torch.cat(logits, dim=-1)
#         else:
#             logits = self.lm_head(hidden_states)
#         logits = logits.float()

#         loss = None
#         if labels is not None:
#             # Shift so that tokens < n predict n
#             shift_logits = logits[..., :-1, :].contiguous()
#             shift_labels = labels[..., 1:].contiguous()
#             # Flatten the tokens
#             loss_fct = CrossEntropyLoss()
#             shift_logits = shift_logits.view(-1, self.config.vocab_size)
#             shift_labels = shift_labels.view(-1)
#             # Enable model parallelism
#             shift_labels = shift_labels.to(shift_logits.device)
#             loss = loss_fct(shift_logits, shift_labels)

#         if not return_dict:
#             output = (logits,) + outputs[1:]
#             return (loss,) + output if loss is not None else output

#         return CausalLMOutputWithPast(
#             loss=loss,
#             logits=logits,
#             past_key_values=outputs.past_key_values,
#             hidden_states=outputs.hidden_states,
#             attentions=outputs.attentions,
#         )


# class CustomMistralModel(MistralModel):
#     def __init__(self):
#         super().__init__()


# class CustomMistralModelForCausalLM(CustomLlamaModelForCausalLM):

#     def __init__(self, config):
#         super().__init__(config)
#         self.config = config



from transformers import (
    LlamaModel,
    LlamaForCausalLM,
)

from transformers.models.llama.modeling_llama import (
    LlamaDecoderLayer, LlamaAttention
)
from transformers.modeling_attn_mask_utils import AttentionMaskConverter

from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)

from transformers import Cache, DynamicCache, StaticCache
from transformers.utils import (
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from transformers.utils import logging

import math
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union

logger = logging.get_logger(__name__)
_CONFIG_FOR_DOC = "LlamaConfig"


LLAMA_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            If `past_key_values` is used, optionally only the last `input_ids` have to be input (see
            `past_key_values`).

        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`.

        past_key_values (`Cache` or `tuple(tuple(torch.FloatTensor))`, *optional*):
            Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used to speed up sequential decoding.

        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Instead of passing `input_ids` you can choose to directly pass an embedded representation.

        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding.

        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers.

        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers.

        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.

        cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
            Indices depicting the position of the input sequence tokens in the sequence. This differs from
            `position_ids` in that it is not affected by any padding. It is used to update the cache in the correct
            position and to infer the complete sequence length.
"""


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """
    Applies Rotary Position Embedding to the query and key tensors.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    # (q, k) shape: [batch, n_heads, seq_len, head_dim]
    # cos/sin shape: [batch, seq_len, head_dim] => unsqueeze to [batch, n_heads, seq_len, head_dim] or similar
    def _rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat([-x2, x1], dim=-1)

    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Equivalent to torch.repeat_interleave(x, dim=1, repeats=n_rep).
    Goes from (batch, num_key_value_heads, seqlen, head_dim)
    to       (batch, num_attention_heads,  seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class CustomLlamaAttention(LlamaAttention):
    """
    Only changed part: we do NOT call `self.rotary_emb` inside this attention.
    Instead, we accept `(cos, sin)` from upper layer (position_embeddings).
    And then we do the same Q, K, V + head masking logic as before.
    """

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,  # not used now
            past_key_value: Optional[Cache] = None,
            output_attentions: bool = False,
            use_cache: bool = False,
            cache_position: Optional[torch.LongTensor] = None,
            head_mask: Optional[dict] = None,
            mask_type: Optional[str] = None,
            scale_factor: Optional[float] = None,
            mask_para: Optional[bool] = None,
            head_dim: Optional[int] = None,
            position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
            **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

        if mask_type not in [None, "scale_mask", "mean_mask"]:
            raise ValueError('`mask_type` must be in [None, "scale_mask", "mean_mask"].')

        if mask_type == 'scale_mask' and scale_factor is None:
            scale_factor = 0.0

        bsz, q_len, _ = hidden_states.size()

        # ============ Q K V Projection ============
        if self.config.pretraining_tp > 1:
            # (TP で分割しているとき)
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
            value_states = torch.cat(value_states, dim=-1)

        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        # (bsz, seq_len, num_heads*head_dim) => (bsz, num_heads, seq_len, head_dim)
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states   = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # ============ (Optional) Q, K, V mask その1 ============
        #   (mask_para=False のときは、この段階でQ/K/Vをいじる)
        if head_mask is not None and (mask_para is False or mask_para is None):
            for (layer_idx_, head_idx_), qkv_list in head_mask.items():
                if layer_idx_ == self.layer_idx:  # このレイヤーの場合
                    for qkv in qkv_list:
                        if qkv == "q":
                            if mask_type == 'scale_mask':
                                query_states[:, head_idx_, :, :] *= scale_factor
                            elif mask_type == 'mean_mask':
                                # 全head平均
                                query_states[:, head_idx_, :, :] = query_states.mean(dim=1, keepdim=False)
                        elif qkv == 'k':
                            if mask_type == 'scale_mask':
                                key_states[:, head_idx_, :, :] *= scale_factor
                            elif mask_type == 'mean_mask':
                                key_states[:, head_idx_, :, :] = key_states.mean(dim=1, keepdim=False)
                        # Vは後段(Softmax後)で処理するのでここでは何もしない

        # ============ Rotary Embedding ============
        # 公式実装と同じく、すでに計算された (cos, sin) を受け取って適用する
        if position_embeddings is None:
            # 万一ユーザが渡していなかった場合、fallback (ただし公式コードと出力ズレる可能性大)
            cos, sin = self.rotary_emb(hidden_states, position_ids)
        else:
            cos, sin = position_embeddings

        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # ============ (Optional) past_key_values (KV-Cache) ============
        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # ============ Expand Key/Value to all heads (if needed) ============
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # ============ QK^T / sqrt(d) + mask ============
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)

        # ============ Attention (Softmax後に V を乗ずる) ============
        attn_output = torch.matmul(attn_weights, value_states)

        # ============ (Optional) Q, K, V mask その2 (Vへの処理) ============
        #   (mask_para=False/None のときの V対応 or head_maskされているならここ)
        if head_mask is not None and (mask_para is False or mask_para is None):
            for (layer_idx_, head_idx_), qkv_list in head_mask.items():
                if layer_idx_ == self.layer_idx:
                    for qkv in qkv_list:
                        if qkv == 'v':
                            if mask_type == 'scale_mask':
                                attn_output[:, head_idx_, :, :] *= scale_factor
                            elif mask_type == 'mean_mask':
                                attn_output[:, head_idx_, :, :] = attn_output.mean(dim=1, keepdim=False)

        # ============ [batch, num_heads, seq_len, head_dim] => [batch, seq_len, num_heads*head_dim] ============
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, -1)

        if self.config.pretraining_tp > 1:
            # TP 分割している場合
            attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class CustomLlamaDecoderLayer(LlamaDecoderLayer):
    """
    公式実装と同じ構造だが、self_attn が CustomLlamaAttention に置き換わっている。
    """
    def __init__(self, config, layer_idx: int):
        super().__init__(config, layer_idx)
        self.self_attn = CustomLlamaAttention(config=config, layer_idx=layer_idx)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        head_mask: Optional[dict] = None,
        mask_type: Optional[str] = None,
        scale_factor: Optional[float] = None,
        mask_para: Optional[bool] = None,
        head_dim: Optional[int] = None,
        # ここで position_embeddings を引数として受け取る
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[torch.FloatTensor], Optional[Tuple[torch.FloatTensor]]]:

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Self-Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            head_mask=head_mask,
            mask_type=mask_type,
            scale_factor=scale_factor,
            mask_para=mask_para,
            head_dim=head_dim,
            position_embeddings=position_embeddings,  # 追加
        )
        hidden_states = residual + hidden_states

        # Feed-Forward
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)
        if use_cache:
            outputs += (present_key_value,)

        return outputs


class CustomLlamaModel(LlamaModel):
    """
    LlamaModel と同じ構造だが、DecoderLayer を CustomLlamaDecoderLayer に置き換える。
    また、公式実装と同じく最初に rotary_emb を計算して、各レイヤーに (cos, sin) を渡す。
    """
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        # LlamaModel ではここで self.layers を作るが、差し替え:
        self.layers = nn.ModuleList(
            [CustomLlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            cache_position: Optional[torch.LongTensor] = None,
            head_mask: Optional[dict] = None,
            mask_type: Optional[str] = None,
            scale_factor: Optional[float] = None,
            mask_para: Optional[bool] = None,
            head_dim: Optional[int] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        return_legacy_cache = False
        if use_cache and not isinstance(past_key_values, Cache):  # BC (非 Cache フォーマットへの対応)
            return_legacy_cache = True
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # attention_mask の 4次元拡張など公式のロジック
        causal_mask = self._update_causal_mask(
            attention_mask,
            inputs_embeds,
            cache_position,
            past_key_values,
            output_attentions
        )

        hidden_states = inputs_embeds

        # === 公式実装と同じく、最初に RoPE を計算して (cos, sin) を得る ===
        #   LlamaRotaryEmbedding の forward は (x, position_ids) を受け取って (cos, sin) を返す
        position_embeddings = self.rotary_emb(hidden_states, position_ids)  # 返り値 (cos, sin)

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for layer_index, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            # 勾配チェックポイントを使う場合
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    head_mask,
                    mask_type,
                    scale_factor,
                    mask_para,
                    head_dim,
                    position_embeddings
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    head_mask=head_mask,
                    mask_type=mask_type,
                    scale_factor=scale_factor,
                    mask_para=mask_para,
                    head_dim=head_dim,
                    position_embeddings=position_embeddings,  # ここで (cos, sin) を渡す
                )
            hidden_states = layer_outputs[0]

            if use_cache:
                # layer_outputs = (hidden_states, self_attn_weights, present_key_value) になる
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if return_legacy_cache and next_cache is not None:
            next_cache = next_cache.to_legacy_cache()

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Optional[Cache],
        output_attentions: bool,
    ):
        """
        公式実装にある Causal Mask の更新部分をそのままコピー。
        """
        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and (attention_mask == 0.0).any():
                return attention_mask
            return None

        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        using_static_cache = isinstance(past_key_values, StaticCache)

        # SDPA の場合で static cache ではなく、かつ output_attentions=False なら causal_mask を省略可能かチェック
        if self.config._attn_implementation == "sdpa" and not using_static_cache and not output_attentions:
            if AttentionMaskConverter._ignore_causal_mask_sdpa(
                attention_mask,
                inputs_embeds=input_tensor,
                past_key_values_length=past_seen_tokens,
                is_training=self.training,
            ):
                return None

        dtype, device = input_tensor.dtype, input_tensor.device
        sequence_length = input_tensor.shape[1]

        if using_static_cache:
            target_length = past_key_values.get_max_cache_shape()
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        # 2D -> 4D に拡張
        causal_mask = self._prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=target_length,
            dtype=dtype,
            device=device,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
        )

        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type == "cuda"
            and not output_attentions
        ):
            # F.scaled_dot_product_attention 用に、完全に 0 でマスクされている位置を補正
            min_dtype = torch.finfo(dtype).min
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask

    @staticmethod
    def _prepare_4d_causal_attention_mask_with_cache_position(
        attention_mask: torch.Tensor,
        sequence_length: int,
        target_length: int,
        dtype: torch.dtype,
        device: torch.device,
        cache_position: torch.Tensor,
        batch_size: int,
        **kwargs,
    ):
        if attention_mask is not None and attention_mask.dim() == 4:
            return attention_mask

        min_dtype = torch.finfo(dtype).min
        causal_mask = torch.full(
            (sequence_length, target_length),
            fill_value=min_dtype,
            dtype=dtype,
            device=device
        )
        if sequence_length != 1:
            causal_mask = torch.triu(causal_mask, diagonal=1)

        causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
        causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)

        if attention_mask is not None:
            causal_mask = causal_mask.clone()
            mask_length = attention_mask.shape[-1]
            padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
            padding_mask = padding_mask == 0
            causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(padding_mask, min_dtype)

        return causal_mask


class CustomLlamaModelForCausalLM(LlamaForCausalLM):
    """
    LlamaForCausalLM と同じ構造だが、内部の model を CustomLlamaModel に差し替えている。
    """
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.model = CustomLlamaModel(config)

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            cache_position: Optional[torch.LongTensor] = None,
            head_mask: Optional[dict] = None,
            mask_type: Optional[str] = None,
            scale_factor: Optional[float] = None,
            mask_para: Optional[bool] = None,
            head_dim: Optional[int] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        """
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                If passed, will be used to compute the language modeling loss.

        Returns:
            CausalLMOutputWithPast or tuple
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # ============ Forward (Decoder) ============
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            head_mask=head_mask,
            mask_type=mask_type,
            scale_factor=scale_factor,
            mask_para=mask_para,
            head_dim=head_dim,
        )
        hidden_states = outputs[0]

        # ============ LM head ============
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()  # (batch, seq_len, vocab_size)

        loss = None
        if labels is not None:
            # shift logits/labels => next token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1).to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    