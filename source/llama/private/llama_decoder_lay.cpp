#pragma once

#include "llama_decoder_lay.h"

LlamaDecoderLayerImpl::LlamaDecoderLayerImpl(const LlamaConfig& config) 
    : config(config),
    hidden_size(config.hidden_size),
    self_attn(std::make_shared<LlamaAttentionImpl>(config)),
    mlp(std::make_shared<LlamaMLPImpl>(config)),
    input_layernorm(std::make_shared<LlamaRMSNormImpl>(config.hidden_size, config.rms_norm_eps)),
    post_attention_layernorm(std::make_shared<LlamaRMSNormImpl>(config.hidden_size, config.rms_norm_eps))
{
    register_module("self_attn", self_attn);
    register_module("mlp", mlp);
    register_module("input_layernorm", input_layernorm);
    register_module("post_attention_layernorm", post_attention_layernorm);
}


std::tuple<torch::Tensor, c10::optional<torch::Tensor>, c10::optional<std::tuple<torch::Tensor, torch::Tensor>>> LlamaDecoderLayerImpl::forward(
    torch::Tensor& hidden_states,
    const c10::optional<torch::Tensor>& attention_mask,
    const c10::optional<torch::Tensor>& position_ids,
    c10::optional<std::tuple<torch::Tensor, torch::Tensor>>& past_key_value,
    bool output_attentions,
    bool use_cache)
{
    auto residual = hidden_states;

    hidden_states = input_layernorm->forward(hidden_states);

    // self attention
    c10::optional<torch::Tensor> self_attn_outputs;
    c10::optional<std::tuple<torch::Tensor, torch::Tensor>> present_key_value;

    std::tie(hidden_states, self_attn_outputs, present_key_value) = self_attn->forward(
        hidden_states,
        attention_mask,
        position_ids,
        past_key_value,
        output_attentions,
        use_cache);

    hidden_states = residual + hidden_states;

    // fully connected

    residual = hidden_states;
    hidden_states = post_attention_layernorm->forward(hidden_states);
    hidden_states = mlp->forward(hidden_states);
    hidden_states = residual + hidden_states;

    return std::make_tuple(hidden_states, self_attn_outputs, present_key_value);


}

