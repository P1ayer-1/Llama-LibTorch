#pragma once
#ifndef LLAAM_DECODER_LAY_H
#define LLAAM_DECODER_LAY_H

#include <torch/torch.h>
#include "llama_attn.h"
#include "llama_mlp.h"
#include "llama_rms.h"
#include "llama_config.h"


class LlamaDecoderLayerImpl : public torch::nn::Module {
public:
    LlamaDecoderLayerImpl(const LlamaConfig& config);

    std::tuple<torch::Tensor, c10::optional<torch::Tensor>, c10::optional<std::tuple<torch::Tensor, torch::Tensor>>> forward(
        torch::Tensor& hidden_states,
        const c10::optional<torch::Tensor>& attention_mask, // dont need to use * here
        const c10::optional<torch::Tensor>& position_ids,
        c10::optional<std::tuple<torch::Tensor, torch::Tensor>>& past_key_value,
        bool output_attentions = false,
        bool use_cache = false);

private:
    LlamaConfig config;
    int hidden_size;


    // self attention
    std::shared_ptr<LlamaAttentionImpl> self_attn;

    // mlp 
    std::shared_ptr<LlamaMLPImpl> mlp;

    // rms norms
    std::shared_ptr<LlamaRMSNormImpl> input_layernorm;
    std::shared_ptr<LlamaRMSNormImpl> post_attention_layernorm;

};

TORCH_MODULE(LlamaDecoderLayer);

#endif // LLAAM_DECODER_LAY_H