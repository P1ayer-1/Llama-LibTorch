#pragma once
#ifndef LLAMA_CAUSAL_LM_H
#define LLAMA_CAUSAL_LM_H

#include <torch/torch.h>
#include <tuple>
#include <vector>
#include "llama_config.h"
#include "llama_model.h"


class LlamaCausalLMImpl : public torch::nn::Module {
public:
    LlamaCausalLMImpl(const LlamaConfig& config);

    std::tuple<torch::Tensor, c10::optional<torch::Tensor>, std::vector<std::tuple<at::Tensor, at::Tensor>>, std::vector<torch::Tensor>, std::vector<c10::optional<torch::Tensor>>> forward(
        const c10::optional<torch::Tensor> input_ids = {},
        c10::optional<torch::Tensor> attention_mask = {},
        c10::optional<torch::Tensor> position_ids = {},
        c10::optional<torch::Tensor> inputs_embeds = {},
        c10::optional<torch::Tensor> labels = {},
        const std::vector<std::tuple<torch::Tensor, torch::Tensor>>& past_key_values = {},
        bool output_attentions = false,
        bool output_hidden_states = false,
        bool use_cache = false);

    torch::Tensor LlamaCausalLMImpl::generate(
        torch::Tensor& input_ids,
        const int32_t num_new_tokens = 10
    );

    LlamaConfig config;

private:
    LlamaModel model = nullptr;
    torch::nn::Linear lm_head = nullptr;

};

TORCH_MODULE(LlamaCausalLM);

#endif // LLAMA_CAUSAL_LM_H