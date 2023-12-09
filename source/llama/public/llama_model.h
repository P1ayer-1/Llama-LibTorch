#pragma once
#ifndef LLAMA_MODEL_H
#define LLAMA_MODEL_H



#include <torch/torch.h>

#include <tuple>
#include <vector>
#include "llama_config.h"
#include "llama_rms.h"
#include <variant>


using KeyValueType = std::variant<c10::optional<at::Tensor>, c10::optional<std::tuple<at::Tensor, at::Tensor>>>;


class LlamaModelImpl : public torch::nn::Module {
public:
    LlamaModelImpl(const LlamaConfig& config);

    std::tuple<torch::Tensor, std::vector<std::tuple<torch::Tensor, torch::Tensor>>, std::vector<torch::Tensor>, std::vector<c10::optional<torch::Tensor>>>
    forward(
        const c10::optional<torch::Tensor> input_ids = {},
        c10::optional<torch::Tensor> attention_mask = {},
        c10::optional<torch::Tensor> position_ids = {},
        c10::optional<torch::Tensor> inputs_embeds = {},
        const std::vector<std::tuple<torch::Tensor, torch::Tensor>>& past_key_values = {},
        c10::optional<bool> output_attentions = {},
        c10::optional<bool> output_hidden_states = {},
        c10::optional<bool> use_cache = {});


    // basic generate
    torch::Tensor generate(
        const torch::Tensor& input_ids,
        const int32_t num_new_tokens
    );

private:
    LlamaConfig config;

    torch::nn::Embedding word_embeddings = nullptr;
    torch::nn::ModuleList layers = nullptr;
    LlamaRMSNorm rms_norm = nullptr;


};

TORCH_MODULE(LlamaModel);

#endif // LLAMA_MODEL_H
    