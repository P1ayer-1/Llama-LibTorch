#pragma once

#include <torch/torch.h>
#include "llama_config.h"

#ifndef LLAMA_MLP_H
#define LLAMA_MLP_H

class LlamaMLPImpl : public torch::nn::Module {
public:
    LlamaMLPImpl(const LlamaConfig& config);

    torch::Tensor forward(const torch::Tensor& x);


private:
    int hidden_size;
    int intermediate_size;

    torch::nn::Linear gate_proj, up_proj, down_proj = nullptr;


};

TORCH_MODULE(LlamaMLP);

#endif // LLAMA_MLP_H