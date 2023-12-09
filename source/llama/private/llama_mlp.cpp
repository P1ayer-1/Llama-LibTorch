#include <llama_mlp.h>


LlamaMLPImpl::LlamaMLPImpl(const LlamaConfig& config) 
    : hidden_size(config.hidden_size), intermediate_size(config.intermediate_size),
    gate_proj(torch::nn::Linear(torch::nn::LinearOptions(hidden_size, intermediate_size).bias(false))),
    up_proj(torch::nn::Linear(torch::nn::LinearOptions(hidden_size, intermediate_size).bias(false))),
    down_proj(torch::nn::Linear(torch::nn::LinearOptions(intermediate_size, hidden_size).bias(false)))

{
    register_module("gate_proj", gate_proj);
    register_module("up_proj", up_proj);
    register_module("down_proj", down_proj);
    // gate_proj = register_module("gate_proj", torch::nn::Linear(hidden_size, intermediate_size));
    // up_proj = register_module("up_proj", torch::nn::Linear(hidden_size, intermediate_size));
    // down_proj = register_module("down_proj", torch::nn::Linear(intermediate_size, hidden_size));

}

torch::Tensor LlamaMLPImpl::forward(const torch::Tensor& x) {
    auto gate = torch::silu(gate_proj->forward(x));
    auto up = up_proj->forward(x); 
    return down_proj->forward(gate * up);
}