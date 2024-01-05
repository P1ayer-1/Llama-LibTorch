#include <llama_rms.h>

LlamaRMSNormImpl::LlamaRMSNormImpl(int hidden_size, double eps) : variance_epsilon(eps) {
    weight = torch::ones({hidden_size});
    register_parameter("weight", weight);

}

torch::Tensor LlamaRMSNormImpl::forward(torch::Tensor& x) {
    // dtype
    auto const input_dtype = x.dtype();
    x = x.to(torch::kFloat32);
    auto const variance = x.pow(2).mean(-1, true);
    x = torch::mul(x, torch::rsqrt(variance + variance_epsilon));
    return torch::mul(x.to(input_dtype), weight);
}