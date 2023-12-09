#include <llama_rms.h>

LlamaRMSNormImpl::LlamaRMSNormImpl(int hidden_size, double eps) : variance_epsilon(eps) {
    torch::ones({hidden_size});
    weight = register_parameter("weight", torch::ones({hidden_size}), false);

}

torch::Tensor LlamaRMSNormImpl::forward(torch::Tensor& x) {
    // dtype
    auto input_dtype = x.dtype();
    x = x.to(torch::kFloat32);
    auto variance = x.pow(2).mean({-1}, true);
    x = x * torch::sqrt(variance + variance_epsilon);
    return weight * x.to(input_dtype);
}