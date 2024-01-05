#pragma once
#include <type_traits>
#include <torch/torch.h>
#include <string>
#include <tuple>

#ifndef LLAMA_UTILS_H
#define LLAMA_UTILS_H

torch::Tensor rotate_half(const torch::Tensor& x);


std::tuple<torch::Tensor, torch::Tensor> apply_rotary_pos_emb(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& cos,
    const torch::Tensor& sin,
    const c10::optional<torch::Tensor>& position_ids,
    int64_t unsqueeze_dim = 1);

torch::Tensor repeat_kv(const torch::Tensor& hidden_states, int n_rep);


torch::Tensor makeCausalMask(
    const std::tuple<int64_t, int64_t>& input_ids_shape,
    const at::ScalarType& dtype,
    const torch::Device& device,
    int64_t past_key_values_length = 0
);

template<typename scalar_t>
double constexpr getLowest() {
    return std::numeric_limits<typename at::scalar_value_type<scalar_t>::type>::lowest();
}

// getMinValue
double constexpr getMinValue(const at::ScalarType& dtype) {
    if (dtype == at::ScalarType::Float) {
        return getLowest<float>();
    } else if (dtype == at::ScalarType::Half) {
        return getLowest<at::Half>();
    } else if (dtype == at::ScalarType::BFloat16) {
        return getLowest<at::BFloat16>();
    } else {
        throw std::invalid_argument("Unsupported dtype");
    }
}

torch::Tensor expandMask(
    const torch::Tensor& mask,
    const at::ScalarType& dtype,
    c10::optional<int64_t> tgt_len_opt = c10::nullopt
);

// to 4d
torch::Tensor to4D(
    const torch::Tensor& attention_mask_2d,
    const int64_t query_length,
    bool is_causal,
    const at::ScalarType& dtype,
    const c10::optional<int64_t> key_value_length = c10::nullopt
);

// to_causal_4d
c10::optional<torch::Tensor> toCausal4D(
    const int64_t batch_size,
    const int64_t query_length,
    const int64_t key_value_length,
    const at::ScalarType& dtype,
    const torch::Device& device = torch::kCPU
);

c10::optional<torch::Tensor> prepare4DCausalMask(
    c10::optional<torch::Tensor> attention_mask,
    std::tuple<int64_t, int64_t>& input_shape,
    const int64_t past_key_values_length,
    const at::ScalarType& dtype,
    const torch::Device& device = torch::kCPU
);


std::vector<char> GetTheBytes(std::string filename);


torch::serialize::OutputArchive toOutputArchive(
    c10::impl::GenericDict dict
    // torch::serialize::OutputArchive archive = torch::serialize::OutputArchive()

);



#endif // ROTARY_EMBEDDING_H

