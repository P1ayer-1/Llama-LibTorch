#pragma once
#include "llama_utils.h"
#include <limits>
#include <fstream> 
#include <memory>



torch::Tensor rotate_half(const torch::Tensor& x) {
    auto x1 = x.slice(/*dim=*/-1, /*start=*/0, /*end=*/x.size(-1) / 2);
    auto x2 = x.slice(/*dim=*/-1, /*start=*/x.size(-1) / 2);
    return torch::cat({-x2, x1}, /*dim=*/-1);
}

std::tuple<torch::Tensor, torch::Tensor> apply_rotary_pos_emb(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& cos,
    const torch::Tensor& sin,
    const c10::optional<torch::Tensor>& position_ids,
    int64_t unsqueeze_dim) {



    auto cos_expanded = cos.index({position_ids}).unsqueeze(unsqueeze_dim);
    auto sin_expanded = sin.index({position_ids}).unsqueeze(unsqueeze_dim);


    (rotate_half(q) * sin_expanded);
    auto q_embed = (q * cos_expanded) + (rotate_half(q) * sin_expanded);
    auto k_embed = (k * cos_expanded) + (rotate_half(k) * sin_expanded);

    return std::make_tuple(q_embed, k_embed);
}



torch::Tensor repeat_kv(const torch::Tensor& hidden_states, int n_rep) {
    auto batch = hidden_states.size(0);
    auto num_key_value_heads = hidden_states.size(1);
    auto slen = hidden_states.size(2);
    auto head_dim = hidden_states.size(3);

    if (n_rep == 1) {
        return hidden_states;
    }

    // In C++, the equivalent of Python's None is torch::indexing::None.
    auto expanded = hidden_states.unsqueeze(2).expand({batch, num_key_value_heads, n_rep, slen, head_dim});
    return expanded.reshape({batch, num_key_value_heads * n_rep, slen, head_dim});
}


torch::Tensor makeCausalMask(
    const std::tuple<int64_t, int64_t>& input_ids_shape,
    const at::ScalarType& dtype,
    const torch::Device& device,
    int64_t past_key_values_length
) {
    int64_t bsz = std::get<0>(input_ids_shape);
    int64_t tgt_len = std::get<1>(input_ids_shape);

    
    double min_val = getMinValue(dtype);
    auto options = torch::TensorOptions().dtype(dtype).device(device);
    auto mask = torch::full({tgt_len, tgt_len}, min_val, options);


    auto mask_cond = torch::arange(mask.size(-1), device);
    mask = mask.masked_fill_(mask_cond < (mask_cond + 1).view({mask.size(-1), 1}), 0);

    mask = mask.to(dtype);

    if (past_key_values_length > 0) {
        mask = torch::cat({torch::zeros({tgt_len, past_key_values_length}, options), mask}, -1);
    }


    return mask.unsqueeze(0).unsqueeze(0).expand({bsz, 1, tgt_len, tgt_len + past_key_values_length});
}



torch::Tensor expandMask(
    const torch::Tensor& mask,
    const at::ScalarType& dtype,
    c10::optional<int64_t> tgt_len_opt
) {
    double min_val = getMinValue(dtype);
    auto bsz = mask.size(0);
    auto src_len = mask.size(1);
    int64_t tgt_len = tgt_len_opt.value_or(src_len);

    auto options = torch::TensorOptions().dtype(dtype);
    auto expanded_mask = mask.unsqueeze(1).unsqueeze(2).expand({bsz, 1, tgt_len, src_len}).to(dtype);

    auto inverted_mask = 1.0 - expanded_mask;
    return inverted_mask.masked_fill(inverted_mask.to(torch::kBool), min_val);
}


torch::Tensor to4D(
    const torch::Tensor& attention_mask_2d,
    const int64_t query_length,
    bool is_causal,
    const at::ScalarType& dtype,
    const c10::optional<int64_t> key_value_length
) {
    auto input_shape = std::make_tuple(attention_mask_2d.size(0), query_length);


    auto expanded_attn_mask = expandMask(attention_mask_2d, dtype, query_length);

    // create causal mask
    // [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
    torch::Tensor expanded_4d_mask;

    if (query_length > 1 && is_causal) {
        if (!key_value_length.has_value()) {
            throw std::invalid_argument("key_value_length must be provided when is_causal is true");
        }
        expanded_4d_mask = makeCausalMask(input_shape, dtype, attention_mask_2d.device(), key_value_length.value());
    } else {
        expanded_4d_mask = expanded_attn_mask;
    }

    return expanded_4d_mask;


}


c10::optional<torch::Tensor> toCausal4D(
    const int64_t batch_size,
    const int64_t query_length,
    const int64_t key_value_length,
    const at::ScalarType& dtype,
    const torch::Device& device
) {
    if (query_length == 1) {
        return c10::nullopt;
    }

    auto input_shape = std::make_tuple(batch_size, query_length);

    const int64_t past_key_values_length = key_value_length - query_length;

    auto mask = makeCausalMask(input_shape, dtype, device, past_key_values_length);
    return mask;
}


c10::optional<torch::Tensor> prepare4DCausalMask(
    c10::optional<torch::Tensor> attention_mask,
    std::tuple<int64_t, int64_t>& input_shape,
    const int64_t past_key_values_length,
    const at::ScalarType& dtype,
    const torch::Device& device
) {
    const int64_t key_value_length = std::get<1>(input_shape) + past_key_values_length;

    if (attention_mask.has_value()) {
        auto attention_mask_2d = attention_mask.value();
        auto expanded_4d_mask = to4D(attention_mask_2d, std::get<1>(input_shape), key_value_length, dtype, true);
        return expanded_4d_mask;
    } else {
        auto causal_4d_mask = toCausal4D(std::get<0>(input_shape), std::get<1>(input_shape), key_value_length, dtype, device);
        return causal_4d_mask;
    }
}

std::vector<char> GetTheBytes(std::string filename ) {

    std::ifstream input(filename, std::ios::binary);
    std::vector<char> bytes(
        (std::istreambuf_iterator<char>(input)),
        (std::istreambuf_iterator<char>()));

    input.close();
    return bytes;
}