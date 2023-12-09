
#pragma once
#include <cmath>
#include <stdexcept>
#include <llama_attn.h>
#include <llama_utils.h>
#include <torch/torch.h>



LlamaAttentionImpl::LlamaAttentionImpl(const LlamaConfig& config)
    : config(config),
      hidden_size(config.hidden_size),
      num_heads(config.num_attention_heads),
      head_dim(hidden_size / num_heads),
      max_position_embeddings(config.max_position_embeddings),
      rope_theta(config.rope_theta),
      is_causal(true),
      q_proj(torch::nn::Linear(torch::nn::LinearOptions(hidden_size, num_heads * head_dim).bias(false))),
      k_proj(torch::nn::Linear(torch::nn::LinearOptions(hidden_size, config.num_attention_heads * head_dim).bias(false))),
      v_proj(torch::nn::Linear(torch::nn::LinearOptions(hidden_size, config.num_attention_heads * head_dim).bias(false))),
      o_proj(torch::nn::Linear(torch::nn::LinearOptions(num_heads * head_dim, hidden_size).bias(false)))
{



    // Constructor implementation
    // Ensure the module is registered
    register_module("q_proj", q_proj);
    register_module("k_proj", k_proj);
    register_module("v_proj", v_proj);
    register_module("o_proj", o_proj);

    if ((head_dim * num_heads) != hidden_size) {
        throw std::runtime_error("hidden_size must be divisible by num_heads");
    }


    // init rotary embedding

    rotary = std::make_shared<LlamaRotaryEmbeddingImpl>(LlamaRotaryEmbeddingImpl(head_dim, max_position_embeddings, rope_theta));


}


// forward function
std::tuple<torch::Tensor, c10::optional<torch::Tensor>, c10::optional<std::tuple<torch::Tensor, torch::Tensor>>>
LlamaAttentionImpl::forward(
    const torch::Tensor& hidden_states,
    const c10::optional<torch::Tensor>& attention_mask,
    const c10::optional<torch::Tensor>& position_ids,
    c10::optional<std::tuple<torch::Tensor, torch::Tensor>>& past_key_value,
    bool output_attentions,
    bool use_cache)
{
    // Forward implementation
    // Extract sequence length from shape
    int64_t seq_len = hidden_states.size(-2);
    int64_t bsz = hidden_states.size(0);

    // Calculate query, key, value
    torch::Tensor q = q_proj->forward(hidden_states);
    torch::Tensor k = k_proj->forward(hidden_states);
    torch::Tensor v = v_proj->forward(hidden_states);

    // Split into num_heads
    q = q.view({-1, seq_len, num_heads, head_dim}).transpose(1, 2);
    k = k.view({-1, seq_len, num_heads, head_dim}).transpose(1, 2);
    v = v.view({-1, seq_len, num_heads, head_dim}).transpose(1, 2);

    int64_t kv_seq_len = k.size(-2);

    if (past_key_value.has_value()) {
        kv_seq_len += std::get<0>(past_key_value.value()).size(-2);
    }

    // Apply rotary embedding
    auto [cos, sin] = rotary->forward(v, kv_seq_len);

    // apply rotary pos emb and get q, k
    std::tie(q, k) = apply_rotary_pos_emb(q, k, cos, sin, position_ids);

    // check if past_key_value is not empty
// check if past_key_value contains a value
    if (past_key_value.has_value()) {
        // If past_key_value contains a value, then access it using the value() method
        // [bs, num_heads, seq_len, head_dim]
        k = torch::cat({std::get<0>(past_key_value.value()), k}, 2);
        v = torch::cat({std::get<1>(past_key_value.value()), v}, 2);
    } 

    if (use_cache) {
        // If use_cache is true, then create a tuple of tensors and assign it to past_key_value
        past_key_value = std::make_tuple(k, v);
    }

    k = repeat_kv(k, 1);
    v = repeat_kv(v, 1);

    // calculate attention scores
    torch::Tensor attention_scores = torch::matmul(q, k.transpose(2, 3)) / std::sqrt(head_dim);

    // Check if the size of attention_scores matches the expected dimensions
    if (!(attention_scores.size(0) == bsz && 
          attention_scores.size(1) == num_heads &&
          attention_scores.size(2) == seq_len && 
          attention_scores.size(3) == kv_seq_len)) {
        std::stringstream ss;
        ss << "Attention weights should be of size (" << bsz << ", " << num_heads << ", " << seq_len << ", " << kv_seq_len 
           << "), but is (" << attention_scores.size(0) << ", " << attention_scores.size(1) << ", "
           << attention_scores.size(2) << ", " << attention_scores.size(3) << ")";
        throw std::runtime_error(ss.str());
    }

    if (attention_mask) {
        if (attention_mask->size(0) != bsz || attention_mask->size(1) != 1 ||
            attention_mask->size(2) != seq_len || attention_mask->size(3) != kv_seq_len) {
            std::stringstream ss;
            ss << "Attention mask should be of size (" << bsz << ", 1, " << seq_len << ", " << kv_seq_len 
            << "), but is (" << attention_mask->size(0) << ", " << attention_mask->size(1) << ", "
            << attention_mask->size(2) << ", " << attention_mask->size(3) << ")";
            throw std::runtime_error(ss.str());
        }
        attention_scores = attention_scores + *attention_mask;
    }

    // upcast attention to fp32 using softmax
    attention_scores = torch::softmax(attention_scores, -1, torch::kFloat32).to(q.dtype());


    torch::Tensor attn_output = torch::matmul(attention_scores, v);


        // Check if the size of attn_output matches the expected dimensions
    if (attn_output.size(0) != bsz || 
        attn_output.size(1) != num_heads || 
        attn_output.size(2) != seq_len || 
        attn_output.size(3) != head_dim) {
        std::stringstream ss;
        ss << "`attn_output` should be of size (" << bsz << ", " << num_heads << ", " << seq_len << ", " << head_dim 
           << "), but is (" << attn_output.size(0) << ", " << attn_output.size(1) << ", "
           << attn_output.size(2) << ", " << attn_output.size(3) << ")";
        throw std::runtime_error(ss.str());
    }

    attn_output = attn_output.transpose(1, 2).contiguous();
    attn_output = attn_output.reshape({bsz, seq_len, hidden_size});

    attn_output = o_proj->forward(attn_output);

    if (output_attentions) {
        return std::make_tuple(attn_output, attention_scores, past_key_value);
    } else {
        return std::make_tuple(attn_output, c10::nullopt, past_key_value);
    }


}








