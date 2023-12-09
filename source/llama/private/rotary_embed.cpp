#include <rotary_embed.h>

LlamaRotaryEmbeddingImpl::LlamaRotaryEmbeddingImpl(int64_t dim, int64_t max_position_embeddings, double base)
    : dim_(dim),
      max_position_embeddings_(max_position_embeddings),
      base_(base) {
  // Calculate inverse frequencies
  inv_freq_ = torch::pow(base, -torch::arange(0, dim, 2).to(torch::kFloat32) / dim);
  register_buffer("inv_freq", inv_freq_);

  // Initialize the cosine and sine cache
  _set_cos_sin_cache(max_position_embeddings_);
}

void LlamaRotaryEmbeddingImpl::_set_cos_sin_cache(int64_t seq_len) {
  max_seq_len_cached_ = seq_len;
  torch::Tensor t = torch::arange(max_seq_len_cached_).to(inv_freq_.device());

  torch::Tensor freqs = torch::einsum("i,j->ij", {t, inv_freq_});
  torch::Tensor emb = torch::cat({freqs, freqs}, -1);
  cos_cached_ = register_buffer("cos_cached", emb.cos().to(torch::kFloat32));
  sin_cached_ = register_buffer("sin_cached", emb.sin().to(torch::kFloat32));
}

std::tuple<torch::Tensor, torch::Tensor> LlamaRotaryEmbeddingImpl::forward(torch::Tensor x, int64_t seq_len) {
  // [bs, num_attention_heads, seq_len, head_size]
  // Extract sequence length from shape

  if (seq_len > max_seq_len_cached_) {
    _set_cos_sin_cache(seq_len);
  }

  return std::make_tuple(cos_cached_.slice(0, 0, seq_len).to(x.dtype()),
                         sin_cached_.slice(0, 0, seq_len).to(x.dtype()));
}
