
#pragma once
#include <string>

#include <c10/util/Optional.h>

struct LlamaConfig {
    int vocab_size = 32000;
    int hidden_size = 4096;
    int intermediate_size = 11008;
    int num_hidden_layers = 32;
    int num_attention_heads = 32;
    std::string hidden_act = "silu";
    int max_position_embeddings = 2048;
    double initializer_range = 0.02;
    double rms_norm_eps = 1e-05;
    bool use_cache = true;
    c10::optional<int> pad_token_id = c10::nullopt;
    int bos_token_id = 1;
    int eos_token_id = 2;

    int pretraining_tp = 1;
    bool tie_word_embeddings = false;
    double rope_theta = 10000.0;
    bool attention_bias = false;

    bool output_hidden_states = false; 
    bool output_attentions = false;

    // torch dtype
    torch::Dtype dtype = torch::kBFloat16;

};
