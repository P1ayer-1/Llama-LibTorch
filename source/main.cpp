#pragma once
#include "llama_causal_lm.h"


void main(int argc, const char* argv[]) {


    // create a config for small model just for testing
    int vocab_size = 10000;
    int hidden_size = 64;
    int num_hidden_layers = 2;
    int num_attention_heads = 2;
    int intermediate_size = 512;
    int max_position_embeddings = 512;

    LlamaConfig config = LlamaConfig();
    config.vocab_size = vocab_size;
    config.hidden_size = hidden_size;
    config.num_hidden_layers = num_hidden_layers;
    config.num_attention_heads = num_attention_heads;
    config.intermediate_size = intermediate_size;
    config.max_position_embeddings = max_position_embeddings;


    auto model = LlamaCausalLM(config);

    // test generate function

    // set eval mode
    model->eval();

    // call generate
    auto input_ids = torch::randint(0, vocab_size, {1, 1});

    auto output = model->generate(input_ids, 10);

    std::cout << output << std::endl;

}