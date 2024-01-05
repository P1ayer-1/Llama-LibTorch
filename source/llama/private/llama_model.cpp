#pragma once

#include "llama_model.h"
#include "llama_decoder_lay.h"
#include "llama_utils.h"

LlamaModelImpl::LlamaModelImpl(const LlamaConfig& config) 
    : config(config),
    rms_norm(LlamaRMSNorm(config.hidden_size, config.rms_norm_eps)),
    layers(torch::nn::ModuleList())
{
    rms_norm->to(config.dtype);
    // word embedding options
    auto options = torch::nn::EmbeddingOptions(config.vocab_size, config.hidden_size);
    if (config.pad_token_id.has_value()) {
        options = options.padding_idx(config.pad_token_id.value());
    }
    word_embeddings = torch::nn::Embedding(options);
    word_embeddings->to(config.dtype);

    register_module("embed_tokens", word_embeddings);
    register_module("norm", rms_norm);

    register_module("layers", layers);
    for (int i = 0; i < config.num_hidden_layers; ++i) {
        auto layer = LlamaDecoderLayer(config);
        layer->to(config.dtype);
        layers->push_back(layer);
    }
}




std::tuple<torch::Tensor, std::vector<std::tuple<torch::Tensor, torch::Tensor>>, std::vector<torch::Tensor>, std::vector<c10::optional<torch::Tensor>>>
LlamaModelImpl::forward(
    const c10::optional<torch::Tensor> input_ids,
    c10::optional<torch::Tensor> attention_mask,
    c10::optional<torch::Tensor> position_ids,
    c10::optional<torch::Tensor> inputs_embeds,
    const std::vector<std::tuple<torch::Tensor, torch::Tensor>>& past_key_values,
    c10::optional<bool> output_attentions,
    c10::optional<bool> output_hidden_states,
    c10::optional<bool> use_cache)
{
    // check output atts
    bool resolved_output_attentions; 
    if (output_attentions.has_value()) {
        resolved_output_attentions = output_attentions.value();
    } else {
        resolved_output_attentions = config.output_attentions;
    }

    // check output hidden states
    bool resolved_output_hidden_states;
    if (output_hidden_states.has_value()) {
        resolved_output_hidden_states = output_hidden_states.value();
    } else {
        resolved_output_hidden_states = config.output_hidden_states;
    }

    // check use cache
    bool resolved_use_cache;
    if (use_cache.has_value()) {
        resolved_use_cache = use_cache.value();
    } else {
        resolved_use_cache = config.use_cache;
    }

    int64_t batch_size = 0;
    int64_t seq_length = 0;

    // retrieve input_ids and inputs_embeds
    if (input_ids.has_value()) {
        // input_ids
        auto input_shape = input_ids->sizes();
        batch_size = input_shape[0];
        seq_length = input_shape[1];
    } else if (inputs_embeds.has_value()) {
        // inputs_embeds
        auto input_shape = inputs_embeds->sizes();
        batch_size = input_shape[0];
        seq_length = input_shape[1];
    } else {
        throw std::invalid_argument("You have to specify either input_ids or inputs_embeds");
    }

    // print seq_length
    std::cout << "seq_length: " << seq_length << std::endl;

    int past_key_values_length = 0;
    if (past_key_values.size() > 0) {
        // get shape of dim 2 of the first tuple
        auto past_key_values_shape = std::get<0>(past_key_values[0]).sizes();
        past_key_values_length = past_key_values_shape[2];
    }

    // position ids
    if (!position_ids.has_value()) {
        auto options = torch::TensorOptions().dtype(torch::kInt64);
        if (input_ids.has_value()) {
            options = options.device(input_ids->device());
        } else if (inputs_embeds.has_value()) {
            options = options.device(inputs_embeds->device());
        }
        position_ids = torch::arange(past_key_values_length, seq_length + past_key_values_length, options).unsqueeze(0);
        // print position_ids
        std::cout << "position_ids: " << position_ids << std::endl;
    }

    if (!inputs_embeds.has_value()) {
        inputs_embeds = word_embeddings->forward(input_ids.value());
    }

    // attention mask
    // input shape
    auto input_shape = std::make_tuple(batch_size, seq_length);
    auto scalar_type = inputs_embeds->scalar_type();

    attention_mask = prepare4DCausalMask(attention_mask, input_shape, past_key_values_length, scalar_type, inputs_embeds->device());

    torch::Tensor hidden_states = inputs_embeds.value();

    // decoder layers
    std::vector<torch::Tensor> all_hidden_states;
    std::vector<c10::optional<torch::Tensor>> all_self_attentions;
    std::vector<std::tuple<at::Tensor, at::Tensor>> next_decoder_cache;


    // loop through layers


    for (int i = 0; i < config.num_hidden_layers; i++) {
        
        // if output_hidden_states, add hidden_states to all_hidden_states
        if (resolved_output_hidden_states) {
            all_hidden_states.push_back(hidden_states);
        }

        
        // get past_key_value

        c10::optional<std::tuple<torch::Tensor, torch::Tensor>> past_key_value;

        if (past_key_values.size() > 0) {
            past_key_value = past_key_values[i];
        } else {
            past_key_value = c10::nullopt;
        }

        auto layer_outputs = layers[i]->as<LlamaDecoderLayerImpl>()->forward(
            hidden_states,
            past_key_value,
            attention_mask,
            position_ids,
            resolved_output_attentions,
            resolved_use_cache
        );

        hidden_states = std::get<0>(layer_outputs);

        if (resolved_use_cache) {
            // check if resolved outputs attentions to know which key to use
            auto present_key_value = std::get<2>(layer_outputs);
            next_decoder_cache.push_back(present_key_value.value());


        }

        if (resolved_output_attentions) {
            all_self_attentions.push_back(std::get<1>(layer_outputs));
        }
    }
    
    hidden_states = rms_norm->forward(hidden_states);

    // add last hidden state
    if (resolved_output_hidden_states) {
        all_hidden_states.push_back(hidden_states);
    }
     


    return std::make_tuple(hidden_states, next_decoder_cache, all_hidden_states, all_self_attentions);
}



    

    
    