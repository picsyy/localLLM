// zipfEngine.cpp - FIXED VERSION
// Improved prompt engineering, better sampling parameters, and refined generation logic

#include "llama.h"
#include "llama-sampling.h"
#include "llama-vocab.h"
#include "zipf.h"

#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <thread>
#include <cmath>
#include <algorithm>
#include <limits>
#include <unordered_map>
#include <fstream>
#include <sstream>
#include <deque>
#include <unordered_set>
#include <cctype>

// ---- Personality Modes ----
struct PersonalityMode {
    std::string mode_name;
    std::string prompt_modifier;
    int min_tokens;
    int max_tokens;
};

const std::vector<PersonalityMode> ALL_PERSONALITY_MODES = {
    { "friendly",   "You respond with warmth, politeness, and helpfulness. Speak in complete sentences.", 15, 150 },
    { "rude",       "You respond curtly, with irritation, sarcasm, or disrespect. Keep responses brief but complete.", 8, 80 },
    { "suspicious", "You respond with mistrust, guarded language, and evasiveness. Answer hesitantly.", 12, 120 },
    { "deferential","You are very respectful and submissive to the speaker. Use honorifics and speak humbly.", 20, 200 },
    { "stoic",      "You speak briefly with little emotion, but still provide complete thoughts.", 10, 60 }
};

const PersonalityMode* get_mode_by_name(const std::string& name) {
    for (const auto& m : ALL_PERSONALITY_MODES)
        if (m.mode_name == name) return &m;
    return &ALL_PERSONALITY_MODES[0];
}

// ---- NPC Profiles ----
struct NPCProfile {
    std::string name;
    std::string base_prompt;
    std::vector<std::string> allowed_modes;
    std::string background_info;
};

const std::vector<NPCProfile> NPCS = {
    {   // 0
        "Krackle",
        "You are Krackle, the deadly front door guard to the Ramsel Dynasty. You are blunt, experienced, and have no time for nonsense. You've seen many adventurers come and go.",
        {"friendly", "rude", "suspicious"},
        "A veteran guard who has protected the dynasty for decades. Wears battle-scarred armor and carries an ancient sword."
    },
    {   // 1
        "Mira",
        "You are Mira, a world-weary but kind tavernkeeper who welcomes all sorts but is slow to trust. You've heard countless stories from travelers.",
        {"friendly", "suspicious", "stoic"},
        "Runs 'The Weary Traveler' tavern. Has graying hair and knowing eyes that have seen much of the world through her patrons."
    },
    {   // 2
        "Feylan",
        "You are Feylan, an anxious young court scribe. You are always deferential to those in authority and eager to help with your knowledge of court matters and records.",
        {"deferential", "friendly", "stoic"},
        "A young scholar with ink-stained fingers and nervous habits. Knows the history and procedures of the royal court intimately."
    }
};

struct GameState {
    std::string player_name;
    std::string player_class;
    std::string relationship;      // "stranger", "friend", "foe"
    int player_level;
    std::string recent_action;     // e.g. "threaten", "ask for help"
};

std::string to_lower(const std::string& s) {
    std::string out = s;
    std::transform(out.begin(), out.end(), out.begin(),
        [](unsigned char c){ return std::tolower(c); });
    return out;
}

// --- Smart Mode Selection ---
std::string pick_mode_for_npc(const NPCProfile& npc, const GameState& state, const std::string& user_input) {
    // THREAT OVERRIDE: If player threatens, always go "rude" if allowed
    std::string input_lc = to_lower(user_input + " " + state.recent_action);
    if ((input_lc.find("threaten") != std::string::npos ||
         input_lc.find("kill") != std::string::npos ||
         input_lc.find("harm") != std::string::npos ||
         input_lc.find("attack") != std::string::npos) &&
         std::find(npc.allowed_modes.begin(), npc.allowed_modes.end(), "rude") != npc.allowed_modes.end())
    {
        return "rude";
    }

    // RELATIONSHIP (friend/foe/stranger)
    if (state.relationship == "friend") {
        if (std::find(npc.allowed_modes.begin(), npc.allowed_modes.end(), "friendly") != npc.allowed_modes.end())
            return "friendly";
    } else if (state.relationship == "foe") {
        if (std::find(npc.allowed_modes.begin(), npc.allowed_modes.end(), "rude") != npc.allowed_modes.end())
            return "rude";
        if (std::find(npc.allowed_modes.begin(), npc.allowed_modes.end(), "suspicious") != npc.allowed_modes.end())
            return "suspicious";
    } else if (state.relationship == "stranger") {
        if (std::find(npc.allowed_modes.begin(), npc.allowed_modes.end(), "suspicious") != npc.allowed_modes.end())
            return "suspicious";
    }

    if (input_lc.find("thank") != std::string::npos && 
        std::find(npc.allowed_modes.begin(), npc.allowed_modes.end(), "friendly") != npc.allowed_modes.end())
        return "friendly";

    if ((input_lc.find("king") != std::string::npos || input_lc.find("queen") != std::string::npos || 
         input_lc.find("majesty") != std::string::npos || input_lc.find("lord") != std::string::npos) &&
        std::find(npc.allowed_modes.begin(), npc.allowed_modes.end(), "deferential") != npc.allowed_modes.end())
        return "deferential";

    return npc.allowed_modes[0];
}

// --- IMPROVED Prompt Construction ---
std::string inject_prompt_context(const NPCProfile& npc, const PersonalityMode& mode,
                                  const GameState& state, const std::string& user_input) {
    std::ostringstream oss;
    
    // Use a more conversational format instead of strict ### headers
    oss << "You are " << npc.name << ". " << npc.base_prompt << "\n\n";
    oss << "Background: " << npc.background_info << "\n\n";
    oss << "Current situation: ";
    oss << "You are speaking with " << state.player_name;
    oss << " (a level " << state.player_level << " " << state.player_class << ") ";
    oss << "who is a " << state.relationship << " to you.\n\n";
    
    oss << "Your current mood/behavior: " << mode.prompt_modifier << "\n\n";
    
    oss << "Important rules:\n";
    oss << "- Respond as " << npc.name << " would, staying in character\n";
    oss << "- Give thoughtful, complete responses (not just one word)\n";
    oss << "- Do not speak for the other person or continue their dialogue\n";
    oss << "- Respond naturally as if in a real conversation\n\n";
    
    oss << state.player_name << " says: \"" << user_input << "\"\n\n";
    oss << npc.name << " responds: \"";
    
    return oss.str();
}

std::string sanitize_token_text(const std::string & input) {
    std::string output;
    bool last_was_space = false;
    for (size_t i = 0; i < input.size(); i++) {
        unsigned char c = input[i];
        char ch = (c == '\n' || (c >= 32 && c <= 126)) ? c : ' ';
        if (ch == ' ') {
            if (!last_was_space) {
                output.push_back(ch);
                last_was_space = true;
            }
        } else {
            output.push_back(ch);
            last_was_space = false;
        }
    }
    return output;
}

// --- Enhanced truncation that preserves sentence completion ---
std::string truncate_at_forbidden_speaker(const std::string& output, const GameState& state) {
    std::vector<std::string> cues = {
        "Adventurer:", "User:", "You say", "### Input:", "### Instruction:", 
        "### Response:", "### Assistant:", "### Human:", state.player_name + ":"
    };
    
    size_t cut = std::string::npos;
    for (const auto& cue : cues) {
        size_t pos = output.find(cue, 1);
        if (pos != std::string::npos && (cut == std::string::npos || pos < cut)) {
            cut = pos;
        }
    }
    
    std::string result = (cut != std::string::npos) ? output.substr(0, cut) : output;
    
    // Try to end at a complete sentence
    size_t last_period = result.find_last_of(".!?");
    if (last_period != std::string::npos && last_period > result.length() * 0.7) {
        result = result.substr(0, last_period + 1);
    }
    
    return result;
}

// --- IMPROVED Sampling Constants ---
#define DEFAULT_MAX_OUTPUT_TOKENS 300
#define DEFAULT_MAX_TOKENS 4096
#define DEFAULT_N_CTX 4096  // Increased context window
#define TOP_K 40            // Reduced for better quality
#define TOP_P 0.95f         // Increased for more variety
#define TEMP 0.8f           // Reduced temperature for better coherence
#define ALPHA 0.1f          // Reduced alpha
#define BASELINE 3.0f       // Adjusted baseline
#define MIN_SCALING 0.7f
#define MAX_SCALING 1.3f
#define COMMON_TOKEN_THRESHOLD 100
#define HIGH_COMMON_PENALTY 0.7f
#define LOW_COMMON_PENALTY 0.9f
#define EARLY_STOP_STREAK_THRESHOLD 15
#define MIN_RESPONSE_TOKENS 8

// DynamicSamplingParams dynamic_params;


int main(int argc, char** argv) {
    std::cout << "Choose NPC to converse with:\n";
    for (size_t i = 0; i < NPCS.size(); ++i)
        std::cout << "  " << i << ": " << NPCS[i].name << " - " << NPCS[i].base_prompt << "\n";
    int npc_idx = 0;
    std::cout << "Enter NPC number: ";
    std::cin >> npc_idx; std::cin.ignore();
    if (npc_idx < 0 || npc_idx >= (int)NPCS.size()) npc_idx = 0;
    const NPCProfile& npc = NPCS[npc_idx];

    GameState state;
    std::cout << "Enter your character's name: "; std::getline(std::cin, state.player_name);
    std::cout << "Enter your character's class: "; std::getline(std::cin, state.player_class);
    std::cout << "Enter your level: "; std::cin >> state.player_level; std::cin.ignore();
    std::cout << "How do you stand to " << npc.name << "? (stranger/friend/foe):  ";
    std::getline(std::cin, state.relationship);
    if (state.relationship.empty()) state.relationship = "stranger";
    std::cout << "What was your recent action (e.g., 'threaten', 'greet', 'ask for help')? ";
    std::getline(std::cin, state.recent_action);

    llama_backend_init();

    const char* model_path = "model/mistral-7b-instruct-v0.1.Q4_K_M.gguf";
    llama_model_params model_params = llama_model_default_params();
    model_params.use_mmap = false;  // Re-enable mmap for better performance
    // model_params.n_gpu_layers = 35; // Increased GPU layers
    llama_model* model = llama_model_load_from_file(model_path, model_params);
    if (!model) {
        std::cerr << "Failed to load model" << std::endl;
        return 1;
    }

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_threads = std::thread::hardware_concurrency();
    ctx_params.n_threads_batch = std::thread::hardware_concurrency();
    ctx_params.n_ctx = DEFAULT_N_CTX;
    ctx_params.flash_attn = true;  // Enable flash attention if available

    llama_context* ctx = llama_init_from_model(model, ctx_params);
    if (!ctx) {
        std::cerr << "Failed to initialize context" << std::endl;
        llama_model_free(model);
        return 1;
    }

    const llama_vocab* vocab = llama_model_get_vocab(model);

    // Initialize Zipf accelerator
    ZipfAccelerator zipf;
    zipf.initialize(vocab);

    // Remove old role-specific token logic (handled by ZipfAccelerator)

    // Pre-compute simpler logit weights (optional, can be removed if not used elsewhere)
    // std::vector<float> precomputed_log_weight(vocab->n_tokens());
    // for (int token_id = 0; token_id < vocab->n_tokens(); ++token_id) {
    //     int rank = zipf.get_rank(token_id);
    //     precomputed_log_weight[token_id] = 1.0f / std::sqrt(rank + 1.0f);  // Gentler penalty
    // }

    // Initialize sampler chain
    llama_sampler_chain_params chain_params = llama_sampler_chain_default_params();
    struct llama_sampler * sampler_chain = llama_sampler_chain_init(chain_params);
    llama_sampler_chain_add(sampler_chain, llama_sampler_init_top_k(TOP_K));
    llama_sampler_chain_add(sampler_chain, llama_sampler_init_top_p(TOP_P, 1));
    llama_sampler_chain_add(sampler_chain, llama_sampler_init_temp(TEMP));
    llama_sampler_chain_add(sampler_chain, llama_sampler_init_greedy());

    int n_vocab = vocab->n_tokens();
    std::vector<llama_token_data> candidates(n_vocab);
    std::vector<int> token_counts(n_vocab, 0);

    std::ostringstream log_buffer;
    auto log_and_print = [&](const std::string& msg) {
        std::cout << msg;
        log_buffer << msg;
    };

    log_and_print("\nImproved character chat (type 'exit' to quit):\n");

    while (true) {
        log_and_print("\nYou: ");
        std::string user_input;
        std::getline(std::cin, user_input);
        if (user_input == "exit") break;
        if (user_input.empty()) continue;

        std::string mode_name = pick_mode_for_npc(npc, state, user_input);
        const PersonalityMode* mode = get_mode_by_name(mode_name);

        // Reinitialize samplers for each turn
        llama_sampler_free(sampler_chain);
        sampler_chain = llama_sampler_chain_init(chain_params);
        llama_sampler_chain_add(sampler_chain, llama_sampler_init_top_k(TOP_K));
        llama_sampler_chain_add(sampler_chain, llama_sampler_init_top_p(TOP_P, 1));
        llama_sampler_chain_add(sampler_chain, llama_sampler_init_temp(TEMP));
        llama_sampler_chain_add(sampler_chain, llama_sampler_init_greedy());

        llama_kv_cache_clear(ctx);

        // Update Zipf context for this turn
        zipf.update_context(npc.name, mode_name, vocab);

        auto start_time = std::chrono::steady_clock::now();

        std::string full_prompt = inject_prompt_context(npc, *mode, state, user_input);
        std::vector<llama_token> prompt_tokens(DEFAULT_MAX_TOKENS);
        int32_t n_prompt = llama_tokenize(vocab, full_prompt.c_str(), full_prompt.size(),
                                          prompt_tokens.data(), DEFAULT_MAX_TOKENS, true, true);
        if (n_prompt <= 0) {
            std::cerr << "Tokenization failed" << std::endl;
            continue;
        }
        prompt_tokens.resize(n_prompt);
        llama_batch prompt_batch = llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size());
        if (llama_decode(ctx, prompt_batch) != 0) {
            std::cerr << "Error decoding prompt" << std::endl;
            continue;
        }

        std::vector<llama_token> assistant_tokens;
        token_counts.assign(n_vocab, 0);
        int common_token_streak = 0;
        int min_tokens = std::max(MIN_RESPONSE_TOKENS, mode->min_tokens);
        int max_tokens = std::min(DEFAULT_MAX_OUTPUT_TOKENS, mode->max_tokens);

        std::string detok_so_far;
        bool found_forbidden_speaker = false;
        bool found_closing_quote = false;

        for (int i = 0; i < max_tokens; ++i) {
            float* logits = llama_get_logits(ctx);

            // Apply Zipf acceleration (biases, role/mood, etc.)
            zipf.accelerate_logits(logits, i, max_tokens - i);

            // Reset candidates array
            for (int token_id = 0; token_id < n_vocab; token_id++) {
                candidates[token_id] = llama_token_data{token_id, logits[token_id], 0.0f};
            }
            llama_token_data_array candidates_arr = { candidates.data(), (size_t)n_vocab, false };

            // Apply repetition penalty using ZipfAccelerator
            for (int token_id = 0; token_id < n_vocab; ++token_id) {
                if (token_counts[token_id] > 0) {
                    float penalty = zipf.get_repetition_penalty(token_id, token_counts[token_id]);
                    logits[token_id] *= penalty;
                }
            }

            llama_sampler_apply(sampler_chain, &candidates_arr);
            llama_token next_token = llama_sampler_sample(sampler_chain, ctx, -1);

            if (next_token == llama_vocab_eos(vocab) || next_token == LLAMA_TOKEN_NULL) {
                if (i >= min_tokens) break;
                // If we haven't hit minimum tokens, try to continue
                continue;
            }

            assistant_tokens.push_back(next_token);
            token_counts[next_token]++;

            // Check for natural stopping points
            char token_buf[128] = {0};
            llama_detokenize(vocab, &next_token, 1, token_buf, sizeof(token_buf), true, false);
            std::string token_str(token_buf);
            detok_so_far += token_str;
            
            // Look for closing quote (natural end of dialogue)
            if (token_str.find("\"") != std::string::npos) {
                found_closing_quote = true;
                if (i >= min_tokens) break;
            }
            
            // Check for forbidden speaker cues
            std::string detok_lower = to_lower(detok_so_far);
            if (detok_lower.find(to_lower(state.player_name) + ":") != std::string::npos ||
                detok_lower.find("you say") != std::string::npos ||
                detok_lower.find("adventurer:") != std::string::npos) {
                found_forbidden_speaker = true;
                break;
            }

            // Update context
            std::vector<llama_token> token_vec = { next_token };
            llama_batch token_batch = llama_batch_get_one(token_vec.data(), token_vec.size());
            if (llama_decode(ctx, token_batch) != 0) {
                std::cerr << "\nDecoding error during generation" << std::endl;
                break;
            }
            
            llama_sampler_accept(sampler_chain, next_token);
        }

        // Detokenize and clean up output
        char output_buf[8192] = {0};
        int32_t n_chars = llama_detokenize(vocab, assistant_tokens.data(), assistant_tokens.size(),
                                           output_buf, sizeof(output_buf), true, false);
        std::string output;
        if (n_chars < 0) {
            output = "Detokenization failed\n";
            log_and_print(output);
        } else {
            output = sanitize_token_text(std::string(output_buf, n_chars));
            output = truncate_at_forbidden_speaker(output, state);
            
            // Clean up the output
            if (output.empty() || output == "\"") {
                output = "I... I'm not sure what to say.";
            }
            
            // Remove leading/trailing quotes if present
            if (output.front() == '"') output = output.substr(1);
            if (!output.empty() && output.back() == '"') output.pop_back();
            
            log_and_print(npc.name + ": \"" + output + "\"\n");
        }
        
        auto end_time = std::chrono::steady_clock::now();
        auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
        double elapsed_sec = elapsed_ms / 1000.0;
        double tokens_per_sec = (elapsed_sec > 0.0) ? (assistant_tokens.size() / elapsed_sec) : 0.0;
        std::string gen_stats = "[Gen " + std::to_string(elapsed_ms) + " ms | " 
                                + std::to_string(tokens_per_sec) + " tok/s]\n";
        log_and_print(gen_stats);

        // Save conversation
        std::ofstream outfile("lastPrompt.txt", std::ios::app);
        outfile << "You: " << user_input << "\n";
        outfile << npc.name << ": \"" + output + "\"\n";
        outfile << gen_stats;
        outfile.close();
    }

    llama_sampler_free(sampler_chain);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();

    return 0;
}
// ---- End of improved zipfEngine.cpp ----
