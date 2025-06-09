// zipf.h - Redesigned for actual performance impact
#pragma once

#include "llama-vocab.h"
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <cmath>
#include <string>
#include <deque>
#include <numeric>

class ZipfAccelerator {
private:
    // Pre-computed token categories for O(1) lookup
    std::vector<float> base_logit_bias;           // Per-token bias based on rank
    std::unordered_set<llama_token> common_tokens; // Top 500 most frequent
    std::unordered_set<llama_token> rare_tokens;   // Bottom 20% least frequent
    std::unordered_set<llama_token> punctuation;   // Sentence enders
    std::unordered_set<llama_token> dialogue_tokens; // Conversation-specific
    
    // Context-aware token sets (rebuilt per conversation turn)
    std::unordered_set<llama_token> current_role_tokens;
    std::unordered_set<llama_token> current_mood_tokens;
    
    int vocab_size;
    bool initialized = false;

    // Conversation state tracking
    struct ConversationState {
        int turn_count = 0;
        int avg_response_length = 0;
        float engagement_score = 0.0f;
        std::deque<int> recent_lengths{};
        std::unordered_map<llama_token, float> turn_frequencies;
    } conv_state;

    // Dynamic adjustment parameters
    struct DynamicParams {
        float complexity_factor = 1.0f;
        float engagement_modifier = 1.0f;
        float pattern_strength = 1.0f;
    } params;

    // Add fast-path optimization
    std::vector<uint8_t> token_flags;  // Bit flags for O(1) category checks
    
    // Flag bits
    static constexpr uint8_t IS_COMMON = 1;
    static constexpr uint8_t IS_RARE = 2;
    static constexpr uint8_t IS_PUNCT = 4;
    static constexpr uint8_t IS_DIALOGUE = 8;
    
public:
    // Fast initialization - only compute what we actually use
    void initialize(const llama_vocab* vocab) {
        vocab_size = vocab->n_tokens();
        base_logit_bias.resize(vocab_size, 0.0f);
        
        // Build frequency ranking
        std::vector<std::pair<llama_token, float>> token_scores;
        token_scores.reserve(vocab_size);
        
        for (llama_token id = 0; id < vocab_size; ++id) {
            float score = vocab->token_get_score(id);
            token_scores.emplace_back(id, score);
        }
        
        std::sort(token_scores.begin(), token_scores.end(), 
                  [](const auto& a, const auto& b) { return a.second > b.second; });
        
        // Pre-compute categories and biases
        common_tokens.clear();
        rare_tokens.clear();
        punctuation.clear();
        dialogue_tokens.clear();
        
        int common_cutoff = std::min(500, vocab_size / 10);
        int rare_cutoff = vocab_size * 4 / 5; // Bottom 20%
        
        for (int rank = 0; rank < vocab_size; ++rank) {
            llama_token token = token_scores[rank].first;
            std::string token_text = vocab->token_get_text(token);
            
            // Categorize tokens
            if (rank < common_cutoff) {
                common_tokens.insert(token);
            }
            if (rank > rare_cutoff) {
                rare_tokens.insert(token);
            }
            
            // Find punctuation and dialogue markers
            if (token_text.find_first_of(".!?\"'") != std::string::npos) {
                punctuation.insert(token);
                if (token_text.find('"') != std::string::npos) {
                    dialogue_tokens.insert(token);
                }
            }
            
            // Pre-compute Zipfian bias (more aggressive than current)
            float zipf_factor = 1.0f / std::pow(rank + 1.0f, 0.3f);
            base_logit_bias[token] = std::log(zipf_factor);
        }
        
        // Setup fast-path flags
        token_flags.resize(vocab_size, 0);
        for (llama_token token : common_tokens) token_flags[token] |= IS_COMMON;
        for (llama_token token : rare_tokens) token_flags[token] |= IS_RARE;
        for (llama_token token : punctuation) token_flags[token] |= IS_PUNCT;
        for (llama_token token : dialogue_tokens) token_flags[token] |= IS_DIALOGUE;
        
        initialized = true;
    }
    
    // Context-aware token set updates (called once per turn)
    void update_context(const std::string& role, const std::string& mood, const llama_vocab* vocab) {
        current_role_tokens.clear();
        current_mood_tokens.clear();
        
        // More comprehensive token detection
        std::vector<std::string> role_keywords = get_role_keywords(role);
        std::vector<std::string> mood_keywords = get_mood_keywords(mood);
        
        for (llama_token id = 0; id < vocab_size; ++id) {
            std::string token_text = vocab->token_get_text(id);
            std::string lower_text = to_lower(token_text);
            
            // Check role matches
            for (const auto& keyword : role_keywords) {
                if (lower_text.find(keyword) != std::string::npos) {
                    current_role_tokens.insert(id);
                    break;
                }
            }
            
            // Check mood matches
            for (const auto& keyword : mood_keywords) {
                if (lower_text.find(keyword) != std::string::npos) {
                    current_mood_tokens.insert(id);
                    break;
                }
            }
        }

        // Update conversation state
        conv_state.turn_count++;
        if (conv_state.recent_lengths.size() >= 5) {
            conv_state.recent_lengths.pop_front();
        }
        
        // Adjust complexity based on recent interaction patterns
        update_complexity_factor();
    }
    
    // MAIN ACCELERATION FUNCTION - applies all biases at once
    void accelerate_logits(float* logits, int context_length, int min_tokens_remaining) {
        const int MAX_RESPONSE_LENGTH = 200; // Max response length for this context

        if (!initialized) return;

        // Apply base biases with dynamic scaling
        for (int i = 0; i < vocab_size; ++i) {
            logits[i] += base_logit_bias[i] * params.complexity_factor;
        }

        // Adaptive role/mood boosts based on engagement
        float role_boost = 0.5f * params.engagement_modifier;
        float mood_boost = 0.3f * params.engagement_modifier;
        float dialogue_boost = 0.4f * params.pattern_strength;

        // Stronger boosts early in generation
        if (context_length < 10) {
            role_boost *= 1.5f;
            mood_boost *= 1.5f;
        }
        
        // Apply boosted tokens
        for (llama_token token : current_role_tokens) {
            logits[token] += role_boost;
        }
        for (llama_token token : current_mood_tokens) {
            logits[token] += mood_boost;
        }
        
        // Dynamic dialogue flow
        float completion_ratio = 1.0f - (float)min_tokens_remaining / MAX_RESPONSE_LENGTH;
        if (completion_ratio < 0.6f) {
            suppress_dialogue_enders(logits);
        } else {
            boost_dialogue_enders(logits, dialogue_boost * completion_ratio);
        }

        apply_conversation_patterns(logits, context_length);
    }
    
    // Fast quality check - returns true if token seems appropriate
    bool is_contextually_appropriate(llama_token token) const {
        // Quick rejection of very rare tokens
        if (rare_tokens.count(token)) return false;
        
        // Quick approval of role/mood tokens
        if (current_role_tokens.count(token) || current_mood_tokens.count(token)) {
            return true;
        }
        
        // Common tokens are generally OK
        return common_tokens.count(token) > 0;
    }
    
    // Adaptive repetition penalty based on token frequency
    float get_repetition_penalty(llama_token token, int count) const {
        float base_penalty = 0.9f; // Standard repetition penalty
        
        if (common_tokens.count(token)) {
            // Common tokens can repeat more
            return std::pow(base_penalty, count * 0.7f);
        } else {
            // Rare tokens get penalized harder for repetition
            return std::pow(base_penalty, count * 1.3f);
        }
    }

private:
    std::string to_lower(const std::string& s) const {
        std::string result = s;
        std::transform(result.begin(), result.end(), result.begin(), ::tolower);
        return result;
    }
    
    std::vector<std::string> get_role_keywords(const std::string& role) const {
        std::unordered_map<std::string, std::vector<std::string>> role_map = {
            {"guard", {"guard", "watch", "protect", "duty", "patrol", "secure", "defend"}},
            {"tavernkeeper", {"tavern", "ale", "drink", "brew", "welcome", "inn", "guest", "room"}},
            {"scribe", {"scroll", "write", "record", "ink", "quill", "document", "archive", "knowledge"}},
            {"merchant", {"gold", "coin", "trade", "sell", "buy", "price", "goods", "wares"}},
            {"knight", {"honor", "sword", "shield", "oath", "noble", "quest", "chivalry"}},
            {"wizard", {"magic", "spell", "arcane", "tome", "staff", "enchant", "ritual"}}
        };
        
        auto it = role_map.find(role);
        return (it != role_map.end()) ? it->second : std::vector<std::string>{};
    }
    
    std::vector<std::string> get_mood_keywords(const std::string& mood) const {
        std::unordered_map<std::string, std::vector<std::string>> mood_map = {
            {"friendly", {"pleased", "welcome", "glad", "happy", "kind", "warm", "cheerful"}},
            {"rude", {"annoyed", "irritated", "bah", "hmph", "whatever", "fool", "waste"}},
            {"suspicious", {"wary", "careful", "suspicious", "doubt", "trust", "watch", "unsure"}},
            {"deferential", {"sir", "madam", "honor", "respect", "please", "apologize", "forgive"}},
            {"stoic", {"indeed", "understood", "very well", "quite", "certainly"}}
        };
        
        auto it = mood_map.find(mood);
        return (it != mood_map.end()) ? it->second : std::vector<std::string>{};
    }

    void update_complexity_factor() {
        // Analyze recent response lengths
        float avg_length = 0.0f;
        if (!conv_state.recent_lengths.empty()) {
            avg_length = std::accumulate(conv_state.recent_lengths.begin(), 
                                       conv_state.recent_lengths.end(), 0.0f) / 
                                       conv_state.recent_lengths.size();
        }

        // Adjust complexity based on interaction patterns
        if (avg_length < 20.0f) {
            params.complexity_factor *= 0.9f;  // Simplify
        } else if (avg_length > 50.0f) {
            params.complexity_factor *= 1.1f;  // Allow more complexity
        }
        
        params.complexity_factor = std::clamp(params.complexity_factor, 0.5f, 2.0f);
    }

    void apply_conversation_patterns(float* logits, int context_length) {
        // Boost frequently used tokens in successful exchanges
        for (const auto& [token, freq] : conv_state.turn_frequencies) {
            if (freq > 0.1f) {  // Token appears in >10% of successful turns
                logits[token] += 0.2f * params.pattern_strength;
            }
        }
    }

    void suppress_dialogue_enders(float* logits) {
        for (llama_token token : dialogue_tokens) {
            if (punctuation.count(token)) {
                logits[token] -= 2.0f;
            }
        }
    }

    void boost_dialogue_enders(float* logits, float boost) {
        for (llama_token token : dialogue_tokens) {
            logits[token] += boost;
        }
    }
};

// Usage example for integration into rolled.cpp:
/*
// Global instance
ZipfAccelerator zipf_accel;

// In main(), after loading model:
zipf_accel.initialize(vocab);

// At start of each conversation turn:
zipf_accel.update_context(npc.name, mode_name, vocab);

// In generation loop, replace your current Zipf logic with:
zipf_accel.accelerate_logits(logits, i, max_tokens - i);

// For repetition penalty:
if (token_counts[token_id] > 0) {
    float penalty = zipf_accel.get_repetition_penalty(token_id, token_counts[token_id]);
    logits[token_id] *= penalty;
}
*/