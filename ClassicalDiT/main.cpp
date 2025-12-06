// main.cpp
#include <iostream>
#include <fstream>
// Helper function to append logs to CSV
void append_log(const std::string& filename, int epoch, double beta, double mse, double time_sec) {
    std::ofstream log_file;
    bool file_exists = std::ifstream(filename).good();

    log_file.open(filename, std::ios::app);
    if (!file_exists) {
        log_file << "epoch,beta,mse,epoch_time_sec\n";
    }
    log_file << epoch << "," << beta << "," << mse << "," << time_sec << "\n";
    log_file.close();
}
#include <sstream>
#include <vector>
#include <map>
#include <unordered_map>
#include <algorithm>
#include <random>
#include <string>
#include <cstdint>
#include <queue>
#include <memory>
#include <cstring>
#include <limits>
#include "BetaSchedule.hpp"
#include "NormalDist.hpp"
#include "GaussianDiffusion.hpp"
#include "Diffusion_model.hpp"
#include <chrono>

// =====================================================
// --- SIMPLE BPE IMPLEMENTATION (TRAIN + TOKENIZE) ---
// =====================================================

struct BPE {
    std::map<std::pair<std::string, std::string>, int> pair_count;
    std::map<std::string, std::vector<std::string>> vocab;
    std::vector<std::pair<std::string, std::string>> merges;

    static std::vector<std::string> chars(const std::string& w) {
        std::vector<std::string> c;
        for (char ch : w) c.push_back(std::string(1, ch));
        return c;
    }

    void count_pairs() {
        pair_count.clear();
        for (auto& it : vocab) {
            const auto& symbols = it.second;
            for (size_t i = 0; i + 1 < symbols.size(); i++) {
                pair_count[{symbols[i], symbols[i+1]}]++;
            }
        }
    }

    void merge(const std::pair<std::string, std::string>& p) {
        merges.push_back(p);

        for (auto& item : vocab) {
            std::vector<std::string>& symbols = item.second;
            std::vector<std::string> new_syms;

            for (size_t i = 0; i < symbols.size();) {
                if (i + 1 < symbols.size() && symbols[i] == p.first && symbols[i+1] == p.second) {
                    new_syms.push_back(p.first + p.second);
                    i += 2;
                } else {
                    new_syms.push_back(symbols[i]);
                    i++;
                }
            }
            symbols = new_syms;
        }
    }

    void train(const std::string& text, int merge_ops = 200) {
        vocab.clear();
        merges.clear();

        std::stringstream ss(text);
        std::string w;
        while (ss >> w) vocab[w] = chars(w);

        for (int i = 0; i < merge_ops; i++) {
            count_pairs();
            if (pair_count.empty()) break;

            auto best = std::max_element(
                pair_count.begin(), pair_count.end(),
                [](auto& a, auto& b) { return a.second < b.second; }
            )->first;

            merge(best);
        }
    }

    std::vector<std::string> tokenize(const std::string& text) const {
        std::stringstream ss(text);
        std::string w;
        std::vector<std::string> out;

        while (ss >> w) {
            std::vector<std::string> t = chars(w);
            for (auto& m : merges) {
                std::vector<std::string> newt;
                for (size_t i = 0; i < t.size();) {
                    if (i + 1 < t.size() && t[i] == m.first && t[i+1] == m.second) {
                        newt.push_back(m.first + m.second);
                        i += 2;
                    } else {
                        newt.push_back(t[i]);
                        i++;
                    }
                }
                t = newt;
            }
            out.insert(out.end(), t.begin(), t.end());
        }

        return out;
    }
};

// =====================================================
// --- SIMPLE EMBEDDING LAYER (dense float matrix) -----
// =====================================================

struct Embedding {
    int dim;
    std::unordered_map<std::string, std::vector<float>> table;

    Embedding(int d) : dim(d) {}

    const std::vector<float>& get(const std::string& tok) {
        auto it = table.find(tok);
        if (it != table.end()) return it->second;

        std::vector<float> v(dim);
        uint32_t h = 2166136261u;
        for (char c : tok) h = (h ^ (unsigned char)c) * 16777619u;

        for (int i = 0; i < dim; i++)
            v[i] = ((h >> ((i*7) % 24)) & 0xFF) / 255.0f;

        auto r = table.emplace(tok, std::move(v));
        return r.first->second;
    }
};

// ==============================
// --- FP16 (IEEE 754 half) -----
// ==============================

uint16_t float_to_fp16(float f) {
    uint32_t x;
    std::memcpy(&x, &f, sizeof(x));
    uint32_t sign = (x >> 16) & 0x8000;
    int32_t exponent = ((x >> 23) & 0xFF) - 127 + 15;
    uint32_t mantissa = x & 0x7FFFFF;

    if (exponent <= 0) {
        if (exponent < -10) return (uint16_t)sign;
        mantissa = (mantissa | 0x800000) >> (1 - exponent);
        return (uint16_t)(sign | (mantissa >> 13));
    } else if (exponent >= 31) {
        return (uint16_t)(sign | 0x7C00);
    }

    return (uint16_t)(sign | (exponent << 10) | (mantissa >> 13));
}

float fp16_to_float(uint16_t h) {
    uint32_t sign = (h & 0x8000) << 16;
    uint32_t exp = (h & 0x7C00) >> 10;
    uint32_t mant = (h & 0x03FF);

    uint32_t out;
    if (exp == 0) {
        if (mant == 0) {
            out = sign;
        } else {
            // subnormal
            int e = -14;
            float m = (float)mant / 1024.0f;
            float val = std::ldexp(m, e);
            std::memcpy(&out, &val, sizeof(out));
            out |= sign;
        }
    } else if (exp == 31) {
        out = sign | 0x7F800000 | (mant << 13);
    } else {
        int e = (int)exp - 15 + 127;
        out = sign | (e << 23) | (mant << 13);
    }
    float rf;
    std::memcpy(&rf, &out, sizeof(rf));
    return rf;
}

// =====================================================
// ------- CLIFFORD ALGEBRA COMPRESSION (GA) -----------
// =====================================================

struct CliffordMultivector {
    std::vector<float> scalars_fp32;
    std::vector<uint16_t> scalars_fp16;
    std::vector<float> scalars_fp32_full;
};

CliffordMultivector clifford_compress(const std::vector<float>& v) {
    CliffordMultivector mv;

    int k = std::min<int>(8, (int)v.size());
    mv.scalars_fp32.resize(k);
    mv.scalars_fp16.resize(k);

    for (int i = 0; i < k; i++) {
        mv.scalars_fp32[i] = v[i];
        mv.scalars_fp16[i] = float_to_fp16(v[i]);
    }

    mv.scalars_fp32_full = v;
    return mv;
}

// =====================================================
// ------------------ SERIALIZATION HELPERS -------------
// =====================================================

template<typename T>
void append_bytes(std::vector<uint8_t>& dst, const T* data, size_t count) {
    const uint8_t* p = reinterpret_cast<const uint8_t*>(data);
    dst.insert(dst.end(), p, p + sizeof(T) * count);
}

void append_bytes_raw(std::vector<uint8_t>& dst, const void* data, size_t nbytes) {
    const uint8_t* p = reinterpret_cast<const uint8_t*>(data);
    dst.insert(dst.end(), p, p + nbytes);
}

// =====================================================
// ---------------- HUFFMAN COMPRESSOR ------------------
// =====================================================

struct HuffNode {
    uint8_t byte;               // valid if leaf
    uint64_t freq;
    std::shared_ptr<HuffNode> left;
    std::shared_ptr<HuffNode> right;

    HuffNode(uint8_t b, uint64_t f) : byte(b), freq(f) {}
    HuffNode(std::shared_ptr<HuffNode> l, std::shared_ptr<HuffNode> r) : byte(0), freq(l->freq + r->freq), left(l), right(r) {}
};

struct NodeCmp {
    bool operator()(const std::shared_ptr<HuffNode>& a, const std::shared_ptr<HuffNode>& b) const {
        return a->freq > b->freq;
    }
};

// build codes map
void build_codes(const std::shared_ptr<HuffNode>& node, std::vector<std::string>& codes, std::string cur = "") {
    if (!node) return;
    if (!node->left && !node->right) {
        // leaf
        codes[node->byte] = cur.empty() ? "0" : cur; // handle single-symbol case
        return;
    }
    if (node->left) build_codes(node->left, codes, cur + "0");
    if (node->right) build_codes(node->right, codes, cur + "1");
}

// main Huffman compress function: writes to outPath
bool huffman_compress_and_write(const std::vector<uint8_t>& input, const std::string& outPath) {
    // 1) frequency table
    std::vector<uint64_t> freq(256, 0);
    for (uint8_t b : input) freq[b]++;

    // 2) build priority queue
    std::priority_queue<std::shared_ptr<HuffNode>, std::vector<std::shared_ptr<HuffNode>>, NodeCmp> pq;
    for (int i = 0; i < 256; ++i) {
        if (freq[i] > 0) {
            pq.push(std::make_shared<HuffNode>((uint8_t)i, freq[i]));
        }
    }

    // Edge-case: if input empty, write empty freq table and return
    if (input.empty()) {
        std::ofstream out(outPath, std::ios::binary);
        if (!out) return false;
        for (int i = 0; i < 256; ++i) {
            uint64_t z = 0;
            out.write(reinterpret_cast<char*>(&z), sizeof(z));
        }
        return true;
    }

    // If only one unique byte, create a single node (special handling)
    if (pq.size() == 1) {
        auto only = pq.top(); pq.pop();
        // create a dummy partner with zero freq
        auto dummy = std::make_shared<HuffNode>((uint8_t)((only->byte + 1) & 0xFF), (uint64_t)0);
        pq.push(only);
        pq.push(dummy);
    }

    while (pq.size() > 1) {
        auto a = pq.top(); pq.pop();
        auto b = pq.top(); pq.pop();
        auto parent = std::make_shared<HuffNode>(a, b);
        pq.push(parent);
    }

    auto root = pq.top();

    // 3) generate codes
    std::vector<std::string> codes(256);
    build_codes(root, codes);

    // 4) encode input to bits
    std::vector<uint8_t> bitstream;
    bitstream.reserve((input.size() * 3) / 2 + 16); // estimate

    uint8_t cur_byte = 0;
    int cur_bits = 0;

    for (uint8_t b : input) {
        const std::string& code = codes[b];
        for (char bit : code) {
            cur_byte = (cur_byte << 1) | (bit == '1' ? 1 : 0);
            cur_bits++;
            if (cur_bits == 8) {
                bitstream.push_back(cur_byte);
                cur_byte = 0;
                cur_bits = 0;
            }
        }
    }
    // flush remaining bits (pad with zeros on the right)
    if (cur_bits > 0) {
        cur_byte <<= (8 - cur_bits);
        bitstream.push_back(cur_byte);
    }

    // 5) write file: frequency table (256 x uint64_t) followed by uint32_t bitstream size and bitstream bytes
    std::ofstream out(outPath, std::ios::binary);
    if (!out) return false;

    // write frequency table (so decoder can rebuild identical tree)
    for (int i = 0; i < 256; ++i) {
        uint64_t f = freq[i];
        out.write(reinterpret_cast<char*>(&f), sizeof(f));
    }

    // write 64-bit original input length in bytes (for sanity)
    uint64_t orig_len = input.size();
    out.write(reinterpret_cast<char*>(&orig_len), sizeof(orig_len));

    // write 64-bit bitstream length in bytes
    uint64_t bits_bytes = bitstream.size();
    out.write(reinterpret_cast<char*>(&bits_bytes), sizeof(bits_bytes));

    // write packed bitstream
    if (!bitstream.empty())
        out.write(reinterpret_cast<const char*>(bitstream.data()), bitstream.size());

    out.close();
    return true;
}

// Vectorized log_prob_from_predictions
std::vector<double> log_prob_batch(const std::vector<double>& y,
                                   const std::vector<double>& x_start_pred,
                                   const std::vector<double>& eps_pred) {
    size_t n = y.size();
    if (x_start_pred.size() != n || eps_pred.size() != n) {
        throw std::invalid_argument("Input vector sizes must match");
    }
    std::vector<double> log_probs(n);
    for (size_t i = 0; i < n; ++i) {
        double mean = NormalDist::compute_mean(x_start_pred[i], eps_pred[i]);
        double sigma = NormalDist::compute_sigma(x_start_pred[i], eps_pred[i]);
        if (sigma <= 0) {
            throw std::invalid_argument("Sigma must be positive");
        }
        log_probs[i] = NormalDist::log_prob(y[i], mean, sigma);
    }
    return log_probs;
}

// Vectorized negative log likelihood
std::vector<double> nll_batch(const std::vector<double>& y,
                              const std::vector<double>& x_start_pred,
                              const std::vector<double>& eps_pred) {
    std::vector<double> log_probs = log_prob_batch(y, x_start_pred, eps_pred);
    for (auto& lp : log_probs) lp = -lp; // NLL = -log_prob
    return log_probs;
}

// Mean Squared Error (mse) Helper
double compute_mse(const std::vector<double>& predicted, const std::vector<double>& target) {
    if (predicted.size() != target.size()) {
        throw std::invalid_argument("Vectors must be the same size for MSE");
    }
    double mse = 0.0;
    for (size_t i = 0; i < predicted.size(); ++i) {
        double diff = predicted[i] - target[i];
        mse += diff * diff;
    }
    return mse / predicted.size();
}

// Fidelity Error Reporting in Batch
double compute_batch_mse(const std::vector<std::vector<double>>& preds,
                        const std::vector<std::vector<double>>& targets) {
    if (preds.size() != targets.size()) {
        throw std::invalid_argument("Batch sizes must match");
    }
    double total_mse = 0.0;
    for (size_t i = 0; i < preds.size(); ++i) {
        total_mse += compute_mse(preds[i], targets[i]);
    }
    return total_mse / preds.size();
}

// =====================================================
// ---------------------- MAIN -------------------------
// =====================================================

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: ./MyProject <dataset.txt> <output.huff>\n";
        return 1;
    }

    std::string inputPath  = argv[1];
    std::string outputPath = argv[2];

    // Read dataset
    std::ifstream in(inputPath);
    if (!in) {
        std::cerr << "Could not open dataset: " << inputPath << "\n";
        return 1;
    }
    std::stringstream buf;
    buf << in.rdbuf();
    std::string text = buf.str();

   // 1) Train BPE
    std::cout << "[1] Training BPE tokenizer...\n";
    BPE bpe;
    bpe.train(text, 200);

    // 2) Tokenize dataset
    std::cout << "[2] Tokenizing...\n";
    auto tokens = bpe.tokenize(text);
    std::cout << "   Tokens: " << tokens.size() << "\n";

    // 3) Create embedding table
    Embedding embed(64);

    // 4) Embed, Clifford compress, serialize
    std::cout << "[3] Embedding + Clifford compression + serialize...\n";
    std::vector<uint8_t> buffer;
    buffer.reserve(tokens.size() * 320);

    for (const auto& t : tokens) {
        const auto& vec = embed.get(t);
        auto mv = clifford_compress(vec);

        // ---- FP32[8] ----
        uint8_t n32 = static_cast<uint8_t>(mv.scalars_fp32.size());
        append_bytes(buffer, &n32, 1);
        append_bytes(buffer, mv.scalars_fp32.data(),
                 n32 * sizeof(float));

        // ---- FP16[8] ----
        uint8_t n16 = static_cast<uint8_t>(mv.scalars_fp16.size());
        append_bytes(buffer, &n16, 1);
        append_bytes(buffer, mv.scalars_fp16.data(),
                 n16 * sizeof(uint16_t));

        // ---- FULL FP32[64] ----
        uint16_t nfull = static_cast<uint16_t>(mv.scalars_fp32_full.size());
        append_bytes(buffer, &nfull, sizeof(uint16_t));
        append_bytes(buffer, mv.scalars_fp32_full.data(),
                 nfull * sizeof(float));
    }

    std::cout << "[4] Raw serialized bytes: " << buffer.size() << "\n";

    // 5) Huffman compress + write dataset
    std::cout << "[5] Building Huffman and writing to: " << outputPath << "\n";
    if (!huffman_compress_and_write(buffer, outputPath)) {
        std::cerr << "Huffman compression / write failed.\n";
        return 1;
    }

    // 6) Initialize BetaSchedule and DiffusionModel, and run training loop
    const int NUM_EPOCHS = 100;
    const double INITIAL_BETA = 0.1;
    const int BATCH_SIZE = 64;
    std::cout << "[6] Initializing BetaSchedule and DiffusionModel, starting training loop...\n";

    BetaSchedule beta_schedule(NUM_EPOCHS, INITIAL_BETA);

    // Helper struct for batch conversion
    struct Multivector {
        std::vector<float> scalars_fp32_full;
    };
    // Properly compress embeddings using Clifford compression
    std::vector<Multivector> compressed_mv;
    compressed_mv.clear();
    for (const auto& t : tokens) {
        const auto& vec = embed.get(t);
        CliffordMultivector mv = clifford_compress(vec);
        Multivector cmv;
        cmv.scalars_fp32_full.assign(mv.scalars_fp32_full.begin(), mv.scalars_fp32_full.end());
        compressed_mv.push_back(cmv);
    }
    // Use compressed_mv for training

    // Helper: convert batch Multivector to vector<vector<double>>
    auto batch_to_vectors = [](const std::vector<Multivector>& batch_mv) {
        std::vector<std::vector<double>> batch_vectors;
        batch_vectors.reserve(batch_mv.size());
        for (const auto& mv : batch_mv) {
            std::vector<double> v(mv.scalars_fp32_full.begin(), mv.scalars_fp32_full.end());
            batch_vectors.push_back(std::move(v));
        }
        return batch_vectors;
    };

    // Vectorized loss function
    auto compute_nll_batch = [](const std::vector<std::vector<double>>& y_batch,
                                const std::vector<std::vector<double>>& x_start_pred_batch,
                                const std::vector<std::vector<double>>& eps_pred_batch) {
        std::vector<double> losses;
        losses.reserve(y_batch.size());
        for (size_t i = 0; i < y_batch.size(); ++i) {
            const auto& y = y_batch[i];
            const auto& x_start_pred = x_start_pred_batch[i];
            const auto& eps_pred = eps_pred_batch[i];
            double sample_loss = 0.0;
            for (size_t j = 0; j < y.size(); ++j) {
                try {
                    double logp = NormalDist::log_prob_from_predictions(y[j], x_start_pred[j], eps_pred[j]);
                    sample_loss += -logp;
                } catch (const std::invalid_argument& e) {
                    sample_loss += 1e6;
                }
            }
            losses.push_back(sample_loss / y.size());
        }
        return losses;
    };

    // ===========================
    // --- DIFFUSION TRAINING ----
    // ===========================

    // Prepare training data as vector<vector<double>>
    std::vector<std::vector<double>> training_data;
    training_data.reserve(compressed_mv.size());
    for (const auto& mv : compressed_mv) {
        training_data.emplace_back(mv.scalars_fp32_full.begin(), mv.scalars_fp32_full.end());
    }

    // Initialize diffusion model parameters vector
    std::vector<double> model_params(training_data[0].size(), 0.1);  // example init

    // Initialize GaussianDiffusion
    int num_timesteps = 1000;
    double beta_start = 0.0001;
    double beta_end = 0.02;
    GaussianDiffusion diffusion(num_timesteps, beta_start, beta_end);

    std::vector<double> nll_losses;
    std::vector<double> entropy_losses;

    // Define model epsilon prediction lambda for train function
    auto model_predict_epsilon = [](const std::vector<double>& x_t, int t, const std::vector<double>& params) -> std::vector<double> {
        // TODO: Replace with your real model prediction logic
        // For now, dummy prediction: just return params as epsilon prediction vector
        return params;
    };

    std::string log_filename = "training_log.csv";

    for (int epoch = 0; epoch < NUM_EPOCHS; ++epoch) {
        auto start = std::chrono::high_resolution_clock::now();

        diffusion.train(training_data, 1, nll_losses, entropy_losses, model_params, model_predict_epsilon);

        // Mock prediction for MSE (replace with your real predicted batch vectors)
        std::vector<std::vector<double>> predicted_batch = training_data; // dummy: perfect prediction

        double mse = compute_batch_mse(predicted_batch, training_data);

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;

        beta_schedule.update(nll_losses, entropy_losses, epoch);

        std::cout << "Epoch " << (epoch + 1) << "/" << NUM_EPOCHS
                  << ", Beta current: " << beta_schedule.getCurrentBeta()
                  << ", MSE: " << mse
                  << ", Epoch time: " << elapsed.count() << "s\n";

        append_log(log_filename, epoch + 1, beta_schedule.getCurrentBeta(), mse, elapsed.count());
    }

    // Run a prediction on first training sample and save to file
    std::vector<double> sample_input = training_data[0];

    // Use your model_predict_epsilon function to get predicted noise
    std::vector<double> predicted_epsilon = model_predict_epsilon(sample_input, 0, model_params);

    // Save predicted epsilon vector to a CSV file for visualization
    std::ofstream pred_out("sample_prediction.csv");
    pred_out << "index,value\n";
    for (size_t i = 0; i < predicted_epsilon.size(); ++i) {
        pred_out << i << "," << predicted_epsilon[i] << "\n";
    }
    pred_out.close();

    std::cout << "Sample prediction saved to sample_prediction.csv\n";

    return 0;
}