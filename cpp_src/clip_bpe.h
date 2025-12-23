

#ifndef TOKENIZER_HPP
#define TOKENIZER_HPP

// clip_bpe.cpp
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <regex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using json = nlohmann::json;
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>


// 用于 unordered_map 的 std::pair 哈希函数
struct PairHash
{
    template<class T1, class T2>
    std::size_t operator()(const std::pair<T1, T2>& p) const
    {
        auto h1 = std::hash<T1>{}(p.first);
        auto h2 = std::hash<T2>{}(p.second);
        // 一种简单的哈希组合方法
        return h1 ^ (h2 << 1);
    }
};

class Tokenizer
{
public:
    Tokenizer(const std::string& vocab_file, const std::string& merges_file, int max_length);
    std::vector<int>                              encode(std::string& prompt);
    std::pair<std::vector<int>, std::vector<int>> encode_with_mask(std::string& prompt);

private:
    // 词汇表和 BPE 排名
    json                                                                     encoder;
    std::unordered_map<std::string, int>                                     added_tokens_decoder;
    std::unordered_map<std::pair<std::wstring, std::wstring>, int, PairHash> bpe_ranks;

    // 字节编码/解码映射
    std::map<unsigned int, wchar_t> byte_encoder;

    // 常量
    int       model_max_length = 32;
    const int bos_token_id     = 49406;
    const int eos_token_id     = 49407;

    // 初始化方法
    void create_byte_encoder();
    void load_merges(const std::string& merges_file);

    // BPE 核心逻辑
    std::set<std::pair<std::wstring, std::wstring>> get_pairs(const std::vector<std::wstring>& word);
    std::wstring                                    bpe(const std::wstring& token);

    // 辅助方法
    int convert_token_to_id(const std::wstring& token);
};

#endif   // TOKENIZER_HPP