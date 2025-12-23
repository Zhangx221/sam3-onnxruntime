#include "clip_bpe.h"
#include <algorithm>
#include <codecvt>
#include <fstream>
#include <iostream>
#include <locale>
#include <regex>

Tokenizer::Tokenizer(const std::string& vocab_file, const std::string& merges_file, int max_length)
{
    // 1. 创建字节到 Unicode 的映射
    create_byte_encoder();
    model_max_length = max_length;
    // 2. 从 JSON 加载词汇表
    std::ifstream vocab_stream(vocab_file);
    if (!vocab_stream.is_open()) {
        throw std::runtime_error("Cannot open vocab file: " + vocab_file);
    }
    vocab_stream >> encoder;

    // 3. 加载特殊标记
    added_tokens_decoder["<|startoftext|>"] = 49406;
    added_tokens_decoder["<|endoftext|>"]   = 49407;

    // 4. 加载 BPE 合并规则
    load_merges(merges_file);

    for (auto it = bpe_ranks.begin(); it != bpe_ranks.end(); ++it) {
        const auto& pair            = it->first;
        const auto& rank            = it->second;
        const auto& [first, second] = pair;
        // std::wcout << "Pair: (" << first << ", " << second << "), Rank: " << rank << std::endl;
    }
}

void Tokenizer::create_byte_encoder()
{
    // 初始化基本字符范围
    std::vector<int> bs;
    // 添加可打印ASCII字符 (! 到 ~)
    for (int i = '!'; i <= '~'; ++i) {
        bs.push_back(i);
    }
    // 添加拉丁扩展字符 (¡ 到 ¬)
    for (int i = 0xA1; i <= 0xAC; ++i) {
        bs.push_back(i);
    }
    // 添加更多拉丁字符 (® 到 ÿ)
    for (int i = 0xAE; i <= 0xFF; ++i) {
        bs.push_back(i);
    }

    std::vector<int> cs = bs;
    int              n  = 0;

    // 处理所有可能的字节值 (0-255)
    for (int b = 0; b < 256; ++b) {
        bool found = false;
        for (int val : bs) {
            if (val == b) {
                found = true;
                break;
            }
        }

        if (!found) {
            bs.push_back(b);
            cs.push_back(256 + n);
            ++n;
        }
    }

    // 创建映射字典

    for (size_t i = 0; i < bs.size(); ++i) {
        byte_encoder[bs[i]] = static_cast<wchar_t>(cs[i]);
        // std::wcout << bs[i] << L"  " << byte_encoder[bs[i]] << std::endl;
    }
}

// std::unordered_map<std::pair<std::string, std::string>, int, PairHash> bpe_ranks;

void Tokenizer::load_merges(const std::string& merges_file)
{
    std::ifstream merges_stream(merges_file);
    if (!merges_stream.is_open()) {
        throw std::runtime_error("Cannot open merges file: " + merges_file);
    }

    std::string line;
    // 跳过第一行（标题）
    std::getline(merges_stream, line);

    int rank = 0;
    // Python 切片为 [1 : 49152 - 256 - 2 + 1]，意味着 48895 行
    for (int i = 0; i < 48895 && std::getline(merges_stream, line); ++i) {
        size_t space_pos = line.find(' ');
        if (space_pos != std::string::npos) {
            std::string w1_str = line.substr(0, space_pos);
            std::string w2_str = line.substr(space_pos + 1);

            // 将 std::string (UTF-8) 转换为 std::wstring
            std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;
            std::wstring                                     w1 = converter.from_bytes(w1_str);
            std::wstring                                     w2 = converter.from_bytes(w2_str);

            bpe_ranks[std::make_pair(w1, w2)] = rank++;
        }
    }
}

std::set<std::pair<std::wstring, std::wstring>> Tokenizer::get_pairs(const std::vector<std::wstring>& word)
{
    std::set<std::pair<std::wstring, std::wstring>> pairs;
    if (word.size() < 2)
        return pairs;
    for (size_t i = 0; i < word.size() - 1; ++i) {
        pairs.insert(std::make_pair(word[i], word[i + 1]));
    }
    return pairs;
}

std::wstring Tokenizer::bpe(const std::wstring& token)
{
    if (token.empty())
        return L"";

    std::vector<std::wstring> word;
    for (size_t i = 0; i < token.length() - 1; ++i) {
        word.push_back(std::wstring(1, token[i]));
    }
    word.push_back(std::wstring(1, token.back()) + L"</w>");

    auto pairs = get_pairs(word);
    if (pairs.empty()) {
        return token + L"</w>";
    }

    while (true) {
        auto min_it = std::min_element(pairs.begin(), pairs.end(), [this](const auto& a, const auto& b) {
            auto rank_a = bpe_ranks.count(a) ? bpe_ranks.at(a) : std::numeric_limits<int>::max();
            auto rank_b = bpe_ranks.count(b) ? bpe_ranks.at(b) : std::numeric_limits<int>::max();
            return rank_a < rank_b;
        });

        const auto& bigram = *min_it;

        if (bpe_ranks.find(bigram) == bpe_ranks.end()) {
            break;
        }

        const std::wstring&       first  = bigram.first;
        const std::wstring&       second = bigram.second;
        std::vector<std::wstring> new_word;
        size_t                    i = 0;
        while (i < word.size()) {
            auto it = std::find(word.begin() + i, word.end(), first);
            if (it == word.end()) {
                new_word.insert(new_word.end(), word.begin() + i, word.end());
                break;
            }
            new_word.insert(new_word.end(), word.begin() + i, it);
            i = std::distance(word.begin(), it);

            if (i < word.size() - 1 && word[i] == first && word[i + 1] == second) {
                new_word.push_back(first + second);
                i += 2;
            }
            else {
                new_word.push_back(word[i]);
                i += 1;
            }
        }
        word = new_word;
        if (word.size() == 1) {
            break;
        }
        pairs = get_pairs(word);
    }

    std::wstring result;
    for (size_t i = 0; i < word.size(); ++i) {
        result += word[i] + (i == word.size() - 1 ? L"" : L" ");
    }
    return result;
}

int Tokenizer::convert_token_to_id(const std::wstring& wtoken)
{
    // Convert wstring back to UTF-8 string for JSON key lookup
    std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;
    std::string                                      token = converter.to_bytes(wtoken);

    if (added_tokens_decoder.count(token)) {
        return added_tokens_decoder.at(token);
    }
    if (encoder.contains(token)) {
        return encoder[token].get<int>();
    }
    // Fallback to unknown token '<|endoftext|>' if present
    if (encoder.contains("<|endoftext|>")) {
        return encoder["<|endoftext|>"].get<int>();
    }
    // As a last resort, return eos_token_id constant
    return eos_token_id;
}

std::vector<int> Tokenizer::encode(std::string& prompt)
{
    // 1. Lowercase the prompt
    std::transform(prompt.begin(), prompt.end(), prompt.begin(), [](unsigned char c) { return std::tolower(c); });

    // 2. Regex tokenization
    // NOTE: std::wregex is used for better Unicode property support (\p{L}, \p{N})
    std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;
    std::wstring                                     wprompt = converter.from_bytes(prompt);
    // std::wregex pat(L"<\\|startoftext\\|>|<\\|endoftext\\|>|'s|'t|'re|'ve|'m|'ll|'d|[\\p{L}]+|[\\p{N}]|[^\\s\\p{L}\\p{N}]+",
    // std::regex_constants::icase); 修正后的正则表达式模式，模拟Python的行为
    std::wregex pattern(LR"(<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[a-zA-ZÀ-ÿ]+|[0-9]+|[^\s\w]+)", std::regex_constants::icase);
    auto        words_begin = std::wsregex_iterator(wprompt.begin(), wprompt.end(), pattern);
    auto        words_end   = std::wsregex_iterator();

    // std::vector<std::wstring> wtokens;
    // while (words_begin != words_end)
    // {
    //     wtokens.push_back(words_begin->str());
    //     ++words_begin;
    // }
    // // 将宽字符结果转换回UTF-8字符串
    // std::vector<std::string> tokens;
    // std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> converter1;
    // for (const auto& wtoken : wtokens) {
    //     tokens.push_back(converter1.to_bytes(wtoken));
    //     std::cout<<converter1.to_bytes(wtoken)<<std::endl;
    // }

    std::vector<std::wstring> bpe_tokens_ws;
    for (std::wsregex_iterator i = words_begin; i != words_end; ++i) {
        std::wstring raw_token = i->str();
        // 3. Apply byte encoder
        // std::string encoded_token;
        // std::string utf8_token = converter.to_bytes(raw_token);
        // for (unsigned char b : utf8_token)
        // {
        //     encoded_token += byte_encoder.at(b);
        // }

        // 4. Apply BPE and split the result
        std::wstring bpe_result = bpe(raw_token);
        // std::wcout << bpe_result << std::endl;

        std::wstringstream ss(bpe_result);
        std::wstring       bpe_sub_token;
        while (ss >> bpe_sub_token) {
            bpe_tokens_ws.push_back(bpe_sub_token);
        }
    }

    // 5. Convert BPE tokens to IDs/
    std::vector<int> ids;
    for (const auto& token : bpe_tokens_ws) {
        ids.push_back(convert_token_to_id(token));
    }

    // 6. Add special tokens, pad, and truncate
    std::vector<int> final_ids;
    final_ids.push_back(bos_token_id);
    final_ids.insert(final_ids.end(), ids.begin(), ids.end());
    final_ids.push_back(eos_token_id);

    if (final_ids.size() > static_cast<size_t>(model_max_length)) {
        final_ids.resize(model_max_length - 1);
        final_ids.push_back(eos_token_id);
    }
    else {
        size_t padding_needed = model_max_length - final_ids.size();
        final_ids.insert(final_ids.end(), padding_needed, eos_token_id);
    }

    return final_ids;
}


std::pair<std::vector<int>, std::vector<int>> Tokenizer::encode_with_mask(std::string& prompt)
{

    /* 1~5 步与原来 encode 完全相同，直接拷贝即可 */
    std::transform(prompt.begin(), prompt.end(), prompt.begin(), [](unsigned char c) { return std::tolower(c); });

    std::wstring_convert<std::codecvt_utf8<wchar_t>> cvt;
    std::wstring                                     wp = cvt.from_bytes(prompt);

    std::wregex pat(LR"(<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[a-zA-ZÀ-ÿ]+|[0-9]+|[^\s\w]+)", std::regex_constants::icase);

    std::vector<std::wstring> bpe_tokens_ws;
    for (std::wsregex_iterator it(wp.begin(), wp.end(), pat), end; it != end; ++it) {
        std::wstring       raw     = it->str();
        std::wstring       bpe_out = bpe(raw);   // 已有函数
        std::wstringstream ss(bpe_out);
        std::wstring       sub;
        while (ss >> sub)
            bpe_tokens_ws.push_back(sub);
    }

    std::vector<int> ids;
    for (auto& t : bpe_tokens_ws)
        ids.push_back(convert_token_to_id(t));

    /* 6. 构造最终 ids + attention_mask */
    std::vector<int> final_ids;
    final_ids.reserve(model_max_length);
    final_ids.push_back(bos_token_id);
    final_ids.insert(final_ids.end(), ids.begin(), ids.end());
    final_ids.push_back(eos_token_id);

    size_t valid_len = final_ids.size();   // 含 BOS/EOS
    if (valid_len > static_cast<size_t>(model_max_length)) {
        final_ids.resize(model_max_length - 1);
        final_ids.push_back(eos_token_id);
        valid_len = model_max_length;
    }
    else {
        final_ids.resize(model_max_length, eos_token_id);
    }

    std::vector<int> mask(valid_len, 1);
    mask.resize(model_max_length, 0);

    return {final_ids, mask};
}