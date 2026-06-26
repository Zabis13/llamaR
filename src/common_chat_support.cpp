// Minimal slice of llama.cpp's common/common.cpp, vendored for llamaR.
//
// The full common.cpp pulls in CLI argument parsing, model downloading, an
// HTTP client and the sampling stack — none of which the chat/template layer
// needs. The common_chat_* closure (chat.cpp, chat-parser*, peg-parser,
// json-schema-to-grammar, jinja/*) only calls a handful of string helpers and
// common_token_to_piece. We define exactly those here so the closure links
// without dragging in the rest.
//
// All of these are *declared* in common.h (which we vendor and include); this
// file supplies only their definitions. Keep the bodies byte-for-byte in sync
// with upstream common/common.cpp so behaviour matches llama-server.

#include "common.h"

#include "llama.h"
#include "ggml.h"

#include <cstdarg>
#include <cstdio>
#include <cctype>
#include <climits>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

// --- string helpers (upstream common.cpp) ------------------------------------

std::string string_format(const char * fmt, ...) {
    va_list ap;
    va_list ap2;
    va_start(ap, fmt);
    va_copy(ap2, ap);
    int size = vsnprintf(NULL, 0, fmt, ap);
    GGML_ASSERT(size >= 0 && size < INT_MAX); // NOLINT
    std::vector<char> buf(size + 1);
    int size2 = vsnprintf(buf.data(), size + 1, fmt, ap2);
    GGML_ASSERT(size2 == size);
    va_end(ap2);
    va_end(ap);
    return std::string(buf.data(), size);
}

std::string string_strip(const std::string & str) {
    size_t start = 0;
    size_t end = str.size();
    while (start < end && std::isspace(str[start])) {
        start++;
    }
    while (end > start && std::isspace(str[end - 1])) {
        end--;
    }
    return str.substr(start, end - start);
}

void string_replace_all(std::string & s, const std::string & search, const std::string & replace) {
    if (search.empty()) {
        return;
    }
    std::string builder;
    builder.reserve(s.length());
    size_t pos = 0;
    size_t last_pos = 0;
    while ((pos = s.find(search, last_pos)) != std::string::npos) {
        builder.append(s, last_pos, pos - last_pos);
        builder.append(replace);
        last_pos = pos + search.length();
    }
    builder.append(s, last_pos, std::string::npos);
    s = std::move(builder);
}

bool string_ends_with(const std::string_view & str, const std::string_view & suffix) {
    return str.size() >= suffix.size() && str.compare(str.size()-suffix.size(), suffix.size(), suffix) == 0;
}

bool string_remove_suffix(std::string & str, const std::string_view & suffix) {
    bool has_suffix = string_ends_with(str, suffix);
    if (has_suffix) {
        str = str.substr(0, str.size() - suffix.size());
    }
    return has_suffix;
}

size_t string_find_partial_stop(const std::string_view & str, const std::string_view & stop) {
    if (!str.empty() && !stop.empty()) {
        const char text_last_char = str.back();
        for (int64_t char_index = stop.size() - 1; char_index >= 0; char_index--) {
            if (stop[char_index] == text_last_char) {
                const auto current_partial = stop.substr(0, char_index + 1);
                if (string_ends_with(str, current_partial)) {
                    return str.size() - char_index - 1;
                }
            }
        }
    }

    return std::string::npos;
}

std::string regex_escape(const std::string & s) {
    static const std::regex special_chars("[.^$|()*+?\\[\\]{}\\\\]");
    return std::regex_replace(s, special_chars, "\\$&");
}

std::string string_join(const std::vector<std::string> & values, const std::string & separator) {
    std::ostringstream result;
    for (size_t i = 0; i < values.size(); ++i) {
        if (i > 0) {
            result << separator;
        }
        result << values[i];
    }
    return result.str();
}

std::vector<std::string> string_split(const std::string & str, const std::string & delimiter) {
    std::vector<std::string> parts;
    size_t start = 0;
    size_t end = str.find(delimiter);

    while (end != std::string::npos) {
        parts.push_back(str.substr(start, end - start));
        start = end + delimiter.length();
        end = str.find(delimiter, start);
    }

    parts.push_back(str.substr(start));

    return parts;
}

std::string string_repeat(const std::string & str, size_t n) {
    if (n == 0) {
        return "";
    }

    std::string result;
    result.reserve(str.length() * n);

    for (size_t i = 0; i < n; ++i) {
        result += str;
    }

    return result;
}

// --- token detokenisation (upstream common.cpp) ------------------------------

std::string common_token_to_piece(const struct llama_vocab * vocab, llama_token token, bool special) {
    std::string piece;
    piece.resize(piece.capacity());  // using string internal cache, 15 bytes + '\n'
    const int n_chars = llama_token_to_piece(vocab, token, &piece[0], piece.size(), 0, special);
    if (n_chars < 0) {
        piece.resize(-n_chars);
        int check = llama_token_to_piece(vocab, token, &piece[0], piece.size(), 0, special);
        GGML_ASSERT(check == -n_chars);
    } else {
        piece.resize(n_chars);
    }

    return piece;
}

std::string common_token_to_piece(const struct llama_context * ctx, llama_token token, bool special) {
    const llama_model * model = llama_get_model(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);
    return common_token_to_piece(vocab, token, special);
}

// --- logging support ---------------------------------------------------------

// Referenced by log.cpp. Inside an R session there is no useful TTY (log output
// is routed through REprintf), so colour escapes would only corrupt the
// console. Always disable them rather than vendoring the full tty probe.
bool tty_can_use_colors() {
    return false;
}
