// R bindings for the vendored common_chat tool-aware template + parsing layer.
//
// Two entry points, mirroring how llama-server uses common/chat.cpp:
//
//   r_llama_chat_build(model, messages_json, tools_json, tool_choice,
//                      json_schema, add_generation_prompt, enable_thinking)
//       Applies the model's chat template (Jinja path) to OpenAI-shaped
//       messages + tools, returning the prompt string plus the grammar and
//       format id needed to constrain and later parse the model's output.
//
//   r_llama_chat_parse(input, format_id, is_partial)
//       Parses raw model output for the given format id back into
//       content / reasoning_content / tool_calls.
//
// messages_json and tools_json are JSON strings in OpenAI's chat-completions
// shape; we hand them to common_chat_msgs_parse_oaicompat /
// common_chat_tools_parse_oaicompat, which is exactly what the server does.

#include <string>
#include <vector>

// IMPORTANT: include the heavy C++ headers (which pull in <filesystem>,
// <locale>, etc.) BEFORE R headers. Rinternals.h defines a `length(x)` macro
// that otherwise mangles those STL headers (e.g. codecvt::length).
#include "llama.h"
#include "chat.h"
#include <nlohmann/json.hpp>

#include <R.h>
#include <Rinternals.h>

using json = nlohmann::ordered_json;

namespace {

// Extract a UTF-8 std::string from the first element of an R character vector.
std::string sexp_to_string(SEXP x) {
    if (Rf_isNull(x) || TYPEOF(x) != STRSXP || Rf_length(x) == 0) {
        return std::string();
    }
    SEXP c = STRING_ELT(x, 0);
    if (c == NA_STRING) return std::string();
    return std::string(Rf_translateCharUTF8(c));
}

SEXP make_utf8_string(const std::string & s) {
    SEXP r = PROTECT(Rf_allocVector(STRSXP, 1));
    SET_STRING_ELT(r, 0, Rf_mkCharLenCE(s.c_str(), (int) s.size(), CE_UTF8));
    UNPROTECT(1);
    return r;
}

} // namespace

// ----------------------------------------------------------------------------
// r_llama_chat_build
// ----------------------------------------------------------------------------

extern "C" SEXP r_llama_chat_build(SEXP r_model, SEXP r_messages_json,
                                   SEXP r_tools_json, SEXP r_tool_choice,
                                   SEXP r_json_schema, SEXP r_add_gen,
                                   SEXP r_enable_thinking) {
    const llama_model * model = (const llama_model *) R_ExternalPtrAddr(r_model);
    if (model == nullptr) {
        Rf_error("llamaR: chat_build received a null model pointer");
    }

    const std::string messages_json = sexp_to_string(r_messages_json);
    const std::string tools_json    = sexp_to_string(r_tools_json);
    const std::string tool_choice   = sexp_to_string(r_tool_choice);
    const std::string json_schema   = sexp_to_string(r_json_schema);
    const bool add_gen        = Rf_asLogical(r_add_gen) == TRUE;
    const bool enable_thinking = Rf_asLogical(r_enable_thinking) == TRUE;

    try {
        common_chat_templates_ptr tmpls =
            common_chat_templates_init(model, /* chat_template_override = */ "");

        common_chat_templates_inputs inputs;
        inputs.use_jinja = true;
        inputs.add_generation_prompt = add_gen;
        inputs.enable_thinking = enable_thinking;

        if (!messages_json.empty()) {
            inputs.messages = common_chat_msgs_parse_oaicompat(json::parse(messages_json));
        }
        if (!tools_json.empty()) {
            inputs.tools = common_chat_tools_parse_oaicompat(json::parse(tools_json));
        }
        if (!tool_choice.empty()) {
            inputs.tool_choice = common_chat_tool_choice_parse_oaicompat(tool_choice);
        }
        if (!json_schema.empty()) {
            inputs.json_schema = json_schema;
        }

        common_chat_params params = common_chat_templates_apply(tmpls.get(), inputs);

        // Convert grammar_triggers into the (trigger_patterns, trigger_tokens)
        // form expected by llama_sampler_init_grammar_lazy_patterns(). This
        // mirrors common/sampling.cpp exactly so lazy grammars (e.g. Mistral /
        // Ministral, which only constrain output after a [TOOL_CALLS] trigger)
        // behave like llama-server. R passes these back into the generation call.
        std::vector<std::string> trigger_patterns;
        std::vector<int>         trigger_tokens;
        for (const auto & trigger : params.grammar_triggers) {
            switch (trigger.type) {
                case COMMON_GRAMMAR_TRIGGER_TYPE_WORD:
                    trigger_patterns.push_back(regex_escape(trigger.value));
                    break;
                case COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN:
                    trigger_patterns.push_back(trigger.value);
                    break;
                case COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN_FULL: {
                    const auto & pattern = trigger.value;
                    std::string anchored = "^$";
                    if (!pattern.empty()) {
                        anchored = (pattern.front() != '^' ? "^" : "")
                            + pattern
                            + (pattern.back() != '$' ? "$" : "");
                    }
                    trigger_patterns.push_back(anchored);
                    break;
                }
                case COMMON_GRAMMAR_TRIGGER_TYPE_TOKEN:
                    trigger_tokens.push_back((int) trigger.token);
                    break;
            }
        }

        // Pack: prompt, grammar, format(int), grammar_lazy(bool),
        //       additional_stops, preserved_tokens, parser(serialized PEG arena),
        //       trigger_patterns(chr), trigger_tokens(int)
        //
        // `parser` is the serialized PEG parser arena (common_peg_arena::save()).
        // PEG-based formats (PEG_SIMPLE/NATIVE/CONSTRUCTED) are parsed by
        // common_chat_peg_parse(), which needs this arena; chat_parse() passes
        // it back in. Empty for non-PEG formats.
        const char * names[] = {"prompt", "grammar", "format", "grammar_lazy",
                                "additional_stops", "preserved_tokens", "parser",
                                "trigger_patterns", "trigger_tokens", ""};
        SEXP result = PROTECT(Rf_mkNamed(VECSXP, names));

        SET_VECTOR_ELT(result, 0, make_utf8_string(params.prompt));
        SET_VECTOR_ELT(result, 1, make_utf8_string(params.grammar));
        SET_VECTOR_ELT(result, 2, Rf_ScalarInteger((int) params.format));
        SET_VECTOR_ELT(result, 3, Rf_ScalarLogical(params.grammar_lazy));

        SEXP stops = PROTECT(Rf_allocVector(STRSXP, (R_xlen_t) params.additional_stops.size()));
        for (size_t i = 0; i < params.additional_stops.size(); i++) {
            SET_STRING_ELT(stops, (R_xlen_t) i,
                Rf_mkCharLenCE(params.additional_stops[i].c_str(),
                               (int) params.additional_stops[i].size(), CE_UTF8));
        }
        SET_VECTOR_ELT(result, 4, stops);

        SEXP preserved = PROTECT(Rf_allocVector(STRSXP, (R_xlen_t) params.preserved_tokens.size()));
        for (size_t i = 0; i < params.preserved_tokens.size(); i++) {
            SET_STRING_ELT(preserved, (R_xlen_t) i,
                Rf_mkCharLenCE(params.preserved_tokens[i].c_str(),
                               (int) params.preserved_tokens[i].size(), CE_UTF8));
        }
        SET_VECTOR_ELT(result, 5, preserved);

        // parser arena: serialized JSON (empty string for non-PEG formats)
        SET_VECTOR_ELT(result, 6, make_utf8_string(params.parser));

        SEXP tpat = PROTECT(Rf_allocVector(STRSXP, (R_xlen_t) trigger_patterns.size()));
        for (size_t i = 0; i < trigger_patterns.size(); i++) {
            SET_STRING_ELT(tpat, (R_xlen_t) i,
                Rf_mkCharLenCE(trigger_patterns[i].c_str(),
                               (int) trigger_patterns[i].size(), CE_UTF8));
        }
        SET_VECTOR_ELT(result, 7, tpat);

        SEXP ttok = PROTECT(Rf_allocVector(INTSXP, (R_xlen_t) trigger_tokens.size()));
        for (size_t i = 0; i < trigger_tokens.size(); i++) {
            INTEGER(ttok)[i] = trigger_tokens[i];
        }
        SET_VECTOR_ELT(result, 8, ttok);

        UNPROTECT(5);  // result, stops, preserved, tpat, ttok
        return result;
    } catch (const std::exception & e) {
        Rf_error("llamaR: chat_build failed: %s", e.what());
    }
    return R_NilValue;  // unreachable
}

// ----------------------------------------------------------------------------
// r_llama_chat_parse
// ----------------------------------------------------------------------------

extern "C" SEXP r_llama_chat_parse(SEXP r_input, SEXP r_format, SEXP r_is_partial,
                                   SEXP r_parser) {
    const std::string input  = sexp_to_string(r_input);
    const std::string parser = sexp_to_string(r_parser);
    const int format_id      = Rf_asInteger(r_format);
    const bool is_partial    = Rf_asLogical(r_is_partial) == TRUE;

    if (format_id < 0 || format_id >= COMMON_CHAT_FORMAT_COUNT) {
        Rf_error("llamaR: chat_parse received an invalid format id: %d", format_id);
    }

    try {
        common_chat_parser_params syntax;
        syntax.format = (common_chat_format) format_id;
        syntax.parse_tool_calls = true;
        // PEG formats need the serialized parser arena from chat_build().
        if (!parser.empty()) {
            syntax.parser.load(parser);
        }

        common_chat_msg msg = common_chat_parse(input, is_partial, syntax);

        // Pack: content, reasoning_content,
        //       tool_calls = data.frame-ish list(name, arguments, id)
        const char * names[] = {"content", "reasoning_content",
                                "tool_names", "tool_arguments", "tool_ids", ""};
        SEXP result = PROTECT(Rf_mkNamed(VECSXP, names));

        SET_VECTOR_ELT(result, 0, make_utf8_string(msg.content));
        SET_VECTOR_ELT(result, 1, make_utf8_string(msg.reasoning_content));

        const R_xlen_t n = (R_xlen_t) msg.tool_calls.size();
        SEXP tnames = PROTECT(Rf_allocVector(STRSXP, n));
        SEXP targs  = PROTECT(Rf_allocVector(STRSXP, n));
        SEXP tids   = PROTECT(Rf_allocVector(STRSXP, n));
        for (R_xlen_t i = 0; i < n; i++) {
            const common_chat_tool_call & tc = msg.tool_calls[(size_t) i];
            SET_STRING_ELT(tnames, i, Rf_mkCharLenCE(tc.name.c_str(), (int) tc.name.size(), CE_UTF8));
            SET_STRING_ELT(targs,  i, Rf_mkCharLenCE(tc.arguments.c_str(), (int) tc.arguments.size(), CE_UTF8));
            SET_STRING_ELT(tids,   i, Rf_mkCharLenCE(tc.id.c_str(), (int) tc.id.size(), CE_UTF8));
        }
        SET_VECTOR_ELT(result, 2, tnames);
        SET_VECTOR_ELT(result, 3, targs);
        SET_VECTOR_ELT(result, 4, tids);

        UNPROTECT(4);  // tnames, targs, tids, result
        return result;
    } catch (const std::exception & e) {
        Rf_error("llamaR: chat_parse failed: %s", e.what());
    }
    return R_NilValue;  // unreachable
}
