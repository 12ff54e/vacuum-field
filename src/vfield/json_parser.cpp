// json_parser.cpp — the single TU that defines the JsonParser implementation.
//
// JsonParser.h (copied from cuMES, with its #ifndef guards and snake_case
// API intact) is a header-only JSON library gated on
// ZQ_JSON_PARSER_IMPLEMENTATION. This TU (and ONLY this TU) defines that
// macro so the implementation is compiled once into the vfield_host target.
// Every consumer includes the header without the macro and picks the symbols
// up by linking vfield_host, which PUBLIC-links this TU.
#define ZQ_JSON_PARSER_IMPLEMENTATION
#include "vfield/JsonParser.h"
