// json_parser.cpp — the single TU that defines the JsonParser implementation.
//
// JsonParser.h is a header-only JSON library gated on
// ZQ_JSON_PARSER_IMPLEMENTATION. This TU (and ONLY this TU) defines that macro
// so the implementation is compiled once into the vfield_host target. Every
// consumer includes the header without the macro and picks the symbols up by
// linking cumes_json — the host library PUBLIC-links it, so the host-model
// reader (src/cumes/config/json_reader.cpp) and everything downstream get
// them automatically.
#define ZQ_JSON_PARSER_IMPLEMENTATION
#include "vfield/JsonParser.h"
