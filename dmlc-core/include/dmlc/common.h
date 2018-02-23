/*!
 *  Copyright (c) 2015 by Contributors
 * \file common.h
 * \brief defines some common utility function.
 */
#ifndef DMLC_COMMON_H_
#define DMLC_COMMON_H_

#include <vector>
#include <string>
#include <sstream>

#include "nvToolsExt.h"

#ifndef PUSH_RANGE
#define PUSH_RANGE0(name,cid,FUNCTION_NAME) { \
    int color_id = cid; \
static const uint32_t colors[] = { 0x0000ff00, 0x000000ff, 0x00ffff00, 0x00ff00ff, 0x0000ffff, 0x00ff0000, 0x00ffffff };\
static const int num_colors = sizeof(colors)/sizeof(uint32_t); \
    color_id = color_id%num_colors;\
    nvtxEventAttributes_t eventAttrib = {0}; \
    eventAttrib.version = NVTX_VERSION; \
    eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE; \
    eventAttrib.colorType = NVTX_COLOR_ARGB; \
    eventAttrib.color = colors[color_id]; \
    eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII; \
    eventAttrib.message.ascii = name " in " #  FUNCTION_NAME; \
    nvtxRangePushEx(&eventAttrib); \
}
#define PUSH_RANGE(name,cid) PUSH_RANGE0(name, cid,FUNCTION_NAME)
#define POP_RANGE() nvtxRangePop();
#endif

namespace dmlc {
/*!
 * \brief Split a string by delimiter
 * \param s String to be splitted.
 * \param delim The delimiter.
 * \return a splitted vector of strings.
 */
inline std::vector<std::string> Split(const std::string& s, char delim) {
  std::string item;
  std::istringstream is(s);
  std::vector<std::string> ret;
  while (std::getline(is, item, delim)) {
    ret.push_back(item);
  }
  return ret;
}

/*!
 * \brief hash an object and combines the key with previous keys
 */
template<typename T>
inline size_t HashCombine(size_t key, const T& value) {
  std::hash<T> hash_func;
  return key ^ (hash_func(value) + 0x9e3779b9 + (key << 6) + (key >> 2));
}

/*!
 * \brief specialize for size_t
 */
template<>
inline size_t HashCombine<size_t>(size_t key, const size_t& value) {
  return key ^ (value + 0x9e3779b9 + (key << 6) + (key >> 2));
}

}  // namespace dmlc

#endif  // DMLC_COMMON_H_
