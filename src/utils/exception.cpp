#include "utils/exception.h"

#include <cstring>
#include <sstream>

namespace Autoalg {

char Error::msg_[1024] = {0};

Error::Error(const char *file, const char *func, unsigned int line)
    : file_(file), func_(func), line_(line) {}

const char *Error::what() const noexcept {
  std::stringstream s;
  s << std::endl;
  s << file_ << ":" << line_ << ": ";
  s << "In function " << func_ << "()." << std::endl;
  s << msg_;
  auto &&str = s.str();
  memcpy(msg_, str.c_str(), str.length());
  return msg_;
}

}  // namespace Autoalg