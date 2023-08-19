#ifndef MULTIDIMENSIONAL_ARRAYS_INCLUDE_UTILS_LOG_H
#define MULTIDIMENSIONAL_ARRAYS_INCLUDE_UTILS_LOG_H

#include <iomanip>
#include <iostream>

#define OUT_PUT_MULTIDIMENSIONAL_ARRAYS_LOG true

#ifdef OUT_PUT_MULTIDIMENSIONAL_ARRAYS_LOG
#define LOG_MDA_INFO(x)                                                 \
  std::cout << std::setprecision(15) << "[INFO] [" << __FILE__ << "] [" \
            << __func__ << "] [" << __LINE__ << "] " << x << std::endl;
#define LOG_MDA_ERROR(x)                                                 \
  std::cout << std::setprecision(15) << "[ERROR] [" << __FILE__ << "] [" \
            << __func__ << "] [" << __LINE__ << "] " << x << std::endl;
#define LOG_MDA_WARNING(x)                                                 \
  std::cout << std::setprecision(15) << "[WARNING] [" << __FILE__ << "] [" \
            << __func__ << "] [" << __LINE__ << "] " << x << std::endl;
#define LOG_MDA_FATAL(x)                                                 \
  std::cout << std::setprecision(15) << "[FATAL] [" << __FILE__ << "] [" \
            << __func__ << "] [" << __LINE__ << "] " << x << std::endl;
#else
#define LOG_MDA_INFO(x)
#define LOG_MDA_ERROR(x)
#define LOG_MDA_WARNING(x)
#define LOG_MDA_FATAL(x)
#endif

#undef OUT_PUT_MULTIDIMENSIONAL_ARRAYS_LOG

#endif
