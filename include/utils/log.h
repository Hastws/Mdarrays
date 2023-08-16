#ifndef MULTIDIMENSIONAL_ARRAYS_INCLUDE_UTILS_LOG_H
#define MULTIDIMENSIONAL_ARRAYS_INCLUDE_UTILS_LOG_H

#include <iomanip>
#include <iostream>

#define OUT_PUT_LOG 1

#ifdef OUT_PUT_LOG
#define LOG_INFO(x) std::cout << std::setprecision(15) << x << std::endl;
#define LOG_ERROR(x) std::cout << std::setprecision(15) << x << std::endl;
#define LOG_WARNING(x) std::cout << std::setprecision(15) << x << std::endl;
#define LOG_FATAL(x) std::cout << std::setprecision(15) << x << std::endl;
#else
#define LOG_INFO(x)
#define LOG_ERROR(x)
#define LOG_WARNING(x)
#define LOG_FATAL(x)
#endif

#undef OUT_PUT_LOG

#endif
