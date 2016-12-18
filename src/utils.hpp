#ifndef __UTILS_HPP__
#define __UTILS_HPP__

#include <chrono>

#ifdef __GNUC__
#define unused __attribute__((unused))
#else // __GNUC__
#define uinused
#endif // __GNUC__

template <typename T>
static bool check_success(const T &err);

template <>
bool check_success<herr_t>(const herr_t &err) {
	  const auto res = err >= static_cast<herr_t>(0);
	    if (res == true) {
		        return res;
			  }
			    std::cout << "Failed in HDFS..." << std::endl;
			      assert(res);
			        return res;
}

#endif
