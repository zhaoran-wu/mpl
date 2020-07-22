#pragma once
#include <mutex>
namespace mpl {
namespace debug {

template <typename FuncPtr, typename... Args, typename T>
void execute_accoding_to_config(bool& project_config, std::mutex& config_mutex, FuncPtr f, T obj_ptr,
                                const Args&... args) {
    if (project_config) {
        config_mutex.lock();
        if (project_config) {
            config_mutex.unlock();
            (obj_ptr->*f)(args...);  // accordubg to cpp standard, can not use member function as template param,but
                                     // function pointer
            return;
        }
        config_mutex.unlock();
    }
}

}  // namespace debug

}  // namespace mpl
