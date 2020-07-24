#pragma once
#include <mutex>
namespace mpl {
namespace debug {

/**
 * @brief function template use to execute a debug member function f ,wenn config is true
 * use double check stategy to save lock time!
 *
 * @tparam FuncPtr the debug function ptr,e.g a image draw and show function,logging function
 * @tparam Args function argumente package
 * @tparam T the class type
 * @param project_config : the config in the config.h
 * @param config_mutex : mutex to protect tje config, writed in visualization thread, read in this function
 * @param f function ptr
 * @param obj_ptr point to class obeject
 * @param args the function argument
 */
template <typename FuncPtr, typename... Args, typename T>
void execute_mem_according_to_config(bool& project_config, std::mutex& config_mutex, FuncPtr f, T obj_ptr,
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

// same as the upper template but for non-member function
template <typename Func, typename... Args>
void execute_func_according_to_config(bool& project_config, std::mutex& config_mutex, Func f, const Args&... args) {
    if (project_config) {
        config_mutex.lock();
        if (project_config) {
            config_mutex.unlock();
            f(args...);  // accordubg to cpp standard, can not use member function as template param,but
                         // function pointer
            return;
        }
        config_mutex.unlock();
    }
}

}  // namespace debug

}  // namespace mpl
