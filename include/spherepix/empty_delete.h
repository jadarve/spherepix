/**
 * empty_delete.h
 */

#ifndef SPHEREPIX_EMPTY_DELETE_H_
#define SPHEREPIX_EMPTY_DELETE_H_

namespace spherepix {

template<typename T>
struct array_empty_deleter {
    void operator()(T const* p) {
        // nothing to do
    }
};

template<typename T>
struct empty_deleter {
    void operator()(T const* p) {
        // nothing to do
    }
};


}; // namespace spherepix

#endif // SPHEREPIX_EMPTY_DELETE_H_