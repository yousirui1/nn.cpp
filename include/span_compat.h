#ifndef __SPAN_COMPAT_H__
#define __SPAN_COMPAT_H__

template<typename T>
struct simple_span{
    T *ptr;
    size_t count;

    simple_span(T *p, size_t n) : ptr(p), count(n) {}

    T *begin() { return ptr; }
    T *end() { return  ptr + count; }
    const T *begin() const { return ptr; }
    const T *end() const { return ptr + count; }
    const T *cbegin() const { return ptr; }
    const T *cend() const { return ptr + count; }

};


#endif //  __SPAN_COMPAT_H__
