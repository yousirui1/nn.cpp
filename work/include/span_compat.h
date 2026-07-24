#pragma once

// Minimal std::span replacement for C++17 (VS2017 compatible)
template<typename T>
struct simple_span {
    T* ptr;
    size_t count;

    simple_span(T* p, size_t n) : ptr(p), count(n) {}

    T* begin() { return ptr; }
    T* end() { return ptr + count; }
    const T* begin() const { return ptr; }
    const T* end() const { return ptr + count; }
    const T* cbegin() const { return ptr; }
    const T* cend() const { return ptr + count; }
};
