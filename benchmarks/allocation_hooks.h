// Standalone benchmark instrumentation: include in one translation unit per executable.
#pragma once
#include <cstddef>
#include <cstdlib>
#include <new>
static thread_local bool counting = false;
static thread_local std::size_t allocations = 0, new_allocations = 0;
extern "C" void* __real_malloc(std::size_t);
extern "C" void* __real_calloc(std::size_t, std::size_t);
extern "C" void* __wrap_malloc(std::size_t n) {
    if (counting) ++allocations;
    return __real_malloc(n);
}
extern "C" void* __wrap_calloc(std::size_t n, std::size_t s) {
    if (counting) ++allocations;
    return __real_calloc(n, s);
}
void* operator new(std::size_t n) {
    if (counting) ++new_allocations;
    if (auto* p = __real_malloc(n ? n : 1)) return p;
    throw std::bad_alloc();
}
void* operator new[](std::size_t n) { return ::operator new(n); }
void operator delete(void* p) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t) noexcept { std::free(p); }
void* operator new(std::size_t n, std::align_val_t alignment) {
    if (counting) ++new_allocations;
    void* p = nullptr;
    if (posix_memalign(&p, std::size_t(alignment), n ? n : 1) == 0) return p;
    throw std::bad_alloc();
}
void* operator new[](std::size_t n, std::align_val_t alignment) { return ::operator new(n, alignment); }
void operator delete(void* p, std::align_val_t) noexcept { std::free(p); }
void operator delete[](void* p, std::align_val_t) noexcept { std::free(p); }
void operator delete(void* p, std::size_t, std::align_val_t) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t, std::align_val_t) noexcept { std::free(p); }
