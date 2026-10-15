#pragma once

#include <vector>
#include <atomic>
#include <cstdint>
#include <algorithm>
#include <utility>

namespace mqfast {

// Minimal spin lock for very short critical sections.
struct SpinLock {
    std::atomic_flag f = ATOMIC_FLAG_INIT;

    inline void lock() noexcept {
        while (f.test_and_set(std::memory_order_acquire)) {
            // Busy-wait.
        }
    }

    inline bool try_lock() noexcept {
        return !f.test_and_set(std::memory_order_acquire);
    }

    inline void unlock() noexcept {
        f.clear(std::memory_order_release);
    }
};

// Very fast RNG: xorshift64*.
static inline uint64_t xorshift64star(uint64_t& x) noexcept {
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    return x * 2685821657736338717ULL;
}

// High-throughput approximate max-priority queue.
//
// Design:
// - Q independent binary max-heaps, each protected by a SpinLock.
// - Each queue keeps an optimistic cached top-key (atomic<double>) to avoid dual-lock pop.
// - try_pop(): sample two queues, pick the larger cached top, try_lock+pop from it.
//
// Requirements:
// - KeyFn(e) returns a non-negative finite key (e.g., residual norm).
// - Empty queue cache key is 0.0.
template <class T, class KeyFn>
class MultiQueueFast {
public:
    explicit MultiQueueFast(int num_queues, KeyFn keyFn = KeyFn())
        : Q_(num_queues > 0 ? num_queues : 1), keyFn_(keyFn), qs_(static_cast<size_t>(Q_)) {}

    int numQueues() const noexcept { return Q_; }

    // Reserve heap capacity per queue to avoid runtime allocations.
    void reserve_per_queue(size_t n) {
        for (auto& q : qs_) q.heap.reserve(n);
    }

    inline void push(const T& x) {
        TLS& st = tls();
        const int i = static_cast<int>(xorshift64star(st.rng) % static_cast<uint64_t>(Q_));
        Queue& q = qs_[static_cast<size_t>(i)];
        q.lk.lock();
        heap_push_(q.heap, x);
        q.top_key.store(keyFn_(q.heap[0]), std::memory_order_release);
        q.lk.unlock();
    }

    inline void push(T&& x) {
        TLS& st = tls();
        const int i = static_cast<int>(xorshift64star(st.rng) % static_cast<uint64_t>(Q_));
        Queue& q = qs_[static_cast<size_t>(i)];
        q.lk.lock();
        heap_push_(q.heap, std::move(x));
        q.top_key.store(keyFn_(q.heap[0]), std::memory_order_release);
        q.lk.unlock();
    }

    // Approximate pop-max. Returns false if empty (best-effort).
    inline bool try_pop(T& out) {
        if (Q_ == 1) return try_pop_from_(0, out);

        TLS& st = tls();
        int a = static_cast<int>(xorshift64star(st.rng) % static_cast<uint64_t>(Q_));
        int b = static_cast<int>(xorshift64star(st.rng) % static_cast<uint64_t>(Q_));
        if (a == b) b = (b + 1) % Q_;

        const double ka = qs_[static_cast<size_t>(a)].top_key.load(std::memory_order_acquire);
        const double kb = qs_[static_cast<size_t>(b)].top_key.load(std::memory_order_acquire);

        const int first  = (kb > ka) ? b : a;
        const int second = (first == a) ? b : a;

        if (try_pop_from_(first, out)) return true;
        if (try_pop_from_(second, out)) return true;

        // Small fallback to reduce false-empty due to stale caches.
        for (int t = 0; t < 2; ++t) {
            int i = static_cast<int>(xorshift64star(st.rng) % static_cast<uint64_t>(Q_));
            if (try_pop_from_(i, out)) return true;
        }
        return false;
    }

private:
    struct alignas(64) Queue {
        SpinLock lk;
        std::vector<T> heap;
        std::atomic<double> top_key{0.0};
    };

    struct TLS {
        uint64_t rng;
    };

    static inline uint64_t seed_() noexcept {
        // Cheap per-thread seed: address mix.
        uint64_t x = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(&seed_));
        x ^= 0x9e3779b97f4a7c15ULL;
        // Ensure non-zero.
        return x ? x : 88172645463325252ULL;
    }

    static inline TLS& tls() {
        thread_local TLS st{seed_()};
        // Stir once per thread.
        (void)xorshift64star(st.rng);
        return st;
    }

    inline bool less_(const T& a, const T& b) const noexcept {
        return keyFn_(a) < keyFn_(b);
    }

    template <class U>
    inline void heap_push_(std::vector<T>& h, U&& x) {
        h.emplace_back(std::forward<U>(x));
        size_t i = h.size() - 1;
        while (i > 0) {
            size_t p = (i - 1) >> 1;
            if (!less_(h[p], h[i])) break;
            std::swap(h[p], h[i]);
            i = p;
        }
    }

    inline void heap_pop_(std::vector<T>& h) {
        const size_t n = h.size();
        if (n == 1) {
            h.pop_back();
            return;
        }
        h[0] = std::move(h.back());
        h.pop_back();
        size_t i = 0;
        const size_t m = h.size();
        while (true) {
            size_t l = (i << 1) + 1;
            if (l >= m) break;
            size_t r = l + 1;
            size_t c = (r < m && less_(h[l], h[r])) ? r : l;
            if (!less_(h[i], h[c])) break;
            std::swap(h[i], h[c]);
            i = c;
        }
    }

    inline bool try_pop_from_(int idx, T& out) {
        Queue& q = qs_[static_cast<size_t>(idx)];
        if (!q.lk.try_lock()) return false;

        if (q.heap.empty()) {
            q.top_key.store(0.0, std::memory_order_release);
            q.lk.unlock();
            return false;
        }

        out = std::move(q.heap[0]);
        heap_pop_(q.heap);
        if (q.heap.empty()) q.top_key.store(0.0, std::memory_order_release);
        else q.top_key.store(keyFn_(q.heap[0]), std::memory_order_release);
        q.lk.unlock();
        return true;
    }

private:
    int Q_;
    KeyFn keyFn_;
    std::vector<Queue> qs_;
};

} // namespace mqfast
