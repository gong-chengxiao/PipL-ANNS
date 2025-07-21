#pragma once

#include <cstdint>
#include <vector>
#include "../common.hpp"
#include "../utils/memory.hpp"

namespace symqg {

class Hash {
private:
    static constexpr PID EMPTY = kPidMax;
    
    size_t size_ = 0;
    size_t mask_ = 0;
    std::vector<PID, memory::AlignedAllocator<PID>> table_;

    [[nodiscard]] auto hash1(const PID value) const { return value & mask_; }

public:
    Hash() = default;

    explicit Hash(size_t ef) {
        size_t size = 1;
        while (size < (ef << 2)) {
            size <<= 1;
        }
        size_ = size;
        mask_ = size_ - 1;
        table_.resize(size_, EMPTY);
    }

    void clear() {
        std::fill(table_.begin(), table_.end(), EMPTY);
    }

    [[nodiscard]] bool get(PID key) const {
        size_t h1 = hash1(key);
       
        for (size_t i = 0; i < size_; i++) {
            size_t pos = (h1 + i) & mask_;
            if (table_[pos] == key) {
                return true;
            }
            if (table_[pos] == EMPTY) {
                return false;
            }
        }
    }

    void set(PID key) {
        size_t h1 = hash1(key);
        
        for (size_t i = 0; i < size_; i++) {
            size_t pos = (h1 + i) & mask_;
            if (table_[pos] == key || table_[pos] == EMPTY) {
                table_[pos] = key;
                return;
            }
        }
    }
};

}  // namespace symqg