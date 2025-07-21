#pragma once

#include <cstddef>
#include <vector>
#include <cfloat>
#include <bit>
#include <span>
#include <new>
#include "../common.hpp"
#include "./memory.hpp"

#define NOT_FOUND 1U << 31

namespace symqg::buffer {
class SearchBuffer {
private:
    std::vector<Candidate<float>, memory::AlignedAllocator<Candidate<float>>> data_;
    size_t capacity_, size_, max_capacity_;

    [[nodiscard]] static size_t parent(size_t i) { return (i - 1) / 2; }
    [[nodiscard]] static size_t grandparent(size_t i) { return (i > 2) ? (i - 3) / 4 : 0; }
    [[nodiscard]] static size_t left_child(size_t i) { return 2 * i + 1; }
    [[nodiscard]] static bool has_left_child(size_t i, size_t n) { return left_child(i) < n; }
    [[nodiscard]] static bool is_on_min_level(size_t i) {
        if (i == 0) return true;
        /* Check level depth. Even depth (0, 2, 4...) is a min level. */
        return (static_cast<int>(floor(log2(i + 1))) % 2) == 0;
    }

    void swap(size_t i, size_t j) {
        std::swap(data_[i], data_[j]);
    }

    /* Restores heap property by moving an element up */
    void sift_up(size_t i) {
        if (i == 0) return;

        size_t p = parent(i);
        if (is_on_min_level(i)) {
            if (data_[i] > data_[p]) {
                swap(i, p);
                sift_up_max(p);
            } else {
                sift_up_min(i);
            }
        } else {
            if (data_[i] < data_[p]) {
                swap(i, p);
                sift_up_min(p);
            } else {
                sift_up_max(i);
            }
        }
    }
    
    void sift_up_min(size_t i) {
        while (i > 2) {
            size_t gp = grandparent(i);
            if (data_[i] < data_[gp]) {
                swap(i, gp);
                i = gp;
            } else {
                break;
            }
        }
    }
    void sift_up_max(size_t i) {
        while (i > 2) {
            size_t gp = grandparent(i);
            if (data_[i] > data_[gp]) {
                swap(i, gp);
                i = gp;
            } else {
                break;
            }
        }
    }

    void sift_down(size_t i) {
        if (is_on_min_level(i)) {
            sift_down_min(i);
        } else {
            sift_down_max(i);
        }
    }

    void sift_down_min(size_t i) {
        while (has_left_child(i, size_)) {
            size_t lc = left_child(i);
            size_t rc = lc + 1;
            size_t m = lc;

            if (rc < size_ && data_[rc] < data_[m]) {
                m = rc;
            }

            for (size_t j = 0; j < 2 && has_left_child(lc + j, size_); ++j) {
                size_t gc_start = left_child(lc + j);
                for (size_t k = 0; k < 2 && (gc_start + k) < size_; ++k) {
                    if (data_[gc_start + k] < data_[m]) {
                        m = gc_start + k;
                    }
                }
            }
            
            bool is_grandchild = (m >= left_child(left_child(i)));

            if (is_grandchild) {
                if (data_[m] < data_[i]) {
                    swap(m, i);
                    if (data_[m] > data_[parent(m)]) {
                        swap(m, parent(m));
                    }
                    i = m;
                } else {
                    break;
                }
            } else {
                if (data_[m] < data_[i]) {
                    swap(m, i);
                }
                break;
            }
        }
    }

    void sift_down_max(size_t i) {
        while (has_left_child(i, size_)) {
            size_t lc = left_child(i);
            size_t rc = lc + 1;
            size_t m = lc;

            if (rc < size_ && data_[rc] > data_[m]) {
                m = rc;
            }

            for (size_t j = 0; j < 2 && has_left_child(lc + j, size_); ++j) {
                size_t gc_start = left_child(lc + j);
                for (size_t k = 0; k < 2 && (gc_start + k) < size_; ++k) {
                    if (data_[gc_start + k] > data_[m]) {
                        m = gc_start + k;
                    }
                }
            }
            
            bool is_grandchild = (m >= left_child(left_child(i)));

            if (is_grandchild) {
                if (data_[m] > data_[i]) {
                    swap(m, i);
                    if (data_[m] < data_[parent(m)]) {
                        swap(m, parent(m));
                    }
                    i = m;
                } else {
                    break;
                }
            } else {
                if (data_[m] > data_[i]) {
                    swap(m, i);
                }
                break;
            }
        }
    }

    void delete_max() {
        if (size_ < 2) {
            if (size_ == 1) size_--;
            return;
        }

        size_t max_idx = 1;
        if (size_ > 2 && data_[2] > data_[1]) {
            max_idx = 2;
        }

        size_--;
        if (size_ == max_idx) {
            return;
        }
        
        swap(max_idx, size_);
        sift_down(max_idx);
    }

public:
    SearchBuffer() = default;

    explicit SearchBuffer(size_t capacity) : capacity_(capacity), size_(0), max_capacity_(capacity) {
        data_.reserve(capacity_);
    }

    void insert(Candidate<float> candidate) {
        if (size_ < capacity_) {
            data_[size_] = candidate;
            sift_up(size_);
            size_++;
        } else if (candidate < find_max()) {
            delete_max();
            data_[size_] = candidate;
            sift_up(size_);
            size_++;
        }
    }

    void insert(PID data_id, float dist) {
        insert(Candidate<float>(data_id, dist));
    }

    /* Get and remove the closest (minimum distance) candidate */
    PID pop() {
        PID pid = data_[0].id;
        size_--;
        capacity_--;
        if (size_ > 0) {
            swap(0, size_);
            sift_down(0);
        }
        return pid;
    }

    void clear() {
        capacity_ = max_capacity_;
        size_ = 0;
    }

    [[nodiscard]] auto find_min() const -> const Candidate<float>& {
        return data_[0];
    }
    
    [[nodiscard]] auto find_max() const -> const Candidate<float>& {
        if (size_ == 1) return data_[0];
        if (size_ == 2) return data_[1];
        return (data_[1] > data_[2]) ? data_[1] : data_[2];
    }

    [[nodiscard]] auto next_id() const { return find_min().id; }
    [[nodiscard]] auto next_dist() const { return find_min().distance; }
    [[nodiscard]] auto has_next() const -> bool { return size_ > 0; }
    [[nodiscard]] auto size() const { return size_; }
    [[nodiscard]] auto is_full(float dist) const { return size_ == capacity_ && (capacity_ == 0 || find_max().distance < dist); }

    void resize(size_t new_size) {
        this->max_capacity_ = new_size;
        this->capacity_ = new_size;
        data_.reserve(new_size);
    }
};

class BucketBuffer {
   private:
    size_t h_bucket_;   /* height of bucket */
    size_t h_buffer_;   /* height of buffer */
    size_t num_scanners_;

    SearchBuffer bucket_;
    std::vector<PID> buffer_;

    static void set_checked(PID& data_id) { data_id |= (1U << 31); }

    [[nodiscard]] static auto is_checked(PID data_id) -> bool {
        return static_cast<bool>(data_id >> 31);
    }

    [[nodiscard]] bool buffer_has_next(size_t scanner_id) const {
        for (size_t i = scanner_id; i < this->h_buffer_; i += this->num_scanners_) {
            if (!is_checked(this->buffer_[i])) {
                return true;
            }
        }
        return false;
    }

   public:
    BucketBuffer() = default;

    explicit BucketBuffer(size_t h_bucket, size_t h_buffer, size_t num_scanners)
        : h_bucket_(h_bucket),
          h_buffer_(h_buffer),
          num_scanners_(num_scanners)
        {
            if ((this->h_buffer_ & (this->h_buffer_ - 1)) != 0) {
                throw std::invalid_argument("h_buffer_ must be power of 2");
            }

            this->bucket_ = SearchBuffer(this->h_bucket_);
            this->buffer_ = std::vector<PID>(this->h_buffer_);
            for (size_t i = 0; i < this->h_buffer_; ++i) {
                this->buffer_[i] = NOT_FOUND;
            }
        }

    void clear() {
        this->bucket_.clear();
        for (size_t i = 0; i < this->h_buffer_; ++i) {
            this->buffer_[i] = NOT_FOUND;
        }
    }

    [[nodiscard]] auto is_full(float dist) const -> bool {
        return this->bucket_.is_full(dist);
    }

    [[nodiscard]] auto has_next(size_t scanner_id) const -> bool {
        return this->buffer_has_next(scanner_id) || this->bucket_.has_next();
    }

    void resize(size_t new_size) {
        this->h_bucket_ = new_size;
        this->bucket_.resize(this->h_bucket_);
    }

    void resize_buffer(size_t new_size) {
        this->buffer_ = std::vector<PID>(new_size);
        for (size_t i = 0; i < new_size; ++i) {
            this->buffer_[i] = NOT_FOUND;
        }
        this->h_buffer_ = new_size;
    }

    void set_num_scanners(size_t num_scanners) {
        if (num_scanners == 0) {
            std::cerr << "num_scanners must be greater than 0" << std::endl;
            throw std::invalid_argument("num_scanners must be greater than 0");
        }
        if (num_scanners > this->h_buffer_) {
            std::cerr << "num_scanners must be less than h_buffer_" << std::endl;
            throw std::invalid_argument("num_scanners must be less than h_buffer_");
        }

        this->num_scanners_ = num_scanners;
    }

    PID try_pop(size_t scanner_id) {
        for (size_t i = scanner_id; i < this->h_buffer_; i += this->num_scanners_) {
            PID pid = this->buffer_[i];
            if (!is_checked(pid)) {
                set_checked(this->buffer_[i]);
                return pid;
            }
        }
        return NOT_FOUND;
    }

    [[nodiscard]] auto pop_from_bucket() -> PID {
        return this->bucket_.pop();
    }

    [[nodiscard]] auto next_id_from_bucket() -> PID {
        return this->bucket_.next_id();
    }

    [[nodiscard]] auto bucket_has_next() const -> bool {
        return this->bucket_.has_next();
    }

    void insert(PID data_id, float dist) {
        this->bucket_.insert(data_id, dist);
    }

    void insert(Candidate<float> candidate) {
        this->bucket_.insert(candidate);
    }

    void try_promote() {
        for (size_t i = 0; i < this->h_buffer_; ++i) {
            if (is_checked(this->buffer_[i]) && this->bucket_.has_next()) {
                this->buffer_[i] = this->bucket_.pop();
            }
        }
    }
    
};

class Pool {
   private:
    static constexpr size_t cacheline_size_ = 64;
    size_t block_size_;
    size_t block_size_bit_;
    size_t size_;
    size_t num_blocks_;
    size_t num_scanners_;
    alignas(cacheline_size_) std::atomic<size_t> collector_pos_;
    std::vector<std::atomic<size_t>> scanners_pos_;
    std::vector<Candidate<float>, memory::AlignedAllocator<Candidate<float>>> candidates_;

   public:
    Pool() = default;

    explicit Pool(size_t num_scanners, size_t insert_limit, size_t num_blocks)
        : num_blocks_(num_blocks),
        num_scanners_(num_scanners) {
            size_t cacheline_capacity = cacheline_size_ / sizeof(Candidate<float>);
            this->block_size_ = (insert_limit + cacheline_capacity - 1) & ~(cacheline_capacity - 1);
            this->block_size_bit_ = std::bit_width(block_size_) - 1;
            this->size_ = block_size_ * num_blocks_;

            if ((size_ & (size_ - 1)) != 0) {
                std::cerr << "size_ must be power of 2" << std::endl;
                throw std::invalid_argument("size_ must be power of 2");
            }
            if (size_ < 2) {
                std::cerr << "size_ must be greater than 1" << std::endl;
                throw std::invalid_argument("size_ must be greater than 1");
            }
            if ((num_scanners & (num_scanners - 1)) != 0) {
                std::cerr << "num_scanners must be power of 2" << std::endl;
                throw std::invalid_argument("num_scanners must be power of 2");
            }

            this->candidates_ = std::vector<Candidate<float>, memory::AlignedAllocator<Candidate<float>>>(this->size_, Candidate<float>(0, FLT_MAX));
            this->collector_pos_ = 0;
            this->scanners_pos_ = std::vector<std::atomic<size_t>>(num_scanners);
            for (size_t i = 0; i < scanners_pos_.size(); ++i) {
                scanners_pos_[i] = i << block_size_bit_;
            }

            std::fill(candidates_.begin(), candidates_.end(), Candidate<float>(0, FLT_MAX));
        }


    void clear() {
        this->collector_pos_ = 0;
        for (size_t i = 0; i < scanners_pos_.size(); ++i) {
            scanners_pos_[i] = i << block_size_bit_;
        }

        std::fill(candidates_.begin(), candidates_.end(), Candidate<float>(0, FLT_MAX));
    }

    void set_num_scanners(size_t num_scanners) {
        if ((num_scanners & (num_scanners - 1)) != 0) {
            std::cerr << "num_scanners must be power of 2" << std::endl;
            throw std::invalid_argument("num_scanners must be power of 2");
        }
        if (num_scanners > this->num_blocks_) {
            std::cerr << "num_scanners must be less than num_blocks_" << std::endl;
            throw std::invalid_argument("num_scanners must be less than num_blocks_");
        }

        this->num_scanners_ = num_scanners;
        this->scanners_pos_ = std::vector<std::atomic<size_t>>(num_scanners);
        for (size_t i = 0; i < num_scanners; ++i) {
            this->scanners_pos_[i] = i << block_size_bit_;
        }
    }

    void set_insert_limit(size_t insert_limit) {
        size_t cacheline_capacity = cacheline_size_ / sizeof(Candidate<float>);
        this->block_size_ = (insert_limit + cacheline_capacity - 1) & ~(cacheline_capacity - 1);
        this->block_size_bit_ = std::bit_width(block_size_) - 1;
        this->size_ = block_size_ * num_blocks_;
        this->candidates_ = std::vector<Candidate<float>, memory::AlignedAllocator<Candidate<float>>>(this->size_, Candidate<float>(0, FLT_MAX));
    }

    void resize(size_t num_blocks) {
        if (this->num_scanners_ > num_blocks) {
            std::cerr << "num_scanners must be less than num_blocks" << std::endl;
            throw std::invalid_argument("num_scanners must be less than num_blocks");
        }

        this->num_blocks_ = num_blocks;
        this->size_ = block_size_ * num_blocks;
        this->candidates_ = std::vector<Candidate<float>, memory::AlignedAllocator<Candidate<float>>>(this->size_, Candidate<float>(0, FLT_MAX));
    }

    [[nodiscard]] auto get_block_size() const -> size_t { return block_size_; }

    /* get a block buffer for put */
    [[nodiscard]] Candidate<float>* get_put_block_buffer(size_t scanner_id) {
        const size_t cur_pos = this->scanners_pos_[scanner_id].load(std::memory_order_acquire);
        if (cur_pos >= collector_pos_ + size_) {
            return nullptr;
        }

        memory::mem_prefetch_l1(reinterpret_cast<const char*>(this->candidates_.data() + (cur_pos & (size_ - 1))), 1);
        std::fill(this->candidates_.data() + (cur_pos & (size_ - 1)), this->candidates_.data() + (cur_pos & (size_ - 1)) + block_size_, Candidate<float>(0, FLT_MAX));
        return this->candidates_.data() + (cur_pos & (size_ - 1));
    }

    void submit_put_block_buffer(size_t scanner_id) {
        this->scanners_pos_[scanner_id].fetch_add(num_scanners_ << block_size_bit_, std::memory_order_release);
    }

    /* get a block buffer for retrieve */
    [[nodiscard]] Candidate<float>* get_retrieve_block_buffer()  {
        const size_t cur_pos = this->collector_pos_.load(std::memory_order_acquire);
        const size_t scanner_id = (cur_pos >> block_size_bit_) & (num_scanners_ - 1);
        const size_t scanner_pos = this->scanners_pos_[scanner_id].load(std::memory_order_acquire);
        if (cur_pos == (scanner_pos & ~(block_size_ - 1))) {
            return nullptr;
        }
        
        memory::mem_prefetch_l1(reinterpret_cast<const char*>(this->candidates_.data() + (cur_pos & (size_ - 1))), 1);
        return this->candidates_.data() + (cur_pos & (size_ - 1));
    }

    void submit_retrieve_block_buffer() {
        this->collector_pos_.fetch_add(block_size_, std::memory_order_release);
    }

};

// sorted linear buffer to store search results
class ResultBuffer {
   public:
    explicit ResultBuffer(size_t capacity)
        : ids_(capacity + 1), 
        distances_(capacity + 1), 
        capacity_(capacity)
        {}
    
    explicit ResultBuffer() {}

    void insert(PID data_id, float dist) {
        if (size_ == capacity_ && dist > distances_[size_ - 1]) {
            return;
        }
        size_t lo = std::lower_bound(distances_.begin(), distances_.begin() + size_, dist) - distances_.begin();
        std::memmove(&ids_[lo + 1], &ids_[lo], (size_ - lo) * sizeof(PID));
        ids_[lo] = data_id;
        std::memmove(&distances_[lo + 1], &distances_[lo], (size_ - lo) * sizeof(float));
        distances_[lo] = dist;
        size_ += static_cast<size_t>(size_ < capacity_);
    }

    void merge(const ResultBuffer& other) {
        for (size_t i = 0; i < other.size_; ++i) {
            this->insert(other.ids_[i], other.distances_[i]);
        }
    }

    [[nodiscard]] auto is_full() const -> bool { return size_ == capacity_; }

    const std::vector<PID, memory::AlignedAllocator<PID>>& ids() { return ids_; }

    void copy_results(PID* knn) const { std::copy(ids_.begin(), ids_.end() - 1, knn); }

   private:
    std::vector<PID, memory::AlignedAllocator<PID>> ids_;
    std::vector<float, memory::AlignedAllocator<float>> distances_;
    size_t size_ = 0, capacity_;
};
}  // namespace symqg::buffer