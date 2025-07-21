#pragma once

#include <omp.h>

#include <cassert>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <condition_variable>
#include <queue>

#include "../common.hpp"
#include "../quantization/rabitq.hpp"
#include "../space/l2.hpp"
#include "../third/ngt/hashset.hpp"
#include "../third/svs/array.hpp"
#include "../utils/buffer.hpp"
#include "../utils/io.hpp"
#include "../utils/memory.hpp"
#include "../utils/rotator.hpp"
#include "../utils/hash.hpp"
#include "./qg_query.hpp"
#include "./qg_scanner.hpp"

namespace symqg {
/**
 * @brief this Factor only for illustration, the true storage is continous
 * degree_bound_*triple_x + degree_bound_*factor_dq + degree_bound_*factor_vq
 *
 */
struct Factor {
    float triple_x;   // Sqr of distance to centroid + 2 * x * x1 / x0
    float factor_dq;  // Factor of delta * ||q_r|| * (FastScanRes - sum_q)
    float factor_vq;  // Factor of v_l * ||q_r||
};

class QuantizedGraph {
    friend class QGBuilder;

   private:
    size_t num_points_ = 0;    // num points
    size_t degree_bound_ = 0;  // degree bound
    size_t dimension_ = 0;     // dimension
    size_t padded_dim_ = 0;    // padded dimension
    PID entry_point_ = 0;      // Entry point of graph

    data::Array<
        float,
        std::vector<size_t>,
        memory::AlignedAllocator<
            float,
            1 << 22,
            true>>
        data_;  // vectors + graph + quantization codes
    QGScanner scanner_;
    FHTRotator rotator_;
    // HashBasedBooleanSet visited_;
    // BloomFilter visited_;
    Hash visited_;
    buffer::SearchBuffer search_pool_;
    std::vector<buffer::ResultBuffer> result_pools_;
    std::atomic<size_t> num_finished_scanners;

    /*
     * Position of different data in each row
     *      RawData + QuantizationCodes + Factors + neighborIDs
     * Since we guarantee the degree for each vertex equals degree_bound (multiple of 32),
     * we do not need to store the degree for each vertex
     */
    size_t code_offset_ = 0;      // pos of packed code
    size_t factor_offset_ = 0;    // pos of Factor
    size_t neighbor_offset_ = 0;  // pos of Neighbors
    size_t row_offset_ = 0;       // length of entire row

    buffer::Pool pool_;
    buffer::BucketBuffer bucket_buffer_;
    size_t h_buffer_ = 4;
    size_t num_pool_blocks_ = 8;
    size_t insert_limit_ = 2;
    size_t num_scanners_ = 4;
    float filter_alpha_ = 2;

    void initialize();

    // search on quantized graph
    void search_qg(
        const float* __restrict__ query, uint32_t knn, uint32_t* __restrict__ results
    );

    void copy_vectors(const float*);

    void search_qg_parallel(
        const float* __restrict__ query, uint32_t knn, uint32_t* __restrict__ results
    );

    void buckets_prepare(const QGQuery& q_obj);

    void scanner_task(const QGQuery& q_obj, const size_t scanner_id);

    void collector_task();

    [[nodiscard]] float* get_vector(PID data_id) {
        return &data_.at(row_offset_ * data_id);
    }

    __attribute__((noinline)) [[nodiscard]] const float* get_vector(PID data_id) const {
        return &data_.at(row_offset_ * data_id);
    }

    [[nodiscard]] uint8_t* get_packed_code(PID data_id) {
        return reinterpret_cast<uint8_t*>(&data_.at((row_offset_ * data_id) + code_offset_)
        );
    }

    [[nodiscard]] const uint8_t* get_packed_code(PID data_id) const {
        return reinterpret_cast<const uint8_t*>(
            &data_.at((row_offset_ * data_id) + code_offset_)
        );
    }

    [[nodiscard]] float* get_factor(PID data_id) {
        return &data_.at((row_offset_ * data_id) + factor_offset_);
    }

    [[nodiscard]] const float* get_factor(PID data_id) const {
        return &data_.at((row_offset_ * data_id) + factor_offset_);
    }

    [[nodiscard]] PID* get_neighbors(PID data_id) {
        return reinterpret_cast<PID*>(&data_.at((row_offset_ * data_id) + neighbor_offset_)
        );
    }

    [[nodiscard]] const PID* get_neighbors(PID data_id) const {
        return reinterpret_cast<const PID*>(
            &data_.at((row_offset_ * data_id) + neighbor_offset_)
        );
    }

    void
    find_candidates(PID, size_t, std::vector<Candidate<float>>&, HashBasedBooleanSet&, const std::vector<uint32_t>&)
        const;

    void update_qg(PID, const std::vector<Candidate<float>>&);

    void update_results(buffer::ResultBuffer&, const float*);

    float scan_neighbors(
        const QGQuery& q_obj,
        const float* cur_data,
        float* appro_dist,
        buffer::SearchBuffer& search_pool,
        uint32_t cur_degree
    ) const;

   public:
    explicit QuantizedGraph(size_t, size_t, size_t);

    [[nodiscard]] auto num_vertices() const { return this->num_points_; }

    [[nodiscard]] auto dimension() const { return this->dimension_; }

    [[nodiscard]] auto degree_bound() const { return this->degree_bound_; }

    [[nodiscard]] auto entry_point() const { return this->entry_point_; }

    void set_ep(PID entry) { this->entry_point_ = entry; };

    void save_index(const char*) const;

    void load_index(const char*);

    void set_ef(size_t);

    void set_buffer_size(size_t buffer_size);

    void set_num_pool_blocks(size_t num_pool_blocks);

    void set_insert_limit(size_t limit);

    void set_num_scanners(size_t num_scanners);

    void set_filter_alpha(float filter_alpha);

    /* search and copy results to KNN */
    void search(
        const float* __restrict__ query, uint32_t knn, uint32_t* __restrict__ results
    );
};

inline QuantizedGraph::QuantizedGraph(size_t num, size_t max_deg, size_t dim)
    : num_points_(num)
    , degree_bound_(max_deg)
    , dimension_(dim)
    , padded_dim_(1 << ceil_log2(dim))
    , scanner_(padded_dim_, degree_bound_)
    , rotator_(dimension_)
    , visited_(100)
    , search_pool_(0)
    , result_pools_(0)
    , pool_(4, 2, 8)
    , bucket_buffer_(0, 4, 2) {
    initialize();
}

inline void QuantizedGraph::copy_vectors(const float* data) {
#pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < num_points_; ++i) {
        const float* src = data + (dimension_ * i);
        float* dst = get_vector(i);
        std::copy(src, src + dimension_, dst);
    }
    std::cout << "\tVectors Copied\n";
}

inline void QuantizedGraph::save_index(const char* filename) const {
    std::cout << "Saving quantized graph to " << filename << '\n';
    std::ofstream output(filename, std::ios::binary);
    assert(output.is_open());

    /* Basic variants */
    output.write(reinterpret_cast<const char*>(&entry_point_), sizeof(PID));

    /* Data */
    data_.save(output);

    /* Rotator */
    this->rotator_.save(output);

    output.close();
    std::cout << "\tQuantized graph saved!\n";
}

inline void QuantizedGraph::load_index(const char* filename) {
    std::cout << "loading quantized graph " << filename << '\n';

    /* Check existence */
    if (!file_exists(filename)) {
        std::cerr << "Index does not exist!\n";
        abort();
    }

    /* Check file size */
    size_t filesize = get_filesize(filename);
    size_t correct_size = sizeof(PID) + (sizeof(float) * num_points_ * row_offset_) +
                          (sizeof(float) * padded_dim_);
    if (filesize != correct_size) {
        std::cerr << "Index file size error! Please make sure the index and "
                     "init parameters are correct\n";
        abort();
    }

    std::ifstream input(filename, std::ios::binary);
    assert(input.is_open());

    /* Basic variants */
    input.read(reinterpret_cast<char*>(&entry_point_), sizeof(PID));

    /* Data */
    data_.load(input);

    /* Rotator */
    this->rotator_.load(input);

    input.close();
    std::cout << "Quantized graph loaded!\n";
}

inline void QuantizedGraph::set_buffer_size(size_t buffer_size) {
    this->h_buffer_ = buffer_size;
    this->bucket_buffer_.resize_buffer(buffer_size);
}

inline void QuantizedGraph::set_num_pool_blocks(size_t num_pool_blocks) {
    this->num_pool_blocks_ = num_pool_blocks;
    this->pool_.resize(num_pool_blocks);
}

inline void QuantizedGraph::set_insert_limit(size_t limit) {
    this->insert_limit_ = limit;
    this->pool_.set_insert_limit(limit);
}

inline void QuantizedGraph::set_num_scanners(size_t num_scanners) {
    this->num_scanners_ = num_scanners;
    this->bucket_buffer_.set_num_scanners(num_scanners);
    this->pool_.set_num_scanners(num_scanners);
    this->result_pools_.resize(num_scanners);
}

inline void QuantizedGraph::set_filter_alpha(float filter_alpha) {
    this->filter_alpha_ = filter_alpha;
}

inline void QuantizedGraph::set_ef(size_t cur_ef) {
    this->bucket_buffer_.resize(cur_ef);
    this->visited_ = Hash(cur_ef);
}

/*
 * search single query
 */
inline void QuantizedGraph::search(
    const float* __restrict__ query, uint32_t knn, uint32_t* __restrict__ results
) {
    /* Init query matrix */
    this->visited_.clear();
    this->search_pool_.clear();
    this->bucket_buffer_.clear();
    this->pool_.clear();

    search_qg_parallel(query, knn, results);
}

inline void QuantizedGraph::search_qg_parallel(
    const float* __restrict__ query, uint32_t knn, uint32_t* __restrict__ results
) {
    this->result_pools_.resize(num_scanners_);
    for (size_t i = 0; i < num_scanners_; ++i) {
        this->result_pools_[i] = buffer::ResultBuffer(knn);
    }

    // query preparation
    QGQuery q_obj(query, padded_dim_);
    q_obj.query_prepare(rotator_, scanner_);
    buckets_prepare(q_obj);

    num_finished_scanners.store(0, std::memory_order_release);
    #pragma omp parallel num_threads(num_scanners_ + 1)
    {
        #pragma omp single
        {
            for (size_t i = 0; i < num_scanners_; ++i) {
                #pragma omp task
                {
                    scanner_task(q_obj, i);
                }
            }
            #pragma omp task
            {
                collector_task();
            }

            #pragma omp taskwait
        }
    }

    for (size_t i = 1; i < num_scanners_; ++i) {
        result_pools_[0].merge(result_pools_[i]);
    }

    update_results(result_pools_[0], query);
    result_pools_[0].copy_results(results);
}

/* fill buckets of all collectors */
inline void QuantizedGraph::buckets_prepare(
    const QGQuery& q_obj
) {
    PID cur_node = this->entry_point_;
    std::vector<float> appro_dist(degree_bound_);
    const float* cur_data = get_vector(cur_node);

    float sqr_y = space::l2_sqr(q_obj.query_data(), cur_data, dimension_);

    /* Compute approximate distance by Fast Scan */
    const auto* packed_code = reinterpret_cast<const uint8_t*>(&cur_data[code_offset_]);
    const auto* factor = &cur_data[factor_offset_];
    this->scanner_.scan_neighbors(
        appro_dist.data(),
        q_obj.lut().data(),
        sqr_y,
        q_obj.lower_val(),
        q_obj.width(),
        q_obj.sumq(),
        packed_code,
        factor
    );

    const PID* ptr_nb = reinterpret_cast<const PID*>(&cur_data[neighbor_offset_]);

    for (size_t j = 0; j < degree_bound_; ++j) {
        PID cur_neighbor = ptr_nb[j];
        float tmp_dist = appro_dist[j];
        bucket_buffer_.insert(cur_neighbor, tmp_dist);
    }
    result_pools_[0].insert(cur_node, sqr_y);
}

inline void QuantizedGraph::scanner_task(
    const QGQuery& q_obj,
    const size_t scanner_id
) {
    float filter_threshold = FLT_MAX;
    float last_threshold = FLT_MAX;
    size_t block_size = pool_.get_block_size();
    std::vector<float> appro_dist(degree_bound_);
    while (bucket_buffer_.has_next(scanner_id)) {
        const PID cur_node = bucket_buffer_.try_pop(scanner_id);
        if (cur_node == NOT_FOUND) {
            continue;
        }
        if (visited_.get(cur_node)) {
            continue;
        }
        visited_.set(cur_node);

        const float* cur_data = get_vector(cur_node);
        const auto* packed_code = reinterpret_cast<const uint8_t*>(&cur_data[code_offset_]);
        const auto* factor = &cur_data[factor_offset_];

        float sqr_y = space::l2_sqr(q_obj.query_data(), cur_data, dimension_);
        size_t remain_count = this->scanner_.scan_neighbors_with_pre_discard(
            appro_dist.data(),
            q_obj.lut().data(),
            sqr_y,
            q_obj.lower_val(),
            q_obj.width(),
            q_obj.sumq(),
            packed_code,
            factor,
            filter_threshold
        );
        if (remain_count < insert_limit_) {
            filter_threshold += 1.1 * filter_threshold;
        }
        result_pools_[scanner_id].insert(cur_node, sqr_y);
    
        const PID* ptr_nb = reinterpret_cast<const PID*>(&cur_data[neighbor_offset_]);
        Candidate<float>* put_buffer = pool_.get_put_block_buffer(scanner_id);
        while (put_buffer == nullptr) {
            std::atomic_thread_fence(std::memory_order_acquire);
            put_buffer = pool_.get_put_block_buffer(scanner_id);
        }
        for (uint32_t i = 0; i < degree_bound_; ++i) {
            PID cur_neighbor = ptr_nb[i];
            if (appro_dist[i] == 0 || visited_.get(cur_neighbor)) {
                continue;
            }
            auto it = std::lower_bound(put_buffer, put_buffer + block_size, Candidate<float>(cur_neighbor, appro_dist[i]));
            if (it == put_buffer + block_size) {
                continue;
            }
            *it = Candidate<float>(cur_neighbor, appro_dist[i]);
        }
        if (put_buffer[(size_t)(filter_alpha_ * insert_limit_) - 1].distance < filter_threshold) {
            last_threshold = filter_threshold;
            filter_threshold = put_buffer[size_t(filter_alpha_ * insert_limit_) - 1].distance;
        }
        // reset put_buffer[insert_limit_:w]
        std::fill(put_buffer + insert_limit_, put_buffer + block_size, Candidate<float>(0, FLT_MAX));
        pool_.submit_put_block_buffer(scanner_id);

    }
    num_finished_scanners.fetch_add(1, std::memory_order_release);
    while (num_finished_scanners.load(std::memory_order_acquire) != num_scanners_) {
        Candidate<float>* put_buffer = pool_.get_put_block_buffer(scanner_id);
        if (put_buffer == nullptr) {
            std::atomic_thread_fence(std::memory_order_acquire);
            continue;
        }
        pool_.submit_put_block_buffer(scanner_id);
    }
}

inline void QuantizedGraph::collector_task() {
    size_t block_size = pool_.get_block_size();
    while (num_finished_scanners.load(std::memory_order_acquire) != num_scanners_) {
        bucket_buffer_.try_promote();
        Candidate<float>* candidates = pool_.get_retrieve_block_buffer();
        if (candidates == nullptr) {
            continue;
        }

        for (size_t i = 0; i < block_size; ++i) {
            if (candidates[i].distance == FLT_MAX || bucket_buffer_.is_full(candidates[i].distance)) {
                continue;
            }
            bucket_buffer_.insert(candidates[i]);
        }
        pool_.submit_retrieve_block_buffer();
    }
}

// scan a data row (including data vec and quantization codes for its neighbors)
// return exact distnace for current vertex
inline float QuantizedGraph::scan_neighbors(
    const QGQuery& q_obj,
    const float* cur_data,
    float* appro_dist,
    buffer::SearchBuffer& search_pool,
    uint32_t cur_degree
) const {
    float sqr_y = space::l2_sqr(q_obj.query_data(), cur_data, dimension_);

    /* Compute approximate distance by Fast Scan */
    const auto* packed_code = reinterpret_cast<const uint8_t*>(&cur_data[code_offset_]);
    const auto* factor = &cur_data[factor_offset_];
    this->scanner_.scan_neighbors(
        appro_dist,
        q_obj.lut().data(),
        sqr_y,
        q_obj.lower_val(),
        q_obj.width(),
        q_obj.sumq(),
        packed_code,
        factor
    );

    const PID* ptr_nb = reinterpret_cast<const PID*>(&cur_data[neighbor_offset_]);
    for (uint32_t i = 0; i < cur_degree; ++i) {
        PID cur_neighbor = ptr_nb[i];
        float tmp_dist = appro_dist[i];
        if (search_pool.is_full(tmp_dist) || visited_.get(cur_neighbor)) {
            continue;
        }
        search_pool.insert(cur_neighbor, tmp_dist);
        memory::mem_prefetch_l2(
            reinterpret_cast<const char*>(get_vector(search_pool.next_id())), 10
        );
    }

    return sqr_y;
}

inline void QuantizedGraph::update_results(
    buffer::ResultBuffer& result_pool, const float* query
) {
    if (result_pool.is_full()) {
        return;
    }

    auto ids = result_pool.ids();
    for (PID data_id : ids) {
        PID* ptr_nb = get_neighbors(data_id);
        for (uint32_t i = 0; i < this->degree_bound_; ++i) {
            PID cur_neighbor = ptr_nb[i];
            if (!visited_.get(cur_neighbor)) {
                visited_.set(cur_neighbor);
                result_pool.insert(
                    cur_neighbor, space::l2_sqr(query, get_vector(cur_neighbor), dimension_)
                );
            }
        }
        if (result_pool.is_full()) {
            break;
        }
    }
}

inline void QuantizedGraph::initialize() {
    /* check size */
    assert(padded_dim_ % 64 == 0);
    assert(padded_dim_ >= dimension_);

    this->code_offset_ = dimension_;  // Pos of packed code (aligned)
    this->factor_offset_ =
        code_offset_ + padded_dim_ / 64 * 2 * degree_bound_;  // Pos of Factor
    this->neighbor_offset_ =
        factor_offset_ + sizeof(Factor) * degree_bound_ / sizeof(float);
    this->row_offset_ = neighbor_offset_ + degree_bound_;

    /* Allocate memory of data*/
    data_ = data::
        Array<float, std::vector<size_t>, memory::AlignedAllocator<float, 1 << 22, true>>(
            std::vector<size_t>{num_points_, row_offset_}
        );
}

// find candidate neighbors for cur_id, exclude the vertex itself
inline void QuantizedGraph::find_candidates(
    PID cur_id,
    size_t search_ef,
    std::vector<Candidate<float>>& results,
    HashBasedBooleanSet& vis,
    const std::vector<uint32_t>& degrees
) const {
    const float* query = get_vector(cur_id);
    QGQuery q_obj(query, padded_dim_);
    q_obj.query_prepare(rotator_, scanner_);

    /* Searching pool initialization */
    buffer::SearchBuffer tmp_pool(search_ef);
    tmp_pool.insert(this->entry_point_, 1e10);
    memory::mem_prefetch_l1(
        reinterpret_cast<const char*>(get_vector(this->entry_point_)), 10
    );

    /* Current version of fast scan compute 32 distances */
    std::vector<float> appro_dist(degree_bound_);  // approximate dis
    while (tmp_pool.has_next()) {
        auto cur_candi = tmp_pool.pop();
        if (vis.get(cur_candi)) {
            continue;
        }
        vis.set(cur_candi);
        auto cur_degree = degrees[cur_candi];
        auto sqr_y = scan_neighbors(
            q_obj, get_vector(cur_candi), appro_dist.data(), tmp_pool, cur_degree
        );
        if (cur_candi != cur_id) {
            results.emplace_back(cur_candi, sqr_y);
        }
    }
}

inline void QuantizedGraph::update_qg(
    PID cur_id, const std::vector<Candidate<float>>& new_neighbors
) {
    size_t cur_degree = new_neighbors.size();

    if (cur_degree == 0) {
        return;
    }
    // copy neighbors
    PID* neighbor_ptr = get_neighbors(cur_id);
    for (size_t i = 0; i < cur_degree; ++i) {
        neighbor_ptr[i] = new_neighbors[i].id;
    }

    RowMatrix<float> x_pad(cur_degree, padded_dim_);  // padded neighbors mat
    RowMatrix<float> c_pad(1, padded_dim_);           // padded duplicate centroid mat
    x_pad.setZero();
    c_pad.setZero();

    /* Copy data */
    for (size_t i = 0; i < cur_degree; ++i) {
        auto neighbor_id = new_neighbors[i].id;
        const auto* cur_data = get_vector(neighbor_id);
        std::copy(cur_data, cur_data + dimension_, &x_pad(static_cast<long>(i), 0));
    }
    const auto* cur_cent = get_vector(cur_id);
    std::copy(cur_cent, cur_cent + dimension_, &c_pad(0, 0));

    /* rotate Matrix */
    RowMatrix<float> x_rotated(cur_degree, padded_dim_);
    RowMatrix<float> c_rotated(1, padded_dim_);
    for (long i = 0; i < static_cast<long>(cur_degree); ++i) {
        this->rotator_.rotate(&x_pad(i, 0), &x_rotated(i, 0));
    }
    this->rotator_.rotate(&c_pad(0, 0), &c_rotated(0, 0));

    // Get codes and factors for rabitq
    float* fac_ptr = get_factor(cur_id);
    float* triple_x = fac_ptr;
    float* factor_dq = triple_x + this->degree_bound_;
    float* factor_vq = factor_dq + this->degree_bound_;
    rabitq_codes(
        x_rotated, c_rotated, get_packed_code(cur_id), triple_x, factor_dq, factor_vq
    );
}
}  // namespace symqg