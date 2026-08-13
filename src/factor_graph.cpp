#include "gbp/FactorGraph.h"
#include <cassert>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <cstring>   // std::memcpy
#include <stdexcept>
#include <atomic>
#include <iterator>
#include <omp.h>
#include <unordered_map>

namespace gbp {

namespace {
using SteadyClock = std::chrono::steady_clock;

double elapsedSeconds(const SteadyClock::time_point& start, const SteadyClock::time_point& end) {
    return std::chrono::duration<double>(end - start).count();
}

bool parallelJointAssemblyEnabled() {
    static const bool enabled = []() {
        const char* value = std::getenv("GBP_JOINT_PARALLEL");
        if (value == nullptr || value[0] == '\0') {
            return false;
        }
        return value[0] != '0';
    }();
    return enabled;
}

void buildContiguousWeightedOffsets(
    int n_items,
    const std::vector<double>& rates,
    std::vector<int>& offsets
) {
    const int thread_count = static_cast<int>(rates.size());
    offsets.assign(thread_count + 1, 0);
    if (thread_count == 0 || n_items <= 0) {
        return;
    }

    double total_rate = 0.0;
    for (double rate : rates) {
        total_rate += std::max(rate, 1e-6);
    }
    if (!(total_rate > 0.0)) {
        total_rate = static_cast<double>(thread_count);
    }

    double prefix = 0.0;
    offsets[0] = 0;
    for (int tid = 0; tid < thread_count - 1; ++tid) {
        prefix += std::max(rates[tid], 1e-6);
        int next = static_cast<int>(std::llround(prefix / total_rate * n_items));
        next = std::max(next, offsets[tid]);
        next = std::min(next, n_items);
        offsets[tid + 1] = next;
    }
    offsets[thread_count] = n_items;
    for (int tid = 1; tid <= thread_count; ++tid) {
        offsets[tid] = std::max(offsets[tid], offsets[tid - 1]);
    }
}

void updateThroughputEma(
    const std::vector<int>& offsets,
    const std::vector<double>& work_times,
    std::vector<double>& rates
) {
    const int thread_count = static_cast<int>(rates.size());
    const double alpha = 0.6;
    for (int tid = 0; tid < thread_count; ++tid) {
        const int count = offsets[tid + 1] - offsets[tid];
        const double work = work_times[tid];
        if (count <= 0 || !(work > 0.0)) {
            continue;
        }
        const double inst_rate = static_cast<double>(count) / work;
        rates[tid] = alpha * rates[tid] + (1.0 - alpha) * inst_rate;
    }
}
}  // namespace

FactorGraph::~FactorGraph() {
    if (!locks_initialized) return;
    for (auto& lk : var_locks) omp_destroy_lock(&lk);
    for (auto& lk : fac_locks) omp_destroy_lock(&lk);
    locks_initialized = false;
}

static inline void ensure_locks(
    std::vector<omp_lock_t>& var_locks,
    std::vector<omp_lock_t>& fac_locks,
    bool& initialized,
    int need_vars,
    int need_facs
) {
    if (!initialized) {
        var_locks.resize(need_vars);
        fac_locks.resize(need_facs);
        for (int i = 0; i < need_vars; ++i) omp_init_lock(&var_locks[i]);
        for (int i = 0; i < need_facs; ++i) omp_init_lock(&fac_locks[i]);
        initialized = true;
        return;
    }

    const int old_vars = (int)var_locks.size();
    const int old_facs = (int)fac_locks.size();
    if (old_vars < need_vars) {
        var_locks.resize(need_vars);
        for (int i = old_vars; i < need_vars; ++i) omp_init_lock(&var_locks[i]);
    }
    if (old_facs < need_facs) {
        fac_locks.resize(need_facs);
        for (int i = old_facs; i < need_facs; ++i) omp_init_lock(&fac_locks[i]);
    }
}

// Atomic max for double (returns true if updated).
static inline bool atomic_max_double(std::atomic<double>& a, double v) noexcept {
    double cur = a.load(std::memory_order_relaxed);
    while (v > cur) {
        if (a.compare_exchange_weak(
                cur, v,
                std::memory_order_acq_rel,
                std::memory_order_relaxed)) {
            return true;
        }
        // on failure, cur is updated with the latest value
    }
    return false;
}

VariableNode* FactorGraph::addVariable(int id, int dofs) {
    if ((int)var_nodes.size() <= id) {
        var_nodes.resize(id + 1);
    }
    var_nodes[id] = std::make_unique<VariableNode>(id, dofs);

    if ((int)var_residual.size() <= id) {
        var_residual.resize(id + 1, 0.0);
    }

    // Keep scheduler atomics sized with variables.
    if ((int)var_residual_a.size() <= id) {
        const int old = (int)var_residual_a.size();
        var_residual_a.resize(id + 1);
        for (int i = old; i <= id; ++i) {
            var_residual_a[i].store(0.0, std::memory_order_relaxed);
        }
    }
    if ((int)var_ver.size() <= id) {
        const int old = (int)var_ver.size();
        var_ver.resize(id + 1);
        for (int i = old; i <= id; ++i) {
            var_ver[i].store(0u, std::memory_order_relaxed);
        }
    }
    return var_nodes[id].get();
}

Factor* FactorGraph::addFactor(
    int id,
    const std::vector<VariableNode*>& vars,
    const std::vector<Eigen::VectorXd>& z,
    const std::vector<Eigen::MatrixXd>& measurement_lambda,
    std::function<std::vector<Eigen::VectorXd>(const Eigen::VectorXd&)> meas_fn,
    std::function<std::vector<Eigen::MatrixXd>(const Eigen::VectorXd&)> jac_fn
) {
    factors.push_back(std::make_unique<Factor>(
        id, vars, z, measurement_lambda, std::move(meas_fn), std::move(jac_fn)
    ));
    return factors.back().get();
}

void FactorGraph::connect(Factor* f, VariableNode* v, int local_idx) {
    assert(f && v);
    v->adj_factors.push_back(AdjFactorRef{
        f,
        local_idx,
        f->currentMessageEtaSlot(local_idx),
        f->currentMessageLamSlot(local_idx)
    });
    v->adj_factors_raw.push_back(f);
}

void FactorGraph::synchronousIteration(bool /*robustify*/) {
    const bool use_parallel = true;
    const bool profile = profile_sync_timing;
    const bool profile_threads = profile_sync_thread_utilization && use_parallel;
    SteadyClock::time_point factor_t0;
    SteadyClock::time_point factor_t1;
    SteadyClock::time_point var_t0;
    SteadyClock::time_point var_t1;

    if (use_parallel) {
        if (!profile_threads &&
            !profile &&
            sync_schedule_kind == SyncScheduleKind::Static &&
            sync_weighted_static_partition &&
            sync_num_threads > 1) {
            const int thread_count = sync_num_threads;
            if (!sync_weighted_offsets_ready_ ||
                static_cast<int>(sync_factor_rates_.size()) != thread_count) {
                sync_factor_rates_.assign(thread_count, 1.0);
                sync_variable_rates_.assign(thread_count, 1.0);
                sync_factor_work_last_.assign(thread_count, 0.0);
                sync_variable_work_last_.assign(thread_count, 0.0);
                buildContiguousWeightedOffsets(static_cast<int>(factors.size()), sync_factor_rates_, sync_factor_offsets_);
                buildContiguousWeightedOffsets(static_cast<int>(var_nodes.size()), sync_variable_rates_, sync_variable_offsets_);
                sync_weighted_offsets_ready_ = true;
            } else {
                buildContiguousWeightedOffsets(static_cast<int>(factors.size()), sync_factor_rates_, sync_factor_offsets_);
                buildContiguousWeightedOffsets(static_cast<int>(var_nodes.size()), sync_variable_rates_, sync_variable_offsets_);
            }

            #pragma omp parallel num_threads(sync_num_threads)
            {
                const int tid = omp_get_thread_num();

                const int factor_begin = sync_factor_offsets_[tid];
                const int factor_end = sync_factor_offsets_[tid + 1];
                const double factor_t0 = omp_get_wtime();
                for (int i = factor_begin; i < factor_end; ++i) {
                    auto& fptr = factors[i];
                    if (!fptr || !fptr->active) continue;
                    fptr->computeMessages(eta_damping);
                }
                const double factor_t1 = omp_get_wtime();
                sync_factor_work_last_[tid] = factor_t1 - factor_t0;

                #pragma omp barrier

                const int var_begin = sync_variable_offsets_[tid];
                const int var_end = sync_variable_offsets_[tid + 1];
                const double var_t0 = omp_get_wtime();
                for (int i = var_begin; i < var_end; ++i) {
                    auto& vptr = var_nodes[i];
                    if (!vptr) continue;
                    if (sync_update_means) {
                        vptr->updateBelief();
                    } else {
                        vptr->updateBeliefNoMu();
                    }
                }
                const double var_t1 = omp_get_wtime();
                sync_variable_work_last_[tid] = var_t1 - var_t0;
            }

            updateThroughputEma(sync_factor_offsets_, sync_factor_work_last_, sync_factor_rates_);
            updateThroughputEma(sync_variable_offsets_, sync_variable_work_last_, sync_variable_rates_);
            return;
        }

        if (!profile_threads && sync_schedule_kind == SyncScheduleKind::Static) {
            if (!profile) {
                if (sync_num_threads > 0) {
                    #pragma omp parallel num_threads(sync_num_threads)
                    {
                        #pragma omp for schedule(static)
                        for (int i = 0; i < (int)factors.size(); ++i) {
                            auto& fptr = factors[i];
                            if (!fptr || !fptr->active) continue;
                            fptr->computeMessages(eta_damping);
                        }

                        #pragma omp for schedule(static) nowait
                        for (int i = 0; i < (int)var_nodes.size(); ++i) {
                            auto& vptr = var_nodes[i];
                            if (!vptr) continue;
                            if (sync_update_means) {
                                vptr->updateBelief();
                            } else {
                                vptr->updateBeliefNoMu();
                            }
                        }
                    }
                } else {
                    #pragma omp parallel
                    {
                        #pragma omp for schedule(static)
                        for (int i = 0; i < (int)factors.size(); ++i) {
                            auto& fptr = factors[i];
                            if (!fptr || !fptr->active) continue;
                            fptr->computeMessages(eta_damping);
                        }

                        #pragma omp for schedule(static) nowait
                        for (int i = 0; i < (int)var_nodes.size(); ++i) {
                            auto& vptr = var_nodes[i];
                            if (!vptr) continue;
                            if (sync_update_means) {
                                vptr->updateBelief();
                            } else {
                                vptr->updateBeliefNoMu();
                            }
                        }
                    }
                }
                return;
            }

            if (sync_num_threads > 0) {
                if (profile) factor_t0 = SteadyClock::now();
                #pragma omp parallel for schedule(static) num_threads(sync_num_threads)
                for (int i = 0; i < (int)factors.size(); ++i) {
                    auto& fptr = factors[i];
                    if (!fptr || !fptr->active) continue;
                    fptr->computeMessages(eta_damping);
                }
                if (profile) factor_t1 = SteadyClock::now();

                if (profile) var_t0 = SteadyClock::now();
                #pragma omp parallel for schedule(static) num_threads(sync_num_threads)
                for (int i = 0; i < (int)var_nodes.size(); ++i) {
                    auto& vptr = var_nodes[i];
                    if (!vptr) continue;
                    if (sync_update_means) {
                        vptr->updateBelief();
                    } else {
                        vptr->updateBeliefNoMu();
                    }
                }
                if (profile) var_t1 = SteadyClock::now();
            } else {
                if (profile) factor_t0 = SteadyClock::now();
                #pragma omp parallel for schedule(static)
                for (int i = 0; i < (int)factors.size(); ++i) {
                    auto& fptr = factors[i];
                    if (!fptr || !fptr->active) continue;
                    fptr->computeMessages(eta_damping);
                }
                if (profile) factor_t1 = SteadyClock::now();

                if (profile) var_t0 = SteadyClock::now();
                #pragma omp parallel for schedule(static)
                for (int i = 0; i < (int)var_nodes.size(); ++i) {
                    auto& vptr = var_nodes[i];
                    if (!vptr) continue;
                    if (sync_update_means) {
                        vptr->updateBelief();
                    } else {
                        vptr->updateBeliefNoMu();
                    }
                }
                if (profile) var_t1 = SteadyClock::now();
            }
            if (profile) {
                sync_factor_pass_sec_accum += elapsedSeconds(factor_t0, factor_t1);
                sync_variable_pass_sec_accum += elapsedSeconds(var_t0, var_t1);
            }
            return;
        }

        if (profile_threads) {
            const int thread_count = std::max(1, (sync_num_threads > 0) ? sync_num_threads : omp_get_max_threads());
            std::vector<double> factor_work(thread_count, 0.0);
            std::vector<double> factor_wait(thread_count, 0.0);
            std::vector<double> factor_wall(thread_count, 0.0);
            std::vector<double> variable_work(thread_count, 0.0);
            std::vector<double> variable_wait(thread_count, 0.0);
            std::vector<double> variable_wall(thread_count, 0.0);
            std::atomic<int> observed_threads{0};

            if (sync_num_threads > 0) {
                #pragma omp parallel num_threads(sync_num_threads)
                {
                    const int tid = omp_get_thread_num();
                    const int team_size = omp_get_num_threads();
                    observed_threads.store(team_size, std::memory_order_relaxed);

                    const double factor_loop_t0 = omp_get_wtime();
                    if (sync_schedule_kind == SyncScheduleKind::Dynamic) {
                        #pragma omp for schedule(dynamic) nowait
                        for (int i = 0; i < (int)factors.size(); ++i) {
                            auto& fptr = factors[i];
                            if (!fptr || !fptr->active) continue;
                            fptr->computeMessages(eta_damping);
                        }
                    } else if (sync_schedule_kind == SyncScheduleKind::Guided) {
                        #pragma omp for schedule(guided) nowait
                        for (int i = 0; i < (int)factors.size(); ++i) {
                            auto& fptr = factors[i];
                            if (!fptr || !fptr->active) continue;
                            fptr->computeMessages(eta_damping);
                        }
                    } else {
                        #pragma omp for schedule(static) nowait
                        for (int i = 0; i < (int)factors.size(); ++i) {
                            auto& fptr = factors[i];
                            if (!fptr || !fptr->active) continue;
                            fptr->computeMessages(eta_damping);
                        }
                    }
                    const double factor_work_t1 = omp_get_wtime();
                    #pragma omp barrier
                    const double factor_barrier_t1 = omp_get_wtime();
                    factor_work[tid] = factor_work_t1 - factor_loop_t0;
                    factor_wait[tid] = factor_barrier_t1 - factor_work_t1;
                    factor_wall[tid] = factor_barrier_t1 - factor_loop_t0;

                    const double var_loop_t0 = omp_get_wtime();
                    if (sync_schedule_kind == SyncScheduleKind::Dynamic) {
                        #pragma omp for schedule(dynamic) nowait
                        for (int i = 0; i < (int)var_nodes.size(); ++i) {
                            auto& vptr = var_nodes[i];
                            if (!vptr) continue;
                            if (sync_update_means) {
                                vptr->updateBelief();
                            } else {
                                vptr->updateBeliefNoMu();
                            }
                        }
                    } else if (sync_schedule_kind == SyncScheduleKind::Guided) {
                        #pragma omp for schedule(guided) nowait
                        for (int i = 0; i < (int)var_nodes.size(); ++i) {
                            auto& vptr = var_nodes[i];
                            if (!vptr) continue;
                            if (sync_update_means) {
                                vptr->updateBelief();
                            } else {
                                vptr->updateBeliefNoMu();
                            }
                        }
                    } else {
                        #pragma omp for schedule(static) nowait
                        for (int i = 0; i < (int)var_nodes.size(); ++i) {
                            auto& vptr = var_nodes[i];
                            if (!vptr) continue;
                            if (sync_update_means) {
                                vptr->updateBelief();
                            } else {
                                vptr->updateBeliefNoMu();
                            }
                        }
                    }
                    const double var_work_t1 = omp_get_wtime();
                    #pragma omp barrier
                    const double var_barrier_t1 = omp_get_wtime();
                    variable_work[tid] = var_work_t1 - var_loop_t0;
                    variable_wait[tid] = var_barrier_t1 - var_work_t1;
                    variable_wall[tid] = var_barrier_t1 - var_loop_t0;
                }
            } else {
                #pragma omp parallel
                {
                    const int tid = omp_get_thread_num();
                    const int team_size = omp_get_num_threads();
                    observed_threads.store(team_size, std::memory_order_relaxed);

                    const double factor_loop_t0 = omp_get_wtime();
                    if (sync_schedule_kind == SyncScheduleKind::Dynamic) {
                        #pragma omp for schedule(dynamic) nowait
                        for (int i = 0; i < (int)factors.size(); ++i) {
                            auto& fptr = factors[i];
                            if (!fptr || !fptr->active) continue;
                            fptr->computeMessages(eta_damping);
                        }
                    } else if (sync_schedule_kind == SyncScheduleKind::Guided) {
                        #pragma omp for schedule(guided) nowait
                        for (int i = 0; i < (int)factors.size(); ++i) {
                            auto& fptr = factors[i];
                            if (!fptr || !fptr->active) continue;
                            fptr->computeMessages(eta_damping);
                        }
                    } else {
                        #pragma omp for schedule(static) nowait
                        for (int i = 0; i < (int)factors.size(); ++i) {
                            auto& fptr = factors[i];
                            if (!fptr || !fptr->active) continue;
                            fptr->computeMessages(eta_damping);
                        }
                    }
                    const double factor_work_t1 = omp_get_wtime();
                    #pragma omp barrier
                    const double factor_barrier_t1 = omp_get_wtime();
                    factor_work[tid] = factor_work_t1 - factor_loop_t0;
                    factor_wait[tid] = factor_barrier_t1 - factor_work_t1;
                    factor_wall[tid] = factor_barrier_t1 - factor_loop_t0;

                    const double var_loop_t0 = omp_get_wtime();
                    if (sync_schedule_kind == SyncScheduleKind::Dynamic) {
                        #pragma omp for schedule(dynamic) nowait
                        for (int i = 0; i < (int)var_nodes.size(); ++i) {
                            auto& vptr = var_nodes[i];
                            if (!vptr) continue;
                            if (sync_update_means) {
                                vptr->updateBelief();
                            } else {
                                vptr->updateBeliefNoMu();
                            }
                        }
                    } else if (sync_schedule_kind == SyncScheduleKind::Guided) {
                        #pragma omp for schedule(guided) nowait
                        for (int i = 0; i < (int)var_nodes.size(); ++i) {
                            auto& vptr = var_nodes[i];
                            if (!vptr) continue;
                            if (sync_update_means) {
                                vptr->updateBelief();
                            } else {
                                vptr->updateBeliefNoMu();
                            }
                        }
                    } else {
                        #pragma omp for schedule(static) nowait
                        for (int i = 0; i < (int)var_nodes.size(); ++i) {
                            auto& vptr = var_nodes[i];
                            if (!vptr) continue;
                            if (sync_update_means) {
                                vptr->updateBelief();
                            } else {
                                vptr->updateBeliefNoMu();
                            }
                        }
                    }
                    const double var_work_t1 = omp_get_wtime();
                    #pragma omp barrier
                    const double var_barrier_t1 = omp_get_wtime();
                    variable_work[tid] = var_work_t1 - var_loop_t0;
                    variable_wait[tid] = var_barrier_t1 - var_work_t1;
                    variable_wall[tid] = var_barrier_t1 - var_loop_t0;
                }
            }

            double factor_work_sum = 0.0;
            double factor_wait_sum = 0.0;
            double factor_wall_max = 0.0;
            double variable_work_sum = 0.0;
            double variable_wait_sum = 0.0;
            double variable_wall_max = 0.0;
            const int team_size = std::max(1, observed_threads.load(std::memory_order_relaxed));
            for (int tid = 0; tid < team_size; ++tid) {
                factor_work_sum += factor_work[tid];
                factor_wait_sum += factor_wait[tid];
                factor_wall_max = std::max(factor_wall_max, factor_wall[tid]);
                variable_work_sum += variable_work[tid];
                variable_wait_sum += variable_wait[tid];
                variable_wall_max = std::max(variable_wall_max, variable_wall[tid]);
            }
            sync_profiled_threads = std::max(sync_profiled_threads, team_size);
            sync_factor_work_sec_accum += factor_work_sum;
            sync_factor_wait_sec_accum += factor_wait_sum;
            sync_variable_work_sec_accum += variable_work_sum;
            sync_variable_wait_sec_accum += variable_wait_sum;
            sync_factor_pass_sec_accum += factor_wall_max;
            sync_variable_pass_sec_accum += variable_wall_max;
            return;
        }

        if (!profile) {
            if (sync_num_threads > 0) {
                #pragma omp parallel num_threads(sync_num_threads)
                {
                    if (sync_schedule_kind == SyncScheduleKind::Dynamic) {
                        #pragma omp for schedule(dynamic)
                        for (int i = 0; i < (int)factors.size(); ++i) {
                            auto& fptr = factors[i];
                            if (!fptr || !fptr->active) continue;
                            fptr->computeMessages(eta_damping);
                        }

                        #pragma omp for schedule(dynamic) nowait
                        for (int i = 0; i < (int)var_nodes.size(); ++i) {
                            auto& vptr = var_nodes[i];
                            if (!vptr) continue;
                            if (sync_update_means) {
                                vptr->updateBelief();
                            } else {
                                vptr->updateBeliefNoMu();
                            }
                        }
                    } else if (sync_schedule_kind == SyncScheduleKind::Guided) {
                        #pragma omp for schedule(guided)
                        for (int i = 0; i < (int)factors.size(); ++i) {
                            auto& fptr = factors[i];
                            if (!fptr || !fptr->active) continue;
                            fptr->computeMessages(eta_damping);
                        }

                        #pragma omp for schedule(guided) nowait
                        for (int i = 0; i < (int)var_nodes.size(); ++i) {
                            auto& vptr = var_nodes[i];
                            if (!vptr) continue;
                            if (sync_update_means) {
                                vptr->updateBelief();
                            } else {
                                vptr->updateBeliefNoMu();
                            }
                        }
                    } else {
                        #pragma omp for schedule(static)
                        for (int i = 0; i < (int)factors.size(); ++i) {
                            auto& fptr = factors[i];
                            if (!fptr || !fptr->active) continue;
                            fptr->computeMessages(eta_damping);
                        }

                        #pragma omp for schedule(static) nowait
                        for (int i = 0; i < (int)var_nodes.size(); ++i) {
                            auto& vptr = var_nodes[i];
                            if (!vptr) continue;
                            if (sync_update_means) {
                                vptr->updateBelief();
                            } else {
                                vptr->updateBeliefNoMu();
                            }
                        }
                    }
                }
            } else {
                #pragma omp parallel
                {
                    if (sync_schedule_kind == SyncScheduleKind::Dynamic) {
                        #pragma omp for schedule(dynamic)
                        for (int i = 0; i < (int)factors.size(); ++i) {
                            auto& fptr = factors[i];
                            if (!fptr || !fptr->active) continue;
                            fptr->computeMessages(eta_damping);
                        }

                        #pragma omp for schedule(dynamic) nowait
                        for (int i = 0; i < (int)var_nodes.size(); ++i) {
                            auto& vptr = var_nodes[i];
                            if (!vptr) continue;
                            if (sync_update_means) {
                                vptr->updateBelief();
                            } else {
                                vptr->updateBeliefNoMu();
                            }
                        }
                    } else if (sync_schedule_kind == SyncScheduleKind::Guided) {
                        #pragma omp for schedule(guided)
                        for (int i = 0; i < (int)factors.size(); ++i) {
                            auto& fptr = factors[i];
                            if (!fptr || !fptr->active) continue;
                            fptr->computeMessages(eta_damping);
                        }

                        #pragma omp for schedule(guided) nowait
                        for (int i = 0; i < (int)var_nodes.size(); ++i) {
                            auto& vptr = var_nodes[i];
                            if (!vptr) continue;
                            if (sync_update_means) {
                                vptr->updateBelief();
                            } else {
                                vptr->updateBeliefNoMu();
                            }
                        }
                    } else {
                        #pragma omp for schedule(static)
                        for (int i = 0; i < (int)factors.size(); ++i) {
                            auto& fptr = factors[i];
                            if (!fptr || !fptr->active) continue;
                            fptr->computeMessages(eta_damping);
                        }

                        #pragma omp for schedule(static) nowait
                        for (int i = 0; i < (int)var_nodes.size(); ++i) {
                            auto& vptr = var_nodes[i];
                            if (!vptr) continue;
                            if (sync_update_means) {
                                vptr->updateBelief();
                            } else {
                                vptr->updateBeliefNoMu();
                            }
                        }
                    }
                }
            }
            return;
        }

        if (sync_num_threads > 0) {
            if (profile) factor_t0 = SteadyClock::now();
            if (sync_schedule_kind == SyncScheduleKind::Dynamic) {
                #pragma omp parallel for schedule(dynamic) num_threads(sync_num_threads)
                for (int i = 0; i < (int)factors.size(); ++i) {
                    auto& fptr = factors[i];
                    if (!fptr || !fptr->active) continue;
                    fptr->computeMessages(eta_damping);
                }
            } else if (sync_schedule_kind == SyncScheduleKind::Guided) {
                #pragma omp parallel for schedule(guided) num_threads(sync_num_threads)
                for (int i = 0; i < (int)factors.size(); ++i) {
                    auto& fptr = factors[i];
                    if (!fptr || !fptr->active) continue;
                    fptr->computeMessages(eta_damping);
                }
            } else {
                #pragma omp parallel for schedule(static) num_threads(sync_num_threads)
                for (int i = 0; i < (int)factors.size(); ++i) {
                    auto& fptr = factors[i];
                    if (!fptr || !fptr->active) continue;
                    fptr->computeMessages(eta_damping);
                }
            }
            if (profile) factor_t1 = SteadyClock::now();

            if (profile) var_t0 = SteadyClock::now();
            if (sync_schedule_kind == SyncScheduleKind::Dynamic) {
                #pragma omp parallel for schedule(dynamic) num_threads(sync_num_threads)
                for (int i = 0; i < (int)var_nodes.size(); ++i) {
                    auto& vptr = var_nodes[i];
                    if (!vptr) continue;
                    if (sync_update_means) {
                        vptr->updateBelief();
                    } else {
                        vptr->updateBeliefNoMu();
                    }
                }
            } else if (sync_schedule_kind == SyncScheduleKind::Guided) {
                #pragma omp parallel for schedule(guided) num_threads(sync_num_threads)
                for (int i = 0; i < (int)var_nodes.size(); ++i) {
                    auto& vptr = var_nodes[i];
                    if (!vptr) continue;
                    if (sync_update_means) {
                        vptr->updateBelief();
                    } else {
                        vptr->updateBeliefNoMu();
                    }
                }
            } else {
                #pragma omp parallel for schedule(static) num_threads(sync_num_threads)
                for (int i = 0; i < (int)var_nodes.size(); ++i) {
                    auto& vptr = var_nodes[i];
                    if (!vptr) continue;
                    if (sync_update_means) {
                        vptr->updateBelief();
                    } else {
                        vptr->updateBeliefNoMu();
                    }
                }
            }
            if (profile) var_t1 = SteadyClock::now();
        } else {
            if (profile) factor_t0 = SteadyClock::now();
            if (sync_schedule_kind == SyncScheduleKind::Dynamic) {
                #pragma omp parallel for schedule(dynamic)
                for (int i = 0; i < (int)factors.size(); ++i) {
                    auto& fptr = factors[i];
                    if (!fptr || !fptr->active) continue;
                    fptr->computeMessages(eta_damping);
                }
            } else if (sync_schedule_kind == SyncScheduleKind::Guided) {
                #pragma omp parallel for schedule(guided)
                for (int i = 0; i < (int)factors.size(); ++i) {
                    auto& fptr = factors[i];
                    if (!fptr || !fptr->active) continue;
                    fptr->computeMessages(eta_damping);
                }
            } else {
                #pragma omp parallel for schedule(static)
                for (int i = 0; i < (int)factors.size(); ++i) {
                    auto& fptr = factors[i];
                    if (!fptr || !fptr->active) continue;
                    fptr->computeMessages(eta_damping);
                }
            }
            if (profile) factor_t1 = SteadyClock::now();

            if (profile) var_t0 = SteadyClock::now();
            if (sync_schedule_kind == SyncScheduleKind::Dynamic) {
                #pragma omp parallel for schedule(dynamic)
                for (int i = 0; i < (int)var_nodes.size(); ++i) {
                    auto& vptr = var_nodes[i];
                    if (!vptr) continue;
                    if (sync_update_means) {
                        vptr->updateBelief();
                    } else {
                        vptr->updateBeliefNoMu();
                    }
                }
            } else if (sync_schedule_kind == SyncScheduleKind::Guided) {
                #pragma omp parallel for schedule(guided)
                for (int i = 0; i < (int)var_nodes.size(); ++i) {
                    auto& vptr = var_nodes[i];
                    if (!vptr) continue;
                    if (sync_update_means) {
                        vptr->updateBelief();
                    } else {
                        vptr->updateBeliefNoMu();
                    }
                }
            } else {
                #pragma omp parallel for schedule(static)
                for (int i = 0; i < (int)var_nodes.size(); ++i) {
                    auto& vptr = var_nodes[i];
                    if (!vptr) continue;
                    if (sync_update_means) {
                        vptr->updateBelief();
                    } else {
                        vptr->updateBeliefNoMu();
                    }
                }
            }
            if (profile) var_t1 = SteadyClock::now();
        }
        if (profile) {
            sync_factor_pass_sec_accum += elapsedSeconds(factor_t0, factor_t1);
            sync_variable_pass_sec_accum += elapsedSeconds(var_t0, var_t1);
        }
        return;
    }

    if (profile) factor_t0 = SteadyClock::now();
    for (int i = 0; i < (int)factors.size(); ++i) {
        auto& fptr = factors[i];
        if (!fptr || !fptr->active) continue;
        fptr->computeMessages(eta_damping);
    }
    if (profile) factor_t1 = SteadyClock::now();

    if (profile) var_t0 = SteadyClock::now();
    for (int i = 0; i < (int)var_nodes.size(); ++i) {
        auto& vptr = var_nodes[i];
        if (!vptr) continue;
        if (sync_update_means) {
            vptr->updateBelief();
        } else {
            vptr->updateBeliefNoMu();
        }
    }
    if (profile) {
        var_t1 = SteadyClock::now();
        sync_factor_pass_sec_accum += elapsedSeconds(factor_t0, factor_t1);
        sync_variable_pass_sec_accum += elapsedSeconds(var_t0, var_t1);
    }
}

void FactorGraph::synchronousIterationFixedLam(bool /*robustify*/) {
    const bool use_parallel = true;
    const bool profile = profile_sync_timing;
    SteadyClock::time_point factor_t0;
    SteadyClock::time_point factor_t1;
    SteadyClock::time_point var_t0;
    SteadyClock::time_point var_t1;

    if (use_parallel) {
        if (!profile) {
            if (sync_num_threads > 0) {
                #pragma omp parallel num_threads(sync_num_threads)
                {
                    if (sync_schedule_kind == SyncScheduleKind::Dynamic) {
                        #pragma omp for schedule(dynamic)
                        for (int i = 0; i < (int)factors.size(); ++i) {
                            auto& fptr = factors[i];
                            if (!fptr || !fptr->active) continue;
                            fptr->computeMessagesFixedLam(eta_damping);
                        }

                        #pragma omp for schedule(dynamic) nowait
                        for (int i = 0; i < (int)var_nodes.size(); ++i) {
                            auto& vptr = var_nodes[i];
                            if (!vptr) continue;
                            if (sync_update_means) {
                                vptr->updateBelief();
                            } else {
                                vptr->updateBeliefNoMu();
                            }
                        }
                    } else if (sync_schedule_kind == SyncScheduleKind::Guided) {
                        #pragma omp for schedule(guided)
                        for (int i = 0; i < (int)factors.size(); ++i) {
                            auto& fptr = factors[i];
                            if (!fptr || !fptr->active) continue;
                            fptr->computeMessagesFixedLam(eta_damping);
                        }

                        #pragma omp for schedule(guided) nowait
                        for (int i = 0; i < (int)var_nodes.size(); ++i) {
                            auto& vptr = var_nodes[i];
                            if (!vptr) continue;
                            if (sync_update_means) {
                                vptr->updateBelief();
                            } else {
                                vptr->updateBeliefNoMu();
                            }
                        }
                    } else {
                        #pragma omp for schedule(static)
                        for (int i = 0; i < (int)factors.size(); ++i) {
                            auto& fptr = factors[i];
                            if (!fptr || !fptr->active) continue;
                            fptr->computeMessagesFixedLam(eta_damping);
                        }

                        #pragma omp for schedule(static) nowait
                        for (int i = 0; i < (int)var_nodes.size(); ++i) {
                            auto& vptr = var_nodes[i];
                            if (!vptr) continue;
                            if (sync_update_means) {
                                vptr->updateBelief();
                            } else {
                                vptr->updateBeliefNoMu();
                            }
                        }
                    }
                }
            } else {
                #pragma omp parallel
                {
                    if (sync_schedule_kind == SyncScheduleKind::Dynamic) {
                        #pragma omp for schedule(dynamic)
                        for (int i = 0; i < (int)factors.size(); ++i) {
                            auto& fptr = factors[i];
                            if (!fptr || !fptr->active) continue;
                            fptr->computeMessagesFixedLam(eta_damping);
                        }

                        #pragma omp for schedule(dynamic) nowait
                        for (int i = 0; i < (int)var_nodes.size(); ++i) {
                            auto& vptr = var_nodes[i];
                            if (!vptr) continue;
                            if (sync_update_means) {
                                vptr->updateBelief();
                            } else {
                                vptr->updateBeliefNoMu();
                            }
                        }
                    } else if (sync_schedule_kind == SyncScheduleKind::Guided) {
                        #pragma omp for schedule(guided)
                        for (int i = 0; i < (int)factors.size(); ++i) {
                            auto& fptr = factors[i];
                            if (!fptr || !fptr->active) continue;
                            fptr->computeMessagesFixedLam(eta_damping);
                        }

                        #pragma omp for schedule(guided) nowait
                        for (int i = 0; i < (int)var_nodes.size(); ++i) {
                            auto& vptr = var_nodes[i];
                            if (!vptr) continue;
                            if (sync_update_means) {
                                vptr->updateBelief();
                            } else {
                                vptr->updateBeliefNoMu();
                            }
                        }
                    } else {
                        #pragma omp for schedule(static)
                        for (int i = 0; i < (int)factors.size(); ++i) {
                            auto& fptr = factors[i];
                            if (!fptr || !fptr->active) continue;
                            fptr->computeMessagesFixedLam(eta_damping);
                        }

                        #pragma omp for schedule(static) nowait
                        for (int i = 0; i < (int)var_nodes.size(); ++i) {
                            auto& vptr = var_nodes[i];
                            if (!vptr) continue;
                            if (sync_update_means) {
                                vptr->updateBelief();
                            } else {
                                vptr->updateBeliefNoMu();
                            }
                        }
                    }
                }
            }
            return;
        }

        if (sync_num_threads > 0) {
            if (profile) factor_t0 = SteadyClock::now();
            if (sync_schedule_kind == SyncScheduleKind::Dynamic) {
                #pragma omp parallel for schedule(dynamic) num_threads(sync_num_threads)
                for (int i = 0; i < (int)factors.size(); ++i) {
                    auto& fptr = factors[i];
                    if (!fptr || !fptr->active) continue;
                    fptr->computeMessagesFixedLam(eta_damping);
                }
            } else if (sync_schedule_kind == SyncScheduleKind::Guided) {
                #pragma omp parallel for schedule(guided) num_threads(sync_num_threads)
                for (int i = 0; i < (int)factors.size(); ++i) {
                    auto& fptr = factors[i];
                    if (!fptr || !fptr->active) continue;
                    fptr->computeMessagesFixedLam(eta_damping);
                }
            } else {
                #pragma omp parallel for schedule(static) num_threads(sync_num_threads)
                for (int i = 0; i < (int)factors.size(); ++i) {
                    auto& fptr = factors[i];
                    if (!fptr || !fptr->active) continue;
                    fptr->computeMessagesFixedLam(eta_damping);
                }
            }
            if (profile) factor_t1 = SteadyClock::now();

            if (profile) var_t0 = SteadyClock::now();
            if (sync_schedule_kind == SyncScheduleKind::Dynamic) {
                #pragma omp parallel for schedule(dynamic) num_threads(sync_num_threads)
                for (int i = 0; i < (int)var_nodes.size(); ++i) {
                    auto& vptr = var_nodes[i];
                    if (!vptr) continue;
                    if (sync_update_means) {
                        vptr->updateBelief();
                    } else {
                        vptr->updateBeliefNoMu();
                    }
                }
            } else if (sync_schedule_kind == SyncScheduleKind::Guided) {
                #pragma omp parallel for schedule(guided) num_threads(sync_num_threads)
                for (int i = 0; i < (int)var_nodes.size(); ++i) {
                    auto& vptr = var_nodes[i];
                    if (!vptr) continue;
                    if (sync_update_means) {
                        vptr->updateBelief();
                    } else {
                        vptr->updateBeliefNoMu();
                    }
                }
            } else {
                #pragma omp parallel for schedule(static) num_threads(sync_num_threads)
                for (int i = 0; i < (int)var_nodes.size(); ++i) {
                    auto& vptr = var_nodes[i];
                    if (!vptr) continue;
                    if (sync_update_means) {
                        vptr->updateBelief();
                    } else {
                        vptr->updateBeliefNoMu();
                    }
                }
            }
            if (profile) var_t1 = SteadyClock::now();
        } else {
            if (profile) factor_t0 = SteadyClock::now();
            if (sync_schedule_kind == SyncScheduleKind::Dynamic) {
                #pragma omp parallel for schedule(dynamic)
                for (int i = 0; i < (int)factors.size(); ++i) {
                    auto& fptr = factors[i];
                    if (!fptr || !fptr->active) continue;
                    fptr->computeMessagesFixedLam(eta_damping);
                }
            } else if (sync_schedule_kind == SyncScheduleKind::Guided) {
                #pragma omp parallel for schedule(guided)
                for (int i = 0; i < (int)factors.size(); ++i) {
                    auto& fptr = factors[i];
                    if (!fptr || !fptr->active) continue;
                    fptr->computeMessagesFixedLam(eta_damping);
                }
            } else {
                #pragma omp parallel for schedule(static)
                for (int i = 0; i < (int)factors.size(); ++i) {
                    auto& fptr = factors[i];
                    if (!fptr || !fptr->active) continue;
                    fptr->computeMessagesFixedLam(eta_damping);
                }
            }
            if (profile) factor_t1 = SteadyClock::now();

            if (profile) var_t0 = SteadyClock::now();
            if (sync_schedule_kind == SyncScheduleKind::Dynamic) {
                #pragma omp parallel for schedule(dynamic)
                for (int i = 0; i < (int)var_nodes.size(); ++i) {
                    auto& vptr = var_nodes[i];
                    if (!vptr) continue;
                    if (sync_update_means) {
                        vptr->updateBelief();
                    } else {
                        vptr->updateBeliefNoMu();
                    }
                }
            } else if (sync_schedule_kind == SyncScheduleKind::Guided) {
                #pragma omp parallel for schedule(guided)
                for (int i = 0; i < (int)var_nodes.size(); ++i) {
                    auto& vptr = var_nodes[i];
                    if (!vptr) continue;
                    if (sync_update_means) {
                        vptr->updateBelief();
                    } else {
                        vptr->updateBeliefNoMu();
                    }
                }
            } else {
                #pragma omp parallel for schedule(static)
                for (int i = 0; i < (int)var_nodes.size(); ++i) {
                    auto& vptr = var_nodes[i];
                    if (!vptr) continue;
                    if (sync_update_means) {
                        vptr->updateBelief();
                    } else {
                        vptr->updateBeliefNoMu();
                    }
                }
            }
            if (profile) var_t1 = SteadyClock::now();
        }
        if (profile) {
            sync_factor_pass_sec_accum += elapsedSeconds(factor_t0, factor_t1);
            sync_variable_pass_sec_accum += elapsedSeconds(var_t0, var_t1);
        }
        return;
    }

    if (profile) factor_t0 = SteadyClock::now();
    for (int i = 0; i < (int)factors.size(); ++i) {
        auto& fptr = factors[i];
        if (!fptr || !fptr->active) continue;
        fptr->computeMessagesFixedLam(eta_damping);
    }
    if (profile) factor_t1 = SteadyClock::now();

    if (profile) var_t0 = SteadyClock::now();
    for (int i = 0; i < (int)var_nodes.size(); ++i) {
        auto& vptr = var_nodes[i];
        if (!vptr) continue;
        if (sync_update_means) {
            vptr->updateBelief();
        } else {
            vptr->updateBeliefNoMu();
        }
    }
    if (profile) {
        var_t1 = SteadyClock::now();
        sync_factor_pass_sec_accum += elapsedSeconds(factor_t0, factor_t1);
        sync_variable_pass_sec_accum += elapsedSeconds(var_t0, var_t1);
    }
}

void FactorGraph::relinearizeAllFactors() {
    // Explicit Gauss-Newton style re-linearization.
    // For each factor, build the linearization point by concatenating the
    // current means (mu) of its adjacent variables, then recompute the factor.
    for (auto& fup : factors) {
        Factor* f = fup.get();
        if (!f || !f->active) continue;

        int total = 0;
        for (auto* vn : f->adj_var_nodes) {
            if (!vn) continue;
            total += vn->dofs;
        }
        if (total <= 0) continue;

        Eigen::VectorXd linpoint(total);
        int off = 0;
        for (auto* vn : f->adj_var_nodes) {
            if (!vn) continue;
            linpoint.segment(off, vn->dofs) = vn->mu;
            off += vn->dofs;
        }

        // update_self=true: refresh cached quadratic + reset outgoing messages
        // (matches how you use computeFactor elsewhere).
        f->computeFactor(linpoint, true);
    }
}

void FactorGraph::residualIterationVarHeap(int max_updates) {
    constexpr double eps = 1e-12;
    std::atomic<int> n_updates{0};

    int n_vars_list = 0;
    for (auto& vptr : var_nodes) if (vptr) ++n_vars_list;
    if (n_vars_list == 0) return;

    // Ensure scheduler arrays are sized.
    const int n_all = (int)var_nodes.size();
    if ((int)var_residual.size() < n_all) var_residual.resize(n_all, 0.0);
    if ((int)var_residual_a.size() < n_all) {
        const int old = (int)var_residual_a.size();
        var_residual_a.resize(n_all);
        for (int i = old; i < n_all; ++i) var_residual_a[i].store(0.0, std::memory_order_relaxed);
    }
    if ((int)var_ver.size() < n_all) {
        const int old = (int)var_ver.size();
        var_ver.resize(n_all);
        for (int i = old; i < n_all; ++i) var_ver[i].store(0u, std::memory_order_relaxed);
    }

    // Lazily construct MultiQueue scheduler.
    if (!var_mq) {
        const unsigned T = std::max(1u, std::thread::hardware_concurrency());
        const int Q = std::max(1, (int)(4u * T));
        var_mq = std::make_unique<mqfast::MultiQueueFast<HeapEntry, HeapEntryKey>>(Q, HeapEntryKey{});
        // Conservative default. You can tune this later based on profiling.
        var_mq->reserve_per_queue(256);
        var_mq_initialized = false;
    }

    if (!var_mq_initialized) {
        for (int vid = 0; vid < n_all; ++vid) {
            if (!var_nodes[vid]) continue;
            var_residual[vid] = 0.0; // scalar mirror
            var_residual_a[vid].store(0.0, std::memory_order_relaxed);
            const uint32_t v0 = var_ver[vid].fetch_add(1u, std::memory_order_relaxed) + 1u;
            var_mq->push(HeapEntry{0.0, vid, v0});
        }
        var_mq_initialized = true;
    }

    // Ensure per-variable/per-factor locks for safe parallel worker loop.
    ensure_locks(var_locks, fac_locks, locks_initialized, n_all, (int)factors.size());

    // Build a fast pointer->index map for factor locks (Factor IDs are not guaranteed contiguous).
    std::unordered_map<Factor*, int> fac_to_idx;
    fac_to_idx.reserve(factors.size() * 2 + 1);
    for (int i = 0; i < (int)factors.size(); ++i) fac_to_idx.emplace(factors[i].get(), i);

    // With MultiQueue, size/empty are not reliable; drive the loop by try_pop.
    const int cap = (max_updates < 0) ? n_vars_list : std::min(max_updates, n_vars_list);

    // Parallel worker loop: each thread continuously pops a (near-)max entry and processes it.
    // We use locks to avoid data races inside Factor::computeMessages and VariableNode::updateBelief.
    #pragma omp parallel
    {
        // Scratch to avoid repeated allocations.
        std::vector<int> fac_idx_buf;
        std::vector<int> var_idx_buf;
        fac_idx_buf.reserve(16);
        var_idx_buf.reserve(8);

        while (true) {
            if (n_updates.load(std::memory_order_relaxed) >= cap) break;

            HeapEntry e;
            if (!var_mq->try_pop(e)) break;

            const int vid = e.varID;
            VariableNode* v = (vid >= 0 && vid < n_all) ? var_nodes[vid].get() : nullptr;
            if (!v) continue;
            if (!v->active) continue;

            const double r_v   = e.r;
            const uint32_t ver_e = e.ver;

            const uint32_t cur_ver = var_ver[vid].load(std::memory_order_acquire);
            if (cur_ver != ver_e) continue;

            const double cur_r = var_residual_a[vid].load(std::memory_order_acquire);
            if (std::abs(r_v - cur_r) > eps) continue;

            // ----- process all adjacent factors -----
            for (const auto& aref : v->adj_factors) {
                Factor* f = aref.factor;
                if (!f || !f->active) continue;

                auto itf = fac_to_idx.find(f);
                if (itf == fac_to_idx.end()) continue;
                const int fid = itf->second;

                // Lock all vars in this factor (sorted) to prevent concurrent belief writes
                // while computing messages.
                var_idx_buf.clear();
                var_idx_buf.reserve(f->adj_var_nodes.size());
                for (auto* u : f->adj_var_nodes) {
                    if (!u || !u->active) continue;
                    const int uid = u->variableID;
                    if (uid < 0 || uid >= n_all) continue;
                    var_idx_buf.push_back(uid);
                }
                std::sort(var_idx_buf.begin(), var_idx_buf.end());
                var_idx_buf.erase(std::unique(var_idx_buf.begin(), var_idx_buf.end()), var_idx_buf.end());
                for (int uid : var_idx_buf) omp_set_lock(&var_locks[uid]);

                // Lock factor to protect its internal message buffers.
                omp_set_lock(&fac_locks[fid]);
                f->computeMessages(eta_damping);
                omp_unset_lock(&fac_locks[fid]);

                for (int uid : var_idx_buf) omp_unset_lock(&var_locks[uid]);

                // Update beliefs for variables touched by this factor.
                for (auto* u : f->adj_var_nodes) {
                    if (!u || !u->active) continue;
                    const int uid = u->variableID;
                    if (uid < 0 || uid >= n_all) continue;

                    // Lock variable belief updates.
                    omp_set_lock(&var_locks[uid]);

                    // Also lock all adjacent factors of this variable while updateBelief reads messages.
                    fac_idx_buf.clear();
                    for (const auto& arf2 : u->adj_factors) {
                        Factor* f2 = arf2.factor;
                        if (!f2 || !f2->active) continue;
                        auto it2 = fac_to_idx.find(f2);
                        if (it2 != fac_to_idx.end()) fac_idx_buf.push_back(it2->second);
                    }
                    std::sort(fac_idx_buf.begin(), fac_idx_buf.end());
                    fac_idx_buf.erase(std::unique(fac_idx_buf.begin(), fac_idx_buf.end()), fac_idx_buf.end());
                    for (int fidx : fac_idx_buf) omp_set_lock(&fac_locks[fidx]);

                    // Avoid heap allocations in this residual estimate hot-path.
                    // For dofs==2, use fixed-size vectors (stack) instead of VectorXd.
                    const bool is2 = (u->dofs == 2);

                    Eigen::Vector2d old_eta2;
                    Eigen::Vector2d new_eta2;
                    Eigen::VectorXd old_eta;
                    Eigen::VectorXd new_eta;

                    if (is2) {
                        old_eta2 = Eigen::Map<const Eigen::Vector2d>(u->belief.eta().data());
                        u->updateBelief();
                        new_eta2 = Eigen::Map<const Eigen::Vector2d>(u->belief.eta().data());
                    } else {
                        // Non-2D fallback: keep correctness; optimize separately if it becomes hot.
                        old_eta = u->belief.eta();
                        u->updateBelief();
                        new_eta = u->belief.eta();
                    }

                    for (auto rit = fac_idx_buf.rbegin(); rit != fac_idx_buf.rend(); ++rit) {
                        omp_unset_lock(&fac_locks[*rit]);
                    }
                    omp_unset_lock(&var_locks[uid]);

                    const double est_r_u = is2 ? (new_eta2 - old_eta2).norm()
                                             : (new_eta - old_eta).norm();
                    if (est_r_u > eps) {
                        if (atomic_max_double(var_residual_a[uid], est_r_u)) {
                            const uint32_t newv = var_ver[uid].fetch_add(1u, std::memory_order_acq_rel) + 1u;
                            var_mq->push(HeapEntry{est_r_u, uid, newv});
                        }
                    }
                }
            }

            // Reset residual for the processed variable and invalidate any stale entries.
            var_residual_a[vid].store(0.0, std::memory_order_release);
            (void)var_ver[vid].fetch_add(1u, std::memory_order_acq_rel);

            n_updates.fetch_add(1, std::memory_order_relaxed);
        }
    }

}

gbp::FactorGraph::JointInfResult gbp::FactorGraph::jointDistributionInfSparse() const {
    int total = 0;
    int max_id = -1;
    for (const auto& v : var_nodes) {
        if (!v) continue;
        total += v->dofs;
        max_id = std::max(max_id, v->variableID);
    }

    JointInfResult out;
    out.total_dim = total;
    out.eta = Eigen::VectorXd::Zero(total);
    out.var_ix.assign(max_id + 1, -1);

    std::vector<Eigen::Triplet<double>> trips;
    trips.reserve(static_cast<size_t>(total) * 10);

    int offset = 0;
    for (const auto& v : var_nodes) {
        if (!v) continue;
        const int id = v->variableID;
        const int m  = v->dofs;

        out.var_ix[id] = offset;

        out.eta.segment(offset, m) += v->prior.eta();

        const auto L = v->prior.lam();
        for (int r = 0; r < m; ++r) {
            for (int c = 0; c < m; ++c) {
                const double val = L(r, c);
                if (val != 0.0) trips.emplace_back(offset + r, offset + c, val);
            }
        }

        offset += m;
    }

    auto append_factor = [&](
        const Factor& f,
        Eigen::VectorXd& eta_out,
        std::vector<Eigen::Triplet<double>>& trips_out
    ) {
        if (!f.active) return;

        const int k = static_cast<int>(f.adj_var_nodes.size());
        if (k == 0) return;

        const auto f_eta = f.factor.eta();
        const auto f_lam = f.factor.lam();

        int factor_ix = 0;
        for (int a = 0; a < k; ++a) {
            const VariableNode* va = f.adj_var_nodes[a];
            const int ida = va->variableID;
            const int da  = va->dofs;

            const int oa = (ida >= 0 && ida < (int)out.var_ix.size()) ? out.var_ix[ida] : -1;
            if (oa < 0) throw std::runtime_error("jointDistributionInfSparse: var_ix missing ida");

            eta_out.segment(oa, da) += f_eta.segment(factor_ix, da);

            for (int r = 0; r < da; ++r) {
                for (int c = 0; c < da; ++c) {
                    const double val = f_lam(factor_ix + r, factor_ix + c);
                    if (val != 0.0) trips_out.emplace_back(oa + r, oa + c, val);
                }
            }

            int other_factor_ix = 0;
            for (int b = 0; b < k; ++b) {
                const VariableNode* vb = f.adj_var_nodes[b];
                const int idb = vb->variableID;
                const int db  = vb->dofs;

                const int ob = (idb >= 0 && idb < (int)out.var_ix.size()) ? out.var_ix[idb] : -1;
                if (ob < 0) throw std::runtime_error("jointDistributionInfSparse: var_ix missing idb");

                if (idb > ida) {
                    for (int r = 0; r < da; ++r) {
                        for (int c = 0; c < db; ++c) {
                            const double val = f_lam(factor_ix + r, other_factor_ix + c);
                            if (val != 0.0) {
                                trips_out.emplace_back(oa + r, ob + c, val);
                                trips_out.emplace_back(ob + c, oa + r, val);
                            }
                        }
                    }
                }

                other_factor_ix += db;
            }

            factor_ix += da;
        }
    };

    const int joint_threads =
        (sync_num_threads > 1 && parallelJointAssemblyEnabled() &&
         static_cast<int>(factors.size()) >= 256)
            ? sync_num_threads
            : 1;
    if (joint_threads > 1) {
        std::vector<Eigen::VectorXd> eta_locals;
        eta_locals.reserve(static_cast<size_t>(joint_threads));
        for (int t = 0; t < joint_threads; ++t) {
            eta_locals.push_back(Eigen::VectorXd::Zero(total));
        }
        std::vector<std::vector<Eigen::Triplet<double>>> trips_locals(
            static_cast<size_t>(joint_threads)
        );
        const size_t triplet_hint =
            std::max<size_t>(256, (factors.size() * 150) / static_cast<size_t>(joint_threads));
        for (auto& local_trips : trips_locals) {
            local_trips.reserve(triplet_hint);
        }

        #pragma omp parallel num_threads(joint_threads)
        {
            const int tid = omp_get_thread_num();
            Eigen::VectorXd& eta_local = eta_locals[static_cast<size_t>(tid)];
            std::vector<Eigen::Triplet<double>>& trips_local =
                trips_locals[static_cast<size_t>(tid)];
            #pragma omp for schedule(static)
            for (int fi = 0; fi < static_cast<int>(factors.size()); ++fi) {
                append_factor(*factors[static_cast<size_t>(fi)], eta_local, trips_local);
            }
        }

        for (int t = 0; t < joint_threads; ++t) {
            out.eta.noalias() += eta_locals[static_cast<size_t>(t)];
            auto& local_trips = trips_locals[static_cast<size_t>(t)];
            trips.insert(
                trips.end(),
                std::make_move_iterator(local_trips.begin()),
                std::make_move_iterator(local_trips.end())
            );
        }
    } else {
        for (const auto& fptr : factors) {
            append_factor(*fptr, out.eta, trips);
        }
    }

    out.lam.resize(total, total);
    out.lam.setFromTriplets(trips.begin(), trips.end());
    out.lam.makeCompressed();
    return out;
}




Eigen::VectorXd gbp::FactorGraph::jointMAPSparse(double diag_jitter) const {
    JointInfResult joint = jointDistributionInfSparse();
    Eigen::SparseMatrix<double> lambda = joint.lam;
    if (diag_jitter != 0.0) {
        for (int i = 0; i < lambda.rows(); ++i) {
            lambda.coeffRef(i, i) += diag_jitter;
        }
    }
    lambda.makeCompressed();

    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
    solver.compute(lambda);
    if (solver.info() != Eigen::Success) {
        throw std::runtime_error("jointMAPSparse: factorization failed");
    }

    Eigen::VectorXd mean = solver.solve(joint.eta);
    if (solver.info() != Eigen::Success) {
        throw std::runtime_error("jointMAPSparse: solve failed");
    }
    return mean;
}

} // namespace gbp