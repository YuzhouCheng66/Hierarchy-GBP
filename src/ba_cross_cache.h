#pragma once

// Storage liveness only: preserve the existing pair sampler, weights and topology.
struct RootCrossCachePlan {
    std::vector<int> edge_slot;
    std::vector<int> retained_slot;
    int slot_count = 0;
    int sample_cap = 0;
};

RootCrossCachePlan makeRootCrossCachePlan(
    const PackedBlockSchurPattern& pattern,
    const RootHGBPBuildPlan& build_plan,
    int sample_cap
) {
    RootCrossCachePlan cache;
    cache.sample_cap = sample_cap;
    cache.retained_slot.assign(static_cast<size_t>(build_plan.retained_edge_count), -1);
    for (size_t slot = 0; slot + 1 < pattern.slot_pair_row_ptr.size(); ++slot) {
        const int begin = pattern.slot_pair_row_ptr[slot];
        const int ref_count = pattern.slot_pair_row_ptr[slot + 1] - begin;
        const int count = sample_cap > 0 ? std::min(ref_count, sample_cap) : ref_count;
        for (int sample = 0; sample < count; ++sample) {
            const int ref = count < ref_count
                ? begin + static_cast<int>((static_cast<long long>(2 * sample + 1) * ref_count) / (2LL * count))
                : begin + sample;
            cache.retained_slot[pattern.slot_pair_edge_a_global[ref]] = 0;
            cache.retained_slot[pattern.slot_pair_edge_b_global[ref]] = 0;
        }
    }
    for (int& slot : cache.retained_slot) {
        if (slot == 0) slot = cache.slot_count++;
    }
    cache.edge_slot.resize(build_plan.retained_edge_slot.size());
    for (size_t e = 0; e < cache.edge_slot.size(); ++e) {
        const int retained = build_plan.retained_edge_slot[e];
        cache.edge_slot[e] = retained < 0 ? -1 : cache.retained_slot[retained];
    }
    return cache;
}
