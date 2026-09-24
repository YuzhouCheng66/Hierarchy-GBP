#pragma once

using BAWeightedAdjacency = std::vector<std::vector<std::pair<int, double>>>;

BAWeightedAdjacency baWeightedAdjacency(const FastHGBPSystem& sys) {
    BAWeightedAdjacency neighbors(sys.free_cameras);
    for (const auto& edge : sys.ws.edges) {
        const double weight = edge.aij.norm() + edge.aji.norm();
        if (!std::isfinite(weight)) throw std::runtime_error("Nonfinite BA aggregation weight");
        if (weight > 0.0) {
            neighbors[edge.i].emplace_back(edge.j, weight);
            neighbors[edge.j].emplace_back(edge.i, weight);
        }
    }
    return neighbors;
}

int baDisconnectedGroups(const FastHGBPSystem& sys, const std::vector<std::vector<int>>& groups) {
    const auto neighbors = baWeightedAdjacency(sys);
    std::vector<int> gid(sys.free_cameras, -1);
    for (int g = 0; g < static_cast<int>(groups.size()); ++g) {
        for (int i : groups[g]) {
            if (i < 0 || i >= sys.free_cameras || gid[i] >= 0) throw std::runtime_error("Invalid BA aggregate partition");
            gid[i] = g;
        }
    }
    if (std::find(gid.begin(), gid.end(), -1) != gid.end()) throw std::runtime_error("Incomplete BA aggregate partition");
    std::vector<unsigned char> visited(sys.free_cameras, 0);
    std::vector<int> queue;
    int disconnected = 0;
    for (int g = 0; g < static_cast<int>(groups.size()); ++g) {
        if (groups[g].empty()) throw std::runtime_error("Empty BA aggregate");
        queue.clear();
        queue.push_back(groups[g][0]);
        visited[queue[0]] = 1;
        for (size_t head = 0; head < queue.size(); ++head) {
            for (const auto& [j, weight] : neighbors[queue[head]]) {
                (void)weight;
                if (gid[j] == g && !visited[j]) {
                    visited[j] = 1;
                    queue.push_back(j);
                }
            }
        }
        if (queue.size() != groups[g].size()) ++disconnected;
    }
    return disconnected;
}

std::vector<std::vector<int>> buildConnectedBAGroups(const FastHGBPSystem& sys, int target) {
    target = std::max(2, target);
    const int n = sys.free_cameras;
    const auto neighbors = baWeightedAdjacency(sys);
    std::vector<int> order(n), gid(n, -1);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int i, int j) {
        return neighbors[i].size() != neighbors[j].size() ?
            neighbors[i].size() > neighbors[j].size() : i < j;
    });
    std::vector<std::vector<int>> groups;
    std::vector<double> best(n, -1.0);
    for (int seed : order) {
        if (gid[seed] >= 0) continue;
        const int g = static_cast<int>(groups.size());
        groups.emplace_back();
        std::fill(best.begin(), best.end(), -1.0);
        // Negated node id resolves equal weights deterministically.
        std::priority_queue<std::pair<double, int>> frontier;
        frontier.emplace(std::numeric_limits<double>::infinity(), -seed);
        while (!frontier.empty() && static_cast<int>(groups[g].size()) < target) {
            const auto [weight, negated] = frontier.top();
            frontier.pop();
            const int i = -negated;
            if (gid[i] >= 0 || weight < best[i]) continue;
            gid[i] = g;
            groups[g].push_back(i);
            for (const auto& [j, edge_weight] : neighbors[i]) {
                if (gid[j] < 0 && edge_weight > best[j]) {
                    best[j] = edge_weight;
                    frontier.emplace(edge_weight, -j);
                }
            }
        }
    }
    // Absorb small fringe aggregates only across an actual graph edge.
    // The common 2*target capacity prevents a fringe pass forming a giant group.
    std::vector<double> score(groups.size());
    for (int g = 0; g < static_cast<int>(groups.size()); ++g) {
        if (groups[g].empty() || 2 * static_cast<int>(groups[g].size()) >= target) continue;
        std::fill(score.begin(), score.end(), 0.0);
        for (int i : groups[g]) {
            for (const auto& [j, weight] : neighbors[i]) {
                const int h = gid[j];
                if (h != g && groups[h].size() + groups[g].size() <= static_cast<size_t>(2 * target)) score[h] += weight;
            }
        }
        int chosen = -1;
        for (int h = 0; h < static_cast<int>(groups.size()); ++h) {
            if (score[h] > 0.0 && (chosen < 0 || score[h] > score[chosen])) chosen = h;
        }
        if (chosen >= 0) {
            for (int i : groups[g]) {
                gid[i] = chosen;
                groups[chosen].push_back(i);
            }
            groups[g].clear();
        }
    }
    std::vector<std::vector<int>> result;
    for (auto& group : groups) {
        if (!group.empty()) {
            std::sort(group.begin(), group.end());
            result.push_back(std::move(group));
        }
    }
    if (baDisconnectedGroups(sys, result) != 0) throw std::runtime_error("Connected BA aggregation invariant failed");
    return result;
}
