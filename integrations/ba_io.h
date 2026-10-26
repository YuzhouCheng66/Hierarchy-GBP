#pragma once

#include <Eigen/Geometry>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

// File-level adapters only. Neither this header nor the generated entry point
// is compiled into the production solver or changes its numerical policy.
namespace hgbp_io {

class SceneStream {
    std::string prefix_;
    std::ofstream binary_;
    size_t cameras_ = 0, landmarks_ = 0;
    std::vector<size_t> indices_;
    std::vector<int> outers_;
    std::vector<float> last_;
public:
    template<class Problem> void append(const Problem& problem, int outer, bool keep_previous=false) {
        const char* prefix = std::getenv("HGBP_SCENE_PREFIX");
        if (!prefix || !*prefix) return;
        if (prefix_.empty()) {
            prefix_ = prefix;
            const auto parent = std::filesystem::path(prefix_).parent_path();
            if (!parent.empty()) std::filesystem::create_directories(parent);
            if (std::filesystem::exists(prefix_+".bin") || std::filesystem::exists(prefix_+".json"))
                throw std::runtime_error("Refusing to overwrite scene capture");
            binary_.open(prefix_+".bin", std::ios::binary);
            cameras_ = problem.cameras().size();
            landmarks_ = problem.landmarks().size();
            const size_t n = std::min<size_t>(40000, landmarks_);
            for (size_t i=0; i<n; ++i) indices_.push_back(i*landmarks_/n);
        }
        if (!outers_.empty() && outer <= outers_.back()) return;
        if (!keep_previous || last_.empty()) {
            last_.clear();
            last_.reserve(3*(indices_.size()+cameras_));
            for (const auto index : indices_) {
                const auto& p = problem.landmarks()[index].p_w;
                for (int k=0; k<3; ++k) last_.push_back(static_cast<float>(p[k]));
            }
            for (const auto& camera : problem.cameras()) {
                const auto p = camera.T_c_w.inverse().translation().eval();
                for (int k=0; k<3; ++k) last_.push_back(static_cast<float>(p[k]));
            }
        }
        for (const auto value : last_) if (!std::isfinite(value)) throw std::runtime_error("Nonfinite scene capture");
        binary_.write(reinterpret_cast<const char*>(last_.data()), sizeof(float)*last_.size());
        if (!binary_) throw std::runtime_error("Scene capture write failed");
        outers_.push_back(outer);
    }
    void finish() {
        if (prefix_.empty()) return;
        binary_.close();
        std::ofstream meta(prefix_+".json");
        meta << "{\"landmarks\":" << landmarks_ << ",\"points\":" << indices_.size()
             << ",\"cameras\":" << cameras_ << ",\"frames\":[";
        for (size_t i=0;i<outers_.size();++i) meta << (i?",":"") << outers_[i];
        meta << "],\"sampling\":\"uniform landmark index; all camera centers; no error filtering\"}\n";
        if (!meta) throw std::runtime_error("Scene metadata write failed");
        prefix_.clear();
    }
};

inline SceneStream scene;
template<class Problem> void snapshot(const Problem& problem, int outer, bool rejected=false) {
    scene.append(problem, outer, rejected);
}

template<class Problem> void writeBAL(const Problem& problem, const std::string& filename) {
    if (std::filesystem::exists(filename)) throw std::runtime_error("Refusing to overwrite output BAL");
    std::ofstream out(filename);
    if (!out) throw std::runtime_error("Cannot open output BAL");
    out << std::setprecision(17);
    size_t observations=0;
    for (const auto& landmark : problem.landmarks()) observations += landmark.obs.size();
    out << problem.cameras().size() << ' ' << problem.landmarks().size() << ' ' << observations << '\n';
    for (size_t i=0;i<problem.landmarks().size();++i)
        for (const auto& [camera, observation] : problem.landmarks()[i].obs)
            out << camera << ' ' << i << ' ' << observation.pos.x() << ' ' << -observation.pos.y() << '\n';
    Eigen::Matrix3d flip = Eigen::Matrix3d::Identity();
    flip(1,1)=-1; flip(2,2)=-1;
    for (const auto& camera : problem.cameras()) {
        const Eigen::AngleAxisd rotation(flip*camera.T_c_w.so3().matrix());
        const Eigen::Vector3d angle_axis = rotation.angle()*rotation.axis();
        const Eigen::Vector3d translation = flip*camera.T_c_w.translation();
        const auto intrinsics = camera.intrinsics.getParam();
        for (int k=0;k<3;++k) out << angle_axis[k] << '\n';
        for (int k=0;k<3;++k) out << translation[k] << '\n';
        for (int k=0;k<3;++k) out << intrinsics[k] << '\n';
    }
    for (const auto& landmark : problem.landmarks())
        for (int k=0;k<3;++k) out << landmark.p_w[k] << '\n';
    if (!out) throw std::runtime_error("Output BAL write failed");
}

template<class Problem> void finish(const Problem& problem) {
    if (const char* path=std::getenv("HGBP_MODEL_OUTPUT")) if (*path) writeBAL(problem, path);
    scene.finish();
}
} // namespace hgbp_io
