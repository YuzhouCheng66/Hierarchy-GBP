#pragma once

#include <array>
#include <cmath>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include <Eigen/Core>

namespace reviewer_pcg {

struct SE2Traits {
    static constexpr int kDimension = 3;
    using Pose = Eigen::Vector3d;
    using PoseVector = std::vector<Pose, Eigen::aligned_allocator<Pose>>;
    using Vector = Eigen::Vector3d;
    using Matrix = Eigen::Matrix3d;
    using JacobianPair = Eigen::Matrix<double, 3, 6>;

    struct Edge {
        int i = -1;
        int j = -1;
        Pose measurement = Pose::Zero();
        Matrix information = Matrix::Zero();
    };

    struct Problem {
        PoseVector poses;
        std::vector<Edge, Eigen::aligned_allocator<Edge>> edges;
        std::vector<int> original_ids;
        int fixed_index = 0;
    };

    static const char* geometryName() {
        return "se2";
    }

    static double wrapAngle(double angle) {
        return std::atan2(std::sin(angle), std::cos(angle));
    }

    static Eigen::Matrix2d rotation(double theta) {
        const double c = std::cos(theta);
        const double s = std::sin(theta);
        Eigen::Matrix2d result;
        result << c, -s, s, c;
        return result;
    }

    static Pose compose(const Pose& a, const Pose& b) {
        Pose result;
        result.head<2>() = a.head<2>() + rotation(a(2)) * b.head<2>();
        result(2) = wrapAngle(a(2) + b(2));
        return result;
    }

    static Pose inverse(const Pose& pose) {
        Pose result;
        result.head<2>() = -rotation(pose(2)).transpose() * pose.head<2>();
        result(2) = wrapAngle(-pose(2));
        return result;
    }

    static Pose between(const Pose& a, const Pose& b) {
        return compose(inverse(a), b);
    }

    static Pose expmap(const Vector& xi) {
        const double w = xi(2);
        if (std::abs(w) < 1e-12) {
            return Pose(xi(0), xi(1), 0.0);
        }
        const double a = std::sin(w) / w;
        const double b = (1.0 - std::cos(w)) / w;
        Eigen::Matrix2d V;
        V << a, -b, b, a;
        const Eigen::Vector2d translation = V * xi.head<2>();
        return Pose(translation(0), translation(1), wrapAngle(w));
    }

    static Vector logmap(const Pose& pose) {
        const double w = pose(2);
        if (std::abs(w) < 1e-12) {
            return Vector(pose(0), pose(1), 0.0);
        }
        const double a = std::sin(w) / w;
        const double b = (1.0 - std::cos(w)) / w;
        Eigen::Matrix2d inverse_v;
        inverse_v << a, b, -b, a;
        inverse_v /= a * a + b * b;
        const Eigen::Vector2d translation = inverse_v * pose.head<2>();
        return Vector(translation(0), translation(1), wrapAngle(w));
    }

    static Pose plus(const Pose& pose, const Vector& delta) {
        return compose(pose, expmap(delta));
    }

    static Vector residual(const Pose& xi, const Pose& xj, const Pose& measurement) {
        return logmap(compose(inverse(measurement), between(xi, xj)));
    }

    static Matrix jacobianPlus(const Pose& base_pose) {
        Matrix result = Matrix::Zero();
        result.topLeftCorner<2, 2>() = rotation(base_pose(2));
        result(2, 2) = 1.0;
        return result;
    }

    static JacobianPair jacobianBetweenAbsolute(const Pose& xi, const Pose& xj) {
        const Eigen::Matrix2d rt = rotation(xi(2)).transpose();
        const Eigen::Vector2d relative = rt * (xj.head<2>() - xi.head<2>());
        const Eigen::Vector2d rotation_derivative(relative(1), -relative(0));
        JacobianPair result = JacobianPair::Zero();
        result.block<2, 2>(0, 0) = -rt;
        result.block<2, 1>(0, 2) = rotation_derivative;
        result.block<2, 2>(0, 3) = rt;
        result(2, 2) = -1.0;
        result(2, 5) = 1.0;
        return result;
    }

    static Matrix jacobianComposeInverseConstant(const Pose& measurement) {
        Matrix result = Matrix::Zero();
        result.topLeftCorner<2, 2>() = rotation(measurement(2)).transpose();
        result(2, 2) = 1.0;
        return result;
    }

    static Matrix jacobianLog(const Pose& pose) {
        const double tx = pose(0);
        const double ty = pose(1);
        const double w = pose(2);
        Matrix result = Matrix::Zero();
        if (std::abs(w) < 1e-8) {
            result.setIdentity();
            result(0, 2) = 0.5 * ty;
            result(1, 2) = -0.5 * tx;
            return result;
        }

        const double a = std::sin(w) / w;
        const double b = (1.0 - std::cos(w)) / w;
        const double denominator = a * a + b * b;
        const double da = (w * std::cos(w) - std::sin(w)) / (w * w);
        const double db =
            (w * std::sin(w) - (1.0 - std::cos(w))) / (w * w);
        const double ddenominator = 2.0 * (a * da + b * db);
        const double c = a / denominator;
        const double d = b / denominator;
        const double dc =
            (da * denominator - a * ddenominator) / (denominator * denominator);
        const double dd =
            (db * denominator - b * ddenominator) / (denominator * denominator);

        result(0, 0) = c;
        result(0, 1) = d;
        result(1, 0) = -d;
        result(1, 1) = c;
        result(0, 2) = dc * tx + dd * ty;
        result(1, 2) = -dd * tx + dc * ty;
        result(2, 2) = 1.0;
        return result;
    }

    static void linearize(
        const Pose& xi,
        const Pose& xj,
        const Pose& measurement,
        Vector& error,
        JacobianPair& jacobian
    ) {
        const Pose prediction = between(xi, xj);
        const Pose error_pose = compose(inverse(measurement), prediction);
        error = logmap(error_pose);
        Eigen::Matrix<double, 6, 6> plus_jacobian =
            Eigen::Matrix<double, 6, 6>::Zero();
        plus_jacobian.block<3, 3>(0, 0) = jacobianPlus(xi);
        plus_jacobian.block<3, 3>(3, 3) = jacobianPlus(xj);
        jacobian = jacobianLog(error_pose) *
            jacobianComposeInverseConstant(measurement) *
            jacobianBetweenAbsolute(xi, xj) *
            plus_jacobian;
    }

    static Matrix informationFromUpper(const std::array<double, 6>& values) {
        Matrix result;
        result << values[0], values[1], values[2],
                  values[1], values[3], values[4],
                  values[2], values[4], values[5];
        return result;
    }

    static Problem load(const std::string& path) {
        std::ifstream input(path);
        if (!input) {
            throw std::runtime_error("failed to open SE2 g2o file: " + path);
        }

        std::unordered_map<int, Pose> raw_vertices;
        struct RawEdge {
            int i = -1;
            int j = -1;
            Pose measurement = Pose::Zero();
            Matrix information = Matrix::Zero();
        };
        std::vector<RawEdge, Eigen::aligned_allocator<RawEdge>> raw_edges;
        std::vector<int> fixed_ids;
        std::string line;
        int line_number = 0;
        while (std::getline(input, line)) {
            ++line_number;
            std::istringstream stream(line);
            std::string tag;
            if (!(stream >> tag) || tag[0] == '#') {
                continue;
            }
            if (tag == "VERTEX_SE2") {
                int id = -1;
                Pose pose;
                if (!(stream >> id >> pose(0) >> pose(1) >> pose(2))) {
                    throw std::runtime_error(
                        "malformed VERTEX_SE2 at line " + std::to_string(line_number)
                    );
                }
                raw_vertices[id] = pose;
            } else if (tag == "EDGE_SE2") {
                RawEdge edge;
                std::array<double, 6> values{};
                if (!(stream >> edge.i >> edge.j
                    >> edge.measurement(0) >> edge.measurement(1) >> edge.measurement(2)
                    >> values[0] >> values[1] >> values[2]
                    >> values[3] >> values[4] >> values[5])) {
                    throw std::runtime_error(
                        "malformed EDGE_SE2 at line " + std::to_string(line_number)
                    );
                }
                edge.information = informationFromUpper(values);
                raw_edges.push_back(edge);
            } else if (tag == "FIX") {
                int id = -1;
                if (!(stream >> id)) {
                    throw std::runtime_error(
                        "malformed FIX at line " + std::to_string(line_number)
                    );
                }
                fixed_ids.push_back(id);
            } else {
                throw std::runtime_error(
                    "unsupported SE2 g2o tag '" + tag + "' at line " +
                    std::to_string(line_number)
                );
            }
        }
        if (raw_vertices.empty()) {
            throw std::runtime_error("SE2 g2o file has no vertices: " + path);
        }

        Problem problem;
        problem.original_ids.reserve(raw_vertices.size());
        for (const auto& [id, pose] : raw_vertices) {
            static_cast<void>(pose);
            problem.original_ids.push_back(id);
        }
        std::sort(problem.original_ids.begin(), problem.original_ids.end());
        std::unordered_map<int, int> index_by_id;
        index_by_id.reserve(problem.original_ids.size());
        problem.poses.resize(problem.original_ids.size());
        for (size_t index = 0; index < problem.original_ids.size(); ++index) {
            const int id = problem.original_ids[index];
            index_by_id[id] = static_cast<int>(index);
            problem.poses[index] = raw_vertices.at(id);
        }

        const int fixed_id =
            fixed_ids.empty() ? problem.original_ids.front() : fixed_ids.front();
        const auto fixed = index_by_id.find(fixed_id);
        if (fixed == index_by_id.end()) {
            throw std::runtime_error("FIX references an unknown SE2 vertex");
        }
        problem.fixed_index = fixed->second;
        problem.edges.reserve(raw_edges.size());
        for (const RawEdge& raw : raw_edges) {
            const auto i = index_by_id.find(raw.i);
            const auto j = index_by_id.find(raw.j);
            if (i == index_by_id.end() || j == index_by_id.end()) {
                throw std::runtime_error("SE2 edge references an unknown vertex");
            }
            problem.edges.push_back(
                Edge{i->second, j->second, raw.measurement, raw.information}
            );
        }
        return problem;
    }

    static void writePoseJson(std::ostream& out, int id, const Pose& pose) {
        out << "[" << id << ", " << pose(0) << ", " << pose(1) << ", " << pose(2)
            << "]";
    }

    static void writeG2O(
        const Problem& problem,
        const PoseVector& poses,
        const std::string& path
    ) {
        std::ofstream out(path);
        if (!out) {
            throw std::runtime_error("failed to open SE2 pose output: " + path);
        }
        out << std::setprecision(17);
        for (size_t i = 0; i < poses.size(); ++i) {
            out << "VERTEX_SE2 " << problem.original_ids[i] << " "
                << poses[i](0) << " " << poses[i](1) << " " << poses[i](2) << "\n";
        }
        out << "FIX " << problem.original_ids[static_cast<size_t>(problem.fixed_index)]
            << "\n";
        for (const Edge& edge : problem.edges) {
            out << "EDGE_SE2 " << problem.original_ids[static_cast<size_t>(edge.i)] << " "
                << problem.original_ids[static_cast<size_t>(edge.j)] << " "
                << edge.measurement(0) << " " << edge.measurement(1) << " "
                << edge.measurement(2) << " "
                << edge.information(0, 0) << " " << edge.information(0, 1) << " "
                << edge.information(0, 2) << " " << edge.information(1, 1) << " "
                << edge.information(1, 2) << " " << edge.information(2, 2) << "\n";
        }
    }
};

}  // namespace reviewer_pcg
