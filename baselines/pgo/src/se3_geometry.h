#pragma once

#include <array>
#include <cmath>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace reviewer_pcg {

struct SE3Traits {
    static constexpr int kDimension = 6;
    using Vector = Eigen::Matrix<double, 6, 1>;
    using Matrix = Eigen::Matrix<double, 6, 6>;
    using JacobianPair = Eigen::Matrix<double, 6, 12>;

    struct Pose {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        Eigen::Vector3d translation = Eigen::Vector3d::Zero();
        Eigen::Quaterniond rotation = Eigen::Quaterniond::Identity();
    };

    using PoseVector = std::vector<Pose, Eigen::aligned_allocator<Pose>>;

    struct Edge {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        int i = -1;
        int j = -1;
        Pose measurement;
        Matrix information = Matrix::Zero();
    };

    struct Problem {
        PoseVector poses;
        std::vector<Edge, Eigen::aligned_allocator<Edge>> edges;
        std::vector<int> original_ids;
        int fixed_index = 0;
    };

    static const char* geometryName() {
        return "se3";
    }

    static Pose normalized(Pose pose) {
        pose.rotation.normalize();
        if (pose.rotation.w() < 0.0) {
            pose.rotation.coeffs() *= -1.0;
        }
        return pose;
    }

    static Eigen::Matrix3d skew(const Eigen::Vector3d& vector) {
        Eigen::Matrix3d result;
        result << 0.0, -vector.z(), vector.y(),
                  vector.z(), 0.0, -vector.x(),
                  -vector.y(), vector.x(), 0.0;
        return result;
    }

    static Eigen::Matrix3d so3Exp(const Eigen::Vector3d& omega) {
        const double theta = omega.norm();
        const Eigen::Matrix3d W = skew(omega);
        const Eigen::Matrix3d W2 = W * W;
        double a = 1.0;
        double b = 0.5;
        if (theta > 1e-12) {
            a = std::sin(theta) / theta;
            b = (1.0 - std::cos(theta)) / (theta * theta);
        } else {
            const double theta2 = theta * theta;
            a = 1.0 - theta2 / 6.0;
            b = 0.5 - theta2 / 24.0;
        }
        return Eigen::Matrix3d::Identity() + a * W + b * W2;
    }

    static Eigen::Vector3d so3Log(const Eigen::Matrix3d& rotation) {
        constexpr double pi = 3.141592653589793238462643383279502884;
        const double cosine =
            std::clamp(0.5 * (rotation.trace() - 1.0), -1.0, 1.0);
        const double theta = std::acos(cosine);
        const Eigen::Vector3d vee(
            rotation(2, 1) - rotation(1, 2),
            rotation(0, 2) - rotation(2, 0),
            rotation(1, 0) - rotation(0, 1)
        );
        if (theta < 1e-8) {
            return 0.5 * vee;
        }
        if (pi - theta < 1e-6) {
            const Eigen::AngleAxisd angle_axis(rotation);
            return angle_axis.axis() * angle_axis.angle();
        }
        return (0.5 * theta / std::sin(theta)) * vee;
    }

    static Eigen::Matrix3d leftJacobianSO3(const Eigen::Vector3d& omega) {
        const double theta = omega.norm();
        const Eigen::Matrix3d W = skew(omega);
        const Eigen::Matrix3d W2 = W * W;
        double a = 0.5;
        double b = 1.0 / 6.0;
        if (theta > 1e-12) {
            a = (1.0 - std::cos(theta)) / (theta * theta);
            b = (theta - std::sin(theta)) / (theta * theta * theta);
        } else {
            const double theta2 = theta * theta;
            a = 0.5 - theta2 / 24.0;
            b = 1.0 / 6.0 - theta2 / 120.0;
        }
        return Eigen::Matrix3d::Identity() + a * W + b * W2;
    }

    static Eigen::Matrix3d leftJacobianInverseSO3(const Eigen::Vector3d& omega) {
        const double theta = omega.norm();
        const Eigen::Matrix3d W = skew(omega);
        const Eigen::Matrix3d W2 = W * W;
        if (theta < 1e-8) {
            return Eigen::Matrix3d::Identity() - 0.5 * W + (1.0 / 12.0) * W2;
        }
        const double coefficient =
            1.0 / (theta * theta) -
            (1.0 + std::cos(theta)) / (2.0 * theta * std::sin(theta));
        return Eigen::Matrix3d::Identity() - 0.5 * W + coefficient * W2;
    }

    static Eigen::Matrix3d rightJacobianInverseSO3(const Eigen::Vector3d& omega) {
        const double theta2 = omega.squaredNorm();
        if (theta2 <= std::numeric_limits<double>::epsilon()) {
            return Eigen::Matrix3d::Identity();
        }
        const double theta = std::sqrt(theta2);
        const Eigen::Matrix3d W = skew(omega);
        return Eigen::Matrix3d::Identity() + 0.5 * W +
            (1.0 / theta2 -
             (1.0 + std::cos(theta)) / (2.0 * theta * std::sin(theta))) *
                (W * W);
    }

    static Pose compose(const Pose& a, const Pose& b) {
        Pose result;
        result.rotation = a.rotation * b.rotation;
        result.translation =
            a.translation + a.rotation.toRotationMatrix() * b.translation;
        return normalized(result);
    }

    static Pose inverse(const Pose& pose) {
        Pose result;
        result.rotation = pose.rotation.conjugate();
        result.translation =
            -result.rotation.toRotationMatrix() * pose.translation;
        return normalized(result);
    }

    static Pose between(const Pose& a, const Pose& b) {
        return compose(inverse(a), b);
    }

    static Pose expmap(const Vector& xi) {
        Pose result;
        result.translation = leftJacobianSO3(xi.tail<3>()) * xi.head<3>();
        result.rotation = Eigen::Quaterniond(so3Exp(xi.tail<3>()));
        return normalized(result);
    }

    static Vector logmap(const Pose& pose) {
        Vector result;
        result.tail<3>() = so3Log(pose.rotation.normalized().toRotationMatrix());
        result.head<3>() =
            leftJacobianInverseSO3(result.tail<3>()) * pose.translation;
        return result;
    }

    static Pose plus(const Pose& pose, const Vector& delta) {
        return compose(pose, expmap(delta));
    }

    static Vector residual(const Pose& xi, const Pose& xj, const Pose& measurement) {
        return logmap(compose(inverse(measurement), between(xi, xj)));
    }

    static Matrix adjoint(const Pose& pose) {
        const Eigen::Matrix3d rotation = pose.rotation.toRotationMatrix();
        Matrix result = Matrix::Zero();
        result.topLeftCorner<3, 3>() = rotation;
        result.topRightCorner<3, 3>() = skew(pose.translation) * rotation;
        result.bottomRightCorner<3, 3>() = rotation;
        return result;
    }

    static Eigen::Matrix3d expmapDerivativeQ(const Vector& xi) {
        const Eigen::Matrix3d V = skew(xi.head<3>());
        const Eigen::Matrix3d W = skew(xi.tail<3>());
        const Eigen::Matrix3d WVW = W * V * W;
        const double angle = xi.tail<3>().norm();
        if (std::abs(angle) > 1e-5) {
            const double sine = std::sin(angle);
            const double cosine = std::cos(angle);
            const double a2 = angle * angle;
            const double a3 = a2 * angle;
            const double a4 = a3 * angle;
            const double a5 = a4 * angle;
            return -0.5 * V
                + (angle - sine) / a3 * (W * V + V * W - WVW)
                + (1.0 - a2 / 2.0 - cosine) / a4 *
                    (W * W * V + V * W * W - 3.0 * WVW)
                - 0.5 *
                    ((1.0 - a2 / 2.0 - cosine) / a4 -
                     3.0 * (angle - sine - a3 / 6.0) / a5) *
                    (WVW * W + W * WVW);
        }
        return -0.5 * V
            + (1.0 / 6.0) * (W * V + V * W - WVW)
            - (1.0 / 24.0) * (W * W * V + V * W * W - 3.0 * WVW)
            + (1.0 / 120.0) * (WVW * W + W * WVW);
    }

    static Matrix logmapDerivative(const Pose& pose) {
        const Vector xi = logmap(pose);
        const Eigen::Matrix3d jr_inverse =
            rightJacobianInverseSO3(xi.tail<3>());
        const Eigen::Matrix3d q = expmapDerivativeQ(xi);
        Matrix result = Matrix::Zero();
        result.topLeftCorner<3, 3>() = jr_inverse;
        result.topRightCorner<3, 3>() = -jr_inverse * q * jr_inverse;
        result.bottomRightCorner<3, 3>() = jr_inverse;
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
        const Matrix derivative = logmapDerivative(error_pose);
        jacobian.leftCols<6>() = derivative * -adjoint(inverse(prediction));
        jacobian.rightCols<6>() = derivative;
    }

    static Matrix informationFromUpper(const std::array<double, 21>& values) {
        Matrix result = Matrix::Zero();
        int index = 0;
        for (int row = 0; row < 6; ++row) {
            for (int column = row; column < 6; ++column) {
                result(row, column) = values[static_cast<size_t>(index)];
                result(column, row) = values[static_cast<size_t>(index)];
                ++index;
            }
        }
        return result;
    }

    static Problem load(const std::string& path) {
        std::ifstream input(path);
        if (!input) {
            throw std::runtime_error("failed to open SE3 g2o file: " + path);
        }
        std::unordered_map<int, Pose> raw_vertices;
        struct RawEdge {
            EIGEN_MAKE_ALIGNED_OPERATOR_NEW
            int i = -1;
            int j = -1;
            Pose measurement;
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
            if (tag == "VERTEX_SE3:QUAT") {
                int id = -1;
                Pose pose;
                if (!(stream >> id
                    >> pose.translation.x() >> pose.translation.y() >> pose.translation.z()
                    >> pose.rotation.x() >> pose.rotation.y() >> pose.rotation.z()
                    >> pose.rotation.w())) {
                    throw std::runtime_error(
                        "malformed VERTEX_SE3:QUAT at line " +
                        std::to_string(line_number)
                    );
                }
                raw_vertices[id] = normalized(pose);
            } else if (tag == "EDGE_SE3:QUAT") {
                RawEdge edge;
                std::array<double, 21> values{};
                if (!(stream >> edge.i >> edge.j
                    >> edge.measurement.translation.x()
                    >> edge.measurement.translation.y()
                    >> edge.measurement.translation.z()
                    >> edge.measurement.rotation.x()
                    >> edge.measurement.rotation.y()
                    >> edge.measurement.rotation.z()
                    >> edge.measurement.rotation.w())) {
                    throw std::runtime_error(
                        "malformed EDGE_SE3:QUAT pose at line " +
                        std::to_string(line_number)
                    );
                }
                for (double& value : values) {
                    if (!(stream >> value)) {
                        throw std::runtime_error(
                            "malformed EDGE_SE3:QUAT information at line " +
                            std::to_string(line_number)
                        );
                    }
                }
                edge.measurement = normalized(edge.measurement);
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
                    "unsupported SE3 g2o tag '" + tag + "' at line " +
                    std::to_string(line_number)
                );
            }
        }
        if (raw_vertices.empty()) {
            throw std::runtime_error("SE3 g2o file has no vertices: " + path);
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
            throw std::runtime_error("FIX references an unknown SE3 vertex");
        }
        problem.fixed_index = fixed->second;
        problem.edges.reserve(raw_edges.size());
        for (const RawEdge& raw : raw_edges) {
            const auto i = index_by_id.find(raw.i);
            const auto j = index_by_id.find(raw.j);
            if (i == index_by_id.end() || j == index_by_id.end()) {
                throw std::runtime_error("SE3 edge references an unknown vertex");
            }
            problem.edges.push_back(
                Edge{i->second, j->second, raw.measurement, raw.information}
            );
        }
        return problem;
    }

    static void writePoseJson(std::ostream& out, int id, const Pose& pose) {
        const Eigen::Quaterniond quaternion = pose.rotation.normalized();
        out << "[" << id << ", "
            << pose.translation.x() << ", " << pose.translation.y() << ", "
            << pose.translation.z() << ", "
            << quaternion.x() << ", " << quaternion.y() << ", "
            << quaternion.z() << ", " << quaternion.w() << "]";
    }

    static void writeG2O(
        const Problem& problem,
        const PoseVector& poses,
        const std::string& path
    ) {
        std::ofstream out(path);
        if (!out) {
            throw std::runtime_error("failed to open SE3 pose output: " + path);
        }
        out << std::setprecision(17);
        for (size_t i = 0; i < poses.size(); ++i) {
            const Pose pose = normalized(poses[i]);
            out << "VERTEX_SE3:QUAT " << problem.original_ids[i] << " "
                << pose.translation.x() << " " << pose.translation.y() << " "
                << pose.translation.z() << " "
                << pose.rotation.x() << " " << pose.rotation.y() << " "
                << pose.rotation.z() << " " << pose.rotation.w() << "\n";
        }
        out << "FIX " << problem.original_ids[static_cast<size_t>(problem.fixed_index)]
            << "\n";
        for (const Edge& edge : problem.edges) {
            const Pose measurement = normalized(edge.measurement);
            out << "EDGE_SE3:QUAT "
                << problem.original_ids[static_cast<size_t>(edge.i)] << " "
                << problem.original_ids[static_cast<size_t>(edge.j)] << " "
                << measurement.translation.x() << " "
                << measurement.translation.y() << " "
                << measurement.translation.z() << " "
                << measurement.rotation.x() << " "
                << measurement.rotation.y() << " "
                << measurement.rotation.z() << " "
                << measurement.rotation.w();
            for (int row = 0; row < 6; ++row) {
                for (int column = row; column < 6; ++column) {
                    out << " " << edge.information(row, column);
                }
            }
            out << "\n";
        }
    }
};

}  // namespace reviewer_pcg
