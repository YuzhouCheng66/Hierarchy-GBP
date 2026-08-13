#pragma once
#include <Eigen/Dense>
#include <Eigen/Cholesky>
#include <vector>
#include <cassert>
#include <functional>
#include <limits>
#include <stdexcept>
#include "NdimGaussian.h"


namespace gbp {

class VariableNode;

struct FixedLamEta6HotEntry {
    const double* factor_eta = nullptr;
    const double* eta_map0 = nullptr;
    const double* eta_map1 = nullptr;
    const double* belief0_eta = nullptr;
    const double* belief1_eta = nullptr;
    double* msg0_eta = nullptr;
    double* msg1_eta = nullptr;
};

class Factor {
public:
    int factorID = -1;
    bool active = true;

    std::vector<VariableNode*> adj_var_nodes;
    std::vector<int> adj_vIDs;

    // Factor-to-variable messages (ping-pong buffers for C optimization)
    std::vector<utils::NdimGaussian, Eigen::aligned_allocator<utils::NdimGaussian>> messages;
    std::vector<utils::NdimGaussian, Eigen::aligned_allocator<utils::NdimGaussian>> messages_next;

    std::vector<Eigen::VectorXd> measurement;         // z_i
    std::vector<Eigen::MatrixXd> measurement_lambda;  // Λ_i

    std::function<std::vector<Eigen::VectorXd>(const Eigen::VectorXd&)> meas_fn;
    std::function<std::vector<Eigen::MatrixXd>(const Eigen::VectorXd&)>  jac_fn;

    utils::NdimGaussian factor;
    Eigen::VectorXd linpoint;

    double eta_damping_local = 0.0;
    bool fixed_lam_valid_ = false;

    Factor(
        int id,
        const std::vector<VariableNode*>& vars,
        const std::vector<Eigen::VectorXd>& z,
        const std::vector<Eigen::MatrixXd>& lambda,
        std::function<std::vector<Eigen::VectorXd>(const Eigen::VectorXd&)> meas,
        std::function<std::vector<Eigen::MatrixXd>(const Eigen::VectorXd&)>  jac
    );

    void computeFactor(const Eigen::VectorXd& linpoint, bool update_self = true);
    void setLinearFactorInfo(const Eigen::VectorXd& eta, const Eigen::MatrixXd& lam);
    void clearMessages();
    // Jacobian/Lambda cache control (use when structure changes)
    void invalidateJacobianCache();
    void computeMessages(double eta_damping);
    void computeMessagesFixedLam(double eta_damping, bool allow_inverse_cache = true);
    EIGEN_STRONG_INLINE const double* currentMessageEtaData(int local_idx) const noexcept { return msg_eta_ptr_[local_idx]; }
    EIGEN_STRONG_INLINE const double* currentMessageLamData(int local_idx) const noexcept { return msg_lam_ptr_[local_idx]; }
    EIGEN_STRONG_INLINE double* const* currentMessageEtaSlot(int local_idx) const noexcept { return &msg_eta_ptr_[local_idx]; }
    EIGEN_STRONG_INLINE double* const* currentMessageLamSlot(int local_idx) const noexcept { return &msg_lam_ptr_[local_idx]; }
    bool isUnaryFactor() const noexcept { return is_unary_; }
    bool tryExportFixedLamEta6HotEntry(FixedLamEta6HotEntry& out) const noexcept;

private:
    static constexpr double kJitter = 1e-10;
    void refreshHotPathPointers_() noexcept;
    void swapMessageBuffers_() noexcept;

    // ==========================
    // Fixed dimensions (set once)
    // ==========================
    bool is_unary_  = false;
    bool is_binary_ = false;

    int d0_ = 0;   // dofs(var0) if binary, else dofs(var0) for unary
    int d1_ = 0;   // dofs(var1) if binary, else 0
    int D_  = 0;   // total dofs = d0_ + d1_

    // For binary Schur:
    // target==0 (msg to v0): d_o=d0_, d_no=d1_
    // target==1 (msg to v1): d_o=d1_, d_no=d0_
    inline int d_o_(int target)  const { return (target == 0) ? d0_ : d1_; }
    inline int d_no_(int target) const { return (target == 0) ? d1_ : d0_; }

    // ==========================
    // Workspace (allocated once)
    // ==========================
    // Copy of factor (eta, lam) with belief correction applied
    Eigen::VectorXd eta_f_;      // size: D_
    Eigen::MatrixXd lam_f_;      // size: D_ x D_

    // lnono copy (for jitter + factorization), sized to max(d0_, d1_)
    Eigen::MatrixXd lnono_;      // size: max(d0_,d1_) x max(d0_,d1_)

    // Solutions:
    // Y_:  (d_no x d_o)  and y_: (d_no)
    // Allocate at max sizes and use topLeftCorner/head when needed.
    Eigen::MatrixXd Y_;          // size: max_no x max_o
    Eigen::VectorXd y_;          // size: max_no

    Eigen::LLT<Eigen::MatrixXd, Eigen::Upper> llt_;  // reusable factorization object

    // Temporaries used by Eigen fallback paths.
    Eigen::MatrixXd tmpLam_;     // size: max_o x max_o
    Eigen::VectorXd tmpEta_;     // size: max_o

    // ==========================
    // computeFactor workspace
    // ==========================
    Eigen::VectorXd ri_cf_;      // residual per measurement block (m)
    Eigen::VectorXd tmpv_cf_;    // tmp vector: Oi * ri (m)
    Eigen::MatrixXd tmpm_cf_;    // tmp matrix: Oi * Ji (m x D)

    // ==========================
    // Jacobian / Lambda cache (J assumed constant across iterations)
    // ==========================
    bool jcache_valid_ = false;   // whether J/JO/Lambda cache is valid
    bool lamcache_set_ = false;   // whether factor.lam has been set from cache
    std::vector<Eigen::MatrixXd> J_cache_;   // blocks Ji (m x D)
    std::vector<Eigen::MatrixXd> JO_cache_;  // blocks (Ji^T * Oi) (D x m)
    Eigen::MatrixXd lambda_cache_;           // sum_i (Ji^T * Oi * Ji) (D x D)


    // =====================
    // Fixed-lambda cache
    // =====================

    mutable Eigen::LLT<Eigen::Matrix2d, Eigen::Upper> llt0_;  // target=0 消元用
    mutable Eigen::LLT<Eigen::Matrix2d, Eigen::Upper> llt1_;  // target=1 消元用
    mutable Eigen::LLT<Eigen::Matrix3d, Eigen::Upper> llt3_0_;
    mutable Eigen::LLT<Eigen::Matrix3d, Eigen::Upper> llt3_1_;
    bool llt_valid0_ = false;
    bool llt_valid1_ = false;
    Eigen::LLT<Eigen::MatrixXd, Eigen::Upper> fixed_lam_llt0_;
    Eigen::LLT<Eigen::MatrixXd, Eigen::Upper> fixed_lam_llt1_;
    Eigen::LDLT<Eigen::MatrixXd> fixed_lam_ldlt0_;
    Eigen::LDLT<Eigen::MatrixXd> fixed_lam_ldlt1_;
    Eigen::LLT<Eigen::Matrix<double, 6, 6>, Eigen::Upper> fixed_lam_llt6_0_;
    Eigen::LLT<Eigen::Matrix<double, 6, 6>, Eigen::Upper> fixed_lam_llt6_1_;
    Eigen::LDLT<Eigen::Matrix<double, 6, 6>> fixed_lam_ldlt6_0_;
    Eigen::LDLT<Eigen::Matrix<double, 6, 6>> fixed_lam_ldlt6_1_;
    Eigen::Matrix<double, 6, 6> fixed_lam_inv6_0_ = Eigen::Matrix<double, 6, 6>::Zero();
    Eigen::Matrix<double, 6, 6> fixed_lam_inv6_1_ = Eigen::Matrix<double, 6, 6>::Zero();
    Eigen::Matrix<double, 6, 6> fixed_lam_eta_map6_0_ = Eigen::Matrix<double, 6, 6>::Zero();
    Eigen::Matrix<double, 6, 6> fixed_lam_eta_map6_1_ = Eigen::Matrix<double, 6, 6>::Zero();
    bool fixed_lam_target_valid0_ = false;
    bool fixed_lam_target_valid1_ = false;
    bool fixed_lam_target_ldlt0_ = false;
    bool fixed_lam_target_ldlt1_ = false;

    Eigen::Vector3d eta0_3_ = Eigen::Vector3d::Zero();
    Eigen::Vector3d eta1_3_ = Eigen::Vector3d::Zero();
    Eigen::Matrix3d loo0_3_ = Eigen::Matrix3d::Zero();
    Eigen::Matrix3d lono0_3_ = Eigen::Matrix3d::Zero();
    Eigen::Matrix3d lnoo0_3_ = Eigen::Matrix3d::Zero();
    Eigen::Matrix3d lnono0_3_ = Eigen::Matrix3d::Zero();
    Eigen::Matrix3d loo1_3_ = Eigen::Matrix3d::Zero();
    Eigen::Matrix3d lono1_3_ = Eigen::Matrix3d::Zero();
    Eigen::Matrix3d lnoo1_3_ = Eigen::Matrix3d::Zero();
    Eigen::Matrix3d lnono1_3_ = Eigen::Matrix3d::Zero();

    const double* belief_eta_ptr_[2] = {nullptr, nullptr};
    const double* belief_lam_ptr_[2] = {nullptr, nullptr};
    double* msg_eta_ptr_[2] = {nullptr, nullptr};
    double* msg_lam_ptr_[2] = {nullptr, nullptr};
    double* msg_next_eta_ptr_[2] = {nullptr, nullptr};
    double* msg_next_lam_ptr_[2] = {nullptr, nullptr};

private:
    void initWorkspace_();
};

} // namespace gbp
