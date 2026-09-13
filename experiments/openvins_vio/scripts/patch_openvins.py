#!/usr/bin/env python3
"""Apply the explicit live-backend interface to a pinned isolated OpenVINS checkout."""

from pathlib import Path
import argparse, hashlib, json, re, subprocess

HEAD = "69488123ed9362dd44b6f28e7f4680abbff1442b"


def apply(repo, previous=None):
    repo = Path(repo).resolve()

    def git(*a):
        return subprocess.check_output(["git", "-C", str(repo), *a], text=True)

    assert git("rev-parse", "HEAD").strip() == HEAD, "OpenVINS revision mismatch"

    def upstream(name):
        return git("show", HEAD + ":ov_msckf/src/" + name)

    changes = {}
    s = upstream("state/State.h")
    s = (
        s.replace(
            "#include <Eigen/Eigen>",
            '#include <Eigen/Eigen>\n#include "ov_gaussian/Backend.h"',
        )
        if "#include <Eigen/Eigen>" in s
        else s.replace(
            '#include "StateOptions.h"',
            '#include "StateOptions.h"\n#include "ov_gaussian/Backend.h"',
        )
    )
    assert "ov_gaussian/Backend.h" in s
    s = s.replace(
        "int max_covariance_size() { return (int)_Cov.rows(); }",
        "int max_covariance_size() { return _gaussian ? _gaussian->dimension() : (int)_Cov.rows(); }",
    )
    s = s.replace(
        "private:\n  // Define that the state helper",
        "  // Experimental live Gaussian backend; absent for the original algorithm.\n  std::shared_ptr<ov_gaussian::Backend> _gaussian;\n\nprivate:\n  // Define that the state helper",
    )
    assert "_gaussian;" in s
    changes["state/State.h"] = s
    s = upstream("state/State.cpp")
    pos = s.rfind("\n}")
    s = (
        s[:pos]
        + """
  if (ov_gaussian::Backend::enabled()) {
    if (current_id != 15 || _options.max_slam_features != 0 ||
        _options.do_calib_camera_pose || _options.do_calib_camera_intrinsics ||
        _options.do_calib_camera_timeoffset || _options.do_calib_imu_intrinsics ||
        _options.do_calib_imu_g_sensitivity ||
        _options.feat_rep_msckf != LandmarkRepresentation::Representation::GLOBAL_3D)
      throw std::runtime_error("Gaussian backend requires fixed-calibration GLOBAL_3D MSCKF-only state");
    _gaussian = std::make_shared<ov_gaussian::Backend>(ov_gaussian::Backend::selected_mode(), ov_gaussian::Backend::audit_requested());
    _gaussian->initialize(_Cov);
    // Make stale legacy covariance use fail visibly instead of returning a plausible old prior.
    _Cov.resize(0, 0);
  }
"""
        + s[pos:]
    )
    changes["state/State.cpp"] = s
    s = upstream("state/StateHelper.h")
    needle = "class StateHelper {\n\npublic:"
    assert needle in s
    s = s.replace(
        needle,
        needle
        + "\n  static void apply_gaussian_increment(std::shared_ptr<State> state, const Eigen::VectorXd &dx);\n",
    )
    changes["state/StateHelper.h"] = s
    s = upstream("state/StateHelper.cpp")

    def insert(name, body):
        nonlocal s
        definitions = list(
            re.finditer(
                r"^(?:void|bool|Eigen::MatrixXd|std::shared_ptr<Type>) StateHelper::"
                + name
                + r"\(",
                s,
                re.M,
            )
        )
        assert len(definitions) == 1, ("expected one function definition", name)
        start = definitions[0].start()
        pos = s.index("{", start) + 1
        s = s[:pos] + "\n" + body + "\n" + s[pos:]

    insert(
        "EKFPropagation",
        """  if (state->_gaussian) {
    if (order_NEW.size()!=1 || order_OLD.size()!=1 || order_NEW[0]!=state->_imu || order_OLD[0]!=state->_imu)
      throw std::runtime_error("Gaussian backend only supports live 15D IMU propagation");
    state->_gaussian->propagate(Phi, Q); return;
  }""",
    )
    insert(
        "EKFUpdate",
        """  if (state->_gaussian)
    throw std::runtime_error("Gaussian backend must consume the raw MSCKF batch, not a legacy EKFUpdate");""",
    )
    insert(
        "set_initial_covariance",
        """  if (state->_gaussian) {
    if (state->_variables.size()!=1 || state->_variables[0]!=state->_imu || state->max_covariance_size()!=15)
      throw std::runtime_error("Gaussian initialization requires a single live IMU");
    Eigen::MatrixXd initial = state->_gaussian->full_covariance();
    int i_index=0;
    for (const auto &a : order) {
      int j_index=0;
      for (const auto &b : order) {
        initial.block(a->id(),b->id(),a->size(),b->size())=covariance.block(i_index,j_index,a->size(),b->size());
        j_index+=b->size();
      }
      i_index+=a->size();
    }
    state->_gaussian->initialize(initial); return;
  }""",
    )
    insert(
        "get_marginal_covariance",
        """  if (state->_gaussian) {
    ov_gaussian::Indices ids;
    for (const auto &var : small_variables) for (int j=0;j<var->size();++j) ids.push_back(var->id()+j);
    return state->_gaussian->marginal(ids);
  }""",
    )
    insert(
        "get_full_covariance",
        """  if (state->_gaussian) return state->_gaussian->full_covariance();""",
    )
    insert(
        "marginalize",
        """  if (state->_gaussian) {
    if (std::find(state->_variables.begin(),state->_variables.end(),marg)==state->_variables.end())
      throw std::runtime_error("Gaussian marginalization requires an active complete variable");
    int id=marg->id(),size=marg->size();
    state->_gaussian->marginalize(id,size);
    std::vector<std::shared_ptr<Type>> remaining;
    for (const auto &v : state->_variables) if (v!=marg) {
      if (v->id()>id) v->set_local_id(v->id()-size);
      remaining.push_back(v);
    }
    marg->set_local_id(-1);state->_variables=std::move(remaining);return;
  }""",
    )
    insert(
        "clone",
        """  if (state->_gaussian) {
    if (variable_to_clone!=state->_imu->pose()) throw std::runtime_error("Gaussian clone must be the live IMU pose");
    int new_id=state->max_covariance_size();
    auto new_clone=variable_to_clone->clone();new_clone->set_local_id(new_id);
    state->_gaussian->clone_pose(variable_to_clone->id(),variable_to_clone->size());
    state->_variables.push_back(new_clone);return new_clone;
  }""",
    )
    for name in ("initialize", "initialize_invertible"):
        insert(
            name,
            '  if (state->_gaussian) throw std::runtime_error("Landmark initialization is outside the Gaussian MSCKF-only integration");',
        )
    insert(
        "augment_clone",
        """  if (state->_gaussian && state->_options.do_calib_camera_timeoffset)
    throw std::runtime_error("Gaussian clone time-offset calibration is unsupported");""",
    )
    s += """
void StateHelper::apply_gaussian_increment(std::shared_ptr<State> state, const Eigen::VectorXd &dx) {
  if (!state->_gaussian || dx.size()!=state->max_covariance_size() || !dx.allFinite())
    throw std::runtime_error("Invalid live Gaussian tangent increment");
  for (const auto &v : state->_variables) v->update(dx.segment(v->id(),v->size()));
  // OpenVINS uses the identity first-order covariance reset here. The next frame
  // derives fresh Jacobians from these actual updated nominal values and FEJ state.
}
"""
    changes["state/StateHelper.cpp"] = s
    s = upstream("core/VioManager.cpp")
    needle = "  state = std::make_shared<State>(params.state_options);"
    assert needle in s
    s = s.replace(
        needle,
        needle
        + """
  if (state->_gaussian && (params.try_zupt || params.use_aruco || params.init_options.init_dyn_use))
    throw std::runtime_error("Gaussian integration requires static initialization, no ZUPT, and no ArUco");
""",
    )
    changes["core/VioManager.cpp"] = s
    u = upstream("update/UpdaterMSCKF.cpp")
    s = u
    needle = "  // Calculate the max possible measurement size"
    assert needle in s
    s = s.replace(
        needle,
        """  if (state->_gaussian) {
    update_gaussian(state, feature_vec);
    return;
  }

"""
        + needle,
        1,
    )
    # Preserve upstream feature construction and fresh Jacobian evaluation byte-for-byte.
    start = u.index("    // Convert our feature into our current format")
    end = u.index("    // Nullspace project", start)
    feature = u[start:end]
    s += (
        """
void UpdaterMSCKF::update_gaussian(std::shared_ptr<State> state, std::vector<std::shared_ptr<Feature>> &feature_vec) {
  if (feature_vec.empty()) return;
  state->_gaussian->begin_visual();
  auto it2=feature_vec.begin();
  while (it2!=feature_vec.end()) {
"""
        + feature
        + """
    ov_gaussian::Indices columns;
    for (const auto &v : Hx_order) for (int j=0;j<v->size();++j) columns.push_back(v->id()+j);
    int dof=res.rows()-H_f.cols();
    double threshold;
    if (dof<500) threshold=chi_squared_table.at(dof);
    else { boost::math::chi_squared dist(dof); threshold=boost::math::quantile(dist,0.95); }
    auto decision=state->_gaussian->add_feature(H_f,H_x,res,columns,_options.sigma_pix_sq,_options.chi2_multipler*threshold);
    (*it2)->to_delete=true;
    if (!decision.accepted) it2=feature_vec.erase(it2); else ++it2;
  }
  Eigen::VectorXd dx=state->_gaussian->finish_visual();
  if (dx.size()) StateHelper::apply_gaussian_increment(state,dx);
}
"""
    )
    changes["update/UpdaterMSCKF.cpp"] = s
    s = upstream("update/UpdaterMSCKF.h")
    needle = "  void update(std::shared_ptr<State> state, std::vector<std::shared_ptr<ov_core::Feature>> &feature_vec);"
    assert needle in s
    s = s.replace(
        needle,
        needle
        + "\n\n  void update_gaussian(std::shared_ptr<State> state, std::vector<std::shared_ptr<ov_core::Feature>> &feature_vec);",
    )
    changes["update/UpdaterMSCKF.h"] = s
    manifest = {}
    for name, new in changes.items():
        original = upstream(name)
        path = repo / "ov_msckf/src" / name
        current = path.read_text()
        old_hash = (previous or {}).get("files", {}).get(name, {}).get("after")
        assert (
            current in (original, new)
            or hashlib.sha256(current.encode()).hexdigest() == old_hash
        ), ("refuse overwriting unrelated edits", name)
        if current != new:
            path.write_text(new)
        manifest[name] = dict(
            before=hashlib.sha256(original.encode()).hexdigest(),
            after=hashlib.sha256(new.encode()).hexdigest(),
        )
    return dict(
        upstream_commit=HEAD,
        files=manifest,
        algorithm="live root messages and shared EKF; no hierarchy claim",
        recorded_state_inputs=False,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("repo", type=Path)
    p.add_argument("--manifest", type=Path, required=True)
    a = p.parse_args()
    result = apply(a.repo)
    a.manifest.write_text(json.dumps(result, indent=2) + "\n")
    print("Patched", len(result["files"]), "explicit OpenVINS interfaces.")
