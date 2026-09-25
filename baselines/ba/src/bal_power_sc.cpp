/**
BSD 3-Clause License

This file is a Windows-build helper for the RootBA external baseline. It is
equivalent to bal_sc.cpp, but forces the Power Schur Complement solver so that
we do not depend on enum command-line parsing on MSVC.
*/

#include <glog/logging.h>

#include "rootba/bal/ba_log_utils.hpp"
#include "rootba/bal/bal_app_options.hpp"
#include "rootba/cli/bal_cli_utils.hpp"
#include "rootba/solver/bal_bundle_adjustment.hpp"

int main(int argc, char** argv) {
  using namespace rootba;

  FLAGS_logtostderr = true;
  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();

  BalAppOptions options;
  if (!parse_bal_app_arguments(
          "Solve BAL problem with Power Schur Complement solver.", argc, argv,
          options)) {
    return 1;
  }

  options.solver.solver_type =
      SolverOptions::SolverType::POWER_SCHUR_COMPLEMENT;

  if (options.solver.verbosity_level >= 2) {
    LOG(INFO) << "Options:\n" << options;
  }

  BalPipelineSummary summary;

  if (!options.solver.use_double) {
#ifdef ROOTBA_INSTANTIATIONS_FLOAT
    auto bal_problem = load_normalized_bal_problem<float>(
        options.dataset, &summary.dataset, &summary.timing);
    bundle_adjust_manual(bal_problem, options.solver, &summary.solver,
                         &summary.timing);
    bal_problem.postprocress(options.dataset, &summary.timing);
#else
    LOG(FATAL) << "Compiled without float support.";
#endif
  } else {
#ifdef ROOTBA_INSTANTIATIONS_DOUBLE
    auto bal_problem = load_normalized_bal_problem<double>(
        options.dataset, &summary.dataset, &summary.timing);
    bundle_adjust_manual(bal_problem, options.solver, &summary.solver,
                         &summary.timing);
    bal_problem.postprocress(options.dataset, &summary.timing);
#else
    LOG(FATAL) << "Compiled without double support.";
#endif
  }

  BaLog log;
  log_summary(log, summary);
  log.save_json(options.solver.log);

  return 0;
}
