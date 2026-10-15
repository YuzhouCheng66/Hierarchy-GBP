#pragma once

std::pair<Args, HGBPConfig> parseBAArguments(const std::vector<std::string>& options) {
    std::vector<std::string> storage = {"ba_solver", "--problem-file", "unused.bal"};
    storage.insert(storage.end(), options.begin(), options.end());
    std::vector<char*> argv;
    for (auto& value : storage) argv.push_back(value.data());
    std::vector<std::string> filtered_storage;
    std::vector<char*> filtered_argv;
    const auto policy = parseHGBPConfig(static_cast<int>(argv.size()), argv.data(),
        filtered_storage, filtered_argv);
    const auto args = parseArgs(static_cast<int>(filtered_argv.size()), filtered_argv.data());
    return {args, policy};
}

int testUniformPolicyArguments() {
    const auto [defaults, policy] = parseBAArguments({});
    if (defaults.outer != 20 || defaults.mg_cycles != 5 || defaults.pre_sweeps != 3 ||
        defaults.gbp_full_sweeps != 32 || defaults.build_threads != 16 ||
        defaults.gbp_threads != 16 || defaults.message_damping != 1.0 ||
        defaults.coarse_scale != 1.0 || !defaults.normalize_bal ||
        policy.aggregation != "connected" || policy.coarse_operator != "additive" ||
        policy.linear_controller != "fcg" || policy.gbp_model != "normalized" ||
        policy.graph_neighbors != 4 || policy.coarse_groups != 24 ||
        policy.linear_rel_tol != 0.1 || policy.variance_rel_tol != 1e-6 ||
        policy.pair_sample_cap != 16 || policy.initial_lambda != 1e-4 ||
        policy.pair_factor_scale != 1.0 || policy.krylov_start_outer != 1) return 50;

    for (const std::string threads : {"1", "16"}) {
        const auto [args, explicit_policy] = parseBAArguments({
            "--outer", "20", "--mg-cycles", "5", "--pre-sweeps", "3",
            "--gbp-full-sweeps", "32", "--graph-neighbors", "4", "--coarse-groups", "24",
            "--linear-rel-tol", "0.1", "--variance-rel-tol", "1e-6", "--pair-sample-cap", "16",
            "--message-damping", "1.0", "--initial-lambda", "0.0001", "--pair-factor-scale", "1.0",
            "--coarse-scale", "1.0", "--coarse-operator", "additive", "--linear-controller", "fcg",
            "--gbp-model", "normalized", "--krylov-start-outer", "1", "--normalize-bal",
            "--aggregation", "connected", "--fine-smoother", "gbp",
            "--build-threads", threads, "--gbp-threads", threads});
        if (args.outer != defaults.outer || args.mg_cycles != defaults.mg_cycles ||
            args.pre_sweeps != defaults.pre_sweeps || args.gbp_full_sweeps != defaults.gbp_full_sweeps ||
            args.message_damping != defaults.message_damping || args.coarse_scale != defaults.coarse_scale ||
            args.normalize_bal != defaults.normalize_bal || args.build_threads != std::stoi(threads) ||
            args.gbp_threads != std::stoi(threads) || explicit_policy.aggregation != policy.aggregation ||
            explicit_policy.coarse_operator != policy.coarse_operator ||
            explicit_policy.linear_controller != policy.linear_controller || explicit_policy.gbp_model != policy.gbp_model ||
            explicit_policy.graph_neighbors != policy.graph_neighbors || explicit_policy.coarse_groups != policy.coarse_groups ||
            explicit_policy.linear_rel_tol != policy.linear_rel_tol || explicit_policy.variance_rel_tol != policy.variance_rel_tol ||
            explicit_policy.pair_sample_cap != policy.pair_sample_cap || explicit_policy.initial_lambda != policy.initial_lambda ||
            explicit_policy.pair_factor_scale != policy.pair_factor_scale || explicit_policy.krylov_start_outer != policy.krylov_start_outer) return 51;
    }
    const std::vector<std::vector<std::string>> rejected = {
        {"--aggregation", "seed"}, {"--linear-controller", "gcr"}, {"--linear-controller", "energy"},
        {"--gbp-model", "psd"}, {"--gbp-model", "normalized-spectral"},
        {"--gbp-model", "normalized-polar"}, {"--gbp-model", "landmark"},
        {"--fine-smoother", "jacobi"}, {"--coarse-scale", "0"}, {"--coarse-scale", "0.5"},
        {"--pair-factor-scale", "0.3"}, {"--krylov-start-outer", "2"},
        {"--linear-audit", "unused.json"}, {"--group-size", "20"},
        {"--min-pair-observations", "1"}, {"--pair-backbone-min-coverage", "0.9"},
        {"--pair-sample-no-rescale"}, {"--unreduced-unary"}, {"--no-pose-scaling"}, {"--no-normalize-bal"},
        {"--worker-affinity"}, {"--persistent-full-lambda"}, {"--ba-worker-affinity"}, {"--ba-worker-ideal"},
        {"--outer", "bad"}, {"--outer", "20junk"}, {"--graph-neighbors", "4.5"},
        {"--variance-rel-tol", "nan"}, {"--initial-lambda", "inf"}, {"--linear-rel-tol", "0.1junk"},
        {"--outer", "0"}, {"--mg-cycles", "0"}, {"--pre-sweeps", "0"}, {"--gbp-full-sweeps", "0"},
        {"--build-threads", "0"}, {"--gbp-threads", "-1"}, {"--graph-neighbors", "0"},
        {"--coarse-groups", "0"}, {"--pair-sample-cap", "0"}, {"--variance-rel-tol", "0"},
        {"--linear-rel-tol", "1"}, {"--initial-lambda", "-1"}, {"--message-damping", "0"},
        {"--message-damping", "1.1"}, {"--coarse-operator"}, {"--outer"}, {"--unknown"}
    };
    for (const auto& options : rejected) {
        bool failed = false;
        try { (void)parseBAArguments(options); }
        catch (const std::exception&) { failed = true; }
        if (!failed) {
            std::cerr << "Accepted obsolete/invalid argument: " << options.front() << '\n';
            return 52;
        }
    }
    for (const std::string mode : {"legacy", "residual", "exact", "gauge", "gauge16",
            "additive-gauge", "additive-gauge16", "additive-schur-gauge", "additive-schur",
            "additive-split", "additive-split-gauge"}) {
        bool failed = false;
        try { (void)parseBAArguments({"--coarse-operator", mode}); }
        catch (const std::exception&) { failed = true; }
        if (!failed) return 53;
    }
    std::cout << "uniform_policy_defaults_and_cli_checks=passed\n";
    return 0;
}
