// Observes the unchanged official PowerBA iteration log and accepted geometry.
// No damping, stopping, linear-solver, or numerical-update code is replaced.
#include <iostream>
#include <streambuf>
#include <string>
#include "rootba/bal/ba_log_utils.hpp"
#include "rootba/bal/bal_app_options.hpp"
#include "rootba/cli/bal_cli_utils.hpp"
#include "rootba/solver/bal_bundle_adjustment.hpp"
#include "ba_io.h"

template<class Problem> class CaptureLog : public std::streambuf {
    std::streambuf* original_;
    const Problem& problem_;
    std::string line_;
    int iteration_=-1, pending_=-1;
    bool rejected_=false;
    void consume(char c) {
        if (c!='\n') { line_+=c; return; }
        if (line_.rfind("Iteration ",0)==0) {
            iteration_=std::stoi(line_.substr(10));
            if(iteration_==0) pending_=0;
        }
        if(line_.find("[Success]")!=std::string::npos) {pending_=iteration_; rejected_=false;}
        if(line_.find("[Reject]")!=std::string::npos || line_.find("[Invalid]")!=std::string::npos) {
            pending_=iteration_; rejected_=true;
        }
        line_.clear();
    }
public:
    CaptureLog(std::streambuf* original, const Problem& problem):original_(original),problem_(problem) {}
    std::streamsize xsputn(const char* data, std::streamsize n) override {
        const auto written=original_->sputn(data,n);
        for(std::streamsize i=0;i<written;++i) consume(data[i]);
        return written;
    }
    int overflow(int c) override {
        if(c==traits_type::eof()) return traits_type::not_eof(c);
        const int result=original_->sputc(static_cast<char>(c));
        if(result!=traits_type::eof()) consume(static_cast<char>(c));
        return result;
    }
    int sync() override {
        const int result=original_->pubsync();
        if(pending_>=0) {hgbp_io::snapshot(problem_,pending_,rejected_);pending_=-1;}
        return result;
    }
};

template<class Scalar, class... Args> void observed_bundle_adjust(rootba::BalProblem<Scalar>& problem, Args&&... args) {
    CaptureLog<rootba::BalProblem<Scalar>> log(std::cout.rdbuf(), problem);
    auto* original=std::cout.rdbuf(&log);
    try {
        rootba::bundle_adjust_manual(problem, std::forward<Args>(args)...);
        std::cout.rdbuf(original);
        hgbp_io::finish(problem);
    } catch(...) {std::cout.rdbuf(original);throw;}
}

#define bundle_adjust_manual observed_bundle_adjust
#include "bal_power_sc.cpp"
#undef bundle_adjust_manual
