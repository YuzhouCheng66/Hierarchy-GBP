#include "pgo_common.h"
#include "se3_geometry.h"

int main(int argc, char** argv) {
    try {
        return reviewer_pcg::run<reviewer_pcg::SE3Traits>(argc, argv);
    } catch (const std::exception& error) {
        g2o::G2OBatchStatistics::setGlobalStats(nullptr);
        std::cerr << "ERROR: " << error.what() << "\n";
        return 1;
    }
}
