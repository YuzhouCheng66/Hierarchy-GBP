#include "pgo_common.h"
#include "se2_geometry.h"

int main(int argc, char** argv) {
    try {
        return reviewer_pcg::run<reviewer_pcg::SE2Traits>(argc, argv);
    } catch (const std::exception& error) {
        g2o::G2OBatchStatistics::setGlobalStats(nullptr);
        std::cerr << "ERROR: " << error.what() << "\n";
        return 1;
    }
}
