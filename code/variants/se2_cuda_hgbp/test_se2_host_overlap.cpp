#include "se2_host_overlap.h"
#include <iostream>
#include <stdexcept>
#include <thread>

void require(bool value) { if(!value) throw std::runtime_error("overlap callback test failed"); }

int main() {
    int calls=0;
    runSE2HostOverlap();
    setSE2HostOverlap([&]{++calls;});
    bool rejected=false;
    try { setSE2HostOverlap([]{}); } catch(const std::runtime_error&) { rejected=true; }
    require(rejected);
    runSE2HostOverlap();
    runSE2HostOverlap();
    require(calls==1);
    setSE2HostOverlap([&]{++calls;});
    clearSE2HostOverlap();
    runSE2HostOverlap();
    require(calls==1);
    setSE2HostOverlap([]{throw std::runtime_error("intentional");});
    rejected=false;
    try { runSE2HostOverlap(); } catch(const std::runtime_error&) { rejected=true; }
    require(rejected);
    setSE2HostOverlap([&]{++calls;});
    int other_calls=0;
    std::thread other([&]{
        runSE2HostOverlap();
        setSE2HostOverlap([&]{++other_calls;});
        runSE2HostOverlap();
    });
    other.join();
    require(calls==1 && other_calls==1);
    runSE2HostOverlap();
    require(calls==2);
    std::cout<<"overlap callback lifecycle tests passed\n";
}
