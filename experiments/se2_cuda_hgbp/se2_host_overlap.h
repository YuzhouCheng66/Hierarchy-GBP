#pragma once
#include <functional>

void setSE2HostOverlap(std::function<void()> work);
void runSE2HostOverlap();
void clearSE2HostOverlap();
