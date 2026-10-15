// Retained transportsegment implementation; no per-dataset kernel selection.
#define HGBP_CACHED_INCOMING
#define HGBP_PACKED_TRANSPORT
#define HGBP_TRANSPORT_GRAPH
#define HGBP_SEGMENT_EXECUTOR
#include "se2_cuda_sourcewarp.cu"
