// The whole transfer/compute batch is replayed, not 100 separate sweep kernels.
struct TransportGraphStorage {
    cudaStream_t stream=nullptr;
    cudaGraph_t graphs[2]{};
    cudaGraphExec_t execs[2]{};
    cudaEvent_t events[4]{};
    int count_key=-1,n_key=-1,m_key=-1,s_key=-1,device_key=-1;
    int blocks=0,registers=0;
    double jitter_key=0;
    bool all_key=false,timers=false,configured=false;
    ~TransportGraphStorage() {
        reset();
        for(auto e:events) if(e) cudaEventDestroy(e);
        if(stream) cudaStreamDestroy(stream);
    }
    void reset() {
        for(int k=0;k<2;++k) {
            if(execs[k]) { cudaGraphExecDestroy(execs[k]); execs[k]=nullptr; }
            if(graphs[k]) { cudaGraphDestroy(graphs[k]); graphs[k]=nullptr; }
        }
    }
    SE2CudaTiming run(Device d,SE2CudaView raw,const SE2TransferLayout& packet,
        Buffer<unsigned char>& input,Buffer<unsigned char>& output,
        int count,double jitter,bool new_outer,bool new_topology,bool all) {
        const char* text=std::getenv("HGBP_TRANSPORT_EVENTS");
        const bool requested=text && std::strcmp(text,"1")==0;
        const int threads=std::getenv("HGBP_RESIDENT_THREADS")?std::atoi(std::getenv("HGBP_RESIDENT_THREADS")):32;
        if(threads!=32) throw std::runtime_error("Transport graph currently requires 32-thread CTAs");
        if(configured && requested!=timers) throw std::runtime_error("Cannot change graph timers during a solve");
        if(!configured) {
            timers=requested;
            check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
            if(timers) for(auto& e:events) check(cudaEventCreate(&e));
            check(cudaGetDevice(&device_key));
            configured=true;
            std::cout<<"CUDA_TRANSPORT_MODE stage_events="<<timers<<" persistent_batch=1\n";
        }
        int device=0; check(cudaGetDevice(&device));
        if(device!=device_key) throw std::runtime_error("Transport graph cannot switch devices");
        const void* kernel=reinterpret_cast<const void*>(
#ifdef HGBP_SEGMENT_EXECUTOR
            sweepsSegmentKernel
#else
            sweepsGatherKernel
#endif
        );
        if(new_topology || count!=count_key || d.n!=n_key || d.m!=m_key || d.s!=s_key ||
           jitter!=jitter_key || all!=all_key) {
            reset();
            cudaDeviceProp prop{}; cudaFuncAttributes attr{}; int active=0;
            check(cudaGetDeviceProperties(&prop,device));
            check(cudaFuncGetAttributes(&attr,kernel));
            check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,kernel,threads,0));
            blocks=(std::max(d.n,d.s)+threads-1)/threads;
            if(!prop.cooperativeLaunch || blocks>active*prop.multiProcessorCount)
                throw std::runtime_error("Transport cooperative residency unavailable");
            registers=attr.numRegs;
            count_key=count; n_key=d.n; m_key=d.m; s_key=d.s; jitter_key=jitter; all_key=all;
        }
        const int which=new_outer?0:1;
        if(!execs[which]) {
            // Only initial graph setup has default-stream topology uploads.
            // The replay stream is nonblocking, so explicitly finish those.
            check(cudaStreamSynchronize(nullptr));
            check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
            if(timers) check(cudaEventRecordWithFlags(events[0],stream,cudaEventRecordExternal));
            check(cudaMemcpyAsync(input.p,input.pinned,new_outer?packet.input_size:packet.cycle_size,
                cudaMemcpyHostToDevice,stream));
            if(new_outer) {
                check(cudaMemsetAsync(d.indices,0,3*sizeof(int),stream));
                pack<<<(std::max(d.n,d.s)+255)/256,256,0,stream>>>(raw,d);
            } else patchBeliefEta<<<(d.n+255)/256,256,0,stream>>>(raw,d);
            check(cudaGetLastError());
            if(timers) check(cudaEventRecordWithFlags(events[1],stream,cudaEventRecordExternal));
            void* args[]={&d,&count,&jitter};
            check(cudaLaunchCooperativeKernel(kernel,blocks,threads,args,0,stream));
            if(timers) check(cudaEventRecordWithFlags(events[2],stream,cudaEventRecordExternal));
            auto out=raw;
            out.belief_eta=reinterpret_cast<double*>(output.p+packet.out_eta);
            out.belief_lam=reinterpret_cast<double*>(output.p+packet.out_lam);
            unpack<<<(std::max(d.n,d.s)+255)/256,256,0,stream>>>(out,d,all,
                reinterpret_cast<SE2CudaState*>(output.p+packet.out_state));
            check(cudaGetLastError());
            check(cudaMemcpyAsync(output.pinned,output.p,packet.output_size,cudaMemcpyDeviceToHost,stream));
            if(timers) check(cudaEventRecordWithFlags(events[3],stream,cudaEventRecordExternal));
            check(cudaStreamEndCapture(stream,&graphs[which]));
            check(cudaGraphInstantiate(&execs[which],graphs[which],nullptr,nullptr,0));
            size_t nodes=0; check(cudaGraphGetNodes(graphs[which],nullptr,&nodes));
            std::cout<<"CUDA_TRANSPORT_GRAPH outer="<<new_outer<<" nodes="<<nodes
                <<" stage_events="<<timers<<" sweeps="<<count<<'\n';
        }
        check(cudaGraphLaunch(execs[which],stream));
        runSE2HostOverlap();
        // Wait for this replay, not for an event that a previous replay recorded.
        check(cudaStreamSynchronize(stream));
        float upload=0,kernel_ms=0,download=0;
        if(timers) {
            check(cudaEventElapsedTime(&upload,events[0],events[1]));
            check(cudaEventElapsedTime(&kernel_ms,events[1],events[2]));
            check(cudaEventElapsedTime(&download,events[2],events[3]));
        }
        // Zero stage fields mean uninstrumented, not zero device work. Whole
        // solve/lower wall clocks stay active and Nsight provides stage timing.
        return {upload,kernel_ms,download,blocks,threads,registers};
    }
};
