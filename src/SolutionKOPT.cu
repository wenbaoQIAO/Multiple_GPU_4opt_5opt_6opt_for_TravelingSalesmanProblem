#include "config/ConfigParamsCF.h"
#include "random_generator_cf.h"
#include "CalculateurEMST.h"
//#include "Multiout.h"

#include "SolutionKOPT.h"

#define round(x) ((fabs(ceil(x) - (x)) < fabs(floor(x) - (x))) ? ceil(x) : floor(x))

#define EMST_PRINT_CM 1
#define EMST_SQUARED_CM 0

template<std::size_t DimP, std::size_t DimCM>
int SolutionKOPT<DimP, DimCM>::cptInstance = 0;

template<std::size_t DimP, std::size_t DimCM>
void SolutionKOPT<DimP, DimCM>::initialize(char* data, char* sol, char* stats)
{
    fileData = data;
    fileSolution = sol;
    fileStats = stats;

    // retrun optimum
    string str = data;
    optimum = returnOptimal(str);
    traceTSP.optimumLength = optimum;
    cout << " TSP: " << data << ", optimumLength = " << traceTSP.optimumLength  << endl;

    initialize();
}


template<std::size_t DimP, std::size_t DimCM>
void SolutionKOPT<DimP, DimCM>::initialize()
{
    printf(" CUDA Device Query (Runtime API) version (CUDART static linking)\n\n");

    int deviceCount = 0;
    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

    if (error_id != cudaSuccess)
    {
        printf("cudaGetDeviceCount returned %d\n-> %s\n", (int)error_id, cudaGetErrorString(error_id));
        printf("Result = FAIL\n");
        exit(EXIT_FAILURE);
    }

    // This function call returns 0 if there are no CUDA capable devices.
    if (deviceCount == 0)
    {
        printf("There are no available device(s) that support CUDA\n");
    }
    else
    {
        printf("Detected %d CUDA Capable device(s)\n", deviceCount);
    }

    int dev, driverVersion = 0, runtimeVersion = 0;

    for (dev = 0; dev < deviceCount; ++dev)
    {
        cudaSetDevice(dev);
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        printf("\nDevice %d: \"%s\"\n", dev, deviceProp.name);

        // Console log
        cudaDriverGetVersion(&driverVersion);
        cudaRuntimeGetVersion(&runtimeVersion);
        printf("  CUDA Driver Version / Runtime Version          %d.%d / %d.%d\n", driverVersion/1000, (driverVersion%100)/10, runtimeVersion/1000, (runtimeVersion%100)/10);
        printf("  CUDA Capability Major/Minor version number:    %d.%d\n", deviceProp.major, deviceProp.minor);

        char msg[256];
        SPRINTF(msg, "  Total amount of global memory:                 %.0f MBytes (%llu bytes)\n",
                (float)deviceProp.totalGlobalMem/1048576.0f, (unsigned long long) deviceProp.totalGlobalMem);
        printf("%s", msg);

        printf("  (%2d) Multiprocessors, (%3d) CUDA Cores/MP:     %d CUDA Cores\n",
               deviceProp.multiProcessorCount,
               _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor),
               _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) * deviceProp.multiProcessorCount);
        printf("  GPU Max Clock rate:                            %.0f MHz (%0.2f GHz)\n", deviceProp.clockRate * 1e-3f, deviceProp.clockRate * 1e-6f);


#if CUDART_VERSION >= 5000
        // This is supported in CUDA 5.0 (runtime API device properties)
        printf("  Memory Clock rate:                             %.0f Mhz\n", deviceProp.memoryClockRate * 1e-3f);
        printf("  Memory Bus Width:                              %d-bit\n",   deviceProp.memoryBusWidth);

        if (deviceProp.l2CacheSize)
        {
            printf("  L2 Cache Size:                                 %d bytes\n", deviceProp.l2CacheSize);
        }

#else
        // This only available in CUDA 4.0-4.2 (but these were only exposed in the CUDA Driver API)
        int memoryClock;
        getCudaAttribute<int>(&memoryClock, CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE, dev);
        printf("  Memory Clock rate:                             %.0f Mhz\n", memoryClock * 1e-3f);
        int memBusWidth;
        getCudaAttribute<int>(&memBusWidth, CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH, dev);
        printf("  Memory Bus Width:                              %d-bit\n", memBusWidth);
        int L2CacheSize;
        getCudaAttribute<int>(&L2CacheSize, CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE, dev);

        if (L2CacheSize)
        {
            printf("  L2 Cache Size:                                 %d bytes\n", L2CacheSize);
        }

#endif

        printf("  Maximum Texture Dimension Size (x,y,z)         1D=(%d), 2D=(%d, %d), 3D=(%d, %d, %d)\n",
               deviceProp.maxTexture1D   , deviceProp.maxTexture2D[0], deviceProp.maxTexture2D[1],
                deviceProp.maxTexture3D[0], deviceProp.maxTexture3D[1], deviceProp.maxTexture3D[2]);
        printf("  Maximum Layered 1D Texture Size, (num) layers  1D=(%d), %d layers\n",
               deviceProp.maxTexture1DLayered[0], deviceProp.maxTexture1DLayered[1]);
        printf("  Maximum Layered 2D Texture Size, (num) layers  2D=(%d, %d), %d layers\n",
               deviceProp.maxTexture2DLayered[0], deviceProp.maxTexture2DLayered[1], deviceProp.maxTexture2DLayered[2]);


        printf("  Total amount of constant memory:               %lu bytes\n", deviceProp.totalConstMem);
        printf("  Total amount of shared memory per block:       %lu bytes\n", deviceProp.sharedMemPerBlock);
        printf("  Total number of registers available per block: %d\n", deviceProp.regsPerBlock);
        printf("  Warp size:                                     %d\n", deviceProp.warpSize);
        printf("  Maximum number of threads per multiprocessor:  %d\n", deviceProp.maxThreadsPerMultiProcessor);
        printf("  Maximum number of threads per block:           %d\n", deviceProp.maxThreadsPerBlock);
        printf("  Max dimension size of a thread block (x,y,z): (%d, %d, %d)\n",
               deviceProp.maxThreadsDim[0],
                deviceProp.maxThreadsDim[1],
                deviceProp.maxThreadsDim[2]);
        printf("  Max dimension size of a grid size    (x,y,z): (%d, %d, %d)\n",
               deviceProp.maxGridSize[0],
                deviceProp.maxGridSize[1],
                deviceProp.maxGridSize[2]);
        printf("  Maximum memory pitch:                          %lu bytes\n", deviceProp.memPitch);
        printf("  Texture alignment:                             %lu bytes\n", deviceProp.textureAlignment);
        printf("  Concurrent copy and kernel execution:          %s with %d copy engine(s)\n", (deviceProp.deviceOverlap ? "Yes" : "No"), deviceProp.asyncEngineCount);
        printf("  Run time limit on kernels:                     %s\n", deviceProp.kernelExecTimeoutEnabled ? "Yes" : "No");
        printf("  Integrated GPU sharing Host Memory:            %s\n", deviceProp.integrated ? "Yes" : "No");
        printf("  Support host page-locked memory mapping:       %s\n", deviceProp.canMapHostMemory ? "Yes" : "No");
        printf("  Alignment requirement for Surfaces:            %s\n", deviceProp.surfaceAlignment ? "Yes" : "No");
        printf("  Device has ECC support:                        %s\n", deviceProp.ECCEnabled ? "Enabled" : "Disabled");
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
        printf("  CUDA Device Driver Mode (TCC or WDDM):         %s\n", deviceProp.tccDriver ? "TCC (Tesla Compute Cluster Driver)" : "WDDM (Windows Display Driver Model)");
#endif
        printf("  Device supports Unified Addressing (UVA):      %s\n", deviceProp.unifiedAddressing ? "Yes" : "No");
        printf("  Supports Cooperative Kernel Launch:            %s\n", deviceProp.cooperativeLaunch ? "Yes" : "No");
        printf("  Supports MultiDevice Co-op Kernel Launch:      %s\n", deviceProp.cooperativeMultiDeviceLaunch ? "Yes" : "No");
        printf("  Device PCI Domain ID / Bus ID / location ID:   %d / %d / %d\n", deviceProp.pciDomainID, deviceProp.pciBusID, deviceProp.pciDeviceID);

        const char *sComputeMode[] =
        {
            "Default (multiple host threads can use ::cudaSetDevice() with device simultaneously)",
            "Exclusive (only one host thread in one process is able to use ::cudaSetDevice() with this device)",
            "Prohibited (no host thread can use ::cudaSetDevice() with this device)",
            "Exclusive Process (many threads in one process is able to use ::cudaSetDevice() with this device)",
            "Unknown",
            NULL
        };
        printf("  Compute Mode:\n");
        printf("     < %s >\n", sComputeMode[deviceProp.computeMode]);
    }

    // If there are 2 or more GPUs, query to determine whether RDMA is supported
    if (deviceCount >= 2)
    {
        cudaDeviceProp prop[64];
        int gpuid[64]; // we want to find the first two GPUs that can support P2P
        int gpu_p2p_count = 0;

        for (int i=0; i < deviceCount; i++)
        {
            checkCudaErrors(cudaGetDeviceProperties(&prop[i], i));

            // Only boards based on Fermi or later can support P2P
            if ((prop[i].major >= 2)
        #if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
                    // on Windows (64-bit), the Tesla Compute Cluster driver for windows must be enabled to support this
                    && prop[i].tccDriver
        #endif
                    )
            {
                // This is an array of P2P capable GPUs
                gpuid[gpu_p2p_count++] = i;
            }
        }

        // Show all the combinations of support P2P GPUs
        int can_access_peer;

        if (gpu_p2p_count >= 2)
        {
            for (int i = 0; i < gpu_p2p_count; i++)
            {
                for (int j = 0; j < gpu_p2p_count; j++)
                {
                    if (gpuid[i] == gpuid[j])
                    {
                        continue;
                    }
                    checkCudaErrors(cudaDeviceCanAccessPeer(&can_access_peer, gpuid[i], gpuid[j]));
                    printf("> Peer access from %s (GPU%d) -> %s (GPU%d) : %s\n", prop[gpuid[i]].name, gpuid[i],
                            prop[gpuid[j]].name, gpuid[j] ,
                            can_access_peer ? "Yes" : "No");
                }
            }
        }
    }

    // csv masterlog info
    // *****************************
    // exe and CUDA driver name
    printf("\n");
    std::string sProfileString = "deviceQuery, CUDA Driver = CUDART";
    char cTemp[16];

    // driver version
    sProfileString += ", CUDA Driver Version = ";
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
    sprintf_s(cTemp, 10, "%d.%d", driverVersion/1000, (driverVersion%100)/10);
#else
    sprintf(cTemp, "%d.%d", driverVersion/1000, (driverVersion%100)/10);
#endif
    sProfileString +=  cTemp;

    // Runtime version
    sProfileString += ", CUDA Runtime Version = ";
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
    sprintf_s(cTemp, 10, "%d.%d", runtimeVersion/1000, (runtimeVersion%100)/10);
#else
    sprintf(cTemp, "%d.%d", runtimeVersion/1000, (runtimeVersion%100)/10);
#endif
    sProfileString +=  cTemp;

    // Device count
    sProfileString += ", NumDevs = ";
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
    sprintf_s(cTemp, 10, "%d", deviceCount);
#else
    sprintf(cTemp, "%d", deviceCount);
#endif
    sProfileString += cTemp;
    sProfileString += "\n";
    printf("%s", sProfileString.c_str());

    printf("Result = PASS\n");

#if TEST_CODE
    int devID = 0;
    cudaError_t error;
    cudaDeviceProp deviceProp;
    error = cudaGetDevice(&devID);
    if (error != cudaSuccess)
    {
        printf("cudaGetDevice returned error code %d, line(%d)\n", error, __LINE__);
    }
    error = cudaGetDeviceProperties(&deviceProp, devID);
    if (deviceProp.computeMode == cudaComputeModeProhibited)
    {
        fprintf(stderr, "Error: device is running in <Compute Mode Prohibited>, no threads can use ::cudaSetDevice().\n");
        exit(EXIT_SUCCESS);
    }
    if (error != cudaSuccess)
    {
        printf("cudaGetDeviceProperties returned error code %d, line(%d)\n", error, __LINE__);
    }
    else
    {
        printf("GPU Device %d: \"%s\" with compute capability %d.%d\n",
               devID, deviceProp.name, deviceProp.major, deviceProp.minor);
        printf("GPU Device %d: \"%s\" with multi-processors %d\n",
               devID, deviceProp.name, deviceProp.multiProcessorCount);
    }
#endif
}

template<std::size_t DimP, std::size_t DimCM>
void SolutionKOPT<DimP, DimCM>::initialize(NetLink& md_links, PointEuclidean& pMin, PointEuclidean& pMax)
{
    cout << "INITIALISATION START" << std::endl;

    size_t nNodes = 0;
    GLint w = md_links.adaptiveMap.getWidth();
    GLint h = md_links.adaptiveMap.getHeight();
    GLint d = md_links.adaptiveMap.getDepth();
    nNodes = w * h * d;
    cout << "Check read cities's num = " << nNodes  << endl;

    // wb.Q add case check
    if(nNodes == 0)
    {
        cout << "Error: no input available." << endl;
        return;
    }

    //qiao 2024 here build initial TSP tour
    md_links.networkLinks.resize(nNodes, 1);
    md_links.generate2ConectedLinks();
//    md_links.write("originalTSP");
    md_links.writeLinks("originalTSP", fileData);
    int numCityTraversed = 0;
    md_links.fixedMap.resize(nNodes, 1);
    double weightOriginal = md_links.evaluateWeightOfTSP(dist, numCityTraversed);
    cout << "*******=== OriginalTSP md_links total weight: " << weightOriginal << ", Number city traversed: "<< numCityTraversed << endl;

    //opt possibilites
    int numOptPossibilites = md_links.nodeParentMap.getWidth();
    if(numOptPossibilites <= 0)
        cout << "Error Opt possibilites = " << numOptPossibilites << endl;

    //qiao changing corrdinate can be reserved untill final experimenta stage, to see whether it has difference
    //    // Save original map
    //    adaptiveMapOriginal.gpuResize(w, h, d);
    //    md_links.adaptiveMap.gpuCopyHostToDevice(adaptiveMapOriginal);

    //    // Change coordinates
    //#if EMST_SQUARED_CM
    //    GLdouble min_x = pMin[0];
    //    GLdouble min_y = pMin[1];
    //    GLdouble max_x = pMax[0];
    //    GLdouble max_y = pMax[1];

    //    int _w = (GLint) ceil(max_x - min_x)+2;
    //    int _h = (GLint) ceil(max_y - min_y)+2;
    //    cout << "city_area_w , city_area_h " << _w << ", " << _h << endl;

    //    // wb.Q, chansfer the coordinate system
    //    float unitX = sqrt(nNodes) / _w;
    //    float unitY = sqrt(nNodes) / _h;
    //    cout << "unitX = " << unitX << endl;
    //    cout << "unitY = " << unitY << endl;
    //#else
    //    GLdouble wg = pMax[0] - pMin[0];
    //    GLdouble hg = pMax[1] - pMin[1];
    //    GLdouble dg = 0;
    //    if (DimP >= 3)
    //        dg = pMax[2] - pMin[2];

    //    cout << "max original x, y, z " << pMax[0] << ", " << pMax[1];
    //    if (DimP >= 3)
    //        cout << ", " << pMax[2];
    //    cout << endl;
    //    cout << "min original x, y, z " << pMin[0] << ", " << pMin[1];
    //    if (DimP >= 3)
    //        cout << ", " << pMin[2];
    //    cout << endl;
    //    cout << "city_area : " << wg << " * " << hg << " * " << dg << endl;

    //    if (wg == 0)
    //        wg = 1;
    //    if (hg == 0)
    //        hg = 1;
    //    GLdouble _wnd = sqrt(nNodes*wg/hg)*2;
    //    if (DimP >= 3) {
    //        if (dg != 0)
    //            _wnd = pow((nNodes*wg*wg)/(hg*dg), 0.333);
    //        else
    //            dg = 1;
    //    }

    //    GLdouble _wndd = ceil(_wnd) + 1;
    //    GLdouble unitX = _wndd / wg;
    //    cout << "unitX = " << unitX << endl;

    //#endif
    //    // Change coord system
    //    mr_links_cpu.resize(nNodes, 1);
    //    GLdouble max_xNew = -INFINITY;
    //    GLdouble max_yNew = -INFINITY;
    //    GLdouble min_xNew = +INFINITY;
    //    GLdouble min_yNew = +INFINITY;
    //    GLdouble max_zNew = -INFINITY;
    //    GLdouble min_zNew = +INFINITY;
    //    if (DimP < 3) {
    //        max_zNew = 0;
    //        min_zNew = 0;
    //    }
    //    IndexG idx(0);
    //    mr_links_cpu.adaptiveMap.iterInit(idx);
    //    while (mr_links_cpu.adaptiveMap.iterNext(idx)) {
    //        mr_links_cpu.adaptiveMap(idx) = (md_links.adaptiveMap(idx) - pMin) * unitX;// float * double
    //        if (mr_links_cpu.adaptiveMap(idx)[0] >= max_xNew) // float vs double
    //            max_xNew = mr_links_cpu.adaptiveMap(idx)[0];
    //        if (mr_links_cpu.adaptiveMap(idx)[1] >= max_yNew)
    //            max_yNew = mr_links_cpu.adaptiveMap(idx)[1];
    //        if (mr_links_cpu.adaptiveMap(idx)[0] < min_xNew)
    //            min_xNew = mr_links_cpu.adaptiveMap(idx)[0];
    //        if (mr_links_cpu.adaptiveMap(idx)[1] < min_yNew)
    //            min_yNew = mr_links_cpu.adaptiveMap(idx)[1];
    //        if (DimP >= 3) {
    //            if (mr_links_cpu.adaptiveMap(idx)[2] >= max_zNew)
    //                max_zNew = mr_links_cpu.adaptiveMap(idx)[2];
    //            if (mr_links_cpu.adaptiveMap(idx)[2] < min_zNew)
    //                min_zNew = mr_links_cpu.adaptiveMap(idx)[2];
    //        }
    //    }
    //    cout << "max x,y,z New " << max_xNew << ", " << max_yNew << ", " << max_zNew << endl;
    //    cout << "min x,y,z New " << min_xNew << ", "  << min_yNew << ", " << min_zNew << endl;


    //    //qiao 2024 here build initial TSP tour
    //    mr_links_cpu.networkLinks.resize(nNodes, 1);
    //    mr_links_cpu.generate2ConectedLinks();
    //    mr_links_cpu.write("mrlinkTSP");
    //    mr_links_cpu.writeLinks("mrlinkTSP");
    //    numCityTraversed = 0;
    //    weightOriginal = 0;
    //    mr_links_cpu.fixedMap.resize(nNodes, 1);
    //    weightOriginal = mr_links_cpu.evaluateWeightOfTSP(dist, numCityTraversed);
    //    cout << "*******=== total weight changed coordinate mr_links_cpu : " << weightOriginal << ", Number city traversed: "<< numCityTraversed << endl;

    //    //! 2-opt sequetial
    //    md_links_firstSerial.resize(nNodes, 1);
    //    md_links_firstSerial.adaptiveMap.assign(md_links.adaptiveMap);
    //    md_links_firstSerial.networkLinks.assign(md_links.networkLinks);
    //    numCityTraversed = 0;
    //    weightOriginal = 0;
    //    md_links_firstSerial.fixedMap.resize(nNodes, 1);
    //    weightOriginal = md_links_firstSerial.evaluateWeightOfTSP(dist, numCityTraversed);
    //    cout << "*******=== total weight changed coordinate md_links_firstSerial : " << weightOriginal << ", Number city traversed: "<< numCityTraversed << endl;

    //    md_links_BestSerial.resize(nNodes, 1);
    //    md_links_BestSerial.adaptiveMap.assign(md_links.adaptiveMap);
    //    md_links_BestSerial.networkLinks.assign(md_links.networkLinks);
    //    numCityTraversed = 0;
    //    weightOriginal = 0;
    //    md_links_BestSerial.fixedMap.resize(nNodes, 1);
    //    weightOriginal = md_links_BestSerial.evaluateWeightOfTSP(dist, numCityTraversed);
    //    cout << "*******=== total weight changed coordinate md_links_BestSerial : " << weightOriginal << ", Number city traversed: "<< numCityTraversed << endl;


    /**********************begin opt  ****************************************/
    //! temp copy of md_links specialy for md_links_gpu
    md_links_firstPara.adaptiveMap.resize(nNodes, 1);
    md_links_firstPara.fixedMap.resize(nNodes, 1);
    md_links_firstPara.networkLinks.resize(nNodes, 1);
    md_links_firstPara.adaptiveMap.assign(md_links_cpu.adaptiveMap);
    md_links_firstPara.networkLinks.assign(md_links_cpu.networkLinks);
    md_links_firstPara.activeMap.resize(nNodes, 1);
    md_links_firstPara.grayValueMap.resize(nNodes, 1);
    md_links_firstPara.densityMap.resize(nNodes, 1);
    md_links_firstPara.minRadiusMap.resize(nNodes, 1);
    md_links_firstPara.optCandidateMap.resize(nNodes, 1);//qiao for optcandidates

    //record the best TSP tour obtained so for
    tspTourBestObtainedSoFar.resize(nNodes, 1);

    double beforekoptFirstPara = md_links_firstPara.evaluateWeightOfTSP(dist, numCityTraversed);
    cout << endl << ">>  before k-opt parallel with rocki: " << beforekoptFirstPara << endl;
    md_links_firstPara.fixedMap.resetValue(0);

    //! gpu tsp tour
    cudaSetDevice(0);
    md_links_gpu.gpuResize(nNodes, 1); // for rocki parallel one iteration 2-opt search with one 2-opt move
    md_links_gpu.optCandidateMap.gpuResize(nNodes, 1); //qiao for optcandidates
    md_links_firstPara.gpuCopyHostToDevice(md_links_gpu);

    //!copy opt possibilites
    md_links_gpu.nodeParentMap.gpuResize(numOptPossibilites, 1);
    md_links.nodeParentMap.gpuCopyHostToDevice(md_links_gpu.nodeParentMap);


    //    cudaSetDevice(1);
    //    md_links_gpu.gpuResize(nNodes, 1); // for rocki parallel one iteration 2-opt search with one 2-opt move
    //    md_links_gpu.optCandidateMap.gpuResize(nNodes, 1); //qiao for optcandidates
    //    md_links_firstPara.gpuCopyHostToDevice(md_links_gpu);

    //    //!copy opt possibilites
    //    md_links_gpu.nodeParentMap.gpuResize(numOptPossibilites, 1);
    //    md_links.nodeParentMap.gpuCopyHostToDevice(md_links_gpu.nodeParentMap);

    //    cudaSetDevice(0);


    //    //qiao only for test
    //    cout << "Test read opt possiblities on GPU " << endl;
    //    md_links.nodeParentMap.gpuCopyDeviceToHost(md_links_gpu.nodeParentMap);
    //    for(int i = 0; i < md_links.nodeParentMap.getWidth(); i ++)
    //    {
    //        cout << md_links.nodeParentMap[0][i] << " " ;

    //        if(md_links.nodeParentMap.getWidth() == md_links.nodeParentMap.getWidth() *8 && i%8==0 && i!= 0)
    //            cout << endl;
    //        else if(md_links.nodeParentMap.getWidth() == 2080 && i%10==0 && i!= 0)
    //            cout << endl;
    //        else if(md_links.nodeParentMap.getWidth() == 23220 && i%12==0 && i!= 0 )
    //            cout << endl;
    //    }

    //qiao here 4-opt does not need cmd or spiral search, delete them all.
    cout << "INITIALISATION DONE" << std::endl << std::endl;


}

template<std::size_t DimP, std::size_t DimCM>
void SolutionKOPT<DimP, DimCM>::clone(SolutionKOPT* sol)
{
    (*sol).fileData = (*this).fileData;
    (*sol).fileSolution = (*this).fileSolution;
    (*sol).fileStats = (*this).fileStats;

    (*sol).t0 = (*this).t0;
    (*sol).tf = (*this).tf;
    (*sol).x0 = (*this).x0;
    (*sol).xf = (*this).xf;

#ifdef CUDA_CODE
    (*sol).start = (*this).start;
    (*sol).stop = (*this).stop;
#endif
    (*sol).global_objectif = (*this).global_objectif;

    (*this).md_links_cpu.clone((*sol).md_links_cpu);

    // size numbers
    (*sol).pMin = (*this).pMin;
    (*sol).pMax = (*this).pMax;

    (*sol).initialize((*sol).md_links_cpu, (*sol).pMin, (*sol).pMax);

}

template<std::size_t DimP, std::size_t DimCM>
void SolutionKOPT<DimP, DimCM>::setIdentical(SolutionKOPT* sol)
{
    (*sol).global_objectif = (*this).global_objectif;

    (*this).mr_links_cpu.setIdentical((*sol).mr_links_cpu);
    (*this).mr_links_gpu.gpuSetIdentical((*sol).mr_links_gpu);

    (*sol).traceTSP = (*this).traceTSP;

}

template<std::size_t DimP, std::size_t DimCM>
void SolutionKOPT<DimP, DimCM>::initEvaluate()
{
    traceTSP.length = 0;//numeric_limits<double>::max();
    traceTSP.size = 0;//numeric_limits<double>::max();
    this->global_objectif = 0;//numeric_limits<double>::max();
}

template<std::size_t DimP, std::size_t DimCM>
double SolutionKOPT<DimP, DimCM>::evaluate()
{
    //-------------------------------------------------------------------------
    // Mise à jour positions vehicules, chemins, volumes
    //-------------------------------------------------------------------------
    initEvaluate();

    md_links_cpu.networkLinks.assign(tspTourBestObtainedSoFar); // copy best tour

    int numCityTraversed = 0;
    double bestTspTourLength = md_links_cpu.evaluateWeightOfTSP(distEuclidean, numCityTraversed);

    AMObjectives obj(0);
    BOp op;
    op.K_sumReduction(mr_links_cpu.objectivesMap, obj);
    traceTSP.size = numCityTraversed;
    traceTSP.length = bestTspTourLength;

    float PDB = float(traceTSP.length - traceTSP.optimumLength)/traceTSP.optimumLength  * 100;
    traceTSP.pdb = PDB;

    cout << "TSP tour size .................. " << numCityTraversed << endl;
    cout << "TSP tour LENGTH .................. " << bestTspTourLength << endl;
    cout << "TSP tour PDB .................. " << traceTSP.pdb << endl;


    // Calcul objectif global
    computeObjectif();

    return global_objectif;
}//evaluate


template<std::size_t DimP, std::size_t DimCM>
double SolutionKOPT<DimP, DimCM>::evaluateInit()
{
    //-------------------------------------------------------------------------
    // Mise à jour positions vehicules, chemins, volumes
    //-------------------------------------------------------------------------
    initEvaluate();

    int numCityTraversed = 0;
    double bestTspTourLength = md_links_cpu.evaluateWeightOfTSP(distEuclidean, numCityTraversed);

    AMObjectives obj(0);
    BOp op;
    op.K_sumReduction(mr_links_cpu.objectivesMap, obj);
    traceTSP.size = numCityTraversed;
    traceTSP.length = bestTspTourLength;

    float PDB = float(traceTSP.length - traceTSP.optimumLength)/traceTSP.optimumLength  * 100;
    traceTSP.pdb = PDB;

    cout << "TSP tour size .................. " << numCityTraversed << endl;
    cout << "TSP tour LENGTH .................. " << bestTspTourLength << endl;
    cout << "TSP tour PDB .................. " << traceTSP.pdb << endl;


    // Calcul objectif global
    computeObjectif();

    return global_objectif;
}//evaluate

/*!
 * \return valeur de la fonction objectif agregative
 */
template<std::size_t DimP, std::size_t DimCM>
double SolutionKOPT<DimP, DimCM>::computeObjectif(void)
{
    global_objectif = traceTSP.size;//traceTSP.length;

    return global_objectif;
}

/*!
 * \param best SolutionKOPT comparee
 * \return vrai si objectif de l'appelant (ie la SolutionKOPT courante) est inferieur ou egal a celui de la SolutionKOPT comparee
 */
template<std::size_t DimP, std::size_t DimCM>
bool SolutionKOPT<DimP, DimCM>::isBest(SolutionKOPT* best)
{
    bool res = false;

    if (computeObjectif() <= best->computeObjectif())
        res = true;

    return res;
}

/*!
 * \return vrai si SolutionKOPT admissible
 */
template<std::size_t DimP, std::size_t DimCM>
bool SolutionKOPT<DimP, DimCM>::isSolution()
{
    bool res = false;
    if (this->global_objectif <= 0)    {
        res = true;
    }
    return res;
}//isSolutionKOPT

