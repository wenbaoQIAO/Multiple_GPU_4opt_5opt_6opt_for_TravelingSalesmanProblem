#include "config/ConfigParamsCF.h"
#include "random_generator_cf.h"
#include "CalculateurSOM3D.h".h"
//#include "Multiout.h"

#include "SolutionSOM3D.h"

#define round(x) ((fabs(ceil(x) - (x)) < fabs(floor(x) - (x))) ? ceil(x) : floor(x))

#define EMST_PRINT_CM 1
#define EMST_SQUARED_CM 0

template<std::size_t DimP, std::size_t DimCM>
int SolutionSOM3D<DimP, DimCM>::cptInstance = 0;

template<std::size_t DimP, std::size_t DimCM>
void SolutionSOM3D<DimP, DimCM>::initialize(char* data, char* sol, char* stats)
{
    fileData = data;
    fileSolution = sol;
    fileStats = stats;

    initialize();
}

template<std::size_t DimP, std::size_t DimCM>
void SolutionSOM3D<DimP, DimCM>::initialize()
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
void SolutionSOM3D<DimP, DimCM>::initialize(NetLink& md_links, PointEuclidean& pMin, PointEuclidean& pMax)
{
    cout << "INITIALISATION START" << std::endl;

    size_t nNodes = 0;
    GLint w = md_links.adaptiveMap.getWidth();
    GLint h = md_links.adaptiveMap.getHeight();
    GLint d = md_links.adaptiveMap.getDepth();
    nNodes = w * h * d;
    cout << "num cities = " << nNodes  << endl;

    // wb.Q add case check
    if(nNodes == 0)
    {
        cout << "Error: no input available." << endl;
        return;
    }

    // Save original map
    adaptiveMapOriginal.gpuResize(w, h, d);
    md_links.adaptiveMap.gpuCopyHostToDevice(adaptiveMapOriginal);

    // Change coordinates
#if EMST_SQUARED_CM
    GLdouble min_x = pMin[0];
    GLdouble min_y = pMin[1];
    GLdouble max_x = pMax[0];
    GLdouble max_y = pMax[1];

    int _w = (GLint) ceil(max_x - min_x)+2;
    int _h = (GLint) ceil(max_y - min_y)+2;
    cout << "city_area_w , city_area_h " << _w << ", " << _h << endl;

    // wb.Q, chansfer the coordinate system
    float unitX = sqrt(nNodes) / _w;
    float unitY = sqrt(nNodes) / _h;
    cout << "unitX = " << unitX << endl;
    cout << "unitY = " << unitY << endl;
#else
    GLdouble wg = pMax[0] - pMin[0];
    GLdouble hg = pMax[1] - pMin[1];
    GLdouble dg = 0;
    if (DimP >= 3)
        dg = pMax[2] - pMin[2];

    //    if (wg == 0 && hg == 0 && dg = 0) {
    //        wg = hg = dg = 1;
    //    }

    cout << "max original x, y, z " << pMax[0] << ", " << pMax[1];
    if (DimP >= 3)
        cout << ", " << pMax[2];
    cout << endl;
    cout << "min original x, y, z " << pMin[0] << ", " << pMin[1];
    if (DimP >= 3)
        cout << ", " << pMin[2];
    cout << endl;
    cout << "city_area : " << wg << " * " << hg << " * " << dg << endl;

    if (wg == 0)
        wg = 1;
    if (hg == 0)
        hg = 1;
    GLdouble _wnd = sqrt(nNodes*wg/hg)*2;
    if (DimP >= 3) {
        if (dg != 0)
            _wnd = pow((nNodes*wg*wg)/(hg*dg), 0.333);
        else
            dg = 1;
    }
    //GLdouble _hnd = sqrt(nNodes*hg/wg);

    GLdouble _wndd = ceil(_wnd) + 1;
    //GLdouble _hndd = ceil(_hnd) + 1;

    GLdouble unitX = _wndd / wg;
    //GLdouble unitY = unitX;//_hndd / hg;
    cout << "unitX = " << unitX << endl;
    //cout << "unitY = " << unitY << endl;

    //    GLint _wn = (GLint) _wndd + 1;
    //    GLint _hn = (GLint) _hndd + 1;
#endif
    // Change coord system
    mr_links_cpu.resize(nNodes, 1);
    GLdouble max_xNew = -INFINITY;
    GLdouble max_yNew = -INFINITY;
    GLdouble min_xNew = +INFINITY;
    GLdouble min_yNew = +INFINITY;
    GLdouble max_zNew = -INFINITY;
    GLdouble min_zNew = +INFINITY;
    if (DimP < 3) {
        max_zNew = 0;
        min_zNew = 0;
    }
    IndexG idx(0);
    mr_links_cpu.adaptiveMap.iterInit(idx);
    while (mr_links_cpu.adaptiveMap.iterNext(idx)) {
        mr_links_cpu.adaptiveMap(idx) = (md_links.adaptiveMap(idx) - pMin) * unitX;// float * double
        if (mr_links_cpu.adaptiveMap(idx)[0] >= max_xNew) // float vs double
            max_xNew = mr_links_cpu.adaptiveMap(idx)[0];
        if (mr_links_cpu.adaptiveMap(idx)[1] >= max_yNew)
            max_yNew = mr_links_cpu.adaptiveMap(idx)[1];
        if (mr_links_cpu.adaptiveMap(idx)[0] < min_xNew)
            min_xNew = mr_links_cpu.adaptiveMap(idx)[0];
        if (mr_links_cpu.adaptiveMap(idx)[1] < min_yNew)
            min_yNew = mr_links_cpu.adaptiveMap(idx)[1];
        if (DimP >= 3) {
            if (mr_links_cpu.adaptiveMap(idx)[2] >= max_zNew)
                max_zNew = mr_links_cpu.adaptiveMap(idx)[2];
            if (mr_links_cpu.adaptiveMap(idx)[2] < min_zNew)
                min_zNew = mr_links_cpu.adaptiveMap(idx)[2];
        }
    }

    cout << "max x,y,z New " << max_xNew << ", " << max_yNew << ", " << max_zNew << endl;
    cout << "min x,y,z New " << min_xNew << ", "  << min_yNew << ", " << min_zNew << endl;

    mr_links_gpu.gpuResize(w, h);
    distanceMap_cpu.resize(w, h, d);
    distanceMap.gpuResize(w, h, d);

    mr_links_cpu.evtMap.resize(w, h, d);
    mr_links_gpu.evtMap.gpuResize(w, h, d);
    mr_links_cpu.nVisitedMap.resize(w, h, d);
    mr_links_gpu.nVisitedMap.gpuResize(w, h, d);
    mr_links_cpu.nodeParentMap.resize(w, h, d);
    mr_links_gpu.nodeParentMap.gpuResize(w, h, d);
    mr_links_cpu.nodeWinMap.resize(w, h, d);
    mr_links_gpu.nodeWinMap.gpuResize(w, h, d);
    mr_links_cpu.nodeDestMap.resize(w, h, d);
    mr_links_gpu.nodeDestMap.gpuResize(w, h, d);

    minDistMap_cpu.resize(w, h, d);
    minDistMap.gpuResize(w, h, d);
    stateMap_cpu.resize(w, h, d);
    stateMap.gpuResize(w, h, d);

    spiralSearchMap_cpu.resize(w, h, d);
    spiralSearchMap.gpuResize(w, h, d);

    mr_links_cpu.gpuCopyHostToDevice(mr_links_gpu);

    // Cellular matrix creation
    ExtentsCM ext(1);
#if EMST_SQUARED_CM
    int _wn = int(max_xNew - min_xNew) + 3;
    int _hn = int(max_yNew - min_yNew) + 3;
    ext[0] = (GLint) ceil(max_xNew - min_xNew);
    ext[1] = (GLint) ceil(max_yNew - min_yNew);
    if (DimCM >= 3)
        ext[2] = (GLint) ceil(max_zNew - min_zNew)+1;
    cout << "vgd area _w,_h " << _wn << ", " << _hn << endl;
#else
    ext[0] = (GLint) ceil(max_xNew - min_xNew)+1;
    ext[1] = (GLint) ceil(max_yNew - min_yNew)+1;
    if (DimCM >= 3)
        ext[2] = (GLint) ceil(max_zNew - min_zNew)+1;
    //    ext[0] = _wn;
    //    ext[1] = _hn;
#endif
    IndexCM pc;
    pc = ext / 2;
    int _R = g_ConfigParameters->levelRadius;
    vgd = ViewG(pc, ext, _R);

    Index<DimCM> extentsDual = vgd.getExtentsDual();
    Index<DimCM> extentsBase = vgd.getExtentsBase();
    Index<DimCM> extents = vgd.getExtents();

    cout << "vgd dual : "
         << extentsDual << endl
         << "vgd base : "
         << extentsBase << endl
         << "vgd low level : "
         << extents << endl;
    cout << "nNodes " << nNodes << ", sqrt(nNodes) " << sqrt(nNodes) << endl;

    cm_cpu.setViewG(vgd);
    cm_cpu.resize(extentsDual);
    //cm_cpu.K_initialize_cpu(vgd);

    // wb.Q 201906 add adaptive cellular partition
    cma_cpu.setViewG(vgd);
    cma_cpu.resize(extentsDual);
    // wb.Q initialize cellular memebers
    cma_cpu.g_dll.resize(w, h, d);
    cma_cpu.g_cellular.resize(extentsDual);
    cout << "qiao test: cma gdll size " << cma_cpu.g_dll.getWidth() << ", " << cma_cpu.g_dll.getHeight() << endl;
    cout << "qiao test: cma gcellular size " << cma_cpu.g_cellular.getWidth() << ", " << cma_cpu.g_cellular.getHeight() << endl;
    cout << "qiao test: cma size " << cma_cpu.getWidth() << ", " << cma_cpu.getHeight() << endl;

    cout << "CM RESIZE DONE " << cm_cpu.length_in_bytes << " " << cm_cpu.length << endl;
    cout << "CM Adaptive Size RESIZE DONE " << cma_cpu.length_in_bytes << " " << cma_cpu.length << endl;
    cout << "CM size w, h, d " << cma_cpu.getWidth() << ", " << cma_cpu.getHeight() << ", " << cma_cpu.getDepth() << endl;

    // Cellular matrix initialisations
    cm_gpu.setViewG(vgd);
    cm_gpu.gpuResize(extentsDual);
    cm_gpu.K_initialize(vgd);

    // wb.Q 201906 add adaptive cellular partition
    cma_gpu.setViewG(vgd);
    cma_gpu.gpuResize(extentsDual);
    cma_gpu.K_initialize(vgd);

    // wb.Q change cellular partition into 1D array index, build dll with size cellular + input point
    IndexCM cm_boundP(cma_cpu.getWidth()-1, cma_cpu.getHeight()-1, cma_cpu.getDepth()-1);
    GLint cm_boundOffset = cma_cpu.compute_offset(cm_boundP);// last point in cellular index
    cout << "cm bound offset " << cm_boundOffset << endl;
    IndexCM cm_boundPCindex = cma_cpu.back_offset(cm_boundOffset);
    cout << "cm bound back offset origin " << cm_boundOffset << endl;
    cout << "cm bound back offset origin coord " << cm_boundPCindex[0] << ", " << cm_boundPCindex[1] << ", " << cm_boundPCindex[2] << endl;



    // wb.Q each cellula has one dll for all cities
    //  GLint celluarDllSize = w * h * d + cm_boundPC;
    cma_cpu.g_dll.resize(w, h, d);
    cma_cpu.g_cellular.resize(extentsDual);

    cma_gpu.g_dll.gpuResize(w, h, d); // 1D ddll
    cma_gpu.g_cellular.gpuResize(extentsDual);

    // wb.Q init value of cma_gpu.g_dll, g_cellular
    cma_gpu.g_dll.gpuResetValue(-1);
    cma_gpu.g_cellular.gpuResetValue(-1);


    cout << "CM GPU RESIZE DONE " << cm_gpu.length_in_bytes << " " << cm_gpu.length << endl;
    cout << "CM Adaptive Size GPU RESIZE DONE " << cma_gpu.length_in_bytes << " " << cma_gpu.length << endl;

    iteration = 0;

    // wb.Q 2019/08 add initialize 3DSOM
    // wb.Q 201908 initialize md_links for 3D tsp
    md_links.networkLinks.resize(w, h, d);
    cout << "qiao test 1, initialize TSP solution. " << endl;

    //initial original tour of tsp distance, TSP is only two connected;
    md_links.generate2ConectedLinks();
    md_links.writeLinks("originalTSP");
    cout << "qiao test, initialize TSP solution. " << endl;

    double weightOriginal = md_links.evaluateWeightOfTSP();
    cout << "*******=== total weight originalTSP md_links : " << weightOriginal << endl;
//    testTermination();

    // qiao to do: here mr_links should be 2-connected to test SOM, do I need to double the size of mr_links?

    // wb.Q SOM init parameters
    paramSom.readParameters("som_op_tsp");
    cout << "qiao test som para " << paramSom.nGene << ", alphaInitial " << paramSom.alphaInitial  << endl;

    // qiao todo: som initialize memebeers



    // Initialze mr_links attributes
    som3dOp.gpuResetValue(mr_links_gpu);
    som3dOp.cpuResetValue(mr_links_cpu);

    // Initialize disjoint set structure
    som3dOp.K_initDisjointSet(mr_links_gpu.disjointSetMap);

    som3dOp.K_clearLinks(mr_links_gpu.networkLinks);

    distanceMap.gpuResetValue(HUGE_VAL);
    PointCoord pInitial(-1);
    mr_links_gpu.correspondenceMap.gpuResetValue(pInitial);

    // Kernel Time init
    time_next_closest = 0;
    time_find_pair = 0;
    time_connect_union = 0;
    time_refreshCm = 0;
    time_terminate = 0;

    cm_gpu.K_clearCells();
    cma_gpu.K_clearCells();


    //som3dOp.K_refreshCell(cm_gpu, mr_links_gpu.adaptiveMap);


#if CELLULAR_ADAPTIVE
    // Spiral search Grid initialisation
    som3dOp.K_initializeSpiralSearch(cma_gpu,
                                       mr_links_gpu.adaptiveMap,
                                       spiralSearchMap);

    cout << "CMA GPU K_initializeSpiralSearch DONE" << endl;
#if EMST_PRINT_CM

    // wb.Q cma.search traverse the dll of each cell, augment the size and current postion
    // class function of cellular
    cma_gpu.K_searchCMA();
    cma_cpu.gpuCopyDeviceToHost(cma_gpu);
    cout << "CMA gpuCopyDeviceToHost DONE" << endl;

    int numNode = 0;
    int maxSize = 0;
    IndexCM idxcma(0);
    cma_cpu.iterInit(idxcma);
    while (cma_cpu.iterNext(idxcma)) {
        //                if (cma_cpu(idxcma).size > 0)
        //                {
        //                    cout << "cma_cpu " <<  idxcma ;
        //                    cout << " " << cma_cpu(idxcma).size << endl;
        //                }
        numNode += cma_cpu(idxcma).size;
        maxSize = (cma_cpu(idxcma).size > maxSize) ? cma_cpu(idxcma).size : maxSize;
    }
    cout << "=== check num of nodes inserted into cma: " << numNode << " max cell size: "<< maxSize << endl;

#endif

#else
    // Spiral search Grid initialisation
    som3dOp.K_initializeSpiralSearch(cm_gpu,
                                       mr_links_gpu.adaptiveMap,
                                       spiralSearchMap);

    cout << "CM GPU K_initializeSpiralSearch DONE" << endl;
#if EMST_PRINT_CM
    cm_cpu.gpuCopyDeviceToHost(cm_gpu);
    cout << "CM gpuCopyDeviceToHost DONE" << endl;
    int numNode = 0;
    int maxSize = 0;
    IndexCM idxcm(0);
    cm_cpu.iterInit(idxcm);
    while (cm_cpu.iterNext(idxcm)) {
        //        if (cm_cpu(idxcm).size > 0) {
        //            cout << "cmd_cpu " <<  idxcm << endl;
        //            cout << " " << cm_cpu(idxcm).size << endl;
        //        }
        numNode += cm_cpu(idxcm).size;
        maxSize = (cm_cpu(idxcm).size > maxSize) ? cm_cpu(idxcm).size : maxSize;
    }
    cout << "=== check num of nodes inserted into cmd: " << numNode << " max cell size: "<< maxSize << endl;


#endif
#endif

    cout << "INITIALISATION DONE" << std::endl;

}

template<std::size_t DimP, std::size_t DimCM>
void SolutionSOM3D<DimP, DimCM>::clone(SolutionSOM3D* sol)
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
void SolutionSOM3D<DimP, DimCM>::setIdentical(SolutionSOM3D* sol)
{
    (*sol).global_objectif = (*this).global_objectif;

    (*this).mr_links_cpu.setIdentical((*sol).mr_links_cpu);
    (*this).mr_links_gpu.gpuSetIdentical((*sol).mr_links_gpu);

    (*sol).traceParallel3DSom = (*this).traceParallel3DSom;

}

template<std::size_t DimP, std::size_t DimCM>
void SolutionSOM3D<DimP, DimCM>::initEvaluate()
{
    traceParallel3DSom.length = 0;//numeric_limits<double>::max();
    traceParallel3DSom.size = 0;//numeric_limits<double>::max();
    this->global_objectif = 0;//numeric_limits<double>::max();
}

template<std::size_t DimP, std::size_t DimCM>
double SolutionSOM3D<DimP, DimCM>::evaluate()
{
    //-------------------------------------------------------------------------
    // Mise à jour positions vehicules, chemins, volumes
    //-------------------------------------------------------------------------
    initEvaluate();

    som3dOp.K_evaluate_ST(mr_links_gpu.networkLinks, adaptiveMapOriginal, mr_links_gpu.objectivesMap);
    mr_links_cpu.objectivesMap.gpuCopyDeviceToHost(mr_links_gpu.objectivesMap);
    AMObjectives obj(0);
    BOp op;
    op.K_sumReduction(mr_links_cpu.objectivesMap, obj);
    traceParallel3DSom.size = obj[obj_distr]/2;
    traceParallel3DSom.length = obj[obj_length]/2;

    cout << "EMST SIZE .................. " << traceParallel3DSom.size << endl;
    cout << "EMST LENGTH .................. " << traceParallel3DSom.length << endl;
    // Calcul objectif global
    computeObjectif();

    return global_objectif;
}//evaluate

/*!
 * \return valeur de la fonction objectif agregative
 */
template<std::size_t DimP, std::size_t DimCM>
double SolutionSOM3D<DimP, DimCM>::computeObjectif(void)
{
    global_objectif = traceParallel3DSom.size;//traceParallel3DSom.length;

    return global_objectif;
}

/*!
 * \param best SolutionSOM3D comparee
 * \return vrai si objectif de l'appelant (ie la SolutionSOM3D courante) est inferieur ou egal a celui de la SolutionSOM3D comparee
 */
template<std::size_t DimP, std::size_t DimCM>
bool SolutionSOM3D<DimP, DimCM>::isBest(SolutionSOM3D* best)
{
    bool res = false;

    if (computeObjectif() <= best->computeObjectif())
        res = true;

    return res;
}

/*!
 * \return vrai si SolutionSOM3D admissible
 */
template<std::size_t DimP, std::size_t DimCM>
bool SolutionSOM3D<DimP, DimCM>::isSolution()
{
    bool res = false;
    if (this->global_objectif <= 0)    {
        res = true;
    }
    return res;
}//isSolutionSOM3D

