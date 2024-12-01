#include "config/ConfigParamsCF.h"
#include "random_generator_cf.h"
#include "SolutionKOPT.h"
#include "distance_functors.h"
#include "CalculateurEMST.h"

/** Operateurs de changement de SolutionKOPT courante.
 *
 */
#define FULL_GPU 1
#define FULL_GPU_FIND_MIN1 1
#define FULL_GPU_FIND_MIN2 1
#define FULL_GPU_CGU 1
#define FULL_GPU_FLATTENING 1
#define MAXNUMRUNS 2
#define CPUEXECUTE 1
#define NUMRUNSLIMIT 2000
#define FIRST 1

#define EMST_DETECT_CYCLE  0
#define EMST_FIND_MIN_PAIR_LIST 1// Distributed broadcast or distributed linked list

template<std::size_t DimP, std::size_t DimCM>
void SolutionKOPT<DimP, DimCM>::initConstruct()
{
}//initConstruct

/** Construction Sequentielle
 */
template<std::size_t DimP, std::size_t DimCM>
void SolutionKOPT<DimP, DimCM>::constructSolutionSeq()
{
    cout << "CONSTRUCTION SEQUENTIELLE ..." << endl;
    int nNodes = mr_links_cpu.adaptiveMap.getWidth();

    int iteration = 0;// maximum iterations

    int radiusSearchCells = 0;
    g_ConfigParameters->readConfigParameter("test_2opt", "radiusSearchCells", radiusSearchCells);

    float gpuTimingKernels = 0;
    float mstTotalTimeFrequen = 0;

    cout << "CONSTRUCTION done" << endl;
}

/*!
 * \return vrai si l'operateur est applique selon choix aleatoire,
 *  faux si l'operateur n'est pas applique
 */
template<std::size_t DimP, std::size_t DimCM>
bool SolutionKOPT<DimP, DimCM>::operator_1() {
    bool ret = true;


    global_objectif = computeObjectif();

    return ret;
}//operator_1

/*!
 * \return vrai si l'operateur est applique selon choix aleatoire,
 *  faux si l'operateur n'est pas applique
 */
template<std::size_t DimP, std::size_t DimCM>
bool SolutionKOPT<DimP, DimCM>::operator_2() {
    bool noUsed = true;

    return noUsed;
}//operator_1

template<std::size_t DimP, std::size_t DimCM>
bool SolutionKOPT<DimP, DimCM>::generateNeighbor()
{
    int no_op = 0;
    double totalCapacity = 0;
    while (no_op < g_ConfigParameters->probaOperators.size()) {
        totalCapacity += g_ConfigParameters->probaOperators[no_op];
        no_op += 1;
    }

    // Tirage aleatoire par "roulette"
    double d = random_cf::aleat_double(0, totalCapacity);
    cout << "RANDOM VALUE OPERATOR !!!!!!!!!!!!!!!! " << d << endl;
    // Determiner no d'operateur
    no_op = -1;
    double t_sise = 0;
    int size = g_ConfigParameters->probaOperators.size();
    for (int k = 0; k < size; k++) {
        t_sise += g_ConfigParameters->probaOperators[k];
        //cout << "probaOperators " << g_ConfigParameters->probaOperators[k] << endl;
        if (d < t_sise) {
            no_op = k;
            break;
        }
    }
    if (no_op == -1)
        cout << "PB TIRAGE OPERATEUR !!! " << g_ConfigParameters->probaOperators.size() << endl;
    else
        cout << "Choix operator : " << no_op << endl;

    // Appliquer l'operateur ...
    if (applyOperator(no_op))
    {
        this->computeObjectif();
        if (this->global_objectif < 0)
        {
            cout << "ERROR!!! OPERATEUR num." << no_op << " A DONNE OBJECTIF NEGATIF : " << this->global_objectif << endl;
        }
    }
    return true;
}//generateNeighbor

template<std::size_t DimP, std::size_t DimCM>
bool SolutionKOPT<DimP, DimCM>::applyOperator(int i)
{
    bool ret = false;
    switch (i)
    {
    case 0:
        break ;
    case 1:
        break ;
    case 2:
        break ;
    case 3:
        break ;
    case 4:
        break ;
    case 5:
        break ;
    case 6:
        break ;
    case 7:
        break ;
    }
    ret = operator_1();
    return ret;
}

template<std::size_t DimP, std::size_t DimCM>
int SolutionKOPT<DimP, DimCM>::nbrOperators() const
{
    return g_ConfigParameters->probaOperators.size();
}

//! \brief Run et activate
//!
//wb.Q 202206 implement GPU parallel 2-opt
template<std::size_t DimP, std::size_t DimCM>
void SolutionKOPT<DimP, DimCM>::run() {

    int nCity = md_links_cpu.adaptiveMap.getWidth();
    int numRuns = 0;

    int maxOptExecuPerRun = 0;
    int numOptimizedTotal = 0;
    vector<int> vectorNumOptExecuted;
    vector<float> vectorPDB;
    float timeGpuKernel = 0;
    float timeGpuH2D = 0;
    float timeGpuD2H = 0;
    float timeGpuTotal = 0;
    float timeCpuKey = 0;
    float pdb2optEatFirstPara = 0;
    float timeRefreshTour = 0;
    float timeSelect = 0;
    float timeExecute = 0;
    int numInter = 1;
    // trace maxtimeGPUone2-OoptRun
    float maxtimeGpuOptSearch = 0;

    // outfile timeline
    string fileTimePerRun = "Results_"; //str
    fileTimePerRun.append("TimePer2optRun.txt");
    ofstream outfileTimePerRunRun;
    outfileTimePerRunRun.open(fileTimePerRun);

    float evaLastRun = 0;
    float percentageImprove = 999999;

    //! prepare the pre-ordered link + coordinates
    Grid<doubleLinkedEdgeForTSP> linkCoordTourCpu;
    linkCoordTourCpu.resize(nCity, 1);
    Grid<doubleLinkedEdgeForTSP> linkCoordTourGpu;
    linkCoordTourGpu.gpuResize(nCity,1);


    unsigned long maxChecks = nCity*(nCity - 1) / 2; // total number of checks for 2-opt
    unsigned int iter = maxChecks / (BLOCKSIZE * GRIDSIZE);

    //wb.Q 2019 add case detection
    if(SolutionKOPT<DimP, DimCM>::md_links_cpu.adaptiveMap.width == 0)
    {
        cout << "Error: no input available." << endl;
        return;
    }
    else
    {
        //wb.Q 2024 rocki 2-opt
        cout << "TSP tour optimum = " << optimum << endl;
        while (numRuns < 2000  && percentageImprove > 0 )
        {
            activateRocki2opt(numRuns, nCity, maxChecks, iter,
                              maxOptExecuPerRun, numOptimizedTotal,
                              timeGpuKernel, timeGpuH2D, timeGpuD2H,
                              timeGpuTotal, timeCpuKey, vectorNumOptExecuted, vectorPDB,
                              timeRefreshTour, timeSelect, timeExecute, maxtimeGpuOptSearch,
                              outfileTimePerRunRun, evaLastRun, percentageImprove,
                              linkCoordTourCpu,linkCoordTourGpu);

            //            if (g_ConfigParameters->traceActive) {
            //                evaluate();
            //                writeStatisticsToFile(iteration);
            //            }
        }
    }// end activateRocki

    //! free gpu memory
    linkCoordTourGpu.gpuFreeMem();


    //! count time gpu total
    timeGpuTotal += timeGpuH2D + timeGpuD2H + timeGpuKernel;


    //! mean time trace
    timeRefreshTour = timeRefreshTour / numRuns;
    timeSelect = timeSelect / numRuns;
    timeExecute = timeExecute / numRuns;

    // close outfile
    outfileTimePerRunRun.close();


}// end run

// qiao 2024 add operators to GPU parallel 2-opt and massive variable 2-opt moves on global tour
template<std::size_t DimP, std::size_t DimCM>
bool SolutionKOPT<DimP, DimCM>::activateRocki2opt(int& numRuns, int nCity, double maxChecks2opt, double iter,
                                                  int& maxOptExecuPerRun, int& numOptimizedTotal,
                                                  float& timeGpuKernel, float& timeGpuH2D, float& timeGpuD2H,
                                                  float& timeGpuTotal, float& timeCpuKey, vector<int>& vectorNumOptExecuted, vector<float>& vectorPDB,
                                                  float& timeRefresh, float& timeSelect, float& timeExecute,
                                                  float &maxtimeGpuOptSearch, ofstream &outfileTimePerRunRun,
                                                  float& evaLastRun, float& percentageImprove, Grid<doubleLinkedEdgeForTSP> &linkCoordTourCpu,
                                                  Grid<doubleLinkedEdgeForTSP>& linkCoordTourGpu)
{
    cout << endl << "Enter 2-opt iteration=: " << numRuns << endl;
    bool ret = true;
    numRuns ++;

    int numOptimizedOneRun = 0;
    int numCityTraversed = 0;

    //!timing runing time on CPU
    //    __int64 CounterStart = 0;
    //    double pcFreq = 0.0;
    float elapsedTime2opt = 0;
    float pdbOneRun = 0;

    //! random starting point
    int ps_random = randomNum(0, nCity);
    PointCoord ps(ps_random, 0);
    cout << "PS [0] " << ps[0] << endl;

    //! clean md_links_firstPara before mark tour ordering
    md_links_firstPara.activeMap.resetValue(0);
    md_links_firstPara.densityMap.resetValue(initialPrepareValue);// densityMap stores node3
    md_links_firstPara.grayValueMap.resetValue(0);// clean orders
    md_links_firstPara.minRadiusMap.resetValue(initialPrepareValue);//  minRadiusMap stores the changeLinks position

    //!timing runing time on CPU
    __int64 CounterStart = 0;
    double pcFreq = 0.0;
    StartCounter(pcFreq, CounterStart);

    //! mark tour orientation from random starting point ps, index of linkCoordTourCpu should correspond to index of gray value map
    md_links_firstPara.markNetLinkSequenceReloadRoutCoord(ps, numRuns%2, 0, linkCoordTourCpu);// every ps check its two directions

    // end time cpu
    double timeCpuRefreshTour = GetCounter(pcFreq, CounterStart);
    cout << "Time:: Refresh tour order: " << timeCpuRefreshTour << endl;


    // time for GPU memcp HD
    float elapsedTime2optHD = 0;
    cudaEvent_t startHD, stopHD;
    cudaEventCreate(&startHD);
    cudaEventCreate(&stopHD);
    cudaEventRecord(startHD, 0);

    // copy tour ordering to gpu, clean gpu network links
    md_links_firstPara.grayValueMap.gpuCopyHostToDevice(md_links_gpu.grayValueMap);
    linkCoordTourCpu.gpuCopyHostToDevice(linkCoordTourGpu);

    cudaEventRecord(stopHD, 0);
    cudaEventSynchronize(stopHD);
    cudaEventElapsedTime(&elapsedTime2optHD, startHD, stopHD);
    cudaEventDestroy(startHD);
    cudaEventDestroy(stopHD);
    cout << "Time:: memcp H to D tour order : " <<  elapsedTime2optHD << endl;

    md_links_gpu.densityMap.gpuResetValue(initialPrepareValue);// use for node3
    md_links_gpu.minRadiusMap.gpuResetValue(initialPrepareValue); // use for local min change


    //qiao only for test
    cout << "Warning: maxChecks2opt= " << maxChecks2opt << endl;

    // cuda timer
    double time = 46;
    double *d_time;


    //divide and conquer
    double maxChecksoptDivide = 1.27719e+11;


    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);


    double iterDivide = (double)maxChecksoptDivide /(double) (BLOCKSIZE * GRIDSIZE);
    if(maxChecks2opt < maxChecksoptDivide)
        iterDivide = 1;
    double maxStride = (double) maxChecks2opt /  (double)maxChecksoptDivide;
    if(maxStride < 1)
        maxStride = 0;
    for(double iStride = 0; iStride < maxStride+1; iStride++ )
    {

        K_oneThreadOne2opt_Rocki_iterStride(md_links_gpu, linkCoordTourGpu, maxChecks2opt, maxChecksoptDivide, iterDivide, iStride);

        cudaDeviceSynchronize();

        cout << "Inner one time " << iStride << endl << endl;
    }


    //    //! WB.Q parallel check exhaustive 2-opt along the tour for each edge
    //    //                K_oneThreadOne2opt(nn_gpu_links, linkCoordTourGpu, maxChecks, iter);// wb.q backup this kernel use blocking of shared memory
    //    K_oneThreadOne2opt_RockiSmall(md_links_gpu, linkCoordTourGpu, maxChecks2opt, iter);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime2opt, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // find the maximum gpu time for a parallel 2-opt run
    cout << "Time:: GPU side one 2-opt runtime : " << elapsedTime2opt << endl;

    if(maxtimeGpuOptSearch < elapsedTime2opt){
        maxtimeGpuOptSearch = elapsedTime2opt;
    }


    //! sequentially select non-interacted 2-exchanges
    float elapsedTime2opt_DH2 = 0;
    cudaEvent_t startDH2, stopDH2;
    cudaEventCreate(&startDH2);
    cudaEventCreate(&stopDH2);
    cudaEventRecord(startDH2, 0);

    md_links_firstPara.densityMap.gpuCopyDeviceToHost(md_links_gpu.densityMap);// node3

    cudaEventRecord(stopDH2, 0);
    cudaEventSynchronize(stopDH2);
    cudaEventElapsedTime(&elapsedTime2opt_DH2, startDH2, stopDH2);
    cudaEventDestroy(startDH2);
    cudaEventDestroy(stopDH2);
    cout << "Time:: memcp D to H 2-opt candidates: " << elapsedTime2opt_DH2 << endl;

    //! clean for mark non-interacted 2-opt
    md_links_firstPara.activeMap.resetValue(0); // for nodes possessing non interacted 2opt
    md_links_firstPara.fixedMap.resetValue(0); // for nodes in stackB


    //!timing runing time on CPU
    CounterStart = 0;
    pcFreq = 0.0;
    StartCounter(pcFreq, CounterStart);


    //! select and execute non-interacted 2-exchanges
    md_links_firstPara.selectNonIteracted2ExchangeRocki(ps);

    // end time cpu
    double timeCpuSelectNonItera = GetCounter(pcFreq, CounterStart);
    cout << "Time:: select non intera 2-opt: " << timeCpuSelectNonItera << endl;

    double timeCpuExecuteNonItera = 0;
    float timeGpuExecute = 0;
    float elapsedTime2optHD2 = 0;
    float elapsedTime2optHD3 = 0;
    float elapsedTime2opt_DH = 0;
    float elapsedTime2opt_execute = 0;

#if CPUEXECUTE
    //!timing runing time on CPU
    CounterStart = 0;
    pcFreq = 0.0;
    StartCounter(pcFreq, CounterStart);

    md_links_firstPara.executeNonInteract2optOnlyNode3(numOptimizedOneRun);

    // end time cpu
    timeCpuExecuteNonItera = GetCounter(pcFreq, CounterStart);
    cout << "Time:: CPU execute non-intera 2-opt: " << timeCpuExecuteNonItera << endl;

    cudaEvent_t startHD2, stopHD2;
    cudaEventCreate(&startHD2);
    cudaEventCreate(&stopHD2);
    cudaEventRecord(startHD2, 0);

    md_links_firstPara.networkLinks.gpuCopyHostToDevice(md_links_gpu.networkLinks);
    errorCheckCudaThreadSynchronize();

    cudaEventRecord(stopHD2, 0);
    cudaEventSynchronize(stopHD2);
    cudaEventElapsedTime(&elapsedTime2optHD3, startHD2, stopHD2);
    cudaEventDestroy(startHD2);
    cudaEventDestroy(stopHD2);
    cout << "Time:: memcp H to D networkLinks : " <<  elapsedTime2optHD3 << endl;

#else

    //! copy activeMap (selected 2-exchanges) to device HD
    cudaEvent_t startHD2, stopHD2;
    cudaEventCreate(&startHD2);
    cudaEventCreate(&stopHD2);
    cudaEventRecord(startHD2, 0);

    md_links_firstPara.activeMap.gpuCopyHostToDevice(md_links_gpu.activeMap);

    cudaEventRecord(stopHD2, 0);
    cudaEventSynchronize(stopHD2);
    cudaEventElapsedTime(&elapsedTime2optHD2, startHD2, stopHD2);
    cudaEventDestroy(startHD2);
    cudaEventDestroy(stopHD2);
    cout << "memcp H to D activeValueMap : " <<  elapsedTime2optHD2 << endl;

    //! kernel execute selected 2-exchanges
    // cuda timer
    cudaEvent_t start2, stop2;
    cudaEventCreate(&start2);
    cudaEventCreate(&stop2);
    cudaEventRecord(start2, 0);

    K_executeNonItera2ExchangeOnlyWithNode3(md_links_gpu);

    cudaEventRecord(stop2, 0);
    cudaEventSynchronize(stop2);
    cudaEventElapsedTime(&elapsedTime2opt_execute, start2, stop2);
    cudaEventDestroy(start2);
    cudaEventDestroy(stop2);

    cout << "gpu search 2opt in parallel " << elapsedTime2opt << endl;
    cout << " gpu execute non intera 2-exchange time " << elapsedTime2opt_execute << endl;

    //        //! copy new tour to host DH

    cudaEvent_t startDH, stopDH;
    cudaEventCreate(&startDH);
    cudaEventCreate(&stopDH);
    cudaEventRecord(startDH, 0);

    md_links_firstPara.networkLinks.gpuCopyDeviceToHost(md_links_gpu.networkLinks);

    cudaEventRecord(stopDH, 0);
    cudaEventSynchronize(stopDH);
    cudaEventElapsedTime(&elapsedTime2opt_DH, startDH, stopDH);
    cudaEventDestroy(startDH);
    cudaEventDestroy(stopDH);
    cout << "memcp device to host networklinks " << elapsedTime2opt_DH << endl;

#endif


    timeGpuExecute =  elapsedTime2optHD2 + elapsedTime2opt_execute + elapsedTime2opt_DH ;
    cout << "Time:: timeGpu Execute " << timeGpuExecute << endl;

    //! evaluation to stop
    numOptimizedOneRun = testGridNum<int>(md_links_firstPara.activeMap);
    float evaCurrentRun = md_links_firstPara.evaluateWeightOfTSP(dist, numCityTraversed);
    cout << "Evaluate:: After " << numRuns << "'th run, evaluate tsp length =  " << evaCurrentRun << endl;
    cout << "Evaluate:: In this run, num of 2-exchange been executed: "  <<  numOptimizedOneRun << endl;

    float evaActualLength = md_links_firstPara.evaluateWeightOfTSP(distEuclidean, numCityTraversed);
    cout << "Evaluate:: After " << numRuns << "'th run, evaluate tsp length =  " << evaActualLength << endl;

    //statistic pdb
    if(optimum > 1)
    {
        float evaCurrentPDB = md_links_firstPara.evaluateWeightOfTSP(dist, numCityTraversed);
        pdbOneRun = (evaCurrentPDB - optimum)*100/optimum;
    }

    if(numRuns == 1){
        timeGpuH2D += elapsedTime2optHD + elapsedTime2optHD2 + elapsedTime2optHD3;
        timeGpuD2H += elapsedTime2opt_DH + elapsedTime2opt_DH2;
        timeGpuKernel += elapsedTime2opt + elapsedTime2opt_execute;
        timeCpuKey += timeCpuRefreshTour + timeCpuSelectNonItera + timeCpuExecuteNonItera;

        numOptimizedTotal += numOptimizedOneRun;
        if(numOptimizedOneRun > maxOptExecuPerRun)
            maxOptExecuPerRun = numOptimizedOneRun; // trace max optimized 2opt per run
        if(numOptimizedOneRun > 0)
            vectorNumOptExecuted.push_back(numOptimizedOneRun);

        // trace pdb one run
        vectorPDB.push_back(pdbOneRun);
        outfileTimePerRunRun << timeCpuKey + timeGpuKernel << " " << endl;

        //! registrer length of the first run
        evaLastRun = evaCurrentRun;
        //        continue;
    }
    else {
        percentageImprove = ((evaLastRun - evaCurrentRun)*100);
    }


    if(percentageImprove > 0){
        timeGpuH2D += elapsedTime2optHD + elapsedTime2optHD2;
        timeGpuD2H += elapsedTime2opt_DH + elapsedTime2opt_DH2;
        timeGpuKernel += elapsedTime2opt + elapsedTime2opt_execute;
        timeCpuKey += timeCpuRefreshTour + timeCpuSelectNonItera + timeCpuExecuteNonItera;
        evaLastRun = evaCurrentRun;

        numOptimizedTotal += numOptimizedOneRun;
        if(numOptimizedOneRun > maxOptExecuPerRun)
            maxOptExecuPerRun = numOptimizedOneRun;
        if(numOptimizedOneRun > 0)
            vectorNumOptExecuted.push_back(numOptimizedOneRun);
        // trace pdb one run
        vectorPDB.push_back(pdbOneRun);
        outfileTimePerRunRun << timeCpuKey +  timeGpuKernel << " " << endl;
    }
    else{
        numRuns -= 1; // the last run does not optimized the tour
    }

    //test
    cout << "Percentage improve " << percentageImprove << endl << endl;


    // count time refresh
    timeRefresh += (float)timeCpuRefreshTour;
    timeSelect += (float)timeCpuSelectNonItera;
#if CPUEXECUTE
    timeExecute += (float)timeCpuExecuteNonItera;
#else
    timeExecute += timeGpuExecute;
#endif

    return ret;
}//end 2opt




//! \brief Run et activate
//!
//wb.Q 202206 implement sequential 2-opt
template<std::size_t DimP, std::size_t DimCM>
void SolutionKOPT<DimP, DimCM>::runSequtial2opt() {

    int nCity = md_links_cpu.adaptiveMap.getWidth();
    int numRuns = 0;

    int maxOptExecuPerRun = 0;
    int numOptimizedTotal = 0;
    vector<int> vectorNumOptExecuted;
    vector<float> vectorPDB;

    float timeCpuKey = 0;
    float pdb2optEatFirstPara = 0;
    float timeRefreshTour = 0;
    float timeSearch = 0;
    float timeSelect = 0;
    float timeExecute = 0;
    int numInter = 1;
    // trace maxtimeGPUone2-OoptRun
    float maxtimeCpu2optSearch = 0;

    // outfile timeline
    string fileTimePerRun = "Results_"; //str
    fileTimePerRun.append("TimePer2optRun.txt");
    ofstream outfileTimePerRunRun;
    outfileTimePerRunRun.open(fileTimePerRun);

    // outfile pdbline
    string filePdbPerRun = "Results_"; //str
    filePdbPerRun.append("PdbPer2optRun.txt");
    ofstream outfilePdbPerRunRun;
    outfilePdbPerRunRun.open(filePdbPerRun);

    // outfile pdbline
    string fileSearchTimePerRun = "Results_"; //str
    fileSearchTimePerRun.append("searchTimePer2optRun.txt");
    ofstream outfileSearchTimePerRunRun;
    outfileSearchTimePerRunRun.open(fileSearchTimePerRun);

    outfileTimePerRunRun << 0 << " " << endl;
    outfilePdbPerRunRun << 1143.63 << " " << endl;
    outfileSearchTimePerRunRun << 0 << endl;

    float evaLastRun = 0;
    float percentageImprove = 999999;

    //! prepare the pre-ordered link + coordinates
    Grid<doubleLinkedEdgeForTSP> linkCoordTourCpu;
    linkCoordTourCpu.resize(nCity, 1);

    //wb.Q 2019 add case detection
    if(SolutionKOPT<DimP, DimCM>::md_links_cpu.adaptiveMap.width == 0)
    {
        cout << "Error: no input available." << endl;
        return;
    }
    else
    {
        //wb.Q 2024 rocki 2-opt
        cout << "TSP tour optimum = " << optimum << endl;
        while (numRuns < 2000  && percentageImprove > 0 )
        {
            activateSequential2opt(numRuns, nCity,
                                   maxOptExecuPerRun, numOptimizedTotal, timeCpuKey, vectorNumOptExecuted, vectorPDB,
                                   timeRefreshTour,timeSearch, timeSelect, timeExecute, maxtimeCpu2optSearch,
                                   outfileTimePerRunRun, outfilePdbPerRunRun,outfileSearchTimePerRunRun, evaLastRun, percentageImprove,
                                   linkCoordTourCpu);

            if (g_ConfigParameters->traceActive) {
                evaluate();
                writeStatisticsToFile(numRuns, "2optimalTour");
            }
        }
    }// end activateRocki



    //! mean time trace
    timeRefreshTour = timeRefreshTour / numRuns;
    timeSearch = timeSearch / numRuns;
    timeSelect = timeSelect / numRuns;
    timeExecute = timeExecute / numRuns;

    // close outfile
    outfileTimePerRunRun.close();
    outfilePdbPerRunRun.close();
    outfileSearchTimePerRunRun.close();


}// end run sequential 2opt

// qiao 2024 add sequential 2-opt on CPU, with multiple 2opt moves along the global tour
template<std::size_t DimP, std::size_t DimCM>
bool SolutionKOPT<DimP, DimCM>::activateSequential2opt(int& numRuns, int nCity,
                                                       int& maxOptExecuPerRun, int& numOptimizedTotal,
                                                       float& timeCpuKey, vector<int>& vectorNumOptExecuted, vector<float>& vectorPDB,
                                                       float& timeRefresh,float& timeSearch, float& timeSelect, float& timeExecute,
                                                       float &maxtimeCpu2optSearch, ofstream &outfileTimePerRunRun,ofstream &outfilePdbPerRunRun,
                                                       ofstream & outfileSearchTimePerRunRun,
                                                       float& evaLastRun, float& percentageImprove, Grid<doubleLinkedEdgeForTSP> &linkCoordTourCpu)
{
    cout << endl << "Enter 2-opt sequential iteration=: " << numRuns << endl;
    bool ret = true;
    numRuns ++;

    maxtimeCpu2optSearch = 0;

    int numOptimizedOneRun = 0;
    int numCityTraversed = 0;

    float pdbOneRun = 0;

    //! random starting point
    int ps_random = randomNum(0, nCity);
    PointCoord ps(ps_random, 0);
    cout << "PS [0] " << ps[0] << endl;

    //! clean md_links_firstPara before mark tour ordering
    md_links_firstPara.activeMap.resetValue(0);
    md_links_firstPara.densityMap.resetValue(initialPrepareValue);// densityMap stores node3
    md_links_firstPara.grayValueMap.resetValue(0);// clean orders
    md_links_firstPara.minRadiusMap.resetValue(initialPrepareValue);//  minRadiusMap stores the changeLinks position

    //!timing runing time on CPU
    __int64 CounterStart = 0;
    double pcFreq = 0.0;
    StartCounter(pcFreq, CounterStart);

    //! mark tour orientation from random starting point ps, index of linkCoordTourCpu should correspond to index of gray value map
    md_links_firstPara.markNetLinkSequenceReloadRoutCoord(ps, numRuns%2, 0, linkCoordTourCpu);// every ps check its two directions

    // end time cpu
    double timeCpuRefreshTour = GetCounter(pcFreq, CounterStart);
    cout << "Time:: Refresh tour order: " << timeCpuRefreshTour << endl;


    //!timing runing time on CPU
    CounterStart = 0;
    pcFreq = 0.0;
    StartCounter(pcFreq, CounterStart);

#if FIRST

    //qiao sequential 2-opt, loop all node and check node one by one to find multiple candidate 2-opt 保留使用densityMap和GPU一致

    md_links_firstPara.sequential2optFirst(linkCoordTourCpu);// every ps check its two directions

#else
    md_links_firstPara.sequential2optBest(linkCoordTourCpu);// every ps check its two directions

#endif
    // end time cpu
    double timeCpu2opt = GetCounter(pcFreq, CounterStart);
    cout << "Time:: select non intera 2-opt: " << timeCpu2opt << endl;


    //! sequentially select non-interacted 2-exchanges
    //! clean for mark non-interacted 2-opt
    md_links_firstPara.activeMap.resetValue(0); // for nodes possessing non interacted 2opt
    md_links_firstPara.fixedMap.resetValue(0); // for nodes in stackB


    //!timing runing time on CPU
    CounterStart = 0;
    pcFreq = 0.0;
    StartCounter(pcFreq, CounterStart);


    //! select and execute non-interacted 2-exchanges
    md_links_firstPara.selectNonIteracted2ExchangeRocki(ps);

    // end time cpu
    double timeCpuSelectNonItera = GetCounter(pcFreq, CounterStart);
    cout << "Time:: select non intera 2-opt: " << timeCpuSelectNonItera << endl;

    double timeCpuExecuteNonItera = 0;
    //!timing runing time on CPU
    CounterStart = 0;
    pcFreq = 0.0;
    StartCounter(pcFreq, CounterStart);

    md_links_firstPara.executeNonInteract2optOnlyNode3(numOptimizedOneRun);

    // end time cpu
    timeCpuExecuteNonItera = GetCounter(pcFreq, CounterStart);
    cout << "Time:: CPU execute non-intera 2-opt: " << timeCpuExecuteNonItera << endl;



    //! evaluation to stop
    numOptimizedOneRun = testGridNum<int>(md_links_firstPara.activeMap);
    float evaCurrentRun = md_links_firstPara.evaluateWeightOfTSP(distEuclidean, numCityTraversed);
    cout << "Evaluate:: After " << numRuns << "'th run, evaluate tsp length =  " << evaCurrentRun << endl;
    cout << "Evaluate:: In this run, num of 2-exchange been executed: "  <<  numOptimizedOneRun << endl;

    float evaActualLength = md_links_firstPara.evaluateWeightOfTSP(distEuclidean, numCityTraversed);
    cout << "Evaluate:: After " << numRuns << "'th run, evaluate tsp length =  " << evaActualLength << endl;

    //statistic pdb
    if(optimum > 1)
    {
        float evaCurrentPDB = md_links_firstPara.evaluateWeightOfTSP(distEuclidean, numCityTraversed);
        pdbOneRun = (evaCurrentPDB - optimum)*100/optimum;
    }

    if(numRuns == 1){


        //! registrer length of the first run
        evaLastRun = evaCurrentRun;
    }
    else {
        percentageImprove = ((evaLastRun - evaCurrentRun)*100);
    }


    if(percentageImprove > 0){
        timeCpuKey += timeCpu2opt + timeCpuRefreshTour + timeCpuSelectNonItera + timeCpuExecuteNonItera;
        evaLastRun = evaCurrentRun;

        numOptimizedTotal += numOptimizedOneRun;
        if(numOptimizedOneRun > maxOptExecuPerRun)
            maxOptExecuPerRun = numOptimizedOneRun;
        if(numOptimizedOneRun > 0)
            vectorNumOptExecuted.push_back(numOptimizedOneRun);
        // trace pdb one run
        vectorPDB.push_back(pdbOneRun);
        outfileTimePerRunRun << timeCpuKey << " " << endl;
        outfilePdbPerRunRun << pdbOneRun << " " << endl;
        outfileSearchTimePerRunRun << timeCpu2opt << " " << endl;

        traceTSP.timeObtainKoptimal = timeCpuKey;
        //record the best TSP tour obtained so far
        tspTourBestObtainedSoFar.assign(md_links_firstPara.networkLinks);
    }
    else{
        numRuns -= 1; // the last run does not optimized the tour

        // outfile pdbline
        string fileKoptimalTimePerRun = "Results_"; //str
        fileKoptimalTimePerRun.append("2optimal.txt");
        ofstream outfileKoptimalTimePerRunRun;
        outfileKoptimalTimePerRunRun.open(fileKoptimalTimePerRun);

        outfileKoptimalTimePerRunRun << timeCpuKey  << " ,pdb: " << pdbOneRun << " ,searchTime: " << timeCpu2opt<< endl;

        outfileKoptimalTimePerRunRun.close();

    }

    //test
    cout << "Percentage improve " << percentageImprove << endl << endl;


    // count time refresh
    timeRefresh += (float)timeCpuRefreshTour;
    timeSearch += (float)timeCpu2opt;
    timeSelect += (float)timeCpuSelectNonItera;
    timeExecute += (float)timeCpuExecuteNonItera;


    return ret;
}//end 2opt sequential



//! \brief Run qiao 2024 run 4-opt
//!
template<std::size_t DimP, std::size_t DimCM>
void SolutionKOPT<DimP, DimCM>::run4opt() {

    cout << "Begin run 4-opt >>>>>>>>>>>>>>>" << endl;

    int nCity = md_links_cpu.adaptiveMap.getWidth();
    int numRuns = 0;

    int maxOptExecuPerRun = 0;
    int numOptimizedTotal = 0;
    vector<int> vectorNumOptExecuted;
    vector<float> vectorPDB;
    float timeGpuKernel = 0;
    float timeGpuH2D = 0;
    float timeGpuD2H = 0;
    float timeGpuTotal = 0;
    float timeCpuKey = 0;
    float pdb2optEatFirstPara = 0;
    float timeRefreshTour = 0;
    float timeSelect = 0;
    float timeExecute = 0;
    int numInter = 1;
    // trace maxtimeGPUone2-OoptRun
    float maxtimeGpuOptSearch = 0;

    // outfile timeline
    string fileTimePerRun = "Results_"; //str
    fileTimePerRun.append("TimePer4optRun.txt");
    ofstream outfileTimePerRunRun;
    outfileTimePerRunRun.open(fileTimePerRun);

    float evaLastRun = 0;
    float percentageImprove = 999999;

    //! prepare the pre-ordered link + coordinates
    Grid<doubleLinkedEdgeForTSP> linkCoordTourCpu;
    linkCoordTourCpu.resize(nCity, 1);
    Grid<doubleLinkedEdgeForTSP> linkCoordTourGpu;
    linkCoordTourGpu.gpuResize(nCity,1);

    double temp = (double)nCity /(double) 2;
    double maxChecks2opt = temp *(nCity - 1) ; // N

    double temptemp =  (double)maxChecks2opt/ (double)2;
    double maxChecks4opt = temptemp*(maxChecks2opt - 1);
    double iter4opt = (double)maxChecks4opt /(double) (BLOCKSIZE * GRIDSIZE);
    if(iter4opt < 1)
        iter4opt = 1;

    cout << "Check maxChecks4opt = " << maxChecks4opt << ", iter4opt = " << iter4opt << endl;


    //wb.Q 2019 add case detection
    if(SolutionKOPT<DimP, DimCM>::md_links_cpu.adaptiveMap.width == 0)
    {
        cout << "Error: no input available." << endl;
        return;
    }
    else
    {
        //wb.Q 2024 4-opt
        cout << "TSP tour optimum = " << optimum << endl;
        while (numRuns < NUMRUNSLIMIT  && percentageImprove > 0 )
        {
            activateRocki4opt(numRuns, nCity, maxChecks2opt, maxChecks4opt, iter4opt,
                              maxOptExecuPerRun, numOptimizedTotal,
                              timeGpuKernel, timeGpuH2D, timeGpuD2H,
                              timeGpuTotal, timeCpuKey, vectorNumOptExecuted, vectorPDB,
                              timeRefreshTour, timeSelect, timeExecute, maxtimeGpuOptSearch,
                              outfileTimePerRunRun, evaLastRun, percentageImprove,
                              linkCoordTourCpu,linkCoordTourGpu);

            //            if (g_ConfigParameters->traceActive) {
            //                evaluate();
            //                writeStatisticsToFile(iteration);
            //            }
        }
    }// end activateRocki

    //! free gpu memory
    linkCoordTourGpu.gpuFreeMem();


    //! count time gpu total
    timeGpuTotal += timeGpuH2D + timeGpuD2H + timeGpuKernel;


    //! mean time trace
    timeRefreshTour = timeRefreshTour / numRuns;
    timeSelect = timeSelect / numRuns;
    timeExecute = timeExecute / numRuns;

    // close outfile
    outfileTimePerRunRun.close();


}// end run

// qiao 2024 add operators to GPU parallel 23456-opt and massive variable 23456-opt moves on global tour
template<std::size_t DimP, std::size_t DimCM>
bool SolutionKOPT<DimP, DimCM>::activateRocki4opt(int& numRuns, int nCity,double maxChecks2opt,
                                                  double  maxChecks4opt,
                                                  double iter,
                                                  int& maxOptExecuPerRun, int& numOptimizedTotal,
                                                  float& timeGpuKernel, float& timeGpuH2D, float& timeGpuD2H,
                                                  float& timeGpuTotal, float& timeCpuKey, vector<int>& vectorNumOptExecuted, vector<float>& vectorPDB,
                                                  float& timeRefresh, float& timeSelect, float& timeExecute,
                                                  float &maxtimeGpuOptSearch, ofstream &outfileTimePerRunRun,
                                                  float& evaLastRun, float& percentageImprove, Grid<doubleLinkedEdgeForTSP> &linkCoordTourCpu,
                                                  Grid<doubleLinkedEdgeForTSP>& linkCoordTourGpu )
{
    cout << endl << "****>>>>Enter 4-opt activate function: " << numRuns << endl;
    bool ret = true;
    numRuns ++;

    int numOptimizedOneRun = 0;
    int numCityTraversed = 0;

    //!timing runing time on CPU
    //    __int64 CounterStart = 0;
    //    double pcFreq = 0.0;
    float elapsedTime2opt = 0;

    float pdbOneRun = 0;

    //! random starting point
    int ps_random = randomNum(0, nCity);
    PointCoord ps(0, 0);
    cout << "PS [0] " << ps[0] << endl;

    //! clean cityCopy before mark tour ordering
    md_links_firstPara.activeMap.resetValue(0);
    md_links_firstPara.densityMap.resetValue(initialPrepareValue);// densityMap stores node3
    md_links_firstPara.grayValueMap.resetValue(0);// clean orders
    md_links_firstPara.minRadiusMap.resetValue(initialPrepareValue);//  minRadiusMap stores the changeLinks position
    md_links_firstPara.optCandidateMap.resetValue(initialPrepareValue);// optCandidateMap stores opt candidate of 23456-opt

    //!timing runing time on CPU
    __int64 CounterStart = 0;
    double pcFreq = 0.0;
    StartCounter(pcFreq, CounterStart);

    //! mark tour orientation from random starting point ps, index of linkCoordTourCpu should correspond to index of gray value map
    md_links_firstPara.markNetLinkSequenceReloadRoutCoord(ps, numRuns%2, 0, linkCoordTourCpu);// every ps check its two directions

    // end time cpu
    double timeCpuRefreshTour = GetCounter(pcFreq, CounterStart);
    cout << "Time:: Refresh tour order: " << timeCpuRefreshTour << endl;


    // time for GPU memcp HD
    float elapsedTime2optHD = 0;
    cudaEvent_t startHD, stopHD;
    cudaEventCreate(&startHD);
    cudaEventCreate(&stopHD);
    cudaEventRecord(startHD, 0);

    // copy tour ordering to gpu, clean gpu network links
    md_links_firstPara.grayValueMap.gpuCopyHostToDevice(md_links_gpu.grayValueMap);// refresh tsp order gpu side
    linkCoordTourCpu.gpuCopyHostToDevice(linkCoordTourGpu);// refresh doubly linked tour order

    cudaEventRecord(stopHD, 0);
    cudaEventSynchronize(stopHD);
    cudaEventElapsedTime(&elapsedTime2optHD, startHD, stopHD);
    cudaEventDestroy(startHD);
    cudaEventDestroy(stopHD);
    cout << "Time:: memcp H to D grayValueMap : " <<  elapsedTime2optHD << endl;


    md_links_gpu.densityMap.gpuResetValue(initialPrepareValue);// use for node3
    md_links_gpu.minRadiusMap.gpuResetValue(initialPrepareValue); // use for local min change
    md_links_gpu.optCandidateMap.gpuResetValue(initialPrepareValueLL);//qiao use for 23456opt

    // cuda timer
    double time = 46;
    double *d_time;


    //qiao only for test
    cout << "Warning: maxChecks4opt= " << maxChecks4opt << ", Warning: maxChecks2opt= " << maxChecks2opt << endl;


    //divide and conquer
    double maxChecks4optDivide = 1.27719e+11;
    double packSize = BLOCKSIZE * GRIDSIZE;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);


    //! WB.Q parallel check exhaustive 4-opt along the tour for each edge
    {
        double iterDivide = (double)maxChecks4optDivide /(double) (packSize);
        if(maxChecks4opt < packSize)
            iterDivide = 1;
        double maxStride = (double) maxChecks4opt /  (double)maxChecks4optDivide;
        if(maxStride < 1)
            maxStride = 0;

        cout << "Changed maxChecks4optDivide = " << maxChecks4optDivide << ", iterDivide = " << iterDivide << ", maxStride= " << maxStride << endl;
        for(double iStride = 0; iStride < maxStride+1; iStride++ )
        {

            //            double test = maxChecks4optDivide * (iStride+1) ;
            //            cout << "StartID " << test << endl;

            //! WB.Q parallel check exhaustive 4-opt along the tour for each edge
            K_oneThreadOne4opt_qiao_iterStride(md_links_gpu, linkCoordTourGpu, maxChecks2opt, maxChecks4opt, maxChecks4optDivide, iterDivide, iStride);

            cudaDeviceSynchronize();

            cout << "Inner one time " << iStride << endl << endl;
        }
    }


    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime2opt, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // find the maximum gpu time for a parallel 4-opt run
    cout << "Time:: GPU side one 4-opt runtime : " << elapsedTime2opt << endl;

    if(maxtimeGpuOptSearch < elapsedTime2opt){
        maxtimeGpuOptSearch = elapsedTime2opt;
    }


    //! sequentially select non-interacted 4-exchanges
    float elapsedTime2opt_DH2 = 0;
    cudaEvent_t startDH2, stopDH2;
    cudaEventCreate(&startDH2);
    cudaEventCreate(&stopDH2);
    cudaEventRecord(startDH2, 0);

    md_links_firstPara.densityMap.gpuCopyDeviceToHost(md_links_gpu.densityMap);//
    md_links_firstPara.optCandidateMap.gpuCopyDeviceToHost(md_links_gpu.optCandidateMap);//opt candidates

    cudaEventRecord(stopDH2, 0);
    cudaEventSynchronize(stopDH2);
    cudaEventElapsedTime(&elapsedTime2opt_DH2, startDH2, stopDH2);
    cudaEventDestroy(startDH2);
    cudaEventDestroy(stopDH2);
    cout << "Time:: memcp D to H " << elapsedTime2opt_DH2 << endl;



    //qiao only for test
    int numCandidate = 0;
    for(int i = 0; i < md_links_firstPara.optCandidateMap.width; i++ )
    {
        if(md_links_firstPara.optCandidateMap[0][i] > 0)
        {
            numCandidate += 1;
            cout << " candidate order " << md_links_firstPara.grayValueMap[0][i] << endl;
        }

    }
    cout << "After one GPU search num of candidates: " << numCandidate << endl;



    //! clean for mark non-interacted 23456-opt
    md_links_firstPara.activeMap.resetValue(initialPrepareValue); // for nodes possessing non interacted 2opt
    md_links_firstPara.fixedMap.resetValue(initialPrepareValue); // for nodes in stackB


    //!timing runing time on CPU
    CounterStart = 0;
    pcFreq = 0.0;
    StartCounter(pcFreq, CounterStart);

    //! select and execute non-interacted 23456-exchanges
    md_links_firstPara.selectNonIteracted23456ExchangeQiao(ps);

    // end time cpu
    double timeCpuSelectNonItera = GetCounter(pcFreq, CounterStart);
    cout << "Time:: select non intera 4-opt: " << timeCpuSelectNonItera << endl;

    double timeCpuExecuteNonItera = 0;
    float timeGpuExecute = 0;
    float elapsedTime2optHD2 = 0;
    float elapsedTime2optHD3 = 0;
    float elapsedTime2opt_DH = 0;
    float elapsedTime2opt_execute = 0;

#if CPUEXECUTE
    //!timing runing time on CPU
    CounterStart = 0;
    pcFreq = 0.0;
    StartCounter(pcFreq, CounterStart);
    md_links_firstPara.executeNonInteract23456optOnlyNode3(numOptimizedOneRun, md_links_cpu.nodeParentMap);//qiao 2024 need modify

    // end time cpu
    timeCpuExecuteNonItera = GetCounter(pcFreq, CounterStart);
    cout << "Time:: CPU execute non-intera 4-opt: " << timeCpuExecuteNonItera << endl;

    cudaEvent_t startHD2, stopHD2;
    cudaEventCreate(&startHD2);
    cudaEventCreate(&stopHD2);
    cudaEventRecord(startHD2, 0);

    md_links_firstPara.networkLinks.gpuCopyHostToDevice(md_links_gpu.networkLinks);
    errorCheckCudaThreadSynchronize();

    cudaEventRecord(stopHD2, 0);
    cudaEventSynchronize(stopHD2);
    cudaEventElapsedTime(&elapsedTime2optHD3, startHD2, stopHD2);
    cudaEventDestroy(startHD2);
    cudaEventDestroy(stopHD2);
    cout << "Time:: memcp H to D networkLinks: " <<  elapsedTime2optHD3 << endl;

#else

    //! copy activeMap (selected 2-exchanges) to device HD
    cudaEvent_t startHD2, stopHD2;
    cudaEventCreate(&startHD2);
    cudaEventCreate(&stopHD2);
    cudaEventRecord(startHD2, 0);

    md_links_firstPara.activeMap.gpuCopyHostToDevice(md_links_gpu.activeMap);

    cudaEventRecord(stopHD2, 0);
    cudaEventSynchronize(stopHD2);
    cudaEventElapsedTime(&elapsedTime2optHD2, startHD2, stopHD2);
    cudaEventDestroy(startHD2);
    cudaEventDestroy(stopHD2);
    cout << "memcp H to D activeValueMap : " <<  elapsedTime2optHD2 << endl;

    //! kernel execute selected 2-exchanges
    // cuda timer
    cudaEvent_t start2, stop2;
    cudaEventCreate(&start2);
    cudaEventCreate(&stop2);
    cudaEventRecord(start2, 0);

    K_executeNonItera2ExchangeOnlyWithNode3(md_links_gpu);

    cudaEventRecord(stop2, 0);
    cudaEventSynchronize(stop2);
    cudaEventElapsedTime(&elapsedTime2opt_execute, start2, stop2);
    cudaEventDestroy(start2);
    cudaEventDestroy(stop2);

    cout << "gpu search 2opt in parallel " << elapsedTime2opt << endl;
    cout << " gpu execute non intera 2-exchange time " << elapsedTime2opt_execute << endl;

    //        //! copy new tour to host DH

    cudaEvent_t startDH, stopDH;
    cudaEventCreate(&startDH);
    cudaEventCreate(&stopDH);
    cudaEventRecord(startDH, 0);

    md_links_firstPara.networkLinks.gpuCopyDeviceToHost(md_links_gpu.networkLinks);

    cudaEventRecord(stopDH, 0);
    cudaEventSynchronize(stopDH);
    cudaEventElapsedTime(&elapsedTime2opt_DH, startDH, stopDH);
    cudaEventDestroy(startDH);
    cudaEventDestroy(stopDH);
    cout << "memcp device to host networklinks " << elapsedTime2opt_DH << endl;

#endif


    timeGpuExecute =  elapsedTime2optHD2 + elapsedTime2opt_execute + elapsedTime2opt_DH ;
    cout << "Time:: timeGpu Execute " << timeGpuExecute << endl;


    //! evaluation to stop
    float evaCurrentRun = md_links_firstPara.evaluateWeightOfTSP(dist, numCityTraversed);
    cout << "Evaluate:: After " << numRuns << "'th run, evaluate tsp length =  " << evaCurrentRun << endl;
    cout << "Evaluate:: In this run, num of 4-exchange been executed: "  <<  numOptimizedOneRun << endl;

    float evaActualLength = md_links_firstPara.evaluateWeightOfTSP(distEuclidean, numCityTraversed);
    cout << "Evaluate:: After " << numRuns << "'th run, evaluate tsp length =  " << evaActualLength << endl;

    //statistic pdb
    if(optimum > 1)
    {
        float evaCurrentPDB = md_links_firstPara.evaluateWeightOfTSP(dist, numCityTraversed);
        pdbOneRun = (evaCurrentPDB - optimum)*100/optimum;
    }

    if(numRuns == 1){
        timeGpuH2D += elapsedTime2optHD + elapsedTime2optHD2 + elapsedTime2optHD3;
        timeGpuD2H += elapsedTime2opt_DH + elapsedTime2opt_DH2;
        timeGpuKernel += elapsedTime2opt + elapsedTime2opt_execute;
        timeCpuKey += timeCpuRefreshTour + timeCpuSelectNonItera + timeCpuExecuteNonItera;

        numOptimizedTotal += numOptimizedOneRun;
        if(numOptimizedOneRun > maxOptExecuPerRun)
            maxOptExecuPerRun = numOptimizedOneRun; // trace max optimized 2opt per run
        if(numOptimizedOneRun > 0)
            vectorNumOptExecuted.push_back(numOptimizedOneRun);

        // trace pdb one run
        vectorPDB.push_back(pdbOneRun);
        outfileTimePerRunRun << timeCpuKey +  timeGpuKernel << " " << endl;

        //! registrer length of the first run
        evaLastRun = evaCurrentRun;
        //        continue;
    }
    else {
        percentageImprove = ((evaLastRun - evaCurrentRun)*100);
    }


    if(percentageImprove > 0){
        timeGpuH2D += elapsedTime2optHD + elapsedTime2optHD2;
        timeGpuD2H += elapsedTime2opt_DH + elapsedTime2opt_DH2;
        timeGpuKernel += elapsedTime2opt + elapsedTime2opt_execute;
        timeCpuKey += timeCpuRefreshTour + timeCpuSelectNonItera + timeCpuExecuteNonItera;
        evaLastRun = evaCurrentRun;

        numOptimizedTotal += numOptimizedOneRun;
        if(numOptimizedOneRun > maxOptExecuPerRun)
            maxOptExecuPerRun = numOptimizedOneRun; // trace max optimized 2opt per run
        if(numOptimizedOneRun > 0)
            vectorNumOptExecuted.push_back(numOptimizedOneRun);
        // trace pdb one run
        vectorPDB.push_back(pdbOneRun);
        outfileTimePerRunRun << timeCpuKey +  timeGpuKernel << " " << endl;
    }
    else{
        numRuns -= 1; // the last run does not optimized the tour
    }

    //test
    cout << "Percentage improve " << percentageImprove << endl << endl;


    // count time refresh
    timeRefresh += (float)timeCpuRefreshTour;
    timeSelect += (float)timeCpuSelectNonItera;
#if CPUEXECUTE
    timeExecute += (float)timeCpuExecuteNonItera;
#else
    timeExecute += timeGpuExecute;
#endif


    return ret;
}//end 4opt



//! \brief Run qiao 2024 run 4-opt sequential
//!
template<std::size_t DimP, std::size_t DimCM>
void SolutionKOPT<DimP, DimCM>::runSequential4opt() {

    cout << "Begin run 4-opt sequential >>>>>>>>>>>>>>>" << endl;

    int nCity = md_links_cpu.adaptiveMap.getWidth();
    int numRuns = 0;

    int maxOptExecuPerRun = 0;
    int numOptimizedTotal = 0;
    vector<int> vectorNumOptExecuted;
    vector<float> vectorPDB;
    float timeCpuKey = 0;
    float pdb2optEatFirstPara = 0;
    float timeRefreshTour = 0;
    float timeSearch = 0;
    float timeSelect = 0;
    float timeExecute = 0;
    int numInter = 1;
    float maxtimeCpuOptSearch = 0;

    // outfile timeline
    string fileTimePerRun = "Results_"; //str
    fileTimePerRun.append("TimePer4optRun.txt");
    ofstream outfileTimePerRunRun;
    outfileTimePerRunRun.open(fileTimePerRun);

    // outfile pdbline
    string filePdbPerRun = "Results_"; //str
    filePdbPerRun.append("PdbPer4optRun.txt");
    ofstream outfilePdbPerRunRun;
    outfilePdbPerRunRun.open(filePdbPerRun);

    // outfile pdbline
    string fileSearchTimePerRun = "Results_"; //str
    fileSearchTimePerRun.append("searchTimePer4optRun.txt");
    ofstream outfileSearchTimePerRunRun;
    outfileSearchTimePerRunRun.open(fileSearchTimePerRun);

    outfileTimePerRunRun << 0 << " " << endl;
    outfilePdbPerRunRun << 1143.63 << " " << endl;
    outfileSearchTimePerRunRun << 0 << endl;

    float evaLastRun = 0;
    float percentageImprove = 999999;

    //! prepare the pre-ordered link + coordinates
    Grid<doubleLinkedEdgeForTSP> linkCoordTourCpu;
    linkCoordTourCpu.resize(nCity, 1);

    //wb.Q 2019 add case detection
    if(SolutionKOPT<DimP, DimCM>::md_links_cpu.adaptiveMap.width == 0)
    {
        cout << "Error: no input available." << endl;
        return;
    }
    else
    {
        //wb.Q 2024 4-opt
        cout << "TSP tour optimum = " << optimum << endl;
        while (numRuns < NUMRUNSLIMIT  && percentageImprove > 0 )
        {
            activateSequential4opt(numRuns, nCity,
                                   maxOptExecuPerRun, numOptimizedTotal, timeCpuKey, vectorNumOptExecuted, vectorPDB,
                                   timeRefreshTour, timeSearch, timeSelect, timeExecute, maxtimeCpuOptSearch,
                                   outfileTimePerRunRun, outfilePdbPerRunRun, outfileSearchTimePerRunRun, evaLastRun, percentageImprove,
                                   linkCoordTourCpu,0);

            if (g_ConfigParameters->traceActive) {
                evaluate();
                writeStatisticsToFile(numRuns, "4optimalTour");
            }
        }
    }// end activateRocki




    //! mean time trace
    timeRefreshTour = timeRefreshTour / numRuns;
    timeSearch = timeSearch / numRuns;
    timeSelect = timeSelect / numRuns;
    timeExecute = timeExecute / numRuns;

    // close outfile
    outfileTimePerRunRun.close();

    outfilePdbPerRunRun.close();
    outfileSearchTimePerRunRun.close();


}// end run

// qiao 2024 add CPU sequential 4-opt
template<std::size_t DimP, std::size_t DimCM>
bool SolutionKOPT<DimP, DimCM>::activateSequential4opt(int& numRuns, int nCity,
                                                       int& maxOptExecuPerRun, int& numOptimizedTotal,
                                                       float& timeCpuKey, vector<int>& vectorNumOptExecuted, vector<float>& vectorPDB,
                                                       float& timeRefresh, float&timeSearch, float& timeSelect, float& timeExecute, float& maxtimeCpuOptSearch,
                                                       ofstream &outfileTimePerRunRun, ofstream &outfilePdbPerRunRun, ofstream &outfileSearchTimePerRunRun, float& evaLastRun, float& percentageImprove,
                                                       Grid<doubleLinkedEdgeForTSP> &linkCoordTourCpu , bool iterOptimal)
{
    cout << endl << "****>>>>Enter sequential 4-opt activate function: " << numRuns << endl;
    bool ret = true;
    numRuns ++;


    maxtimeCpuOptSearch = 0;

    int numOptimizedOneRun = 0;
    int numCityTraversed = 0;
    float pdbOneRun = 0;

    //! random starting point
    int ps_random = randomNum(0, nCity);
    PointCoord ps(ps_random, 0);
    cout << "PS [0] " << ps[0] << endl;

    //! clean cityCopy before mark tour ordering
    md_links_firstPara.activeMap.resetValue(0);
    md_links_firstPara.densityMap.resetValue(initialPrepareValue);// densityMap stores node3
    md_links_firstPara.grayValueMap.resetValue(0);// clean orders
    md_links_firstPara.minRadiusMap.resetValue(initialPrepareValue);//  minRadiusMap stores the changeLinks position
    md_links_firstPara.optCandidateMap.resetValue(initialPrepareValue);// optCandidateMap stores opt candidate of 23456-opt

    //!timing runing time on CPU
    __int64 CounterStart = 0;
    double pcFreq = 0.0;
    StartCounter(pcFreq, CounterStart);

    //! mark tour orientation from random starting point ps, index of linkCoordTourCpu should correspond to index of gray value map
    md_links_firstPara.markNetLinkSequenceReloadRoutCoord(ps, numRuns%2, 0, linkCoordTourCpu);// every ps check its two directions

    // end time cpu
    double timeCpuRefreshTour = GetCounter(pcFreq, CounterStart);
    cout << "Time:: Refresh tour order: " << timeCpuRefreshTour << endl;

    //!timing runing time on CPU
    CounterStart = 0;
    pcFreq = 0.0;
    StartCounter(pcFreq, CounterStart);

#if FIRST

    //qiao sequential 4-opt, four loop, check each node's 4-opt possibility, and find multiple 4-opt candidates
    md_links_firstPara.sequential4optFirst(linkCoordTourCpu, md_links_cpu.nodeParentMap);// every ps check its two directions

#else
    md_links_firstPara.sequential4optBest(linkCoordTourCpu, md_links_cpu.nodeParentMap);// every ps check its two directions
#endif
    // end time cpu
    double timeCpu4opt = GetCounter(pcFreq, CounterStart);
    cout << "Time:: CPU side one 4-opt runtime : " << timeCpu4opt << endl;

    if(maxtimeCpuOptSearch < timeCpu4opt){
        maxtimeCpuOptSearch = timeCpu4opt;
    }


    //! sequentially select non-interacted 4-exchanges
    md_links_firstPara.activeMap.resetValue(initialPrepareValue); // for nodes possessing non interacted 2opt
    md_links_firstPara.fixedMap.resetValue(initialPrepareValue); // for nodes in stackB


    //!timing runing time on CPU
    CounterStart = 0;
    pcFreq = 0.0;
    StartCounter(pcFreq, CounterStart);

    //! select and execute non-interacted 23456-exchanges
    md_links_firstPara.selectNonIteracted23456ExchangeQiao(ps);

    // end time cpu
    double timeCpuSelectNonItera = GetCounter(pcFreq, CounterStart);
    cout << "Time:: select non intera 4-opt: " << timeCpuSelectNonItera << endl;

    double timeCpuExecuteNonItera = 0;
    float elapsedTime2opt_execute = 0;

    //!timing runing time on CPU
    CounterStart = 0;
    pcFreq = 0.0;
    StartCounter(pcFreq, CounterStart);
    md_links_firstPara.executeNonInteract23456optOnlyNode3(numOptimizedOneRun, md_links_cpu.nodeParentMap);//qiao 2024 need modify

    // end time cpu
    timeCpuExecuteNonItera = GetCounter(pcFreq, CounterStart);
    cout << "Time:: CPU execute non-intera 4-opt: " << timeCpuExecuteNonItera << endl;


    //! evaluation to stop
    float evaCurrentRun = md_links_firstPara.evaluateWeightOfTSP(distEuclidean, numCityTraversed);
    cout << "Evaluate:: After " << numRuns << "'th run, evaluate tsp length =  " << evaCurrentRun << endl;
    cout << "Evaluate:: In this run, num of 4-exchange been executed: "  <<  numOptimizedOneRun << endl;

    float evaActualLength = md_links_firstPara.evaluateWeightOfTSP(distEuclidean, numCityTraversed);
    cout << "Evaluate:: After " << numRuns << "'th run, evaluate tsp length =  " << evaActualLength << endl;

    //statistic pdb
    if(optimum > 1)
    {
        float evaCurrentPDB = md_links_firstPara.evaluateWeightOfTSP(distEuclidean, numCityTraversed);
        pdbOneRun = (evaCurrentPDB - optimum)*100/optimum;
    }

    if(numRuns == 1){

        //! registrer length of the first run
        evaLastRun = evaCurrentRun;
        //        continue;
    }
    else {
        percentageImprove = ((evaLastRun - evaCurrentRun)*100 );
    }


    if(percentageImprove > 0)
    {

        timeCpuKey += timeCpu4opt + timeCpuRefreshTour + timeCpuSelectNonItera + timeCpuExecuteNonItera;
        evaLastRun = evaCurrentRun;

        numOptimizedTotal += numOptimizedOneRun;

        if(numOptimizedOneRun > maxOptExecuPerRun)
            maxOptExecuPerRun = numOptimizedOneRun; // trace max optimized 2opt per run
        if(numOptimizedOneRun > 0)
            vectorNumOptExecuted.push_back(numOptimizedOneRun);
        // trace pdb one run
        vectorPDB.push_back(pdbOneRun);
        outfileTimePerRunRun << timeCpuKey << " " << endl;

        outfilePdbPerRunRun << pdbOneRun << " " << endl;
        outfileSearchTimePerRunRun << timeCpu4opt << " " << endl;

        traceTSP.timeObtainKoptimal = timeCpuKey;
        //record the best TSP tour obtained so far
        tspTourBestObtainedSoFar.assign(md_links_firstPara.networkLinks);

    }
    else{
        numRuns -= 1; // the last run does not optimized the tour

        // outfile pdbline
        string fileKoptimalTimePerRun = "Results_"; //str
        fileKoptimalTimePerRun.append("4optimal.txt");
        ofstream outfileKoptimalTimePerRunRun;
        outfileKoptimalTimePerRunRun.open(fileKoptimalTimePerRun);

        outfileKoptimalTimePerRunRun << timeCpuKey  << " ,pdb: " << pdbOneRun << " ,searchTime: " << timeCpu4opt<< endl;

        outfileKoptimalTimePerRunRun.close();
    }

    //test
    cout << "Percentage improve " << percentageImprove << endl << endl;


    // count time refresh
    timeRefresh += (float)timeCpuRefreshTour;
    timeSearch += (float)timeCpu4opt;
    timeSelect += (float)timeCpuSelectNonItera;
    timeExecute += (float)timeCpuExecuteNonItera;

    return ret;
}//end 4opt sequential





//! \brief Run et activate
//!
//wb.Q 202408 implement 5-opt
template<std::size_t DimP, std::size_t DimCM>
void SolutionKOPT<DimP, DimCM>::run5opt() {

    cout << "Begin run 5-opt >>>>>>>>>>>>>>>" << endl;

    int nCity = md_links_cpu.adaptiveMap.getWidth();
    int numRuns = 0;

    int maxOptExecuPerRun = 0;
    int numOptimizedTotal = 0;
    vector<int> vectorNumOptExecuted;
    vector<float> vectorPDB;
    float timeGpuKernel = 0;
    float timeGpuH2D = 0;
    float timeGpuD2H = 0;
    float timeGpuTotal = 0;
    float timeCpuKey = 0;
    float pdb2optEatFirstPara = 0;
    float timeRefreshTour = 0;
    float timeSelect = 0;
    float timeExecute = 0;
    int numInter = 1;
    // trace maxtimeGPUone2-OoptRun
    float maxtimeGpuOptSearch = 0;

    // outfile timeline
    string fileTimePerRun = "Results_"; //str
    fileTimePerRun.append("TimePerRun.txt");
    ofstream outfileTimePerRunRun;
    outfileTimePerRunRun.open(fileTimePerRun);

    float evaLastRun = 0;
    float percentageImprove = 999999;

    //! prepare the pre-ordered link + coordinates
    Grid<doubleLinkedEdgeForTSP> linkCoordTourCpu;
    linkCoordTourCpu.resize(nCity, 1);
    Grid<doubleLinkedEdgeForTSP> linkCoordTourGpu;
    linkCoordTourGpu.gpuResize(nCity,1);



    double maxChecks2opt = nCity*(nCity - 1) / 2; // total number of checks for 2-opt
    unsigned int iter = maxChecks2opt / (BLOCKSIZE * GRIDSIZE);

    double maxChecks4opt = maxChecks2opt*(maxChecks2opt - 1) / 2;
    unsigned int iter4opt = maxChecks4opt / (BLOCKSIZE * GRIDSIZE);

    //qiao why there is an iter5opt here
    double maxChecks5opt = maxChecks4opt*(maxChecks4opt - 1) / 2;
    unsigned int iter5opt = maxChecks5opt / (BLOCKSIZE * GRIDSIZE);

    cout << "Warning: maxChecks5opt= " << maxChecks5opt <<  " maxChecks4opt= " << maxChecks4opt << " maxChecks2opt= " << maxChecks2opt << endl;

    //wb.Q 2019 add case detection
    if(SolutionKOPT<DimP, DimCM>::md_links_cpu.adaptiveMap.width == 0)
    {
        cout << "Error: no input available." << endl;
        return;
    }
    else
    {
        //wb.Q 2024 rocki 2-opt
        cout << "TSP tour optimum = " << optimum << endl;
        while (numRuns < NUMRUNSLIMIT  && percentageImprove > 0 )
        {
            activateRocki5opt(numRuns, nCity, maxChecks4opt, maxChecks2opt, iter5opt,
                              maxOptExecuPerRun, numOptimizedTotal,
                              timeGpuKernel, timeGpuH2D, timeGpuD2H,
                              timeGpuTotal, timeCpuKey, vectorNumOptExecuted, vectorPDB,
                              timeRefreshTour, timeSelect, timeExecute, maxtimeGpuOptSearch,
                              outfileTimePerRunRun, evaLastRun, percentageImprove,
                              linkCoordTourCpu,linkCoordTourGpu);

            //            if (g_ConfigParameters->traceActive) {
            //                evaluate();
            //                writeStatisticsToFile(iteration);
            //            }
        }
    }// end activateRocki

    //! free gpu memory
    linkCoordTourGpu.gpuFreeMem();


    //! count time gpu total
    timeGpuTotal += timeGpuH2D + timeGpuD2H + timeGpuKernel;


    //! mean time trace
    timeRefreshTour = timeRefreshTour / numRuns;
    timeSelect = timeSelect / numRuns;
    timeExecute = timeExecute / numRuns;

    // close outfile
    outfileTimePerRunRun.close();


}// end run

// qiao 2024 add operators to GPU parallel 23456-opt and massive variable 23456-opt moves on global tour
template<std::size_t DimP, std::size_t DimCM>
bool SolutionKOPT<DimP, DimCM>::activateRocki5opt(int& numRuns, int nCity, double maxChecks4opt,
                                                  double maxChecks2opt,
                                                  unsigned int iter,
                                                  int& maxOptExecuPerRun, int& numOptimizedTotal,
                                                  float& timeGpuKernel, float& timeGpuH2D, float& timeGpuD2H,
                                                  float& timeGpuTotal, float& timeCpuKey, vector<int>& vectorNumOptExecuted, vector<float>& vectorPDB,
                                                  float& timeRefresh, float& timeSelect, float& timeExecute,
                                                  float &maxtimeGpuOptSearch, ofstream &outfileTimePerRunRun,
                                                  float& evaLastRun, float& percentageImprove, Grid<doubleLinkedEdgeForTSP> &linkCoordTourCpu,
                                                  Grid<doubleLinkedEdgeForTSP>& linkCoordTourGpu )
{
    cout << endl << "****>>>>Enter 5-opt activate function: " << numRuns << endl;
    bool ret = true;
    numRuns ++;


    int numOptimizedOneRun = 0;
    int numCityTraversed = 0;

    //!timing runing time on CPU
    //    __int64 CounterStart = 0;
    //    double pcFreq = 0.0;
    float elapsedTime2opt = 0;

    float pdbOneRun = 0;

    //! random starting point
    int ps_random = randomNum(0, nCity);
    PointCoord ps(ps_random, 0);
    cout << "PS [0] " << ps[0] << endl;

    //! clean cityCopy before mark tour ordering
    md_links_firstPara.activeMap.resetValue(0);
    md_links_firstPara.densityMap.resetValue(initialPrepareValue);// densityMap stores node3
    md_links_firstPara.grayValueMap.resetValue(0);// clean orders
    md_links_firstPara.minRadiusMap.resetValue(initialPrepareValue);//  minRadiusMap stores the changeLinks position
    md_links_firstPara.optCandidateMap.resetValue(initialPrepareValue);// optCandidateMap stores opt candidate of 23456-opt

    //!timing runing time on CPU
    __int64 CounterStart = 0;
    double pcFreq = 0.0;
    StartCounter(pcFreq, CounterStart);

    //! mark tour orientation from random starting point ps
    //! reserver, index of linkCoordTourCpu should correspond to index of gray value map
    md_links_firstPara.markNetLinkSequenceReloadRoutCoord(ps, numRuns%2, 0, linkCoordTourCpu);// every ps check its two directions

    // end time cpu
    double timeCpuRefreshTour = GetCounter(pcFreq, CounterStart);
    cout << "Time:: Refresh tour order: " << timeCpuRefreshTour << endl;


    // time for GPU memcp HD
    float elapsedTime2optHD = 0;
    cudaEvent_t startHD, stopHD;
    cudaEventCreate(&startHD);
    cudaEventCreate(&stopHD);
    cudaEventRecord(startHD, 0);

    // copy tour ordering to gpu, clean gpu network links
    md_links_firstPara.grayValueMap.gpuCopyHostToDevice(md_links_gpu.grayValueMap);// refresh tsp order gpu side
    linkCoordTourCpu.gpuCopyHostToDevice(linkCoordTourGpu);// refresh doubly linked tour order

    cudaEventRecord(stopHD, 0);
    cudaEventSynchronize(stopHD);
    cudaEventElapsedTime(&elapsedTime2optHD, startHD, stopHD);
    cudaEventDestroy(startHD);
    cudaEventDestroy(stopHD);
    cout << "Time:: memcp H to D grayValueMap : " <<  elapsedTime2optHD << endl;


    md_links_gpu.densityMap.gpuResetValue(initialPrepareValue);// use for node3
    md_links_gpu.minRadiusMap.gpuResetValue(initialPrepareValue); // use for local min change
    md_links_gpu.optCandidateMap.gpuResetValue(initialPrepareValueLL);//qiao use for 23456opt


    // cuda timer
    double time = 46;
    double *d_time;

    //qiao only for test
    cout << "Warning: maxChecks4opt= " << maxChecks4opt << " maxChecks2opt= " << maxChecks2opt << endl;

    double maxChecks4optDivide = 1.27719e+11;
    double packSize = BLOCKSIZE * GRIDSIZE;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);


    //! WB.Q parallel check exhaustive 5-opt along the tour for each edge
    for(int n = 9; n < md_links_firstPara.adaptiveMap.getWidth(); n++)
    {
        cout << "5-opt n-row = " << n << endl;

        double maxChecks2opt = n*(n - 1) / 2; // total number of checks for 2-opt
        double maxChecks4opt = maxChecks2opt*(maxChecks2opt - 1) / 2;

        double iterDivide = (double)maxChecks4optDivide /(double) (packSize);
        if(maxChecks4opt < packSize)
            iterDivide = 1;
        double maxStride = (double) maxChecks4opt /  (double)maxChecks4optDivide;
        if(maxStride < 1)
            maxStride = 0;

        cout << "maxChecks4opt= " << maxChecks4opt << ", packSize= " << packSize << ", Changed maxChecks4optDivide = " << maxChecks4optDivide << ", iterDivide = " << iterDivide << ", maxStride= " << maxStride << endl;
        for(double iStride = 0; iStride < maxStride+1; iStride++ )
        {

            //            double test = maxChecks4optDivide * (iStride+1) ;
            //            cout << "StartID " << test << endl;

            //! WB.Q parallel check exhaustive 4-opt along the tour for each edge
            K_oneThreadOne5opt_qiao_StrideIter(md_links_gpu, linkCoordTourGpu, n, maxChecks2opt, maxChecks4opt, maxChecks4optDivide, iterDivide, iStride);

            cudaDeviceSynchronize();

            cout << "Inner one time " << iStride << endl;
        }

        cout << " End out one row " << n << endl << endl;
        //        K_oneThreadOne5opt_RockiSmall(md_links_gpu, linkCoordTourGpu, n, maxChecks4opt, maxChecks2opt, iter);//qiao here should be iter4opt

    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime2opt, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // find the maximum gpu time for a parallel 5-opt run
    cout << "Time:: GPU side one 5-opt run : " << elapsedTime2opt << endl;

    if(maxtimeGpuOptSearch < elapsedTime2opt){
        maxtimeGpuOptSearch = elapsedTime2opt;
    }


    //! sequentially select non-interacted 5-exchanges
    float elapsedTime2opt_DH2 = 0;
    cudaEvent_t startDH2, stopDH2;
    cudaEventCreate(&startDH2);
    cudaEventCreate(&stopDH2);
    cudaEventRecord(startDH2, 0);

    md_links_firstPara.densityMap.gpuCopyDeviceToHost(md_links_gpu.densityMap);
    md_links_firstPara.optCandidateMap.gpuCopyDeviceToHost(md_links_gpu.optCandidateMap);//opt candidates


    cudaEventRecord(stopDH2, 0);
    cudaEventSynchronize(stopDH2);
    cudaEventElapsedTime(&elapsedTime2opt_DH2, startDH2, stopDH2);
    cudaEventDestroy(startDH2);
    cudaEventDestroy(stopDH2);
    cout << "Time:: memcp D to H " << elapsedTime2opt_DH2 << endl;


    //qiao only for test
    int numCandidate = 0;
    for(int i = 0; i < md_links_firstPara.optCandidateMap.width; i++ )
    {
        if(md_links_firstPara.optCandidateMap[0][i] > 0)
        {
            numCandidate += 1;
            cout << " candidate order " << md_links_firstPara.grayValueMap[0][i] << endl;
        }

    }
    cout << "After one GPU search num of candidates: " << numCandidate << endl;



    //! clean for mark non-interacted 23456-opt
    md_links_firstPara.activeMap.resetValue(initialPrepareValue); // for nodes possessing non interacted 2opt
    md_links_firstPara.fixedMap.resetValue(initialPrepareValue); // for nodes in stackB


    //!timing runing time on CPU
    CounterStart = 0;
    pcFreq = 0.0;
    StartCounter(pcFreq, CounterStart);


    //! select and execute non-interacted 23456-exchanges
    md_links_firstPara.selectNonIteracted23456ExchangeQiao(ps);

    // end time cpu
    double timeCpuSelectNonItera = GetCounter(pcFreq, CounterStart);
    cout << "Time:: select non intera 5-opt: " << timeCpuSelectNonItera << endl;

    double timeCpuExecuteNonItera = 0;
    float timeGpuExecute = 0;
    float elapsedTime2optHD2 = 0;
    float elapsedTime2optHD3 = 0;
    float elapsedTime2opt_DH = 0;
    float elapsedTime2opt_execute = 0;

#if CPUEXECUTE
    //!timing runing time on CPU
    CounterStart = 0;
    pcFreq = 0.0;
    StartCounter(pcFreq, CounterStart);
    md_links_firstPara.executeNonInteract23456optOnlyNode3(numOptimizedOneRun, md_links_cpu.nodeParentMap);//qiao 2024 need modify

    // end time cpu
    timeCpuExecuteNonItera = GetCounter(pcFreq, CounterStart);
    cout << "Time:: CPU execute non-intera 5-opt: " << timeCpuExecuteNonItera << endl;

    cudaEvent_t startHD2, stopHD2;
    cudaEventCreate(&startHD2);
    cudaEventCreate(&stopHD2);
    cudaEventRecord(startHD2, 0);

    md_links_firstPara.networkLinks.gpuCopyHostToDevice(md_links_gpu.networkLinks);
    errorCheckCudaThreadSynchronize();

    cudaEventRecord(stopHD2, 0);
    cudaEventSynchronize(stopHD2);
    cudaEventElapsedTime(&elapsedTime2optHD3, startHD2, stopHD2);
    cudaEventDestroy(startHD2);
    cudaEventDestroy(stopHD2);
    cout << "Time:: memcp H to D networkLinks: " <<  elapsedTime2optHD3 << endl;

#else

    //! copy activeMap (selected 2-exchanges) to device HD
    cudaEvent_t startHD2, stopHD2;
    cudaEventCreate(&startHD2);
    cudaEventCreate(&stopHD2);
    cudaEventRecord(startHD2, 0);

    md_links_firstPara.activeMap.gpuCopyHostToDevice(md_links_gpu.activeMap);

    cudaEventRecord(stopHD2, 0);
    cudaEventSynchronize(stopHD2);
    cudaEventElapsedTime(&elapsedTime2optHD2, startHD2, stopHD2);
    cudaEventDestroy(startHD2);
    cudaEventDestroy(stopHD2);
    cout << "memcp H to D activeValueMap : " <<  elapsedTime2optHD2 << endl;

    //! kernel execute selected 2-exchanges
    // cuda timer
    cudaEvent_t start2, stop2;
    cudaEventCreate(&start2);
    cudaEventCreate(&stop2);
    cudaEventRecord(start2, 0);

    K_executeNonItera2ExchangeOnlyWithNode3(md_links_gpu);

    cudaEventRecord(stop2, 0);
    cudaEventSynchronize(stop2);
    cudaEventElapsedTime(&elapsedTime2opt_execute, start2, stop2);
    cudaEventDestroy(start2);
    cudaEventDestroy(stop2);

    cout << "gpu search 2opt in parallel " << elapsedTime2opt << endl;
    cout << " gpu execute non intera 2-exchange time " << elapsedTime2opt_execute << endl;

    //        //! copy new tour to host DH

    cudaEvent_t startDH, stopDH;
    cudaEventCreate(&startDH);
    cudaEventCreate(&stopDH);
    cudaEventRecord(startDH, 0);

    md_links_firstPara.networkLinks.gpuCopyDeviceToHost(md_links_gpu.networkLinks);

    cudaEventRecord(stopDH, 0);
    cudaEventSynchronize(stopDH);
    cudaEventElapsedTime(&elapsedTime2opt_DH, startDH, stopDH);
    cudaEventDestroy(startDH);
    cudaEventDestroy(stopDH);
    cout << "memcp device to host networklinks " << elapsedTime2opt_DH << endl;

#endif


    timeGpuExecute =  elapsedTime2optHD2 + elapsedTime2opt_execute + elapsedTime2opt_DH ;
    cout << "Time:: timeGpu Execute " << timeGpuExecute << endl;


    //! evaluation to stop
    float evaCurrentRun = md_links_firstPara.evaluateWeightOfTSP(dist, numCityTraversed);
    cout << "Evaluate:: After " << numRuns << "'th run, evaluate tsp length =  " << evaCurrentRun << endl;
    cout << "Evaluate:: In this run, num of 5-exchange been executed: "  <<  numOptimizedOneRun << endl;

    float evaActualLength = md_links_firstPara.evaluateWeightOfTSP(distEuclidean, numCityTraversed);
    cout << "Evaluate:: After " << numRuns << "'th run, evaluate tsp length =  " << evaActualLength << endl;

    //statistic pdb
    if(optimum > 1)
    {
        float evaCurrentPDB = md_links_firstPara.evaluateWeightOfTSP(dist, numCityTraversed);
        pdbOneRun = (evaCurrentPDB - optimum)*100/optimum;
    }

    if(numRuns == 1){
        timeGpuH2D += elapsedTime2optHD + elapsedTime2optHD2 + elapsedTime2optHD3;
        timeGpuD2H += elapsedTime2opt_DH + elapsedTime2opt_DH2;
        timeGpuKernel += elapsedTime2opt + elapsedTime2opt_execute;
        timeCpuKey += timeCpuRefreshTour + timeCpuSelectNonItera + timeCpuExecuteNonItera;

        numOptimizedTotal += numOptimizedOneRun;
        if(numOptimizedOneRun > maxOptExecuPerRun)
            maxOptExecuPerRun = numOptimizedOneRun; // trace max optimized 2opt per run
        if(numOptimizedOneRun > 0)
            vectorNumOptExecuted.push_back(numOptimizedOneRun);

        // trace pdb one run
        vectorPDB.push_back(pdbOneRun);
        outfileTimePerRunRun << timeCpuKey +  timeGpuKernel << " " << endl;

        //! registrer length of the first run
        evaLastRun = evaCurrentRun;
        //        continue;
    }
    else {
        percentageImprove = ((evaLastRun - evaCurrentRun)*100);
    }


    if(percentageImprove > 0){
        timeGpuH2D += elapsedTime2optHD + elapsedTime2optHD2;
        timeGpuD2H += elapsedTime2opt_DH + elapsedTime2opt_DH2;
        timeGpuKernel += elapsedTime2opt + elapsedTime2opt_execute;
        timeCpuKey += timeCpuRefreshTour + timeCpuSelectNonItera + timeCpuExecuteNonItera;
        evaLastRun = evaCurrentRun;

        numOptimizedTotal += numOptimizedOneRun;
        if(numOptimizedOneRun > maxOptExecuPerRun)
            maxOptExecuPerRun = numOptimizedOneRun; // trace max optimized 2opt per run
        if(numOptimizedOneRun > 0)
            vectorNumOptExecuted.push_back(numOptimizedOneRun);
        // trace pdb one run
        vectorPDB.push_back(pdbOneRun);
        outfileTimePerRunRun << timeCpuKey +  timeGpuKernel << " " << endl;
    }
    else{
        numRuns -= 1; // the last run does not optimized the tour
    }

    //test
    cout << "Percentage improve " << percentageImprove << endl << endl;


    // count time refresh
    timeRefresh += (float)timeCpuRefreshTour;
    timeSelect += (float)timeCpuSelectNonItera;
#if CPUEXECUTE
    timeExecute += (float)timeCpuExecuteNonItera;
#else
    timeExecute += timeGpuExecute;
#endif


    return ret;
}//end 5opt



//! \brief Run et activate
//!
//wb.Q 202408 implement 5-opt sequential
template<std::size_t DimP, std::size_t DimCM>
void SolutionKOPT<DimP, DimCM>::runSequential5opt() {

    cout << "Begin run 5-opt sequential >>>>>>>>>>>>>>>" << endl;

    int nCity = md_links_cpu.adaptiveMap.getWidth();
    int numRuns = 0;

    int maxOptExecuPerRun = 0;
    int numOptimizedTotal = 0;
    vector<int> vectorNumOptExecuted;
    vector<float> vectorPDB;
    float timeCpuKey = 0;
    float pdb2optEatFirstPara = 0;
    float timeRefreshTour = 0;
    float timeSearch = 0;
    float timeSelect = 0;
    float timeExecute = 0;
    int numInter = 1;
    // trace maxtimeCPUone2-OoptRun
    float maxtimeCpuOptSearch = 0;

    // outfile timeline
    string fileTimePerRun = "Results_"; //str
    fileTimePerRun.append("Time5optPerRun.txt");
    ofstream outfileTimePerRunRun;
    outfileTimePerRunRun.open(fileTimePerRun);

    // outfile pdbline
    string filePdbPerRun = "Results_"; //str
    filePdbPerRun.append("PdbPer5optRun.txt");
    ofstream outfilePdbPerRunRun;
    outfilePdbPerRunRun.open(filePdbPerRun);

    // outfile pdbline
    string fileSearchTimePerRun = "Results_"; //str
    fileSearchTimePerRun.append("searchTimePer5optRun.txt");
    ofstream outfileSearchTimePerRunRun;
    outfileSearchTimePerRunRun.open(fileSearchTimePerRun);

    outfileTimePerRunRun << 0 << " " << endl;
    outfilePdbPerRunRun << 1143.63 << " " << endl;
    outfileSearchTimePerRunRun << 0 << endl;


    float evaLastRun = 0;
    float percentageImprove = 999999;

    //! prepare the pre-ordered link + coordinates
    Grid<doubleLinkedEdgeForTSP> linkCoordTourCpu;
    linkCoordTourCpu.resize(nCity, 1);

    //wb.Q 2019 add case detection
    if(SolutionKOPT<DimP, DimCM>::md_links_cpu.adaptiveMap.width == 0)
    {
        cout << "Error: no input available." << endl;
        return;
    }
    else
    {
        //wb.Q 2024 rocki 2-opt
        cout << "TSP tour optimum = " << optimum << endl;
        while (numRuns < NUMRUNSLIMIT  && percentageImprove > 0 )
        {
            activateSequential5opt(numRuns, nCity,
                                   maxOptExecuPerRun, numOptimizedTotal, timeCpuKey, vectorNumOptExecuted, vectorPDB,
                                   timeRefreshTour, timeSearch, timeSelect, timeExecute, maxtimeCpuOptSearch,
                                   outfileTimePerRunRun,outfilePdbPerRunRun, outfileSearchTimePerRunRun, evaLastRun, percentageImprove,
                                   linkCoordTourCpu,0);

            if (g_ConfigParameters->traceActive) {
                evaluate();
                writeStatisticsToFile(numRuns, "5optimalTour");
            }
        }
    }// end activateRocki


    //! mean time trace
    timeRefreshTour = timeRefreshTour / numRuns;
    timeSearch = timeSearch / numRuns;
    timeSelect = timeSelect / numRuns;
    timeExecute = timeExecute / numRuns;

    // close outfile
    outfileTimePerRunRun.close();
    outfilePdbPerRunRun.close();
    outfileSearchTimePerRunRun.close();


}// end run

// qiao 2024 add operators to GPU seqential 5-opt
template<std::size_t DimP, std::size_t DimCM>
bool SolutionKOPT<DimP, DimCM>::activateSequential5opt(int& numRuns, int nCity,
                                                       int& maxOptExecuPerRun, int& numOptimizedTotal,
                                                       float& timeCpuKey, vector<int>& vectorNumOptExecuted, vector<float>& vectorPDB,
                                                       float& timeRefresh, float& timeSearch, float& timeSelect, float& timeExecute,
                                                       float &maxtimeCpuOptSearch, ofstream &outfileTimePerRunRun,ofstream &outfilePdbPerRunRun, ofstream &outfileSearchTimePerRunRun,
                                                       float& evaLastRun, float& percentageImprove, Grid<doubleLinkedEdgeForTSP> &linkCoordTourCpu, bool iterOptimal)
{
    cout << endl << "****>>>>Enter sequential 5-opt activate function: " << numRuns << endl;
    bool ret = true;
    numRuns ++;


    maxtimeCpuOptSearch = 0;

    int numOptimizedOneRun = 0;
    int numCityTraversed = 0;

    //!timing runing time on CPU
    float elapsedTime2opt = 0;

    float pdbOneRun = 0;

    //! random starting point
    int ps_random = randomNum(0, nCity);
    PointCoord ps(ps_random, 0);
    cout << "PS [0] " << ps[0] << endl;

    //! clean cityCopy before mark tour ordering
    md_links_firstPara.activeMap.resetValue(0);
    md_links_firstPara.densityMap.resetValue(initialPrepareValue);// densityMap stores node3
    md_links_firstPara.grayValueMap.resetValue(0);// clean orders
    md_links_firstPara.minRadiusMap.resetValue(initialPrepareValue);//  minRadiusMap stores the changeLinks position
    md_links_firstPara.optCandidateMap.resetValue(initialPrepareValue);// optCandidateMap stores opt candidate of 23456-opt

    //!timing runing time on CPU
    __int64 CounterStart = 0;
    double pcFreq = 0.0;
    StartCounter(pcFreq, CounterStart);

    //! mark tour orientation from random starting point ps
    //! reserver, index of linkCoordTourCpu should correspond to index of gray value map
    md_links_firstPara.markNetLinkSequenceReloadRoutCoord(ps, numRuns%2, 0, linkCoordTourCpu);// every ps check its two directions

    // end time cpu
    double timeCpuRefreshTour = GetCounter(pcFreq, CounterStart);
    cout << "Time:: Refresh tour order: " << timeCpuRefreshTour << endl;

    //!timing runing time on CPU
    CounterStart = 0;
    pcFreq = 0.0;
    StartCounter(pcFreq, CounterStart);


#if FIRST
    //qiao 2024 add sequential 5-opt using 5 loops to find each node's candidate 5-opt move, and get multiple 5-opt moves along the tour
    if(iterOptimal == 0)
        md_links_firstPara.sequential5optFirst(linkCoordTourCpu, md_links_cpu.nodeParentMap);// every ps check its two directions
    else
        md_links_firstPara.sequential5optBest(linkCoordTourCpu, md_links_cpu.nVisitedMap);// every ps check its two directions
#endif


    // end time cpu
    double timeCpu5opt = GetCounter(pcFreq, CounterStart);
    cout << "Time:: CPU side one 5-opt runtime : " << timeCpu5opt << endl;


    if(maxtimeCpuOptSearch < timeCpu5opt){
        maxtimeCpuOptSearch = timeCpu5opt;
    }


    //! sequentially select non-interacted 5-exchanges
    //! clean for mark non-interacted 23456-opt
    md_links_firstPara.activeMap.resetValue(initialPrepareValue); // for nodes possessing non interacted 2opt
    md_links_firstPara.fixedMap.resetValue(initialPrepareValue); // for nodes in stackB


    //!timing runing time on CPU
    CounterStart = 0;
    pcFreq = 0.0;
    StartCounter(pcFreq, CounterStart);

    //! select and execute non-interacted 23456-exchanges
    md_links_firstPara.selectNonIteracted23456ExchangeQiao(ps);

    // end time cpu
    double timeCpuSelectNonItera = GetCounter(pcFreq, CounterStart);
    cout << "Time:: select non intera 5-opt: " << timeCpuSelectNonItera << endl;

    double timeCpuExecuteNonItera = 0;
    //!timing runing time on CPU
    CounterStart = 0;
    pcFreq = 0.0;
    StartCounter(pcFreq, CounterStart);

    if(iterOptimal == 0)
        md_links_firstPara.executeNonInteract23456optOnlyNode3(numOptimizedOneRun, md_links_cpu.nodeParentMap);//qiao 2024 need modify

    else
        md_links_firstPara.executeNonInteract23456optOnlyNode3(numOptimizedOneRun, md_links_cpu.nVisitedMap);//qiao 2024 need modify
    // end time cpu
    timeCpuExecuteNonItera = GetCounter(pcFreq, CounterStart);
    cout << "Time:: CPU execute non-intera 5-opt: " << timeCpuExecuteNonItera << endl;


    //! evaluation to stop
    float evaCurrentRun = md_links_firstPara.evaluateWeightOfTSP(distEuclidean, numCityTraversed);
    cout << "Evaluate:: After " << numRuns << "'th run, evaluate tsp length =  " << evaCurrentRun << endl;
    cout << "Evaluate:: In this run, num of 5-exchange been executed: "  <<  numOptimizedOneRun << endl;

    float evaActualLength = md_links_firstPara.evaluateWeightOfTSP(distEuclidean, numCityTraversed);
    cout << "Evaluate:: After " << numRuns << "'th run, evaluate tsp length =  " << evaActualLength << endl;

    //statistic pdb
    if(optimum > 1)
    {
        float evaCurrentPDB = md_links_firstPara.evaluateWeightOfTSP(distEuclidean, numCityTraversed);
        pdbOneRun = (evaCurrentPDB - optimum)*100/optimum;
    }

    if(numRuns == 1){


        //! registrer length of the first run
        evaLastRun = evaCurrentRun;
        //        continue;
    }
    else {
        percentageImprove = ((evaLastRun - evaCurrentRun)*100);
    }


    if(percentageImprove > 0){

        timeCpuKey +=  timeCpu5opt + timeCpuRefreshTour + timeCpuSelectNonItera + timeCpuExecuteNonItera;
        evaLastRun = evaCurrentRun;

        numOptimizedTotal += numOptimizedOneRun;
        if(numOptimizedOneRun > maxOptExecuPerRun)
            maxOptExecuPerRun = numOptimizedOneRun; // trace max optimized 2opt per run
        if(numOptimizedOneRun > 0)
            vectorNumOptExecuted.push_back(numOptimizedOneRun);

        // trace pdb one run
        vectorPDB.push_back(pdbOneRun);
        outfileTimePerRunRun << timeCpuKey << " " << endl;
        outfilePdbPerRunRun << pdbOneRun << " " << endl;
        outfileSearchTimePerRunRun << timeCpu5opt << endl;


        traceTSP.timeObtainKoptimal = timeCpuKey;
        //record the best TSP tour obtained so far
        tspTourBestObtainedSoFar.assign(md_links_firstPara.networkLinks);


    }
    else{
        numRuns -= 1; // the last run does not optimized the tour

        // outfile pdbline
        string fileKoptimalTimePerRun = "Results_"; //str
        fileKoptimalTimePerRun.append("5optimal.txt");
        ofstream outfileKoptimalTimePerRunRun;
        outfileKoptimalTimePerRunRun.open(fileKoptimalTimePerRun);

        outfileKoptimalTimePerRunRun << timeCpuKey  << " ,pdb: " << pdbOneRun << " ,searchTime: " << timeCpu5opt<< endl;

        outfileKoptimalTimePerRunRun.close();
    }

    //test
    cout << "Percentage improve " << percentageImprove << endl << endl;


    // count time refresh
    timeRefresh += (float)timeCpuRefreshTour;
    timeSearch += (float)timeCpu5opt;
    timeSelect += (float)timeCpuSelectNonItera;
    timeExecute += (float)timeCpuExecuteNonItera;



    return ret;
}//end 5opt



//! \brief Run et activate
//!
//wb.Q 202407 implement GPU 3-opt following rocki's method
template<std::size_t DimP, std::size_t DimCM>
void SolutionKOPT<DimP, DimCM>::run3opt() {

    cout << "Begin run 3-opt >>>>>>>>>>>>>>>" << endl;

    int nCity = md_links_cpu.adaptiveMap.getWidth();
    int numRuns = 0;

    int maxOptExecuPerRun = 0;
    int numOptimizedTotal = 0;
    vector<int> vectorNumOptExecuted;
    vector<float> vectorPDB;
    float timeGpuKernel = 0;
    float timeGpuH2D = 0;
    float timeGpuD2H = 0;
    float timeGpuTotal = 0;
    float timeCpuKey = 0;
    float pdb2optEatFirstPara = 0;
    float timeRefreshTour = 0;
    float timeSelect = 0;
    float timeExecute = 0;
    int numInter = 1;
    // trace maxtimeGPUone2-OoptRun
    float maxtimeGpuOptSearch = 0;

    // outfile timeline
    string fileTimePerRun = "Results_"; //str
    fileTimePerRun.append("TimePer3optRun.txt");
    ofstream outfileTimePerRunRun;
    outfileTimePerRunRun.open(fileTimePerRun);

    float evaLastRun = 0;
    float percentageImprove = 999999;

    //! prepare the pre-ordered link + coordinates
    Grid<doubleLinkedEdgeForTSP> linkCoordTourCpu;
    linkCoordTourCpu.resize(nCity, 1);
    Grid<doubleLinkedEdgeForTSP> linkCoordTourGpu;
    linkCoordTourGpu.gpuResize(nCity,1);


    double temp = (double) nCity / (double)6;
    double maxChecks3opt = temp * (nCity - 1) * (nCity - 2) ; // total number of checks for 3-opt
    double iter = ( maxChecks3opt / (BLOCKSIZE * GRIDSIZE) ) ;//+1 ;//need to +1 to get maximum
    if(iter < 1)
        iter = 1;

    cout << "Check maxChecks3opt = " << maxChecks3opt << ", iter = " << iter << endl;

    //wb.Q 2019 add case detection
    if(SolutionKOPT<DimP, DimCM>::md_links_cpu.adaptiveMap.width == 0)
    {
        cout << "Error: no input available." << endl;
        return;
    }
    else
    {
        //wb.Q 2024 rocki 3-opt
        cout << "TSP tour optimum = " << optimum << endl;
        while (numRuns < NUMRUNSLIMIT  && percentageImprove > 0 )
        {
            activateRocki3opt(numRuns, nCity, maxChecks3opt, iter,
                              maxOptExecuPerRun, numOptimizedTotal,
                              timeGpuKernel, timeGpuH2D, timeGpuD2H,
                              timeGpuTotal, timeCpuKey, vectorNumOptExecuted, vectorPDB,
                              timeRefreshTour, timeSelect, timeExecute, maxtimeGpuOptSearch,
                              outfileTimePerRunRun, evaLastRun, percentageImprove,
                              linkCoordTourCpu,linkCoordTourGpu);

            //            if (g_ConfigParameters->traceActive) {
            //                evaluate();
            //                writeStatisticsToFile(iteration);
            //            }
        }
    }// end activateRocki

    //! free gpu memory
    linkCoordTourGpu.gpuFreeMem();


    //! count time gpu total
    timeGpuTotal += timeGpuH2D + timeGpuD2H + timeGpuKernel;


    //! mean time trace
    timeRefreshTour = timeRefreshTour / numRuns;
    timeSelect = timeSelect / numRuns;
    timeExecute = timeExecute / numRuns;

    // close outfile
    outfileTimePerRunRun.close();


}// end run

// qiao 2024 add operators to GPU parallel 23456-opt and massive variable 23456-opt moves on global tour
template<std::size_t DimP, std::size_t DimCM>
bool SolutionKOPT<DimP, DimCM>::activateRocki3opt(int& numRuns, int nCity,double maxChecks3opt, double iter,
                                                  int& maxOptExecuPerRun, int& numOptimizedTotal,
                                                  float& timeGpuKernel, float& timeGpuH2D, float& timeGpuD2H,
                                                  float& timeGpuTotal, float& timeCpuKey, vector<int>& vectorNumOptExecuted, vector<float>& vectorPDB,
                                                  float& timeRefresh, float& timeSelect, float& timeExecute,
                                                  float &maxtimeGpuOptSearch, ofstream &outfileTimePerRunRun,
                                                  float& evaLastRun, float& percentageImprove, Grid<doubleLinkedEdgeForTSP> &linkCoordTourCpu,
                                                  Grid<doubleLinkedEdgeForTSP>& linkCoordTourGpu )
{
    cout << endl << "****>>>>Enter 3-opt activate function: " << numRuns << endl;
    bool ret = true;
    numRuns ++;

    int numOptimizedOneRun = 0;
    int numCityTraversed = 0;

    //!timing runing time on CPU
    //    __int64 CounterStart = 0;
    //    double pcFreq = 0.0;
    float elapsedTime2opt = 0;

    float pdbOneRun = 0;

    //! random starting point
    int ps_random = randomNum(0, nCity);
    PointCoord ps(ps_random, 0);
    cout << "PS [0] " << ps[0] << endl;

    //! clean md_links_firstPara before mark tour ordering
    md_links_firstPara.activeMap.resetValue(initialPrepareValue);
    md_links_firstPara.densityMap.resetValue(initialPrepareValue);// densityMap stores k value of k-opt
    md_links_firstPara.optCandidateMap.resetValue(initialPrepareValue);// optCandidateMap stores opt candidate of 23456-opt
    md_links_firstPara.grayValueMap.resetValue(initialPrepareValue);// clean orders
    md_links_firstPara.minRadiusMap.resetValue(initialPrepareValue);//  minRadiusMap stores the changeLinks position

    //!timing runing time on CPU
    __int64 CounterStart = 0;
    double pcFreq = 0.0;
    StartCounter(pcFreq, CounterStart);

    //! mark tour orientation from random starting point ps, index of linkCoordTourCpu should correspond to index of gray value map
    md_links_firstPara.markNetLinkSequenceReloadRoutCoord(ps, numRuns%2, 0, linkCoordTourCpu);// every ps check its two directions


    // end time cpu
    double timeCpuRefreshTour = GetCounter(pcFreq, CounterStart);
    cout << "Time:: Refresh tour order: " << timeCpuRefreshTour << endl;


    // time for GPU memcp HD
    float elapsedTime2optHD = 0;
    cudaEvent_t startHD, stopHD;
    cudaEventCreate(&startHD);
    cudaEventCreate(&stopHD);
    cudaEventRecord(startHD, 0);

    // copy tour ordering to gpu, clean gpu network links
    md_links_firstPara.grayValueMap.gpuCopyHostToDevice(md_links_gpu.grayValueMap);// refresh tsp order gpu side
    linkCoordTourCpu.gpuCopyHostToDevice(linkCoordTourGpu);// refresh doubly linked tour order



    cudaEventRecord(stopHD, 0);
    cudaEventSynchronize(stopHD);
    cudaEventElapsedTime(&elapsedTime2optHD, startHD, stopHD);
    cudaEventDestroy(startHD);
    cudaEventDestroy(stopHD);
    cout << "Time:: memcp H to D tour order : " <<  elapsedTime2optHD << endl;

    md_links_gpu.densityMap.gpuResetValue(initialPrepareValue);// use for mark k of k-opt
    md_links_gpu.optCandidateMap.gpuResetValue(initialPrepareValueLL);//qiao use for 23456opt
    md_links_gpu.minRadiusMap.gpuResetValue(initialPrepareValue); // use for local min change

    // cuda timer
    double time = 46;
    double *d_time;


    double maxChecksoptDivide = 1.27719e+11;
    double packSize = BLOCKSIZE * GRIDSIZE;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    //qiao only for test
    cout << "Warning: maxChecks3opt= " << maxChecks3opt << ", width: " << md_links_firstPara.adaptiveMap.width << ", gpu.width="
         << md_links_gpu.adaptiveMap.width << endl;


    double iterDivide = (double)maxChecksoptDivide /(double) (packSize);
    if(maxChecks3opt < packSize)
        iterDivide = 1;
    double maxStride = (double) maxChecks3opt /  (double)maxChecksoptDivide;
    if(maxStride < 1)
        maxStride = 0;
    cout << "Changed maxChecks3optDivide = " << maxChecksoptDivide << ", iterDivide = " << iterDivide << ", maxStride= " << maxStride << endl;

    for(double iStride = 0; iStride < maxStride+1; iStride++ )
    {

        K_oneThreadOne3opt_qiao_stride(md_links_gpu, linkCoordTourGpu, maxChecks3opt, maxChecksoptDivide, iterDivide, iStride);

        cudaDeviceSynchronize();

        cout << "Inner one time " << iStride << endl << endl;

    }

    //    //! WB.Q parallel check exhaustive 3-opt along the tour for each edge
    //        K_oneThreadOne3opt_RockiSmall(md_links_gpu, linkCoordTourGpu, maxChecks3opt, iter);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime2opt, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // find the maximum gpu time for a parallel 2-opt run
    cout << "Time:: GPU side one 3-opt runtime : " << elapsedTime2opt << endl;

    if(maxtimeGpuOptSearch < elapsedTime2opt){
        maxtimeGpuOptSearch = elapsedTime2opt;
    }


    //! sequentially select non-interacted 3-exchanges
    float elapsedTime2opt_DH2 = 0;
    cudaEvent_t startDH2, stopDH2;
    cudaEventCreate(&startDH2);
    cudaEventCreate(&stopDH2);
    cudaEventRecord(startDH2, 0);

    md_links_firstPara.densityMap.gpuCopyDeviceToHost(md_links_gpu.densityMap);// node3
    md_links_firstPara.optCandidateMap.gpuCopyDeviceToHost(md_links_gpu.optCandidateMap);//opt candidates

    cudaEventRecord(stopDH2, 0);
    cudaEventSynchronize(stopDH2);
    cudaEventElapsedTime(&elapsedTime2opt_DH2, startDH2, stopDH2);
    cudaEventDestroy(startDH2);
    cudaEventDestroy(stopDH2);
    cout << "Time:: memcp D to H 3-opt candidates: " << elapsedTime2opt_DH2 << endl;



    //qiao only for test
    int numCandidate = 0;
    for(int i = 0; i < md_links_firstPara.optCandidateMap.width; i++ )
    {
        if(md_links_firstPara.optCandidateMap[0][i] > 0)
        {
            numCandidate += 1;
            cout << " candidate order " << md_links_firstPara.grayValueMap[0][i] << endl;
        }

    }
    cout << "After one GPU search num of candidates: " << numCandidate << endl;




    //! clean for mark non-interacted 23456-opt
    md_links_firstPara.activeMap.resetValue(initialPrepareValue); // for nodes possessing non interacted 2opt
    md_links_firstPara.fixedMap.resetValue(initialPrepareValue); // for nodes in stackB

    //!timing runing time on CPU
    CounterStart = 0;
    pcFreq = 0.0;
    StartCounter(pcFreq, CounterStart);

    //! select and execute non-interacted 23456-exchanges
    md_links_firstPara.selectNonIteracted23456ExchangeQiao(ps);

    // end time cpu
    double timeCpuSelectNonItera = GetCounter(pcFreq, CounterStart);
    cout << "Time:: select non intera 3-opt: " << timeCpuSelectNonItera << endl;

    double timeCpuExecuteNonItera = 0;
    float timeGpuExecute = 0;
    float elapsedTime2optHD2 = 0;
    float elapsedTime2optHD3 = 0;
    float elapsedTime2opt_DH = 0;
    float elapsedTime2opt_execute = 0;

#if CPUEXECUTE
    //!timing runing time on CPU
    CounterStart = 0;
    pcFreq = 0.0;
    StartCounter(pcFreq, CounterStart);

    md_links_firstPara.executeNonInteract23456optOnlyNode3(numOptimizedOneRun, md_links_cpu.nodeParentMap);//qiao 2024 need modify

    // end time cpu
    timeCpuExecuteNonItera = GetCounter(pcFreq, CounterStart);
    cout << "Time:: CPU execute non-intera intera 3-opt: " << timeCpuExecuteNonItera << endl;

    cudaEvent_t startHD2, stopHD2;
    cudaEventCreate(&startHD2);
    cudaEventCreate(&stopHD2);
    cudaEventRecord(startHD2, 0);

    md_links_firstPara.networkLinks.gpuCopyHostToDevice(md_links_gpu.networkLinks);
    errorCheckCudaThreadSynchronize();

    cudaEventRecord(stopHD2, 0);
    cudaEventSynchronize(stopHD2);
    cudaEventElapsedTime(&elapsedTime2optHD3, startHD2, stopHD2);
    cudaEventDestroy(startHD2);
    cudaEventDestroy(stopHD2);
    cout << "Time:: memcp H to D networkLinks: " <<  elapsedTime2optHD3 << endl;

#else

    //! copy activeMap (selected 2-exchanges) to device HD
    cudaEvent_t startHD2, stopHD2;
    cudaEventCreate(&startHD2);
    cudaEventCreate(&stopHD2);
    cudaEventRecord(startHD2, 0);

    md_links_firstPara.activeMap.gpuCopyHostToDevice(md_links_gpu.activeMap);

    cudaEventRecord(stopHD2, 0);
    cudaEventSynchronize(stopHD2);
    cudaEventElapsedTime(&elapsedTime2optHD2, startHD2, stopHD2);
    cudaEventDestroy(startHD2);
    cudaEventDestroy(stopHD2);
    cout << "memcp H to D activeValueMap : " <<  elapsedTime2optHD2 << endl;

    //! kernel execute selected 2-exchanges
    // cuda timer
    cudaEvent_t start2, stop2;
    cudaEventCreate(&start2);
    cudaEventCreate(&stop2);
    cudaEventRecord(start2, 0);

    K_executeNonItera2ExchangeOnlyWithNode3(md_links_gpu);

    cudaEventRecord(stop2, 0);
    cudaEventSynchronize(stop2);
    cudaEventElapsedTime(&elapsedTime2opt_execute, start2, stop2);
    cudaEventDestroy(start2);
    cudaEventDestroy(stop2);

    cout << "gpu search 2opt in parallel " << elapsedTime2opt << endl;
    cout << " gpu execute non intera 2-exchange time " << elapsedTime2opt_execute << endl;

    //        //! copy new tour to host DH

    cudaEvent_t startDH, stopDH;
    cudaEventCreate(&startDH);
    cudaEventCreate(&stopDH);
    cudaEventRecord(startDH, 0);

    md_links_firstPara.networkLinks.gpuCopyDeviceToHost(md_links_gpu.networkLinks);

    cudaEventRecord(stopDH, 0);
    cudaEventSynchronize(stopDH);
    cudaEventElapsedTime(&elapsedTime2opt_DH, startDH, stopDH);
    cudaEventDestroy(startDH);
    cudaEventDestroy(stopDH);
    cout << "memcp device to host networklinks " << elapsedTime2opt_DH << endl;

#endif


    timeGpuExecute =  elapsedTime2optHD2 + elapsedTime2opt_execute + elapsedTime2opt_DH ;
    cout << "Time:: timeGpu Execute " << timeGpuExecute << endl;


    //! evaluation to stop
    float evaCurrentRun = md_links_firstPara.evaluateWeightOfTSP(dist, numCityTraversed);
    cout << "Evaluate:: After " << numRuns << "'th run, evaluate tsp length =  " << evaCurrentRun << endl;
    cout << "Evaluate:: In this run, num of 3-exchange been executed: "  <<  numOptimizedOneRun << endl;

    float evaActualLength = md_links_firstPara.evaluateWeightOfTSP(distEuclidean, numCityTraversed);
    cout << "Evaluate:: After " << numRuns << "'th run, evaluate tsp length =  " << evaActualLength << endl;


    //statistic pdb
    if(optimum > 1)
    {
        float evaCurrentPDB = md_links_firstPara.evaluateWeightOfTSP(dist, numCityTraversed);
        pdbOneRun = (evaCurrentPDB - optimum)*100/optimum;
    }

    if(numRuns == 1){
        timeGpuH2D += elapsedTime2optHD + elapsedTime2optHD2 + elapsedTime2optHD3;
        timeGpuD2H += elapsedTime2opt_DH + elapsedTime2opt_DH2;
        timeGpuKernel += elapsedTime2opt + elapsedTime2opt_execute;
        timeCpuKey += timeCpuRefreshTour + timeCpuSelectNonItera + timeCpuExecuteNonItera;

        numOptimizedTotal += numOptimizedOneRun;
        if(numOptimizedOneRun > maxOptExecuPerRun)
            maxOptExecuPerRun = numOptimizedOneRun; // trace max optimized 2opt per run
        if(numOptimizedOneRun > 0)
            vectorNumOptExecuted.push_back(numOptimizedOneRun);

        // trace pdb one run
        vectorPDB.push_back(pdbOneRun);
        outfileTimePerRunRun << timeCpuKey +  timeGpuKernel << " " << endl;

        //! registrer length of the first run
        evaLastRun = evaCurrentRun;
        //        continue;
    }
    else {
        percentageImprove = ((evaLastRun - evaCurrentRun)*100);
    }


    if(percentageImprove > 0){
        timeGpuH2D += elapsedTime2optHD + elapsedTime2optHD2;
        timeGpuD2H += elapsedTime2opt_DH + elapsedTime2opt_DH2;
        timeGpuKernel += elapsedTime2opt + elapsedTime2opt_execute;
        timeCpuKey += timeCpuRefreshTour + timeCpuSelectNonItera + timeCpuExecuteNonItera;
        evaLastRun = evaCurrentRun;

        numOptimizedTotal += numOptimizedOneRun;
        if(numOptimizedOneRun > maxOptExecuPerRun)
            maxOptExecuPerRun = numOptimizedOneRun; // trace max optimized 2opt per run
        if(numOptimizedOneRun > 0)
            vectorNumOptExecuted.push_back(numOptimizedOneRun);
        // trace pdb one run
        vectorPDB.push_back(pdbOneRun);
        outfileTimePerRunRun << timeCpuKey +  timeGpuKernel << " " << endl;
    }
    else{
        numRuns -= 1; // the last run does not optimized the tour
    }

    //test
    cout << "Percentage improve " << percentageImprove << endl << endl;


    // count time refresh
    timeRefresh += (float)timeCpuRefreshTour;
    timeSelect += (float)timeCpuSelectNonItera;
#if CPUEXECUTE
    timeExecute += (float)timeCpuExecuteNonItera;
#else
    timeExecute += timeGpuExecute;
#endif

    return ret;
}




//! \brief Run et activate
//!
//wb.Q 202409 test running time of sequential 3-opt
template<std::size_t DimP, std::size_t DimCM>
void SolutionKOPT<DimP, DimCM>::runSequential3opt() {

    cout << "Begin run sequential 3-opt >>>>>>>>>>>>>>>" << endl;

    int nCity = md_links_cpu.adaptiveMap.getWidth();
    int numRuns = 0;

    int maxOptExecuPerRun = 0;
    int numOptimizedTotal = 0;
    vector<int> vectorNumOptExecuted;
    vector<float> vectorPDB;
    float timeCpuKey = 0;
    float timeRefreshTour = 0;
    float timeSearch = 0;
    float timeSelect = 0;
    float timeExecute = 0;
    int numInter = 1;
    // trace maxtimeGPUone2-OoptRun
    float maxtimeCpuOptSearch = 0;


    // outfile timeline
    string fileTimePerRun = "Results_"; //str
    fileTimePerRun.append("TimePer3optRun.txt");
    ofstream outfileTimePerRunRun;
    outfileTimePerRunRun.open(fileTimePerRun);

    // outfile pdbline
    string filePdbPerRun = "Results_"; //str
    filePdbPerRun.append("PdbPer3optRun.txt");
    ofstream outfilePdbPerRunRun;
    outfilePdbPerRunRun.open(filePdbPerRun);

    // outfile pdbline
    string fileSearchTimePerRun = "Results_"; //str
    fileSearchTimePerRun.append("searchTimePer3optRun.txt");
    ofstream outfileSearchTimePerRunRun;
    outfileSearchTimePerRunRun.open(fileSearchTimePerRun);

    outfileTimePerRunRun << 0 << " " << endl;
    outfilePdbPerRunRun << 1143.63 << " " << endl;
    outfileSearchTimePerRunRun << 0 << endl;



    float evaLastRun = 0;
    float percentageImprove = 999999;

    //! prepare the pre-ordered link + coordinates
    Grid<doubleLinkedEdgeForTSP> linkCoordTourCpu;
    linkCoordTourCpu.resize(nCity, 1);

    //wb.Q 2019 add case detection
    if(SolutionKOPT<DimP, DimCM>::md_links_cpu.adaptiveMap.width == 0)
    {
        cout << "Error: no input available." << endl;
        return;
    }
    else
    {
        //wb.Q 2024 rocki 3-opt
        cout << "TSP tour optimum = " << optimum << endl;
        while (numRuns < NUMRUNSLIMIT  && percentageImprove > 0 )
        {
            activateSequential3opt(numRuns, nCity,
                                   maxOptExecuPerRun, numOptimizedTotal,
                                   timeCpuKey, vectorNumOptExecuted, vectorPDB,
                                   timeRefreshTour, timeSearch, timeSelect, timeExecute, maxtimeCpuOptSearch,
                                   outfileTimePerRunRun, outfilePdbPerRunRun,outfileSearchTimePerRunRun, evaLastRun, percentageImprove,
                                   linkCoordTourCpu);

            if (g_ConfigParameters->traceActive) {
                evaluate();
                writeStatisticsToFile(numRuns, "3optimalTour");
            }
        }
    }// end activateRocki

    //! mean time trace
    timeRefreshTour = timeRefreshTour / numRuns;
    timeSearch = timeSearch / numRuns;
    timeSelect = timeSelect / numRuns;
    timeExecute = timeExecute / numRuns;

    // close outfile
    outfileTimePerRunRun.close();

    outfilePdbPerRunRun.close();
    outfileSearchTimePerRunRun.close();


}// end run

// qiao 2024 add operators to CPU sequential 3-opt
template<std::size_t DimP, std::size_t DimCM>
bool SolutionKOPT<DimP, DimCM>::activateSequential3opt(int& numRuns, int nCity,
                                                       int& maxOptExecuPerRun, int& numOptimizedTotal,
                                                       float& timeCpuKey, vector<int>& vectorNumOptExecuted, vector<float>& vectorPDB,
                                                       float& timeRefresh, float& timeSearch, float& timeSelect, float& timeExecute,
                                                       float &maxtimeCpuOptSearch, ofstream &outfileTimePerRunRun, ofstream &outfilePdbPerRunRun,  ofstream &outfileSearchTimePerRunRun,
                                                       float& evaLastRun, float& percentageImprove, Grid<doubleLinkedEdgeForTSP> &linkCoordTourCpu)
{
    cout << endl << "****>>>>Enter sequential 3-opt activate function: " << numRuns << endl;
    bool ret = true;
    numRuns ++;

    maxtimeCpuOptSearch = 0;

    int numOptimizedOneRun = 0;
    int numCityTraversed = 0;
    float pdbOneRun = 0;

    //! random starting point
    int ps_random = randomNum(0, nCity);
    PointCoord ps(ps_random, 0);
    cout << "PS [0] " << ps[0] << endl;

    //! clean md_links_firstPara before mark tour ordering
    md_links_firstPara.activeMap.resetValue(initialPrepareValue);
    md_links_firstPara.densityMap.resetValue(initialPrepareValue);// densityMap stores k value of k-opt
    md_links_firstPara.optCandidateMap.resetValue(initialPrepareValue);// optCandidateMap stores opt candidate of 23456-opt
    md_links_firstPara.grayValueMap.resetValue(initialPrepareValue);// clean orders
    md_links_firstPara.minRadiusMap.resetValue(initialPrepareValue);//  minRadiusMap stores the changeLinks position

    //!timing runing time on CPU
    __int64 CounterStart = 0;
    double pcFreq = 0.0;
    StartCounter(pcFreq, CounterStart);

    //! mark tour orientation from random starting point ps, index of linkCoordTourCpu should correspond to index of gray value map
    md_links_firstPara.markNetLinkSequenceReloadRoutCoord(ps, numRuns%2, 0, linkCoordTourCpu);// every ps check its two directions


    // end time cpu
    double timeCpuRefreshTour = GetCounter(pcFreq, CounterStart); /////////////////////////////time record 1
    cout << "Time:: Refresh tour order: " << timeCpuRefreshTour << endl;

    //!timing runing time on CPU
    CounterStart = 0;
    pcFreq = 0.0;
    StartCounter(pcFreq, CounterStart);

#if FIRST

    //qiao add sequential 3-opt and find all possible 3-opt for each node and get multiple candidate 3-opt moves
    md_links_firstPara.sequential3optFirst(linkCoordTourCpu);// every ps check its two directions

#else
    md_links_firstPara.sequential3optBest(linkCoordTourCpu);// every ps check its two directions
#endif

    // end time cpu
    double timeCpu3opt = GetCounter(pcFreq, CounterStart); /////////////////////////////time record 2
    cout << "Time:: CPU side one 3-opt runtime : " << timeCpu3opt << endl;

    if(maxtimeCpuOptSearch < timeCpu3opt){
        maxtimeCpuOptSearch = timeCpu3opt;
    }


    //! sequentially select non-interacted 3-exchanges
    //! clean for mark non-interacted 23456-opt
    md_links_firstPara.activeMap.resetValue(initialPrepareValue); // for nodes possessing non interacted 2opt
    md_links_firstPara.fixedMap.resetValue(initialPrepareValue); // for nodes in stackB

    //!timing runing time on CPU
    CounterStart = 0;
    pcFreq = 0.0;
    StartCounter(pcFreq, CounterStart);

    //! select and execute non-interacted 23456-exchanges
    md_links_firstPara.selectNonIteracted23456ExchangeQiao(ps);

    // end time cpu
    double timeCpuSelectNonItera = GetCounter(pcFreq, CounterStart);  /////////////////////////////time record 3
    cout << "Time:: select non intera 3-opt: " << timeCpuSelectNonItera << endl;

    double timeCpuExecuteNonItera = 0;
    //!timing runing time on CPU
    CounterStart = 0;
    pcFreq = 0.0;
    StartCounter(pcFreq, CounterStart);

    md_links_firstPara.executeNonInteract23456optOnlyNode3(numOptimizedOneRun, md_links_cpu.nodeParentMap);//qiao 2024 need modify

    // end time cpu
    timeCpuExecuteNonItera = GetCounter(pcFreq, CounterStart); /////////////////////////////time record 4
    cout << "Time:: CPU execute non-intera intera 3-opt: " << timeCpuExecuteNonItera << endl;


    //! evaluation to stop
    float evaCurrentRun = md_links_firstPara.evaluateWeightOfTSP(distEuclidean, numCityTraversed);
    cout << "Evaluate:: After " << numRuns << "'th run, evaluate tsp length =  " << evaCurrentRun << endl;
    cout << "Evaluate:: In this run, num of 3-exchange been executed: "  <<  numOptimizedOneRun << endl;

    float evaActualLength = md_links_firstPara.evaluateWeightOfTSP(distEuclidean, numCityTraversed);
    cout << "Evaluate:: After " << numRuns << "'th run, evaluate tsp length =  " << evaActualLength << endl;


    //statistic pdb
    if(optimum > 1)
    {
        float evaCurrentPDB = md_links_firstPara.evaluateWeightOfTSP(distEuclidean, numCityTraversed);
        pdbOneRun = (evaCurrentPDB - optimum)*100/optimum;
    }

    if(numRuns == 1)
    {

        //! registrer length of the first run
        evaLastRun = evaCurrentRun;
    }
    else {
        percentageImprove = ((evaLastRun - evaCurrentRun)*100);
    }


    if(percentageImprove > 0){
        timeCpuKey += timeCpu3opt + timeCpuRefreshTour + timeCpuSelectNonItera + timeCpuExecuteNonItera;
        evaLastRun = evaCurrentRun;

        numOptimizedTotal += numOptimizedOneRun;
        if(numOptimizedOneRun > maxOptExecuPerRun)
            maxOptExecuPerRun = numOptimizedOneRun; // trace max optimized 2opt per run
        if(numOptimizedOneRun > 0)
            vectorNumOptExecuted.push_back(numOptimizedOneRun);
        // trace pdb one run
        vectorPDB.push_back(pdbOneRun);
        outfileTimePerRunRun << timeCpuKey << " " << endl; // total running time by far

        outfilePdbPerRunRun << pdbOneRun << " " << endl;
        outfileSearchTimePerRunRun << timeCpu3opt << " " << endl;


        traceTSP.timeObtainKoptimal = timeCpuKey;
        //record the best TSP tour obtained so far
        tspTourBestObtainedSoFar.assign(md_links_firstPara.networkLinks);


    }
    else{
        numRuns -= 1; // the last run does not optimized the tour

        // outfile pdbline
        string fileKoptimalTimePerRun = "Results_"; //str
        fileKoptimalTimePerRun.append("3optimal.txt");
        ofstream outfileKoptimalTimePerRunRun;
        outfileKoptimalTimePerRunRun.open(fileKoptimalTimePerRun);

        outfileKoptimalTimePerRunRun << timeCpuKey  << " ,pdb: " << pdbOneRun << " ,searchTime: " << timeCpu3opt<< endl;

        outfileKoptimalTimePerRunRun.close();
    }

    //test
    cout << "Percentage improve " << percentageImprove << endl << endl;


    // count time refresh
    timeRefresh += (float)timeCpuRefreshTour;
    timeSearch += (float)timeCpu3opt;
    timeSelect += (float)timeCpuSelectNonItera;
    timeExecute += (float)timeCpuExecuteNonItera;

    return ret;
}


//! \brief Run et activate
//!
//wb.Q 202408 implement 6-opt
template<std::size_t DimP, std::size_t DimCM>
void SolutionKOPT<DimP, DimCM>::run6opt() {

    cout << "Begin run 6-opt >>>>>>>>>>>>>>>" << endl;

    int nCity = md_links_cpu.adaptiveMap.getWidth();
    int numRuns = 0;

    int maxOptExecuPerRun = 0;
    int numOptimizedTotal = 0;
    vector<int> vectorNumOptExecuted;
    vector<float> vectorPDB;
    float timeGpuKernel = 0;
    float timeGpuH2D = 0;
    float timeGpuD2H = 0;
    float timeGpuTotal = 0;
    float timeCpuKey = 0;
    float pdb2optEatFirstPara = 0;
    float timeRefreshTour = 0;
    float timeSelect = 0;
    float timeExecute = 0;
    int numInter = 1;
    // trace maxtimeGPUone2-OoptRun
    float maxtimeGpuOptSearch = 0;

    // outfile timeline
    string fileTimePerRun = "Results_"; //str
    fileTimePerRun.append("TimePerRun.txt");
    ofstream outfileTimePerRunRun;
    outfileTimePerRunRun.open(fileTimePerRun);

    float evaLastRun = 0;
    float percentageImprove = 999999;

    //! prepare the pre-ordered link + coordinates
    Grid<doubleLinkedEdgeForTSP> linkCoordTourCpu;
    linkCoordTourCpu.resize(nCity, 1);
    Grid<doubleLinkedEdgeForTSP> linkCoordTourGpu;
    linkCoordTourGpu.gpuResize(nCity,1);



    double maxChecks3opt = nCity*(nCity - 1)*(nCity - 2) / 6; // total number of checks for 3-opt

    double maxCheck6opt = maxChecks3opt*(maxChecks3opt - 1) / 2;  // total number of checks for 6-opt
    unsigned int iter = maxCheck6opt / (BLOCKSIZE * GRIDSIZE);

    cout << "Check maxChecks6opt = " << maxCheck6opt << ", iter = " << iter << endl;


    //wb.Q 2019 add case detection
    if(SolutionKOPT<DimP, DimCM>::md_links_cpu.adaptiveMap.width == 0)
    {
        cout << "Error: no input available." << endl;
        return;
    }
    else
    {
        //wb.Q 2024 rocki 2-opt
        cout << "TSP tour optimum = " << optimum << endl;
        while (numRuns < NUMRUNSLIMIT  && percentageImprove > 0 )
        {
            activateRocki6opt(numRuns, nCity, maxCheck6opt, maxChecks3opt,iter,
                              maxOptExecuPerRun, numOptimizedTotal,
                              timeGpuKernel, timeGpuH2D, timeGpuD2H,
                              timeGpuTotal, timeCpuKey, vectorNumOptExecuted, vectorPDB,
                              timeRefreshTour, timeSelect, timeExecute, maxtimeGpuOptSearch,
                              outfileTimePerRunRun, evaLastRun, percentageImprove,
                              linkCoordTourCpu,linkCoordTourGpu);

            //            if (g_ConfigParameters->traceActive) {
            //                evaluate();
            //                writeStatisticsToFile(iteration);
            //            }
        }
    }// end activateRocki

    //! free gpu memory
    linkCoordTourGpu.gpuFreeMem();


    //! count time gpu total
    timeGpuTotal += timeGpuH2D + timeGpuD2H + timeGpuKernel;


    //! mean time trace
    timeRefreshTour = timeRefreshTour / numRuns;
    timeSelect = timeSelect / numRuns;
    timeExecute = timeExecute / numRuns;

    // close outfile
    outfileTimePerRunRun.close();



}// end run

// qiao 2024 add operators to GPU parallel 23456-opt and massive variable 23456-opt moves on global tour
template<std::size_t DimP, std::size_t DimCM>
bool SolutionKOPT<DimP, DimCM>::activateRocki6opt(int& numRuns, int nCity, double maxChecks6opt,
                                                  double maxChecks3opt, unsigned int iter,
                                                  int& maxOptExecuPerRun, int& numOptimizedTotal,
                                                  float& timeGpuKernel, float& timeGpuH2D, float& timeGpuD2H,
                                                  float& timeGpuTotal, float& timeCpuKey, vector<int>& vectorNumOptExecuted, vector<float>& vectorPDB,
                                                  float& timeRefresh, float& timeSelect, float& timeExecute,
                                                  float &maxtimeGpuOptSearch, ofstream &outfileTimePerRunRun,
                                                  float& evaLastRun, float& percentageImprove, Grid<doubleLinkedEdgeForTSP> &linkCoordTourCpu,
                                                  Grid<doubleLinkedEdgeForTSP>& linkCoordTourGpu )
{
    cout << endl << "****>>>>Enter 6-opt activate function: " << numRuns << endl;
    bool ret = true;
    numRuns ++;

    int numOptimizedOneRun = 0;
    int numCityTraversed = 0;


    //!timing runing time on CPU
    //    __int64 CounterStart = 0;
    //    double pcFreq = 0.0;
    float elapsedTime2opt = 0;


    float pdbOneRun = 0;

    //! random starting point
    int ps_random = randomNum(0, nCity);
    PointCoord ps(ps_random, 0);
    cout << "PS [0] " << ps[0] << endl;

    //! clean cityCopy before mark tour ordering
    md_links_firstPara.activeMap.resetValue(0);
    md_links_firstPara.densityMap.resetValue(initialPrepareValue);// densityMap stores node3
    md_links_firstPara.optCandidateMap.resetValue(initialPrepareValue);// optCandidateMap stores opt candidate of 23456-opt
    md_links_firstPara.grayValueMap.resetValue(0);// clean orders
    md_links_firstPara.minRadiusMap.resetValue(initialPrepareValue);//  minRadiusMap stores the changeLinks position

    //!timing runing time on CPU
    __int64 CounterStart = 0;
    double pcFreq = 0.0;
    StartCounter(pcFreq, CounterStart);

    //! mark tour orientation from random starting point ps
    //! reserver, index of linkCoordTourCpu should correspond to index of gray value map
    md_links_firstPara.markNetLinkSequenceReloadRoutCoord(ps, numRuns%2, 0, linkCoordTourCpu);// every ps check its two directions

    // end time cpu
    double timeCpuRefreshTour = GetCounter(pcFreq, CounterStart);
    cout << "Time:: Refresh tour order: " << timeCpuRefreshTour << endl;


    // time for GPU memcp HD
    float elapsedTime2optHD = 0;
    cudaEvent_t startHD, stopHD;
    cudaEventCreate(&startHD);
    cudaEventCreate(&stopHD);
    cudaEventRecord(startHD, 0);

    // copy tour ordering to gpu, clean gpu network links
    md_links_firstPara.grayValueMap.gpuCopyHostToDevice(md_links_gpu.grayValueMap);// refresh tsp order gpu side
    linkCoordTourCpu.gpuCopyHostToDevice(linkCoordTourGpu);// refresh doubly linked tour order

    cudaEventRecord(stopHD, 0);
    cudaEventSynchronize(stopHD);
    cudaEventElapsedTime(&elapsedTime2optHD, startHD, stopHD);
    cudaEventDestroy(startHD);
    cudaEventDestroy(stopHD);
    cout << "Time:: memcp H to D grayValueMap : " <<  elapsedTime2optHD << endl;


    md_links_gpu.densityMap.gpuResetValue(initialPrepareValue);// use for node3
    md_links_gpu.minRadiusMap.gpuResetValue(initialPrepareValue); // use for local min change
    md_links_gpu.optCandidateMap.gpuResetValue(initialPrepareValueLL);//qiao use for 23456opt


    // cuda timer
    double time = 46;
    double *d_time;

    double maxChecksoptDivide = 1.27719e+11;
    double packSize = BLOCKSIZE * GRIDSIZE;


    //qiao only for test
    cout << "Warning: maxChecks6opt= " << maxChecks6opt << endl;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);



    double iterDivide = (double)maxChecksoptDivide /(double) (packSize);
    if(maxChecks6opt < packSize)
        iterDivide = 1;
    double maxStride = (double) maxChecks6opt /  (double)maxChecksoptDivide;
    if(maxStride < 1)
        maxStride = 0;
    cout << "Changed maxChecks6optDivide = " << maxChecksoptDivide << ", iterDivide = " << iterDivide << ", maxStride= " << maxStride << endl;

    for(double iStride = 0; iStride < maxStride+1; iStride++ )
    {

        K_oneThreadOne6opt_qiao_iterStride(md_links_gpu, linkCoordTourGpu,maxChecks6opt, maxChecks3opt, maxChecksoptDivide, iterDivide, iStride);

        cudaDeviceSynchronize();

        cout << "Inner one time " << iStride << endl << endl;

    }

    //! WB.Q parallel check exhaustive 6-opt along the tour for each edge
    //    K_oneThreadOne6opt_RockiSmall(md_links_gpu, linkCoordTourGpu, maxChecks6opt,  maxChecks3opt, iter);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime2opt, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // find the maximum gpu time for a parallel 2-opt run
    cout << "Time:: GPU side one 6-opt run : " << elapsedTime2opt << endl;

    if(maxtimeGpuOptSearch < elapsedTime2opt){
        maxtimeGpuOptSearch = elapsedTime2opt;
    }


    //! sequentially select non-interacted 2-exchanges
    float elapsedTime2opt_DH2 = 0;
    cudaEvent_t startDH2, stopDH2;
    cudaEventCreate(&startDH2);
    cudaEventCreate(&stopDH2);
    cudaEventRecord(startDH2, 0);

    md_links_firstPara.densityMap.gpuCopyDeviceToHost(md_links_gpu.densityMap);
    md_links_firstPara.optCandidateMap.gpuCopyDeviceToHost(md_links_gpu.optCandidateMap);//opt candidates


    cudaEventRecord(stopDH2, 0);
    cudaEventSynchronize(stopDH2);
    cudaEventElapsedTime(&elapsedTime2opt_DH2, startDH2, stopDH2);
    cudaEventDestroy(startDH2);
    cudaEventDestroy(stopDH2);
    cout << "Time:: memcp D to H " << elapsedTime2opt_DH2 << endl;


    //qiao only for test
    int numCandidate = 0;
    for(int i = 0; i < md_links_firstPara.optCandidateMap.width; i++ )
    {
        if(md_links_firstPara.optCandidateMap[0][i] > 0)
        {
            numCandidate += 1;
            cout << " candidate order " << md_links_firstPara.grayValueMap[0][i] << endl;
        }

    }
    cout << "After one GPU search num of candidates: " << numCandidate << endl;


    //! clean for mark non-interacted 6-opt
    md_links_firstPara.activeMap.resetValue(0); // for nodes possessing non interacted 2opt
    md_links_firstPara.fixedMap.resetValue(0); // for nodes in stackB


    //!timing runing time on CPU
    CounterStart = 0;
    pcFreq = 0.0;
    StartCounter(pcFreq, CounterStart);

    //! select and execute non-interacted 23456-exchanges
    md_links_firstPara.selectNonIteracted23456ExchangeQiao(ps);

    // end time cpu
    double timeCpuSelectNonItera = GetCounter(pcFreq, CounterStart);
    cout << "Time:: select non intera 6-opt: " << timeCpuSelectNonItera << endl;

    double timeCpuExecuteNonItera = 0;
    float timeGpuExecute = 0;
    float elapsedTime2optHD2 = 0;
    float elapsedTime2optHD3 = 0;
    float elapsedTime2opt_DH = 0;
    float elapsedTime2opt_execute = 0;

#if CPUEXECUTE
    //!timing runing time on CPU
    CounterStart = 0;
    pcFreq = 0.0;
    StartCounter(pcFreq, CounterStart);
    md_links_firstPara.executeNonInteract23456optOnlyNode3(numOptimizedOneRun, md_links_cpu.nodeParentMap);//qiao 2024 need modify

    // end time cpu
    timeCpuExecuteNonItera = GetCounter(pcFreq, CounterStart);
    cout << "Time:: CPU execute non-intera 6-opt: " << timeCpuExecuteNonItera << endl;

    cudaEvent_t startHD2, stopHD2;
    cudaEventCreate(&startHD2);
    cudaEventCreate(&stopHD2);
    cudaEventRecord(startHD2, 0);

    md_links_firstPara.networkLinks.gpuCopyHostToDevice(md_links_gpu.networkLinks);
    errorCheckCudaThreadSynchronize();

    cudaEventRecord(stopHD2, 0);
    cudaEventSynchronize(stopHD2);
    cudaEventElapsedTime(&elapsedTime2optHD3, startHD2, stopHD2);
    cudaEventDestroy(startHD2);
    cudaEventDestroy(stopHD2);
    cout << "Time:: memcp H to D networkLinks:  " <<  elapsedTime2optHD3 << endl;

#else

    //! copy activeMap (selected 2-exchanges) to device HD
    cudaEvent_t startHD2, stopHD2;
    cudaEventCreate(&startHD2);
    cudaEventCreate(&stopHD2);
    cudaEventRecord(startHD2, 0);

    md_links_firstPara.activeMap.gpuCopyHostToDevice(md_links_gpu.activeMap);

    cudaEventRecord(stopHD2, 0);
    cudaEventSynchronize(stopHD2);
    cudaEventElapsedTime(&elapsedTime2optHD2, startHD2, stopHD2);
    cudaEventDestroy(startHD2);
    cudaEventDestroy(stopHD2);
    cout << "memcp H to D activeValueMap : " <<  elapsedTime2optHD2 << endl;

    //! kernel execute selected 2-exchanges
    // cuda timer
    cudaEvent_t start2, stop2;
    cudaEventCreate(&start2);
    cudaEventCreate(&stop2);
    cudaEventRecord(start2, 0);

    K_executeNonItera2ExchangeOnlyWithNode3(md_links_gpu);

    cudaEventRecord(stop2, 0);
    cudaEventSynchronize(stop2);
    cudaEventElapsedTime(&elapsedTime2opt_execute, start2, stop2);
    cudaEventDestroy(start2);
    cudaEventDestroy(stop2);

    cout << "gpu search 2opt in parallel " << elapsedTime2opt << endl;
    cout << " gpu execute non intera 2-exchange time " << elapsedTime2opt_execute << endl;

    //        //! copy new tour to host DH

    cudaEvent_t startDH, stopDH;
    cudaEventCreate(&startDH);
    cudaEventCreate(&stopDH);
    cudaEventRecord(startDH, 0);

    md_links_firstPara.networkLinks.gpuCopyDeviceToHost(md_links_gpu.networkLinks);

    cudaEventRecord(stopDH, 0);
    cudaEventSynchronize(stopDH);
    cudaEventElapsedTime(&elapsedTime2opt_DH, startDH, stopDH);
    cudaEventDestroy(startDH);
    cudaEventDestroy(stopDH);
    cout << "memcp device to host networklinks " << elapsedTime2opt_DH << endl;

#endif


    timeGpuExecute =  elapsedTime2optHD2 + elapsedTime2opt_execute + elapsedTime2opt_DH ;
    cout << "Time:: timeGpu Execute " << timeGpuExecute << endl;


    //! evaluation to stop
    float evaCurrentRun = md_links_firstPara.evaluateWeightOfTSP(dist, numCityTraversed);
    cout << "Evaluate:: After " << numRuns << "'th run, evaluate tsp length =  " << evaCurrentRun << endl;
    cout << "Evaluate:: In this run, num of 6-exchange been executed: "  <<  numOptimizedOneRun << endl;

    float evaActualLength = md_links_firstPara.evaluateWeightOfTSP(distEuclidean, numCityTraversed);
    cout << "Evaluate:: After " << numRuns << "'th run, evaluate tsp length =  " << evaActualLength << endl;

    //statistic pdb
    if(optimum > 1)
    {
        float evaCurrentPDB = md_links_firstPara.evaluateWeightOfTSP(dist, numCityTraversed);
        pdbOneRun = (evaCurrentPDB - optimum)*100/optimum;
    }

    if(numRuns == 1){
        timeGpuH2D += elapsedTime2optHD + elapsedTime2optHD2 + elapsedTime2optHD3;
        timeGpuD2H += elapsedTime2opt_DH + elapsedTime2opt_DH2;
        timeGpuKernel += elapsedTime2opt + elapsedTime2opt_execute;
        timeCpuKey += timeCpuRefreshTour + timeCpuSelectNonItera + timeCpuExecuteNonItera;

        numOptimizedTotal += numOptimizedOneRun;
        if(numOptimizedOneRun > maxOptExecuPerRun)
            maxOptExecuPerRun = numOptimizedOneRun; // trace max optimized 2opt per run
        if(numOptimizedOneRun > 0)
            vectorNumOptExecuted.push_back(numOptimizedOneRun);

        // trace pdb one run
        vectorPDB.push_back(pdbOneRun);
        outfileTimePerRunRun << timeCpuKey +  timeGpuKernel << " " << endl;

        //! registrer length of the first run
        evaLastRun = evaCurrentRun;
        //        continue;
    }
    else {
        percentageImprove = ((evaLastRun - evaCurrentRun)*100);
    }


    if(percentageImprove > 0){
        timeGpuH2D += elapsedTime2optHD + elapsedTime2optHD2;
        timeGpuD2H += elapsedTime2opt_DH + elapsedTime2opt_DH2;
        timeGpuKernel += elapsedTime2opt + elapsedTime2opt_execute;
        timeCpuKey += timeCpuRefreshTour + timeCpuSelectNonItera + timeCpuExecuteNonItera;
        evaLastRun = evaCurrentRun;

        numOptimizedTotal += numOptimizedOneRun;
        if(numOptimizedOneRun > maxOptExecuPerRun)
            maxOptExecuPerRun = numOptimizedOneRun; // trace max optimized 2opt per run
        if(numOptimizedOneRun > 0)
            vectorNumOptExecuted.push_back(numOptimizedOneRun);
        // trace pdb one run
        vectorPDB.push_back(pdbOneRun);
        outfileTimePerRunRun << timeCpuKey +  timeGpuKernel << " " << endl;
    }
    else{
        numRuns -= 1; // the last run does not optimized the tour
    }

    //test
    cout << "Percentage improve " << percentageImprove << endl << endl;


    // count time refresh
    timeRefresh += (float)timeCpuRefreshTour;
    timeSelect += (float)timeCpuSelectNonItera;
#if CPUEXECUTE
    timeExecute += (float)timeCpuExecuteNonItera;
#else
    timeExecute += timeGpuExecute;
#endif

    return ret;
}// end 6opt



//! \brief Run et activate
//!
//wb.Q 202408 test running time of sequential 6-opt
template<std::size_t DimP, std::size_t DimCM>
void SolutionKOPT<DimP, DimCM>::runSequential6opt() {

    cout << "Begin run sequential 6-opt >>>>>>>>>>>>>>>" << endl;

    int nCity = md_links_cpu.adaptiveMap.getWidth();
    int numRuns = 0;

    int maxOptExecuPerRun = 0;
    int numOptimizedTotal = 0;
    vector<int> vectorNumOptExecuted;
    vector<float> vectorPDB;
    float timeCpuKey = 0;
    float pdb2optEatFirstPara = 0;
    float timeRefreshTour = 0;
    float timeSearch = 0;
    float timeSelect = 0;
    float timeExecute = 0;
    int numInter = 1;
    // trace maxtimeGPUone2-OoptRun
    float maxtimeCpuOptSearch = 0;

    // outfile timeline
    string fileTimePerRun = "Results_"; //str
    fileTimePerRun.append("Time6optPerRun.txt");
    ofstream outfileTimePerRunRun;
    outfileTimePerRunRun.open(fileTimePerRun);

    // outfile pdbline
    string filePdbPerRun = "Results_"; //str
    filePdbPerRun.append("PdbPer6optRun.txt");
    ofstream outfilePdbPerRunRun;
    outfilePdbPerRunRun.open(filePdbPerRun);

    // outfile pdbline
    string fileSearchTimePerRun = "Results_"; //str
    fileSearchTimePerRun.append("searchTimePer6optRun.txt");
    ofstream outfileSearchTimePerRunRun;
    outfileSearchTimePerRunRun.open(fileSearchTimePerRun);

    outfileTimePerRunRun << 0 << " " << endl;
    outfilePdbPerRunRun << 1143.63 << " " << endl;
    outfileSearchTimePerRunRun << 0 << endl;

    float evaLastRun = 0;
    float percentageImprove = 999999;

    //! prepare the pre-ordered link + coordinates
    Grid<doubleLinkedEdgeForTSP> linkCoordTourCpu;
    linkCoordTourCpu.resize(nCity, 1);


    //wb.Q 2019 add case detection
    if(SolutionKOPT<DimP, DimCM>::md_links_cpu.adaptiveMap.width == 0)
    {
        cout << "Error: no input available." << endl;
        return;
    }
    else
    {
        //wb.Q 2024 rocki 2-opt
        cout << "TSP tour optimum = " << optimum << endl;
        while (numRuns < NUMRUNSLIMIT  && percentageImprove > 0 )
        {
            activateSequential6opt(numRuns, nCity,
                                   maxOptExecuPerRun, numOptimizedTotal,
                                   timeCpuKey, vectorNumOptExecuted, vectorPDB,
                                   timeRefreshTour, timeSearch, timeSelect, timeExecute, maxtimeCpuOptSearch,
                                   outfileTimePerRunRun,outfilePdbPerRunRun, outfileSearchTimePerRunRun,  evaLastRun, percentageImprove,
                                   linkCoordTourCpu, 0);

            if (g_ConfigParameters->traceActive) {
                evaluate();
                writeStatisticsToFile(numRuns, "6optimalTour");
            }
        }
    }// end activateRocki

    //! mean time trace
    timeRefreshTour = timeRefreshTour / numRuns;
    timeSearch = timeSearch / numRuns;
    timeSelect = timeSelect / numRuns;
    timeExecute = timeExecute / numRuns;

    // close outfile
    outfileTimePerRunRun.close();
    outfilePdbPerRunRun.close();
    outfileSearchTimePerRunRun.close();


}// end run

// qiao 2024 add operators to GPU parallel 23456-opt and massive variable 23456-opt moves on global tour
template<std::size_t DimP, std::size_t DimCM>
bool SolutionKOPT<DimP, DimCM>::activateSequential6opt(int& numRuns, int nCity,
                                                       int& maxOptExecuPerRun, int& numOptimizedTotal,
                                                       float& timeCpuKey, vector<int>& vectorNumOptExecuted, vector<float>& vectorPDB,
                                                       float& timeRefresh, float& timeSearch,float& timeSelect, float& timeExecute,
                                                       float &maxtimeCpuOptSearch, ofstream &outfileTimePerRunRun,ofstream &outfilePdbPerRunRun,ofstream &outfileSearchTimePerRunRun,
                                                       float& evaLastRun, float& percentageImprove, Grid<doubleLinkedEdgeForTSP> &linkCoordTourCpu, bool iterOptimal)
{
    cout << endl << "****>>>>Enter 6-opt activate function: " << numRuns << endl;
    bool ret = true;
    numRuns ++;

    maxtimeCpuOptSearch = 0;

    int numOptimizedOneRun = 0;
    int numCityTraversed = 0;
    float pdbOneRun = 0;

    //! random starting point
    int ps_random = randomNum(0, nCity);
    PointCoord ps(ps_random, 0);
    cout << "PS [0] " << ps[0] << endl;

    //! clean cityCopy before mark tour ordering
    md_links_firstPara.activeMap.resetValue(0);
    md_links_firstPara.densityMap.resetValue(initialPrepareValue);// densityMap stores node3
    md_links_firstPara.optCandidateMap.resetValue(initialPrepareValue);// optCandidateMap stores opt candidate of 23456-opt
    md_links_firstPara.grayValueMap.resetValue(0);// clean orders
    md_links_firstPara.minRadiusMap.resetValue(initialPrepareValue);//  minRadiusMap stores the changeLinks position

    //!timing runing time on CPU
    __int64 CounterStart = 0;
    double pcFreq = 0.0;
    StartCounter(pcFreq, CounterStart);

    //! mark tour orientation from random starting point ps
    //! reserver, index of linkCoordTourCpu should correspond to index of gray value map
    md_links_firstPara.markNetLinkSequenceReloadRoutCoord(ps, numRuns%2, 0, linkCoordTourCpu);// every ps check its two directions

    // end time cpu
    double timeCpuRefreshTour = GetCounter(pcFreq, CounterStart);
    cout << "Time:: Refresh tour order: " << timeCpuRefreshTour << endl;

    //!timing runing time on CPU
    CounterStart = 0;
    pcFreq = 0.0;
    StartCounter(pcFreq, CounterStart);

#if FIRST

    //qiao add sequential 6-opt
    if (iterOptimal ==0)
        md_links_firstPara.sequential6optFirst(linkCoordTourCpu, md_links_cpu.nodeParentMap);// every ps check its two directions
    else
        md_links_firstPara.sequential6optBest(linkCoordTourCpu, md_links_cpu.evtMap);// every ps check its two directions

#endif
    // end time cpu
    double timeCpu6opt = GetCounter(pcFreq, CounterStart);
    cout << "Time:: CPU side one 3-opt runtime : " << timeCpu6opt << endl;

    if(maxtimeCpuOptSearch < timeCpu6opt){
        maxtimeCpuOptSearch = timeCpu6opt;
    }


    //! clean for mark non-interacted 6-opt
    md_links_firstPara.activeMap.resetValue(0); // for nodes possessing non interacted 2opt
    md_links_firstPara.fixedMap.resetValue(0); // for nodes in stackB


    //!timing runing time on CPU
    CounterStart = 0;
    pcFreq = 0.0;
    StartCounter(pcFreq, CounterStart);

    //! select and execute non-interacted 23456-exchanges
    md_links_firstPara.selectNonIteracted23456ExchangeQiao(ps);

    // end time cpu
    double timeCpuSelectNonItera = GetCounter(pcFreq, CounterStart);
    cout << "Time:: select non intera 6-opt: " << timeCpuSelectNonItera << endl;

    double timeCpuExecuteNonItera = 0;
    float elapsedTime2opt_execute = 0;

    //!timing runing time on CPU
    CounterStart = 0;
    pcFreq = 0.0;
    StartCounter(pcFreq, CounterStart);

    if(iterOptimal ==0)
        md_links_firstPara.executeNonInteract23456optOnlyNode3(numOptimizedOneRun, md_links_cpu.nodeParentMap);//qiao 2024 need modify
    else
        md_links_firstPara.executeNonInteract23456optOnlyNode3(numOptimizedOneRun, md_links_cpu.evtMap);//qiao 2024 need modify


    // end time cpu
    timeCpuExecuteNonItera = GetCounter(pcFreq, CounterStart);
    cout << "Time:: CPU execute non-intera 6-opt: " << timeCpuExecuteNonItera << endl;


    //! evaluation to stop
    float evaCurrentRun = md_links_firstPara.evaluateWeightOfTSP(distEuclidean, numCityTraversed);
    cout << "Evaluate:: After " << numRuns << "'th run, evaluate tsp length =  " << evaCurrentRun << endl;
    cout << "Evaluate:: In this run, num of 6-exchange been executed: "  <<  numOptimizedOneRun << endl;

    float evaActualLength = md_links_firstPara.evaluateWeightOfTSP(distEuclidean, numCityTraversed);
    cout << "Evaluate:: After " << numRuns << "'th run, evaluate tsp length =  " << evaActualLength << endl;

    //statistic pdb
    if(optimum > 1)
    {
        float evaCurrentPDB = md_links_firstPara.evaluateWeightOfTSP(distEuclidean, numCityTraversed);
        pdbOneRun = (evaCurrentPDB - optimum)*100/optimum;
    }

    if(numRuns == 1){

        //! registrer length of the first run
        evaLastRun = evaCurrentRun;
        //        continue;
    }
    else {
        percentageImprove = ((evaLastRun - evaCurrentRun)*100);
    }


    if(percentageImprove > 0){

        timeCpuKey += timeCpu6opt + timeCpuRefreshTour + timeCpuSelectNonItera + timeCpuExecuteNonItera;
        evaLastRun = evaCurrentRun;

        numOptimizedTotal += numOptimizedOneRun;
        if(numOptimizedOneRun > maxOptExecuPerRun)
            maxOptExecuPerRun = numOptimizedOneRun; // trace max optimized 2opt per run
        if(numOptimizedOneRun > 0)
            vectorNumOptExecuted.push_back(numOptimizedOneRun);
        // trace pdb one run
        vectorPDB.push_back(pdbOneRun);
        outfileTimePerRunRun << timeCpuKey << " " << endl;

        outfilePdbPerRunRun << pdbOneRun << " " << endl;
        outfileSearchTimePerRunRun << timeCpu6opt << " " << endl;

        traceTSP.timeObtainKoptimal = timeCpuKey;

        //record the best TSP tour obtained so far
        tspTourBestObtainedSoFar.assign(md_links_firstPara.networkLinks);
    }
    else{
        numRuns -= 1; // the last run does not optimized the tour

        // outfile pdbline
        string fileKoptimalTimePerRun = "Results_"; //str
        fileKoptimalTimePerRun.append("6optimal.txt");
        ofstream outfileKoptimalTimePerRunRun;
        outfileKoptimalTimePerRunRun.open(fileKoptimalTimePerRun);

        outfileKoptimalTimePerRunRun << timeCpuKey  << " ,pdb: " << pdbOneRun << " ,searchTime: " << timeCpu6opt<< endl;

        outfileKoptimalTimePerRunRun.close();
    }

    //test
    cout << "Percentage improve " << percentageImprove << endl << endl;


    // count time refresh
    timeRefresh += (float)timeCpuRefreshTour;
    timeSearch += (float)timeCpu6opt;
    timeSelect += (float)timeCpuSelectNonItera;

    timeExecute += (float)timeCpuExecuteNonItera;


    return ret;
}// end 6opt




//! \brief Run et activate
//!
//wb.Q 202408 implement 6-opt sequential
template<std::size_t DimP, std::size_t DimCM>
void SolutionKOPT<DimP, DimCM>::runSerialVariablekopt() {

    cout << "Begin run seiral variable k-opt >>>>>>>>>>>>>>>" << endl;

    int nCity = md_links_cpu.adaptiveMap.getWidth();
    int numRuns = 0;

    int maxOptExecuPerRun = 0;
    int numOptimizedTotal = 0;
    vector<int> vectorNumOptExecuted;
    vector<float> vectorPDB;
    float timeCpuKey = 0;
    float timeSearch = 0;
    float timeRefreshTour = 0;
    float timeSelect = 0;
    float timeExecute = 0;
    int numInter = 1;
    // trace maxtimeGPUone2-OoptRun
    float maxtimeCpuOptSearch = 0;

    // outfile timeline
    string fileTimePerRun = "Results_"; //str
    fileTimePerRun.append("TimePerVariablKoptRun.txt");
    ofstream outfileTimePerRunRun;
    outfileTimePerRunRun.open(fileTimePerRun);

    // outfile pdbline
    string filePdbPerRun = "Results_"; //str
    filePdbPerRun.append("PdbPerVariablKoptRun.txt");
    ofstream outfilePdbPerRunRun;
    outfilePdbPerRunRun.open(filePdbPerRun);

    // outfile pdbline
    string fileSearchTimePerRun = "Results_"; //str
    fileSearchTimePerRun.append("searchTimePer6optRun.txt");
    ofstream outfileSearchTimePerRunRun;
    outfileSearchTimePerRunRun.open(fileSearchTimePerRun);

    outfileTimePerRunRun << 0 << " " << endl;
    outfilePdbPerRunRun << 1143.63 << " " << endl;
    outfileSearchTimePerRunRun << 0 << endl;


    float evaLastRun = 0;
    float percentageImprove = 999999;

    //! prepare the pre-ordered link + coordinates
    Grid<doubleLinkedEdgeForTSP> linkCoordTourCpu;
    linkCoordTourCpu.resize(nCity, 1);


    //wb.Q 2019 add case detection
    if(SolutionKOPT<DimP, DimCM>::md_links_cpu.adaptiveMap.width == 0)
    {
        cout << "Error: no input available." << endl;
        return;
    }
    else
    {
        //wb.Q 2024 rocki 2-opt
        cout << "TSP tour optimum = " << optimum << endl;
        while (numRuns < NUMRUNSLIMIT  && percentageImprove > 0 )
        {
            activateSerialVariablekopt(numRuns, nCity,
                                       maxOptExecuPerRun, numOptimizedTotal,
                                       timeCpuKey, vectorNumOptExecuted, vectorPDB,
                                       timeRefreshTour,timeSearch, timeSelect, timeExecute, maxtimeCpuOptSearch,
                                       outfileTimePerRunRun, outfilePdbPerRunRun, outfileSearchTimePerRunRun, evaLastRun, percentageImprove,
                                       linkCoordTourCpu);

            if (g_ConfigParameters->traceActive) {
                evaluate();
                writeStatisticsToFile(numRuns, "variablekopt");
            }
        }
    }// end activateRocki

    //! mean time trace
    timeRefreshTour = timeRefreshTour / numRuns;
    timeSearch = timeSearch / numRuns;
    timeSelect = timeSelect / numRuns;
    timeExecute = timeExecute / numRuns;

    // close outfile
    outfileTimePerRunRun.close();
    outfilePdbPerRunRun.close();
    outfileSearchTimePerRunRun.close();


}// end run

// qiao 2024 add operators to GPU parallel 23456-opt and massive variable 23456-opt moves on global tour
template<std::size_t DimP, std::size_t DimCM>
bool SolutionKOPT<DimP, DimCM>::activateSerialVariablekopt(int& numRuns, int nCity,
                                                           int& maxOptExecuPerRun, int& numOptimizedTotal,
                                                           float& timeCpuKey, vector<int>& vectorNumOptExecuted, vector<float>& vectorPDB,
                                                           float& timeRefresh, float& timeSearch, float& timeSelect, float& timeExecute,
                                                           float &maxtimeCpuOptSearch, ofstream &outfileTimePerRunRun,ofstream &outfilePdbPerRunRun, ofstream &outfileSearchTimePerRunRun,
                                                           float& evaLastRun, float& percentageImprove, Grid<doubleLinkedEdgeForTSP> &linkCoordTourCpu)
{
    cout << endl << "****>>>>Enter variablek-opt activate function: " << numRuns << endl;
    bool ret = true;
    numRuns ++;


    maxtimeCpuOptSearch = 0;

    int numOptimizedOneRun = 0;
    int numCityTraversed = 0;

    networkLinksCP.resize(nCity,0);

    float pdbOneRun = 0;

    //! random starting point
    int ps_random = randomNum(0, nCity);
    PointCoord ps(ps_random, 0);
    cout << "PS [0] " << ps[0] << endl;

    //! clean cityCopy before mark tour ordering
    md_links_firstPara.activeMap.resetValue(0);
    md_links_firstPara.densityMap.resetValue(initialPrepareValue);// densityMap stores node3
    md_links_firstPara.optCandidateMap.resetValue(initialPrepareValue);// optCandidateMap stores opt candidate of 23456-opt
    md_links_firstPara.grayValueMap.resetValue(0);// clean orders
    md_links_firstPara.minRadiusMap.resetValue(initialPrepareValue);//  minRadiusMap stores the changeLinks position

    //!timing runing time on CPU
    __int64 CounterStart = 0;
    double pcFreq = 0.0;
    StartCounter(pcFreq, CounterStart);

    //! mark tour orientation from random starting point ps
    //! reserver, index of linkCoordTourCpu should correspond to index of gray value map
    md_links_firstPara.markNetLinkSequenceReloadRoutCoord(ps, numRuns%2, 0, linkCoordTourCpu);// every ps check its two directions

    // end time cpu
    double timeCpuRefreshTour = GetCounter(pcFreq, CounterStart);
    cout << "Time:: Refresh tour order: " << timeCpuRefreshTour << endl;


    //!timing runing time on CPU
    CounterStart = 0;
    pcFreq = 0.0;
    StartCounter(pcFreq, CounterStart);

#if FIRST

    //qiao add sequential 6-opt
//        md_links_firstPara.sequentialVariable3optFirst(linkCoordTourCpu, md_links_cpu.nodeParentMap, md_links_cpu.nVisitedMap, md_links_cpu.evtMap);// every ps check its two directions

    md_links_firstPara.sequentialVariable4optFirst(linkCoordTourCpu, md_links_cpu.nodeParentMap, md_links_cpu.nVisitedMap, md_links_cpu.evtMap);// every ps check its two directions

//        md_links_firstPara.sequentialVariable5optFirst(linkCoordTourCpu, md_links_cpu.nodeParentMap, md_links_cpu.nVisitedMap, md_links_cpu.evtMap);// every ps check its two directions

//        md_links_firstPara.sequentialVariable6optFirst(linkCoordTourCpu, md_links_cpu.nodeParentMap, md_links_cpu.nVisitedMap, md_links_cpu.evtMap);// every ps check its two directions


#else

//    md_links_firstPara.sequentialVariable3optBest(linkCoordTourCpu, md_links_cpu.nodeParentMap, md_links_cpu.nVisitedMap, md_links_cpu.evtMap);// every ps check its two directions

    md_links_firstPara.sequentialVariable4optBest(linkCoordTourCpu, md_links_cpu.nodeParentMap, md_links_cpu.nVisitedMap, md_links_cpu.evtMap);// every ps check its two directions

//    md_links_firstPara.sequentialVariable6optBest(linkCoordTourCpu, md_links_cpu.nodeParentMap, md_links_cpu.nVisitedMap, md_links_cpu.evtMap);// every ps check its two directions

#endif

    // end time cpu
    double timeCpuopt = GetCounter(pcFreq, CounterStart);
    cout << "Time:: CPU side one variable-opt runtime : " << timeCpuopt << endl;

    if(maxtimeCpuOptSearch < timeCpuopt){
        maxtimeCpuOptSearch = timeCpuopt;
    }


    //! clean for mark non-interacted 6-opt
    md_links_firstPara.activeMap.resetValue(0); // for nodes possessing non interacted 2opt
    md_links_firstPara.fixedMap.resetValue(0); // for nodes in stackB


    //!timing runing time on CPU
    CounterStart = 0;
    pcFreq = 0.0;
    StartCounter(pcFreq, CounterStart);

    //! select and execute non-interacted 23456-exchanges
    md_links_firstPara.selectNonIteracted23456ExchangeQiao(ps);

    // end time cpu
    double timeCpuSelectNonItera = GetCounter(pcFreq, CounterStart);
    cout << "Time:: select non intera variable-opt: " << timeCpuSelectNonItera << endl;

    double timeCpuExecuteNonItera = 0;
    float elapsedTime2opt_execute = 0;

    //!timing runing time on CPU
    CounterStart = 0;
    pcFreq = 0.0;
    StartCounter(pcFreq, CounterStart);


    //    networkLinksCP.assign(md_links_firstPara.networkLinks);
    md_links_firstPara.executeNonInteract23456optOnlyNode3(numOptimizedOneRun, md_links_cpu.nodeParentMap, md_links_cpu.nVisitedMap, md_links_cpu.evtMap);//qiao 2024 need modify



    // end time cpu
    timeCpuExecuteNonItera = GetCounter(pcFreq, CounterStart);
    cout << "Time:: CPU execute non-intera variable-opt: " << timeCpuExecuteNonItera << endl;


    //! evaluation to stop
    float evaCurrentRun = md_links_firstPara.evaluateWeightOfTSP(dist, numCityTraversed);
    cout << "Evaluate:: After " << numRuns << "'th run, evaluate tsp length =  " << evaCurrentRun << endl;
    cout << "Evaluate:: In this run, num of variable-exchange been executed: "  <<  numOptimizedOneRun << endl;

    float evaActualLength = md_links_firstPara.evaluateWeightOfTSP(distEuclidean, numCityTraversed);
    cout << "Evaluate:: After " << numRuns << "'th run, evaluate tsp length =  " << evaActualLength << endl;

    //statistic pdb
    if(optimum > 1)
    {
        float evaCurrentPDB = md_links_firstPara.evaluateWeightOfTSP(distEuclidean, numCityTraversed);
        pdbOneRun = (evaCurrentPDB - optimum)*100/optimum;
    }

    if(numRuns == 1){

        //! registrer length of the first run
        evaLastRun = evaCurrentRun;
        //        continue;
    }
    else {
        percentageImprove = ((evaLastRun - evaCurrentRun)*100);
    }


    if(percentageImprove > 0){

        timeCpuKey += timeCpuopt + timeCpuRefreshTour + timeCpuSelectNonItera + timeCpuExecuteNonItera;
        evaLastRun = evaCurrentRun;

        numOptimizedTotal += numOptimizedOneRun;
        if(numOptimizedOneRun > maxOptExecuPerRun)
            maxOptExecuPerRun = numOptimizedOneRun; // trace max optimized 2opt per run
        if(numOptimizedOneRun > 0)
            vectorNumOptExecuted.push_back(numOptimizedOneRun);
        // trace pdb one run
        vectorPDB.push_back(pdbOneRun);
        outfileTimePerRunRun << timeCpuKey << " " << endl;
        outfilePdbPerRunRun << pdbOneRun << " " << endl;
        outfileSearchTimePerRunRun << timeCpuopt << " " << endl;


        traceTSP.timeObtainKoptimal = timeCpuKey;

        //record the best TSP tour obtained so far
        tspTourBestObtainedSoFar.assign(md_links_firstPara.networkLinks);
    }
    else{
        numRuns -= 1; // the last run does not optimized the tour

        // outfile pdbline
        string fileKoptimalTimePerRun = "Results_"; //str
        fileKoptimalTimePerRun.append("variableKoptimal.txt");
        ofstream outfileKoptimalTimePerRunRun;
        outfileKoptimalTimePerRunRun.open(fileKoptimalTimePerRun);

        outfileKoptimalTimePerRunRun << timeCpuKey  << " ,pdb: " << pdbOneRun << " ,searchTime: " << timeCpuopt<< endl;

        outfileKoptimalTimePerRunRun.close();
    }

    //test
    cout << "Percentage improve " << percentageImprove << endl << endl;


    // count time refresh
    timeRefresh += (float)timeCpuRefreshTour;
    timeSearch += (float)timeCpuopt;
    timeSelect += (float)timeCpuSelectNonItera;
    timeExecute += (float)timeCpuExecuteNonItera;


    return ret;
}// end variable k-opt


//! \brief Run implement iterative k-optimal
template<std::size_t DimP, std::size_t DimCM>
void SolutionKOPT<DimP, DimCM>::runSerialIterativeKoptimal() {

    cout << "Begin run seiral iterative k-optimal >>>>>>>>>>>>>>>" << endl;

    int nCity = md_links_cpu.adaptiveMap.getWidth();
    int numRuns = 0;

    int maxOptExecuPerRun = 0;
    int numOptimizedTotal = 0;
    vector<int> vectorNumOptExecuted;
    vector<float> vectorPDB;
    float timeCpuKey = 0;
    float pdb2optEatFirstPara = 0;
    float timeRefreshTour = 0;
    float timeSearch = 0;
    float timeSelect = 0;
    float timeExecute = 0;
    int numInter = 1;
    // trace maxtimeGPUone2-OoptRun
    float maxtimeCpuOptSearch = 0;

    //trace the best TSP tour ever computed for one instance


    //trace running time


    //solve the problem of multi-GPU cards

    // outfile timeline
    string fileTimePerRun = "Results_"; //str
    fileTimePerRun.append("TimePerIterOptimalRun.txt");
    ofstream outfileTimePerRunRun;
    outfileTimePerRunRun.open(fileTimePerRun);

    // outfile pdbline
    string filePdbPerRun = "Results_"; //str
    filePdbPerRun.append("PdbPerIterOptimalRun.txt");
    ofstream outfilePdbPerRunRun;
    outfilePdbPerRunRun.open(filePdbPerRun);

    // outfile pdbline
    string fileSearchTimePerRun = "Results_"; //str
    fileSearchTimePerRun.append("searchTimePerIterKoptRun.txt");
    ofstream outfileSearchTimePerRunRun;
    outfileSearchTimePerRunRun.open(fileSearchTimePerRun);


    outfileTimePerRunRun << 0 << " " << endl;
    outfilePdbPerRunRun << 1143.63 << " " << endl;
    outfileSearchTimePerRunRun << 0 << endl;


    float evaLastRun = 0;
    float percentageImprove2opt = 999999;
    float percentageImprove3opt = 999999;
    float percentageImprove4opt = 999999;
    float percentageImprove5opt = 999999;
    float percentageImprove6opt = 999999;

    //! prepare the pre-ordered link + coordinates
    Grid<doubleLinkedEdgeForTSP> linkCoordTourCpu;
    linkCoordTourCpu.resize(nCity, 1);

    //wb.Q 2019 add case detection
    if(SolutionKOPT<DimP, DimCM>::md_links_cpu.adaptiveMap.width == 0)
    {
        cout << "Error: no input available." << endl;
        return;
    }
    else
    {
        //wb.Q 2024 serial 2-opt
        cout << "TSP tour optimum = " << optimum << endl;
        int numCityTraversed = 0;
        bool iterOptimal = 1;

        //!iterative 2-opt until achieve 2-optimal
        while (numRuns < NUMRUNSLIMIT  && percentageImprove2opt > 0 )
        {
            activateSequential2opt(numRuns, nCity,
                                   maxOptExecuPerRun, numOptimizedTotal, timeCpuKey, vectorNumOptExecuted, vectorPDB,
                                   timeRefreshTour,timeSearch, timeSelect, timeExecute, maxtimeCpuOptSearch,
                                   outfileTimePerRunRun,outfilePdbPerRunRun, outfileSearchTimePerRunRun, evaLastRun, percentageImprove2opt,
                                   linkCoordTourCpu);

            if (g_ConfigParameters->traceActive) {
                evaluate();
                writeStatisticsToFile(numRuns, "iterkOptimalTour");
            }

        }

        //cout TSP length of 2-optimal
        float eva2optimal = md_links_firstPara.evaluateWeightOfTSP(distEuclidean,numCityTraversed);
        cout << "Evaluate:: eva2optimal =  " << eva2optimal << endl;

        //!iterative 3-opt until achieve 3-optimal
        while (numRuns < NUMRUNSLIMIT  && percentageImprove3opt > 0 )
        {
            activateSequential3opt(numRuns, nCity,
                                   maxOptExecuPerRun, numOptimizedTotal, timeCpuKey, vectorNumOptExecuted, vectorPDB,
                                   timeRefreshTour, timeSearch,timeSelect, timeExecute, maxtimeCpuOptSearch,
                                   outfileTimePerRunRun, outfilePdbPerRunRun, outfileSearchTimePerRunRun, evaLastRun, percentageImprove3opt,
                                   linkCoordTourCpu);

            if (g_ConfigParameters->traceActive) {
                evaluate();
                writeStatisticsToFile(numRuns, "iterkOptimalTour");
            }
        }

        //cout TSP length of 3-optimal
        float eva3optimal = md_links_firstPara.evaluateWeightOfTSP(distEuclidean, numCityTraversed);
        cout << "Evaluate:: eva3optimal =  " << eva3optimal << endl;

        //!iterative 4-opt until achieve 4-optimal
        while (numRuns < NUMRUNSLIMIT  && percentageImprove4opt > 0 )
        {
            activateSequential4opt(numRuns, nCity,
                                   maxOptExecuPerRun, numOptimizedTotal, timeCpuKey, vectorNumOptExecuted, vectorPDB,
                                   timeRefreshTour, timeSearch,timeSelect, timeExecute, maxtimeCpuOptSearch,
                                   outfileTimePerRunRun, outfilePdbPerRunRun,outfileSearchTimePerRunRun, evaLastRun, percentageImprove4opt,
                                   linkCoordTourCpu, iterOptimal);

            if (g_ConfigParameters->traceActive) {
                evaluate();
                writeStatisticsToFile(numRuns, "iterkOptimalTour");
            }

        }

        //cout TSP length of 4-optimal
        float eva4optimal = md_links_firstPara.evaluateWeightOfTSP(distEuclidean, numCityTraversed);
        cout << "Evaluate:: eva4optimal =  " << eva4optimal << endl;


        //!iterative 5-opt until achieve 5-optimal
        while (numRuns < NUMRUNSLIMIT  && percentageImprove5opt > 0 )
        {
            activateSequential5opt(numRuns, nCity,
                                   maxOptExecuPerRun, numOptimizedTotal, timeCpuKey, vectorNumOptExecuted, vectorPDB,
                                   timeRefreshTour, timeSearch,timeSelect, timeExecute, maxtimeCpuOptSearch,
                                   outfileTimePerRunRun,outfilePdbPerRunRun, outfileSearchTimePerRunRun, evaLastRun, percentageImprove5opt,
                                   linkCoordTourCpu, iterOptimal);

            if (g_ConfigParameters->traceActive) {
                evaluate();
                writeStatisticsToFile(numRuns, "iterkOptimalTour");
            }

        }

        //cout TSP length of 5-optimal
        float eva5optimal = md_links_firstPara.evaluateWeightOfTSP(distEuclidean, numCityTraversed);
        cout << "Evaluate:: eva5optimal =  " << eva5optimal << endl;


        //!iterative 6-opt until achieve 6-optimal
        while (numRuns < NUMRUNSLIMIT  && percentageImprove6opt > 0 )
        {
            activateSequential6opt(numRuns, nCity,
                                   maxOptExecuPerRun, numOptimizedTotal, timeCpuKey, vectorNumOptExecuted, vectorPDB,
                                   timeRefreshTour, timeSearch, timeSelect, timeExecute, maxtimeCpuOptSearch,
                                   outfileTimePerRunRun, outfilePdbPerRunRun,outfileSearchTimePerRunRun, evaLastRun, percentageImprove6opt,
                                   linkCoordTourCpu, iterOptimal);

            if (g_ConfigParameters->traceActive) {
                evaluate();
                writeStatisticsToFile(numRuns, "iterkOptimalTour");
            }

        }

        //cout TSP length of 6-optimal
        float eva6optimal = md_links_firstPara.evaluateWeightOfTSP(distEuclidean, numCityTraversed);
        cout << "Evaluate:: eva6optimal =  " << eva6optimal << endl;

        //records the best TSP tour


    }// end activateRocki

    //! mean time trace
    timeRefreshTour = timeRefreshTour / numRuns;
    timeSearch = timeSearch / numRuns;
    timeSelect = timeSelect / numRuns;
    timeExecute = timeExecute / numRuns;

    // close outfile
    outfileTimePerRunRun.close();
    outfilePdbPerRunRun.close();
    outfileSearchTimePerRunRun.close();


}// end run

