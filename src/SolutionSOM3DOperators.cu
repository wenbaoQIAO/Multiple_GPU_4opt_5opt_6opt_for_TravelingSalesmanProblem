#include "config/ConfigParamsCF.h"
#include "random_generator_cf.h"
#include "SolutionSOM3D.h"

/** Operateurs de changement de SolutionSOM3D courante.
 *
 */
#define FULL_GPU 1
#define FULL_GPU_FIND_MIN1 1
#define FULL_GPU_FIND_MIN2 1
#define FULL_GPU_CGU 1
#define FULL_GPU_FLATTENING 1

#define EMST_DETECT_CYCLE  0
#define EMST_FIND_MIN_PAIR_LIST 1// Distributed broadcast or distributed linked list

template<std::size_t DimP, std::size_t DimCM>
void SolutionSOM3D<DimP, DimCM>::initConstruct()
{
}//initConstruct

/** Construction Sequentielle
 */
template<std::size_t DimP, std::size_t DimCM>
void SolutionSOM3D<DimP, DimCM>::constructSolutionSeq()
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
bool SolutionSOM3D<DimP, DimCM>::operator_1() {
    bool ret = true;


    global_objectif = computeObjectif();

    return ret;
}//operator_1

/*!
 * \return vrai si l'operateur est applique selon choix aleatoire,
 *  faux si l'operateur n'est pas applique
 */
template<std::size_t DimP, std::size_t DimCM>
bool SolutionSOM3D<DimP, DimCM>::operator_2() {
    bool noUsed = true;

    return noUsed;
}//operator_1

template<std::size_t DimP, std::size_t DimCM>
bool SolutionSOM3D<DimP, DimCM>::generateNeighbor()
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
bool SolutionSOM3D<DimP, DimCM>::applyOperator(int i)
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
int SolutionSOM3D<DimP, DimCM>::nbrOperators() const
{
    return g_ConfigParameters->probaOperators.size();
}

//! \brief Run et activate
template<std::size_t DimP, std::size_t DimCM>
void SolutionSOM3D<DimP, DimCM>::run() {

    //wb.Q 2019 add case detection
    if(SolutionSOM3D<DimP, DimCM>::md_links_cpu.adaptiveMap.width == 0)
    {
        cout << "Error: no input available." << endl;
        return;
    }
    else{
        while (SolutionSOM3D<DimP, DimCM>::activate()) {
            if (g_ConfigParameters->traceActive) {
                evaluate();
                writeStatisticsToFile(iteration);
            }
        }
    }
}

template<std::size_t DimP, std::size_t DimCM>
bool SolutionSOM3D<DimP, DimCM>::activate() {
    bool ret = true;

    /*********************************************************
    * REFRESH CELLULAR PARTITION
    * wb.Q: refresh cellular cmr according to mr
    * *******************************************************
    */
    // Time
    float timeGpu;
    cudaEvent_t pcFreq;
    cudaEvent_t pcFreq2;
    cudaEventCreate(&pcFreq);
    cudaEventCreate(&pcFreq2);
    cudaEventRecord(pcFreq, 0);

//    // qiao todo: refresh cellular of mr
//  som3dOp.K_refreshCell(cma_gpu, mr_links_gpu.adaptiveMap);

    // qiao todo: adaptiveMap need to be double size and re-initialized
    som3dOp.K_initializeSpiralSearch(cma_gpu,
                                       mr_links_gpu.adaptiveMap,
                                       spiralSearchMap); // wb.Q spiralSearchMap initialize each point ps and its cell center pc

    // End time
    cudaEventRecord(pcFreq2, 0);
    cudaEventSynchronize(pcFreq2);
    cudaEventElapsedTime(&timeGpu, pcFreq, pcFreq2);
    cout << "Time refresh CM: " << timeGpu << endl;
    time_refreshCm += timeGpu;
    traceParallel3DSom.timeRefreshCm = timeGpu;
    traceParallel3DSom.timeCumulativeFlatten = time_refreshCm;

    /**********************************************************************
    * SOM 3D TRAINING
    **********************************************************************
    */

    // Time
    cudaEventRecord(pcFreq, 0);

    /*begin:SOM3D operators and training procedure*/


    //qiao todo: Set adaptator status in grid Map
//    Som3DTriggerMap.alpha = execParams.alpha;
//    Som3DTriggerMap.radius = execParams.radius;

//    som3dOp.K_trainingTsp(cma_gpu,
//                          mr_links_gpu.adaptiveMap, //md_links_gpu.adaptiveMap,
//                          mr_links_gpu.adaptiveMap,
//                          mr_links_gpu.networkLinks,
//                          mr_links_gpu.fixedMap,
//                          mr_links_gpu.fixedLinks,
//                          spiralSearchMap,
//                          Som3DTriggerMap);


    /*end:SOM3D operators and training procedure*/

    // End time
    cudaEventRecord(pcFreq2, 0);
    cudaEventSynchronize(pcFreq2);
    cudaEventElapsedTime(&timeGpu, pcFreq, pcFreq2);
    cout << "Time Test Termination: " << timeGpu << endl;
    time_terminate += timeGpu;
    traceParallel3DSom.timeTestTermination = timeGpu;
    traceParallel3DSom.timeCumulativeTermination = time_terminate;

    som3dOp.updateParams(paramSom, execParams);

    GLfloat alpha = execParams.alpha;
    GLfloat radius = execParams.radius;

    // Update Params, qiao todo: check if generation/iteration has influence on SOM training.
    som3dOp.K_updateSomParam(Som3DTriggerMap, alpha, radius);
    // Generation finished

    cout << "qiao test update param: " << timeGpu << endl;

    if (iteration == 100) {
        cout << "ACTIVATE FINISHED !!!" << endl;

        return ret = false;
    }

    iteration ++;
    cout << "iteration " << iteration << endl;

    return ret;
}

