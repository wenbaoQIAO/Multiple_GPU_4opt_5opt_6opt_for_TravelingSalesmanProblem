#include "CalculateurEMST.h"

#include "SolutionKOPT.h"
#include "LocalSearch.h"
#include "AgentMetaSolver.h"
#include "random_generator_cf.h"
//#include "Multiout.h"

using namespace std;

#define TEST_CODE   0
#define TEST_ALEAT_DOUBLE   1
#define TEST_3D_GRID        0

typedef SolutionKOPT<2, 2> Solution;

typedef SolutionKOPT<3, 3> Solution3D;

typedef SolutionKOPT<3, 2> Solution2DPoint5;

Solution* sol = NULL;
Solution* sol1 = NULL;
LocalSearch<Solution>* lS = NULL;
//AgentMetaSolver<Solution>* Gm = NULL;

Solution3D* sol3D = NULL;

//Solution2DPoint5* sol2DPoint5 = NULL;

#if TEST_3D_GRID
extern Grid3D grid3D;
extern NN mrH;
//Solution solH;
#endif

//Grid<Point3D>* CalculateurEMST::getAdaptiveMap() {
//    return sol2DPoint5->getAdpativeMap();
//}

//Grid<BufferLinkPointCoord>* CalculateurEMST::getLinks() {
//    return sol2DPoint5->getLinks();
//}

void CalculateurEMST::initialize(char* fileData, char* fileSolution, char* fileStats, config::ConfigParamsCF* params)
{
    g_ConfigParameters = params;

    // Initialise le générateur de nombres aléatoires
    if (!g_ConfigParameters->useSeed) {
        g_ConfigParameters->seedValue = random_cf::aleat_get_time();
        cout << "SEED VALUE " << g_ConfigParameters->seedValue << endl;
    }
    random_cf::aleat_initialize(g_ConfigParameters->seedValue);
#if TEST_ALEAT_DOUBLE
    for (int i = 0; i < 10 ; ++i) {
        cout << random_cf::aleat_double(0, 1) << " ";
    }
    cout << endl;
#endif
#if TEST_3D_GRID
    //NN mrHgpu;
    Grid3D grid3D_gpu;
    //mrH.colorMap.resize(10,10,19);
    grid3D_gpu.gpuResize(6,222,180);
    //mrH.gpuCopyHostToDevice(mrHgpu);
    grid3D_gpu.gpuResetValue(Point3D(5,6,7));
    grid3D.gpuCopyDeviceToHost(grid3D_gpu);
    ofstream fo;
    fo.open("essaiD.txt");
    if (fo) {
        fo << grid3D;
        fo.close();
    }
    else
        cout << "pb file" << endl;
#endif
    // Sélection du mode de fonctionnement
    switch (g_ConfigParameters->functionModeChoice) {
    case EVAL_ONLY:
        cout << "INIT EVALUATE" << endl;

        sol = new Solution();
        sol->initialize(fileData, fileSolution, fileStats);
        sol->readSolution();
        //        sol->initStatisticsFile();
        break;

    case LOCAL_SEARCH:
        cout << "INIT LOCAL SEARCH" << endl;

        lS = new LocalSearch<Solution>();
        lS->initialize(fileData, fileSolution, fileStats);
        break;

    case RUN2OPT:
    {
        cout << "RUN2OPT INITIALIZATION" << endl;

        sol = new Solution();
        sol->initialize(fileData, fileSolution, fileStats);
        sol->readSolution();
        sol->initStatisticsFile("2optimalTour");

        sol->evaluateInit();
        sol->writeStatisticsToFile("2optimalTour");
        sol->writeHeaderStatistics(cout);
        sol->writeStatistics(-1,cout);


    }
        break;

    case RUN3OPT:
    {
        cout << "RUN3OPT INITIALIZATION" << endl;
        sol = new Solution();
        sol->initialize(fileData, fileSolution, fileStats);
        sol->readSolution();
        sol->initStatisticsFile("3optimalTour");

        sol->evaluateInit();
        sol->writeStatisticsToFile("3optimalTour");
        sol->writeHeaderStatistics(cout);
        sol->writeStatistics(-1,cout);

    }
        break;

    case RUN4OPT:
    {
        cout << "RUN4OPT INITIALIZATION" << endl;

        sol = new Solution();
        sol->initialize(fileData, fileSolution, fileStats);
        sol->readSolution(g_ConfigParameters->functionModeChoice); //qiao add
        sol->initStatisticsFile("4optimalTour");

        sol->evaluateInit();
        sol->writeStatisticsToFile("4optimalTour");
        sol->writeHeaderStatistics(cout);
        sol->writeStatistics(-1,cout);


    }
        break;
    case RUN5OPT:
    {
        cout << "RUN5OPT INITIALIZATION" << endl;

        sol = new Solution();
        sol->initialize(fileData, fileSolution, fileStats);
        sol->readSolution(g_ConfigParameters->functionModeChoice);
        sol->initStatisticsFile("5optimalTour");

        sol->evaluateInit();
        sol->writeStatisticsToFile("5optimalTour");
        sol->writeHeaderStatistics(cout);
        sol->writeStatistics(-1,cout);

    }
        break;

    case RUN6OPT:
    {
        cout << "RUN6OPT INITIALIZATION" << endl;

        sol = new Solution();
        sol->initialize(fileData, fileSolution, fileStats);
        sol->readSolution(g_ConfigParameters->functionModeChoice);
        sol->initStatisticsFile("6optimalTour");

        sol->evaluateInit();
        sol->writeStatisticsToFile("6optimalTour");
        sol->writeHeaderStatistics(cout);
        sol->writeStatistics(-1,cout);


    }
    case RUN_KOPT:
    {
        cout << "RUN VARIABLE K-OPT INITIALIZATION" << endl;

        sol = new Solution();
        sol->initialize(fileData, fileSolution, fileStats);
        sol->readSolution(g_ConfigParameters->functionModeChoice);
        sol->initStatisticsFile("variablekopt");

        sol->evaluateInit();
        sol->writeStatisticsToFile("variablekopt");
        sol->writeHeaderStatistics(cout);
        sol->writeStatistics(-1,cout);


    }
    case RUNIterKOPT:
    {
        cout << "RUN ITERATIVE K-OPTIMAL INITIALIZATION" << endl;

        sol = new Solution();
        sol->initialize(fileData, fileSolution, fileStats);
        sol->readSolution(g_ConfigParameters->functionModeChoice);
        sol->initStatisticsFile("iterKoptimalTour");

        sol->evaluateInit();
        sol->writeStatisticsToFile("iterKoptimalTour");
        sol->writeHeaderStatistics(cout);
        sol->writeStatistics(-1,cout);


    }

    default:
        cout << "UNSUPPORTED FUNCTIONMODE initiazlize=" << params->functionModeChoice << " !! " << endl;
        break;
    }

}//initialize

//run paralle version
//void CalculateurEMST::run()
//{
//    // Sélection du mode de fonctionnement
//    switch (g_ConfigParameters->functionModeChoice) {
//    case EVAL_ONLY:
//        cout << "EVALUATE" << endl;

//        sol->initEvaluate();
//        sol->evaluate();

//        sol->writeStatisticsToFile();
//        sol->writeHeaderStatistics(cout);
//        sol->writeStatistics(cout);

//        sol->writeSolution();

//        delete sol;
//        break;

//    case LOCAL_SEARCH:
//        cout << "LOCAL SEARCH" << endl;

//        lS->run();

//        delete lS;
//        break;

//    case RUN2OPT:
//        cout << "RUN2OPT RUN" << endl;

//        sol->run(); //rocki 2-opt with massive moves
//        // Après run
//        sol->evaluate();
//        //sol->writeStatisticsToFile(-1);
//        //sol->setIdentical(sol1);
//        sol->writeStatisticsToFile();
//        sol->writeHeaderStatistics(cout);
//        sol->writeStatistics(cout);
//        sol->writeSolution();
//        delete sol;
//        break;

//    case RUN3OPT:
//        cout << "RUN3OPT RUN" << endl;

//        sol->run3opt(); //rocki 3-opt with massive moves
//        // Après run
//        sol->evaluate();
//        //sol->writeStatisticsToFile(-1);
//        //sol->setIdentical(sol1);
//        sol->writeStatisticsToFile();
//        sol->writeHeaderStatistics(cout);
//        sol->writeStatistics(cout);

//        delete sol1;
//        break;

//    case RUN4OPT:
//        cout << "RUN4OPT RUN" << endl;

//        sol->run4opt(); //rocki 2-opt with massive moves
//        // Après run
//        sol->evaluate();
//        //sol->writeStatisticsToFile(-1);
//        //sol->setIdentical(sol1);
//        sol->writeStatisticsToFile();
//        sol->writeHeaderStatistics(cout);
//        sol->writeStatistics(cout);
//        sol->writeSolution();
//        delete sol;
//        break;

//    case RUN5OPT:
//        cout << "RUN5OPT RUN" << endl;

//        sol->run5opt(); //rocki 2-opt with massive moves
//        // Après run
//        sol->evaluate();
//        //sol->writeStatisticsToFile(-1);
//        //sol->setIdentical(sol1);
//        sol->writeStatisticsToFile();
//        sol->writeHeaderStatistics(cout);
//        sol->writeStatistics(cout);
//        sol->writeSolution();
//        delete sol;
//        break;

//    case RUN6OPT:
//        cout << "RUN6OPT RUN" << endl;

//        sol->run6opt(); //rocki 2-opt with massive moves
//        // Après run
//        sol->evaluate();
//        //sol->writeStatisticsToFile(-1);
//        //sol->setIdentical(sol1);
//        sol->writeStatisticsToFile();
//        sol->writeHeaderStatistics(cout);
//        sol->writeStatistics(cout);
//        sol->writeSolution();
//        delete sol;
//        break;


//    default:
//        cout << "UNSUPPORTED FUNCTIONMODE=" << g_ConfigParameters->functionModeChoice << " !! " << endl;
//        break;
//    }

//}//run



void CalculateurEMST::runSequential()
{
    // Sélection du mode de fonctionnement
    switch (g_ConfigParameters->functionModeChoice) {
    case EVAL_ONLY:
        cout << "EVALUATE" << endl;

        sol->initEvaluate();
        sol->evaluate();

        sol->writeStatisticsToFile();
        sol->writeHeaderStatistics(cout);
        sol->writeStatistics(cout);

        sol->writeSolution();

        delete sol;
        break;

    case LOCAL_SEARCH:
        cout << "LOCAL SEARCH" << endl;

        lS->run();

        delete lS;
        break;

    case RUN2OPT:
        cout << "RUN2OPT RUN SEQUENTIAL" << endl;

        sol->runSequtial2opt(); //Sequential 2-opt with massive moves


        // Après run
        sol->evaluate();// copy the best TSP tour found by current iterative 2-opt
        //sol->setIdentical(sol1);
        sol->writeStatisticsToFile("2optimalTour");
        sol->writeHeaderStatistics(cout);
        sol->writeStatistics(cout);
        sol->writeSolution("2optimalTour"); // write the best TSP tour found
        delete sol;
        break;

    case RUN3OPT:
        cout << "RUN3OPT RUN" << endl;

        sol->runSequential3opt(); //Sequential 3-opt with massive moves
        // Après run
        sol->evaluate();
        //sol->setIdentical(sol1);
        sol->writeStatisticsToFile("3optimalTour");
        sol->writeHeaderStatistics(cout);
        sol->writeStatistics(cout);
        sol->writeSolution("3optimalTour");
        delete sol1;
        break;

    case RUN4OPT:
        cout << "RUN4OPT RUN" << endl;

        sol->runSequential4opt(); //Sequential 4-opt with massive moves
        // Après run
        sol->evaluate();
        //sol->setIdentical(sol1);
        sol->writeStatisticsToFile("4optimalTour");
        sol->writeHeaderStatistics(cout);
        sol->writeStatistics(cout);
        sol->writeSolution("4optimalTour");

        delete sol;
        break;

    case RUN5OPT:
        cout << "RUN5OPT RUN" << endl;

        sol->runSequential5opt(); //Sequential 5-opt with massive moves
        // Après run
        sol->evaluate();
        //sol->setIdentical(sol1);
        sol->writeStatisticsToFile("5optimalTour");
        sol->writeHeaderStatistics(cout);
        sol->writeStatistics(cout);
        sol->writeSolution("5optimalTour");
        delete sol;
        break;

    case RUN6OPT:
        cout << "RUN6OPT RUN" << endl;

        sol->runSequential6opt(); //Sequential 6-opt with massive moves
        // Après run
        sol->evaluate();
        //sol->setIdentical(sol1);
        sol->writeStatisticsToFile("6optimalTour");
        sol->writeHeaderStatistics(cout);
        sol->writeStatistics(cout);
        sol->writeSolution("6optimalTour");
        delete sol;
        break;

    case RUN_KOPT:
        cout << "RUN Serial Variable k-opt " << endl;

        sol->runSerialVariablekopt(); // serially run variable k-opt with massive variable k-opt moves
        // Après run
        sol->evaluate();
        //sol->setIdentical(sol1);
        sol->writeStatisticsToFile("variablekopt");
        sol->writeHeaderStatistics(cout);
        sol->writeStatistics(cout);
        sol->writeSolution("variablekopt");
        delete sol;
        break;

    case RUNIterKOPT:
        cout << "RUN Serial Iterative K-optimal " << endl;

        sol->runSerialIterativeKoptimal(); // serially run k-opt until get k-optimal
        // Après run
        sol->evaluate();
        //sol->writeStatisticsToFile(-1);
        //sol->setIdentical(sol1);
        sol->writeStatisticsToFile("iterKoptimalTour");
        sol->writeHeaderStatistics(cout);
        sol->writeStatistics(cout);
        sol->writeSolution("iterKoptimalTour");
        delete sol;
        break;



    default:
        cout << "UNSUPPORTED FUNCTIONMODE run=" << g_ConfigParameters->functionModeChoice << " !! " << endl;
        break;
    }

}//run

bool CalculateurEMST::activate()
{
    bool ret = true;
    // Sélection du mode de fonctionnement
    switch (g_ConfigParameters->functionModeChoice) {
    case EVAL_ONLY:
        cout << "EVALUATE" << endl;

        sol->initEvaluate();
        sol->evaluate();

        sol->writeStatisticsToFile();
        sol->writeHeaderStatistics(cout);
        sol->writeStatistics(cout);

        sol->writeSolution();

        delete sol;
        break;

    case LOCAL_SEARCH:
        cout << "LOCAL SEARCH" << endl;

        lS->run();

        delete lS;
        break;

    case RUN2OPT:
        cout << "CONSTRUCTION" << endl;

        sol->constructSolutionSeq();
        // Après construction
        sol->evaluate();
        sol->writeStatisticsToFile();
        sol->writeHeaderStatistics(cout);
        sol->writeStatistics(cout);
        sol->writeSolution();
        delete sol;
        break;

    case RUN3OPT:
        cout << "RUN" << endl;

        sol1->run();
        // Après run
        sol1->evaluate();
        //sol->writeStatisticsToFile(-1);
        //sol->setIdentical(sol1);
        sol1->writeStatisticsToFile();
        sol1->writeHeaderStatistics(cout);
        sol1->writeStatistics(cout);
        sol1->writeSolution();
        delete sol;
        break;

    case RUN4OPT:
        cout << "RUN" << endl;

        sol->run();
        // Après run
        sol->evaluate();
        sol->writeStatisticsToFile();
        sol->writeHeaderStatistics(cout);
        sol->writeStatistics(cout);
        sol->writeSolution();
        delete sol;
        break;

    case RUN5OPT:
        cout << "RUN" << endl;

        sol->run();
        // Après run
        sol->evaluate();
        sol->writeStatisticsToFile();
        sol->writeHeaderStatistics(cout);
        sol->writeStatistics(cout);
        sol->writeSolution();
        delete sol;
        break;

    case RUN6OPT:
        cout << "RUN" << endl;

        sol->run();
        // Après run
        sol->evaluate();
        sol->writeStatisticsToFile();
        sol->writeHeaderStatistics(cout);
        sol->writeStatistics(cout);
        sol->writeSolution();
        delete sol;
        break;



    default:
        cout << "UNSUPPORTED FUNCTIONMODE activate=" << g_ConfigParameters->functionModeChoice << " !! " << endl;
        break;
    }

    return ret;

}//activate

//#ifndef SEPARATE_COMPILATION
//#include "..\src\SolutionKOPT.cu"
//#include "..\src\SolutionKOPTRW.cu"
//#include "..\src\SolutionKOPTOperators.cu"
//#endif

#ifndef SEPARATE_COMPILATION
#include "SolutionKOPT.cu"
#include "SolutionKOPTRW.cu"
#include "SolutionKOPTOperators.cu"
#endif
