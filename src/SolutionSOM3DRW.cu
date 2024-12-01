#include <cmath>
#include "SolutionSOM3D.h"
//#include "Multiout.h"

#define TEST_CODE   0

#define round(x) ((fabs(ceil(x) - (x)) < fabs(floor(x) - (x))) ? ceil(x) : floor(x))

//template <typename T> std::string tostr(const T& t)
//{
//    std::ostringstream os; os << t; return os.str();
//}

template<std::size_t DimP, std::size_t DimCM>
std::ofstream* SolutionSOM3D<DimP, DimCM>::OutputStream = NULL;

template<std::size_t DimP, std::size_t DimCM>
void SolutionSOM3D<DimP, DimCM>::readSolution()
{
    readSolution(fileData);
}

template<std::size_t DimP, std::size_t DimCM>
void SolutionSOM3D<DimP, DimCM>::readSolution(const char* file)
{
    cout << "BEGIN READ " << file << std::endl;
    traceParallel3DSom.benchMark = file;

    // Initialize NN netwoks CPU/GPU
    // read the tsp file into a NN (1D) and get the margin values.
    EMST_RW irw;
    irw.readFile(file, md_links_cpu, pMin, pMax);
    initialize(md_links_cpu, pMin, pMax);


}

template<std::size_t DimP, std::size_t DimCM>
void SolutionSOM3D<DimP, DimCM>::writeSolution()
{
    writeSolution(fileSolution);
}

template<std::size_t DimP, std::size_t DimCM>
void SolutionSOM3D<DimP, DimCM>::writeSolution(const char* file)
{
    cout << "BEGIN WRITE " << file << std::endl;

    // output
    mr_links_cpu.write(fileSolution);
    mr_links_cpu.writeLinks(fileSolution);

    IndexCM pc = vgd.getCenter();
    cout << pc << endl;
    IndexCM PC = vgd.F(vgd.getCenterBase());
    cout << PC << endl;
    IndexCM PCD = vgd.FDual(vgd.getCenterDual());
    cout << PCD << endl;

//    cout << vgd.FEuclid(pc) << endl;
//    cout << vgd.FEuclid(PC) << endl;
//    cout << vgd.FEuclid(PCD) << endl;
    cout << "Test Som done" << endl;

    cout << "END WRITE" << std::endl;
}

template<std::size_t DimP, std::size_t DimCM>
void SolutionSOM3D<DimP, DimCM>::openStatisticsFile()
{
    // Ouverture du fichier traitement en mode append
    OutputStream->open(fileStats, ios::app);
    if (!OutputStream->rdbuf()->is_open())
    {
        cerr << "Unable to open file " << fileStats << "CRITICAL ERROR" << endl;
        exit(-1);
    }
}

template<std::size_t DimP, std::size_t DimCM>
void SolutionSOM3D<DimP, DimCM>::closeStatisticsFile()
{
    OutputStream->close();
}

template<std::size_t DimP, std::size_t DimCM>
void SolutionSOM3D<DimP, DimCM>::initStatisticsFile()
{
    time(&t0);
    x0 = clock();
#ifdef CUDA_CODE
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
#endif
    openStatisticsFile();
    writeHeaderStatistics(*OutputStream);
    closeStatisticsFile();
}

template<std::size_t DimP, std::size_t DimCM>
void SolutionSOM3D<DimP, DimCM>::writeHeaderStatistics(std::ostream& o)
{
    ifstream fin(fileStats);
    string s;
    fin >> s;
    if(s.length() == 0)
    {
        o << "iteration" << setw(30)
          << "global_objectif" << setw(30)
          << "duree(s)" << setw(30)
          << "duree(s.xx)" << setw(30)
     #ifdef CUDA_CODE
          << "cuda_duree(ms)" << setw(30)
     #endif
          << "benchMark" << setw(30)
          << "size" << setw(30)
          << "length" << setw(30)
          << "timeFlatten" << setw(30)
          << "timeTestTermination" << setw(30)
          << "timeFindNextClosest" << setw(30)
          << "timeFindMinPair" << setw(30)
          << "timeConnectGraphUnion" << setw(30)
          << "timeCumulativeFindNextClosest" << setw(30)
          << "timeCumulativeFindMinPair" << setw(30)
          << "timeCumulativeConnectUnion" << setw(30)
          << "timeCumulativeFlatten"  << setw(30)
          << "timeCumulativeTerminate" << endl;

        //        o << endl;
    }
}

template<std::size_t DimP, std::size_t DimCM>
void SolutionSOM3D<DimP, DimCM>::writeStatistics(int iteration, std::ostream& o)
{
#ifdef CUDA_CODE
    // cuda timer
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop); //ReSolutionSOM3D ~0.5ms
#endif

    o << iteration << setw(30)
      << this->global_objectif << setw(30)
      << time(&tf) - t0 << setw(30)
      << (clock() - x0) / CLOCKS_PER_SEC << setw(30)
     #ifdef CUDA_CODE
      << elapsedTime << setw(30)
     #endif
      << traceParallel3DSom.benchMark << setw(30)
      << traceParallel3DSom.size << setw(30)
      << traceParallel3DSom.length << setw(30)
      << traceParallel3DSom.timeRefreshCm << setw(30)
      << traceParallel3DSom.timeTestTermination << setw(30)
      << traceParallel3DSom.timeFindNextClosest<< setw(30)
      << traceParallel3DSom.timeFindMinPair << setw(30)
      << traceParallel3DSom.timeConnectGraphUnion << setw(30)
      << traceParallel3DSom.timeCumulativeFindNextClosest << setw(30)
      << traceParallel3DSom.timeCumulativeFindMinPair << setw(30)
      << traceParallel3DSom.timeCumulativeConnetUnion << setw(30)
      << traceParallel3DSom.timeCumulativeFlatten << setw(30)
      << traceParallel3DSom.timeCumulativeTermination << endl;

    o << endl << endl;
}

template<std::size_t DimP, std::size_t DimCM>
void SolutionSOM3D<DimP, DimCM>::writeStatisticsToFile(int iteration)
{
    openStatisticsFile();
    writeStatistics(iteration, *OutputStream);
    closeStatisticsFile();
}

template<std::size_t DimP, std::size_t DimCM>
void SolutionSOM3D<DimP, DimCM>::writeStatisticsToFile()
{
    openStatisticsFile();
    writeStatistics(iteration, *OutputStream);
    closeStatisticsFile();
}

template<std::size_t DimP, std::size_t DimCM>
void SolutionSOM3D<DimP, DimCM>::writeStatistics(std::ostream& o)
{
    openStatisticsFile();
    writeStatistics(iteration, o);
    closeStatisticsFile();
}



