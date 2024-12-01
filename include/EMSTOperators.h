#ifndef EMST_OPERATORS_H
#define EMST_OPERATORS_H
/*
 ***************************************************************************
 *
 * Author : Wenbao Qiao, J.C. Créput
 * Creation date : Sep. 2016
 *
 ***************************************************************************
 */
#include <stdio.h>
#include <string.h>
#include <iostream>
#include <fstream>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <algorithm>

//#include <cuda_runtime.h>
//#include <cuda.h>
//#include <helper_functions.h>
//#include <device_launch_parameters.h>
//#include <curand_kernel.h>
#include <device_atomic_functions.h>
#include <sm_60_atomic_functions.h>
#include <sm_61_intrinsics.h>

#include <vector>
#include <iterator>

#include "macros_cuda.h"
#include "ConfigParams.h"
#include "Node.h"
#include "GridOfNodes.h"
#include "NeuralNet.h"
#include "distances_matching.h"
#include "basic_operations.h"
#include "Cell.h"
#include "adaptator_basics.h"
#include "CellularMatrix.h"
#include "distance_functors.h"
#include "NIter.h"

//! reference EMST components
#include "NeuralNetEMST.h"
#include "NodeEMST.h"

#define EMST_BLOCK_SIZE 128

#define BLOCKSIZE 1024
#define GRIDSIZE 1048576//(4opt) //16777216 //10485760 //2147483640 // 512000 (3opt) //20480 // 1024 for sw24978
#define SHAREDMAXCITIES 1979//734
#define OPTPOSSIBILITES4OPT 200
#define OPTPOSSIBILITES5OPT 2080
#define OPTPOSSIBILITES6OPT 23220

using namespace std;
using namespace components;

struct TspResultInfo
{
    GLfloat length = 0;
    GLint size = 0;
    GLfloat pdb = 0;
    GLfloat timeTestTermination = 0;
    GLfloat timeFindNextClosest = 0;
    GLfloat timeFindMinPair = 0;
    GLfloat timeConnectGraphUnion = 0;
    GLfloat timeCumulativeFindNextClosest = 0;
    GLfloat timeCumulativeFindMinPair = 0;
    // wb.Q add
    GLfloat timeCumulativeConnetUnion = 0;
    GLfloat timeCumulativeFlatten = 0;
    GLfloat timeCumulativeTermination = 0;
    GLfloat timeObtainKoptimal = 0;
    string benchMark = "";
    GLfloat optimumLength = 0;
};

//#include "MstOperator.h"

namespace operators
{


//! WB.Q add to return optimum value for PDB
float returnOptimal(string str){

    if(str == "qa194.tsp")
        return 9352;
    if(str == "ar9152.tsp")
        return 837479;
    if(str == "dj38.tsp")
        return 6656;
    if(str == "uy734.tsp")
        return 79114;
    else if(str == "zi929.tsp")
        return 95345;
    else if(str == "lu980.tsp")
        return 11340;
    else if(str == "rw1621.tsp")
        return 26051;
    else if(str == "mu1979.tsp")
        return 86891;
    else if(str == "UY734.tsp")
        return 79114;
    else if(str == "ZI929.tsp")
        return 95345;
    else if(str == "nu3496.tsp")
        return 96132;
    else if(str == "ca4663.tsp")
        return 1290319;
    else if(str == "tz6117.tsp")
        return 394718;
    else if(str == "eg7146.tsp")
        return 172387;
    else if(str == "ym7663.tsp")
        return 238314;
    else if(str == "ei8246.tsp")
        return 206171;
    else if(str == "ja9847.tsp")
        return 491924;
    else if(str == "gr9882.tsp")
        return 300899;
    else if(str == "kz9976.tsp")
        return 1061882;
    else if(str == "fi10639.tsp")
        return 520527;
    else if(str == "mo14185.tsp")
        return 427377;
    else if(str == "ho14473.tsp")
        return 177105;
    else if(str == "it16862.tsp")
        return 557315;
    else if(str == "vm22775.tsp")
        return 569288;
    else if(str == "sw24978.tsp")
        return 855597;
    else if(str == "bm33708.tsp")
        return 959304;
    else if(str == "ch71009.tsp")
        return 4566563;
    else
        return -1;
}


//!QWB add to test changement in grid
template<class type>
int testGridNum(Grid<type> testGrid){

    int numTest = 0;
    for (int j = 0; j < testGrid.height; j++ )
        for (int i = 0; i < testGrid.width; i++)
        {
            if (testGrid[j][i])
                numTest += 1;// testGrid[j][i];
        }
    return numTest;

}

//! WB.Q add to count time
//! copy source code from https://stackoverflow.com/questions/1739259/how-to-use-queryperformancecounter
//! The StartCounter() function records the number of ticks the performance counter has in the CounterStart variable.
//! The GetCounter() function returns the number of milliseconds since StartCounter() was last called as a double, so if GetCounter() returns 0.001 then it has been about 1 microsecond since StartCounter() was called.
void StartCounter(double& PCFreq, __int64& CounterStart)
{

    LARGE_INTEGER li;
    if(!QueryPerformanceFrequency(&li))
        cout << "QueryPerformanceFrequency failed!\n";

    PCFreq = double(li.QuadPart)/1000.0;// millisecond
    //    PCFreq = double(li.QuadPart); // s
    //    PCFreq = double(li.QuadPart)/1000000.0; // microsecond

    QueryPerformanceCounter(&li);
    CounterStart = li.QuadPart;
}
double GetCounter(double PCFreq, __int64 CounterStart)
{
    LARGE_INTEGER li;
    QueryPerformanceCounter(&li);
    return double(li.QuadPart-CounterStart)/PCFreq;
}


//! WB.Q error check cuda synchronozation
bool errorCheckCudaThreadSynchronize(){
    cudaError_t err = cudaThreadSynchronize();
    if (cudaSuccess != err) {
        fprintf(stderr, "cudaCheckError() failed at %s:%i : %s.\n", __FILE__, __LINE__, cudaGetErrorString( err ) );
        exit(1);
    }
    else
        return 0;
}

//! WB.Q error check cuda synchronozation
bool cudaChk(cudaError_t err){
    if (cudaSuccess != err) {
        fprintf(stderr, "cudaCheckError() failed at %s:%i : %s.\n", __FILE__, __LINE__, cudaGetErrorString( err ) );
        exit(1);
    }
    else
        return 0;
}

/*!
 * \brief 191116 QWB: add compute distance with ordered array
 */
__device__ float dist(int i, int j, doubleLinkedEdgeForTSP* coords){
    float dx, dy;
    dx = coords[i].currentCoord[0] - coords[j].currentCoord[0];
    dy = coords[i].currentCoord[1] - coords[j].currentCoord[1];
    return (dx*dx + dy*dy);
    //     double dist = (dx*dx + dy*dy);
    //     return sqrt(dist);
}


/*!
 * \brief 191116 QWB: add parallel 2-opt with rocki's method
 */
//epecially for small size, copy all cities into shared memory
KERNEL void K_2opt_oneThreadOne2opt_rockiSmall_shared(Grid<float> densityMap,
                                                      Grid<float> minRadiusMap,
                                                      Grid<doubleLinkedEdgeForTSP> arrayTSP,
                                                      unsigned long maxChecks,
                                                      unsigned int iter)
{

    int local_id = threadIdx.x + blockIdx.x * blockDim.x;
    int width =  densityMap.width; // each thread has this register


    if(local_id < maxChecks && threadIdx.x < width)
    {

        int i, j, id;

        __shared__ doubleLinkedEdgeForTSP sharedPoints[BLOCKSIZE]; // each sub-tour has the same length of blockDim.x

        for(int k = threadIdx.x; k < width; k+= blockDim.x)// step through all the points in list, but in blocks
        {
            sharedPoints[threadIdx.x] = arrayTSP[0][threadIdx.x];

            __syncthreads();

        }// end for copy shared

        for(unsigned int nu = 0; nu < iter; nu++)
        {
            id = local_id + nu * BLOCKSIZE * GRIDSIZE;

            if(id < maxChecks)
            {
                //WB.Q this way will produce i = j
                i = int(3 + sqrtf(8.0f * (float)id + 1.0f)) / 2 ;
                j = id - (i-2)*(i-1)/2 + 1;
                if(j > 0  && i < width-1 && j < i && j-1 >= 0 && j <= width && j+1 != i && j+ width != i+1 && i-1 >= 0 )
                {

                    float optimization =  dist(j-1, j, sharedPoints) + dist(i-1, i, sharedPoints) - dist(j-1, i-1, sharedPoints) - dist(j, i, sharedPoints);

                    if(optimization > 0)
                    {
                        // here automic operation is necessary
                        int node1 = (int)sharedPoints[j-1].current;
                        int node3 = (int)sharedPoints[i-1].current;

                        float localMinChange = minRadiusMap[0][node1];

                        if(optimization > localMinChange)
                        {
                            atomicExch(&(densityMap[0][node1]), node3); // WB.Q this way can work for multi-thread operation
                            atomicExch(&(minRadiusMap[0][node1]), optimization);
                        }

                    }

                }

            }

        }// for iter

    }
}// end K_2optOneThreadOne2opt



/*!
 * \brief 191116 QWB: add parallel 2-opt with rocki's method
 */
template <template<typename, typename> class NeuralNetLinks, class BufferDimension, class Point>
KERNEL void K_2opt_oneThreadOne2opt_rockiSmall(NeuralNetLinks<BufferDimension, Point> nn_source,
                                               Grid<doubleLinkedEdgeForTSP> arrayTSP,
                                               unsigned long maxChecks,
                                               unsigned int iter)
{

    int local_id = threadIdx.x + blockIdx.x * blockDim.x;
    int width =  nn_source.adaptiveMap.width; // each thread has this register

    if(local_id < maxChecks){

        //        int packSize = blockDim.x * gridDim.x;
        int i, j, id;

        //        for(int nu = 0; nu <= iter; nu++)
        {

            id = local_id;// + nu * packSize;

            //            //qiao only for test
            //            if(id == maxChecks - 2)
            //                printf("Check inner GPU id == maxCheck2opt %d ", id); //correct

            //            if(id < maxChecks)
            {
                //WB.Q this way will produce i = j
                i = int(3 + sqrt(8.0f * (float)id + 1.0f)) / 2 ;
                j = id - (i-2)*(i-1)/2 + 1;

                //qiao only for test
                if(id == maxChecks-2)
                    printf("maximum 2-opt id id %d,  i, j: %d %d \n", id, i,j);


                if(j > 0 && j < i && j-1 >= 0 && j <= width && j+1 != i && j+ width != i+1 && i-1 >= 0 && i < width-1)
                {
                    //qiao for test to see 2-opt pairs
                    // printf("2-opt selected id %d,  i %d, j %d \n", id, i,j);

                    float oldLength =  dist(j-1, j, arrayTSP[0]) + dist(i-1, i, arrayTSP[0]);
                    float newLength = dist(j-1, i-1, arrayTSP[0]) + dist(j, i, arrayTSP[0]);

                    if(newLength < oldLength)
                    {
                        float optimization = oldLength - newLength;
                        // here automic operation is necessary
                        int node1 = (int)arrayTSP[0][j-1].current;
                        int node3 = (int)arrayTSP[0][i-1].current;

                        float localMinChange = nn_source.minRadiusMap[0][node1];

                        if(optimization > localMinChange)
                        {
                            atomicExch(&(nn_source.densityMap[0][node1]), node3); // WB.Q this way can work for multi-thread operation
                            atomicExch(&(nn_source.minRadiusMap[0][node1]), optimization);
                        }
                    }
                }
            }
        }
    }
    __syncthreads();
}// end K_2optOneThreadOne2opt




/*!
 * \brief 191116 QWB: add parallel 2-opt with rocki's method work correctly
 */
template <template<typename, typename> class NeuralNetLinks, class BufferDimension, class Point>
KERNEL void K_2opt_oneThreadOne2opt_qiaoIterStride_best(NeuralNetLinks<BufferDimension, Point> nn_source,
                                                        Grid<doubleLinkedEdgeForTSP> arrayTSP,
                                                        double max2optChecks, double maxChecksOptDivide,
                                                        double iter, double istride)
{

    double local_id = threadIdx.x + blockIdx.x * blockDim.x;
    int width =  nn_source.adaptiveMap.width; // each thread has this register

    if(local_id < max2optChecks){

        double startId = maxChecksOptDivide * istride;

        if(local_id == 0)
            printf("StartID %f, local_id %f \n", startId, local_id);

        for(double id = local_id*iter ; id < (local_id+1)*(iter); id++)
        {

            id = id + startId;

            if(id > 0 && id < max2optChecks)
            {

                int i, j;
                id = trunc(id);

                //WB.Q this way will produce i = j
                i = int(3 + sqrt(8.0f * (double)id + 1.0f)) / 2 ;
                j = id - (i-2)*(i-1)/2 + 1;

                //qiao only for test
                if(id == max2optChecks-2)
                    printf("maximum 2-opt id id %f,  i, j: %d %d \n", id, i,j);


                if(j > 0 && j < i && j-1 >= 0 && j <= width && j+1 != i && j+ width != i+1 && i-1 >= 0 && i < width-1)
                {
                    //qiao for test to see 2-opt pairs
                    // printf("2-opt selected id %d,  i %d, j %d \n", id, i,j);

                    float oldLength =  dist(j-1, j, arrayTSP[0]) + dist(i-1, i, arrayTSP[0]);
                    float newLength = dist(j-1, i-1, arrayTSP[0]) + dist(j, i, arrayTSP[0]);

                    if(newLength < oldLength)
                    {
                        float optimization = oldLength - newLength;
                        // here automic operation is necessary
                        int node1 = (int)arrayTSP[0][j-1].current;
                        int node3 = (int)arrayTSP[0][i-1].current;

                        float localMinChange = nn_source.minRadiusMap[0][node1];

                        if(optimization > localMinChange)
                        {
                            atomicExch(&(nn_source.densityMap[0][node1]), node3); // WB.Q this way can work for multi-thread operation
                            atomicExch(&(nn_source.minRadiusMap[0][node1]), optimization);
                        }
                    }
                }
            }
        }
    }
    __syncthreads();
}// end K_2optOneThreadIter2opt



/*!
 * \brief 191116 QWB: add parallel 2-opt with rocki's method work correctly
 */
template <template<typename, typename> class NeuralNetLinks, class BufferDimension, class Point>
KERNEL void K_2opt_oneThreadOne2opt_qiaoIterStride(NeuralNetLinks<BufferDimension, Point> nn_source,
                                                   Grid<doubleLinkedEdgeForTSP> arrayTSP,
                                                   double max2optChecks, double maxChecksOptDivide,
                                                   double iter, double istride)
{

    double local_id = threadIdx.x + blockIdx.x * blockDim.x;
    int width =  nn_source.adaptiveMap.width; // each thread has this register

    if(local_id < max2optChecks){

        double startId = maxChecksOptDivide * istride;

        if(local_id == 0)
            printf("StartID %f, local_id %f \n", startId, local_id);

        for(double id = local_id*iter ; id < (local_id+1)*(iter); id++)
        {

            id = id + startId;



            if(id > 0 && id < max2optChecks)
            {

                int i, j;
                id = trunc(id);

                //WB.Q this way will produce i = j
                i = int(3 + sqrt(8.0f * (double)id + 1.0f)) / 2 ;
                j = id - (i-2)*(i-1)/2 + 1;

                //qiao only for test
                if(id == max2optChecks-2)
                    printf("maximum 2-opt id id %f,  i, j: %d %d \n", id, i,j);


                if(j > 0 && j < i && j-1 >= 0 && j <= width && j+1 != i && j+ width != i+1 && i-1 >= 0 && i < width-1)
                {
                    //qiao for test to see 2-opt pairs
                    // printf("2-opt selected id %d,  i %d, j %d \n", id, i,j);

                    bool existingCandidate = 0;
                    if(nn_source.minRadiusMap[0][j-1] == 1 || nn_source.minRadiusMap[0][i-1] == 1 )
                        existingCandidate = 1;

                    if(existingCandidate == 0)
                    {

                        float oldLength =  dist(j-1, j, arrayTSP[0]) + dist(i-1, i, arrayTSP[0]);
                        float newLength = dist(j-1, i-1, arrayTSP[0]) + dist(j, i, arrayTSP[0]);

                        atomicExch(&(nn_source.minRadiusMap[0][i-1]), 1);
                        atomicExch(&(nn_source.minRadiusMap[0][j-1]), 1);

                        if(newLength < oldLength)
                        {
                            // here automic operation is necessary
                            int node1 = (int)arrayTSP[0][j-1].current;
                            int node3 = (int)arrayTSP[0][i-1].current;

                            atomicExch(&(nn_source.densityMap[0][node1]), node3); // WB.Q this way can work for multi-thread operation

                        }

                    }
                }
            }
        }
    }
    __syncthreads();
}// end K_2optOneThreadIter2opt


/*!
 * \brief 191116 QWB: add parallel 2-opt with rocki's method work correctly
 */
template <template<typename, typename> class NeuralNetLinks, class BufferDimension, class Point>
KERNEL void K_2opt_oneThreadOne2opt_qiaoIterStride_best_shared(NeuralNetLinks<BufferDimension, Point> nn_source,
                                                               Grid<doubleLinkedEdgeForTSP> arrayTSP,
                                                               double max2optChecks, double maxChecksOptDivide,
                                                               double iter, double istride)
{

    double local_id = threadIdx.x + blockIdx.x * blockDim.x;
    int width =  nn_source.adaptiveMap.width; // each thread has this register

    if(local_id < max2optChecks){

        double startId = maxChecksOptDivide * istride;

        //        if(local_id == 0)
        //            printf("StartID %f, local_id %f \n", startId, local_id);

        for(double id = local_id*iter ; id < (local_id+1)*(iter); id++)
        {

            id = id + startId;

            if(id > 0 && id < max2optChecks)
            {

                int i, j;
                id = trunc(id);

                //WB.Q this way will produce i = j
                i = int(3 + sqrt(8.0f * (double)id + 1.0f)) / 2 ;
                j = id - (i-2)*(i-1)/2 + 1;

                //qiao only for test
                if(id == max2optChecks-2)
                    printf("maximum 2-opt id id %f,  i, j: %d %d \n", id, i,j);


                if(j > 0 && j < i && j-1 >= 0 && j <= width && j+1 != i && j+ width != i+1 && i-1 >= 0 && i < width-1)
                {
                    //qiao for test to see 2-opt pairs
                    // printf("2-opt selected id %d,  i %d, j %d \n", id, i,j);

                    float oldLength =  dist(j-1, j, arrayTSP[0]) + dist(i-1, i, arrayTSP[0]);
                    float newLength = dist(j-1, i-1, arrayTSP[0]) + dist(j, i, arrayTSP[0]);

                    if(newLength < oldLength)
                    {
                        float optimization = oldLength - newLength;
                        // here automic operation is necessary
                        int node1 = (int)arrayTSP[0][j-1].current;
                        int node3 = (int)arrayTSP[0][i-1].current;

                        float localMinChange = nn_source.minRadiusMap[0][node1];

                        if(optimization > localMinChange)
                        {
                            atomicExch(&(nn_source.densityMap[0][node1]), node3); // WB.Q this way can work for multi-thread operation
                            atomicExch(&(nn_source.minRadiusMap[0][node1]), optimization);
                        }
                    }
                }
            }
        }
    }
    __syncthreads();
}// end K_2optOneThreadIter2opt


/*!
 * \brief 191116 QWB: add parallel 2-opt with rocki's method
 */
template <template<typename, typename> class NeuralNetLinks, class BufferDimension, class Point>
KERNEL void K_2opt_oneThreadOne2opt_rockiSmall_iter(NeuralNetLinks<BufferDimension, Point> nn_source,
                                                    Grid<doubleLinkedEdgeForTSP> arrayTSP,
                                                    unsigned long maxChecks,
                                                    unsigned int iter)
{

    int local_id = threadIdx.x + blockIdx.x * blockDim.x;
    int width =  nn_source.adaptiveMap.width; // each thread has this register

    if(local_id < maxChecks){

        int packSize = blockDim.x * gridDim.x;
        int i, j, id;

        for(int nu = 0; nu <= iter; nu++)
        {

            id = local_id + nu * packSize;

            //            //qiao only for test
            //            if(id == maxChecks - 2)
            //                printf("Check inner GPU id == maxCheck2opt %d ", id); //correct

            //            if(id < maxChecks)
            {
                //WB.Q this way will produce i = j
                i = int(3 + sqrt(8.0f * (float)id + 1.0f)) / 2 ;
                j = id - (i-2)*(i-1)/2 + 1;

                //qiao only for test
                if(id == maxChecks-2)
                    printf("maximum 2-opt id id %d,  i, j: %d %d \n", id, i,j);


                if(j > 0 && j < i && j-1 >= 0 && j <= width && j+1 != i && j+ width != i+1 && i-1 >= 0 && i < width-1)
                {
                    //qiao for test to see 2-opt pairs
                    // printf("2-opt selected id %d,  i %d, j %d \n", id, i,j);

                    float oldLength =  dist(j-1, j, arrayTSP[0]) + dist(i-1, i, arrayTSP[0]);
                    float newLength = dist(j-1, i-1, arrayTSP[0]) + dist(j, i, arrayTSP[0]);

                    if(newLength < oldLength)
                    {
                        float optimization = oldLength - newLength;
                        // here automic operation is necessary
                        int node1 = (int)arrayTSP[0][j-1].current;
                        int node3 = (int)arrayTSP[0][i-1].current;

                        float localMinChange = nn_source.minRadiusMap[0][node1];

                        if(optimization > localMinChange)
                        {
                            atomicExch(&(nn_source.densityMap[0][node1]), node3); // WB.Q this way can work for multi-thread operation
                            atomicExch(&(nn_source.minRadiusMap[0][node1]), optimization);
                        }
                    }
                }
            }
        }
    }
    __syncthreads();
}// end K_2optOneThreadOne2opt





/*!
 * \brief 2024 QWB: add parallel 4-opt
 */
//epecially for small size, copy all cities into shared memory
template <template<typename, typename> class NeuralNetLinks, class BufferDimension, class Point>
KERNEL void K_4opt_oneThreadOne4opt_rockiSmall(NeuralNetLinks<BufferDimension, Point> nn_source,
                                               Grid<doubleLinkedEdgeForTSP> arrayTSP,
                                               double maxChecks2opt, double maxChecks4opt,
                                               unsigned int iter)
{

    double  id = threadIdx.x + blockIdx.x * blockDim.x;
    int width =  nn_source.adaptiveMap.width; // each thread has this register

    if(id < maxChecks4opt)
    {

        //        int packSize = blockDim.x * gridDim.x;
        double  outi, outj;

        //        id = trunc(id);

        if(id > maxChecks4opt-5)//350631671)
            printf("largeID id %f \n", id);
        else if (id <5)
            printf("SmallID id %f \n", id);

        {

            //            id = local_id;// + nu * packSize;

            //            if(id < maxChecks4opt)
            {
                //WB.Q this way will produce i = j
                outi = (3 + sqrt(8.0f * (double )id + 1.0f)) / 2 ;
                outj = id - (outi-2)*(outi-1)/2 + 1;

                if(outi < maxChecks2opt && outj < maxChecks2opt)
                {
                    int k = int(3 + sqrt(8.0f * (double )outi + 1.0f)) / 2 ;
                    int p = outi - (k-2)*(k-1)/2 + 1;

                    int j = int(3 + sqrt(8.0f * (double )outj + 1.0f)) / 2 ;
                    int w = outj - (j-2)*(j-1)/2 + 1;


                    if( k > p && p> j&& j>w&& k< width && p<width && j<width && w<width &&  k > 0 && p > 0 && j > 0 && w > 0 && p+1!=k && w+1!=j && j+1!=p)
                    {

                        //                        if(k > width - 2)
                        //                            printf(" maximum 4-opt id = %d, outi= %d, outj=%d, inner k,p,j,w =(%d, %d, %d, %d, %d), \n", id, outi, outj, k, p, j, w);


                        bool existingCandidate = 0;
                        if(nn_source.minRadiusMap[0][w-1] == 1 || nn_source.minRadiusMap[0][j-1] == 1 ||nn_source.minRadiusMap[0][p-1] == 1 ||nn_source.minRadiusMap[0][k-1] == 1)
                            existingCandidate = 1;

                        if(existingCandidate == 0)
                        {

                            float oldLength = dist(w-1, w, arrayTSP[0]) + dist(j-1, j, arrayTSP[0]) + dist(p-1, p, arrayTSP[0])+ dist(k-1, k, arrayTSP[0]);
                            float newLength;//25 is fixed for 4-opt
                            int array[8];
                            array[0] = w-1;
                            array[1] = w;
                            array[2] = j-1;
                            array[3] = j;
                            array[4] = p-1;
                            array[5] = p;
                            array[6] = k-1;
                            array[7] = k;

                            int finalSelect = -1;
                            float optimiz = -INFINITY;

                            for(int opt = 0; opt < 200; opt +=8) //  4 edges 8 nodes
                            {
                                int nd1 = nn_source.nodeParentMap[0][opt] -1;
                                int nd2 = nn_source.nodeParentMap[0][opt+1] -1;
                                int nd3 = nn_source.nodeParentMap[0][opt+2] -1;
                                int nd4 = nn_source.nodeParentMap[0][opt+3] -1;
                                int nd5 = nn_source.nodeParentMap[0][opt+4] -1;
                                int nd6 = nn_source.nodeParentMap[0][opt+5] -1;
                                int nd7 = nn_source.nodeParentMap[0][opt+6] -1;
                                int nd8 = nn_source.nodeParentMap[0][opt+7] -1;


                                int optCandi = opt / 8;
                                //printf("GPU search nd1-8 %d, %d, %d, %d, %d, %d, %d, %d; optCandi=%d \n", nd1, nd2, nd3, nd4, nd5, nd6, nd7, nd8, optCandi);
                                newLength= dist(array[nd1],array[nd2], arrayTSP[0]) + dist(array[nd3],array[nd4], arrayTSP[0]) + dist(array[nd5],array[nd6], arrayTSP[0])+ dist(array[nd7],array[nd8], arrayTSP[0]);

                                float opti = oldLength - newLength;
                                if(opti > 0 && opti > optimiz)
                                {
                                    finalSelect = optCandi;
                                    optimiz = opti;

                                    atomicExch(&(nn_source.minRadiusMap[0][w-1]), 1);
                                    atomicExch(&(nn_source.minRadiusMap[0][j-1]), 1);
                                    atomicExch(&(nn_source.minRadiusMap[0][p-1]), 1);
                                    atomicExch(&(nn_source.minRadiusMap[0][k-1]), 1);

                                }
                            }


                            if(finalSelect >= 0)
                            {

                                unsigned int node1 = (int)arrayTSP[0][w-1].current;
                                unsigned int node3 = (int)arrayTSP[0][j-1].current;
                                unsigned int node5 = (int)arrayTSP[0][p-1].current;
                                unsigned int node7 = (int)arrayTSP[0][k-1].current;

                                //                            float localMinChange = nn_source.minRadiusMap[0][node1];

                                //                            if(optimiz > localMinChange)
                                {

                                    unsigned long long result = 0;
                                    result = result | node3;
                                    result = result << 16;
                                    result = result | node5;
                                    result = result << 16;
                                    result = result | node7;

                                    float codekopt = finalSelect; //WB.Q 2024 densityMap only mark the k value of k-opt, and mark which mode of k-exchange
                                    codekopt = finalSelect * 100 + 4;

                                    //  printf("GPU search, node1, node3, node5, node7, %d, %d, %d, %d; order(%d,%d,%d,%d), optvalue %lld, codekopt %f \n",
                                    //           node1, node3, node5, node7, nn_source.grayValueMap[0][node1], nn_source.grayValueMap[0][node3], nn_source.grayValueMap[0][node5] , nn_source.grayValueMap[0][node7], result, codekopt);
                                    atomicExch(&(nn_source.optCandidateMap[0][node1]), result); // WB.Q 2024 find a solution to judge non-interacted 23456-opt
                                    atomicExch(&(nn_source.densityMap[0][node1]), codekopt); // WB.Q 2024 densityMap only mark the k value of k-opt, and mark which mode of k-exchange
                                    //                                atomicExch(&(nn_source.minRadiusMap[0][node1]), optimiz);
                                }
                            }

                        }

                    }//end if k<p
                }

            }
        }
    }
    __syncthreads();
}// end K_4optOneThreadOne4opt


/*!
 * \brief 2024 QWB: add parallel 4-opt
 */
template <template<typename, typename> class NeuralNetLinks, class BufferDimension, class Point>
KERNEL void K_4opt_oneThreadOne4opt_qiaoIterStride(NeuralNetLinks<BufferDimension, Point> nn_source,
                                                   Grid<doubleLinkedEdgeForTSP> arrayTSP,
                                                   double maxChecks2opt, double maxChecks4opt, double maxChecks4optDivide,
                                                   double iter, double istride)
{

    double local_id = threadIdx.x + blockIdx.x * blockDim.x;// + gridDim * blockDim.x;
    int width =  nn_source.adaptiveMap.width; // each thread has this register

    if(local_id < maxChecks4opt)
    {
        double startId = maxChecks4optDivide * (istride);

        if(local_id == 0 )
            printf("StartID %f, local_id %f \n", startId, local_id);

        for(double id = local_id*iter ; id < (local_id+1)*(iter); id++)
        {
            id = id + startId;

            if(id > 0 && id < maxChecks4opt)
            {

                double  outi, outj;
                double sqrtOuti = 8.0 * (double)id + 1.0;
                //                outi = int(3 + sqrt(sqrtOuti)) / 2 ;
                outi = (3 + sqrt(sqrtOuti)) / 2 ;
                outi = trunc(outi);
                outj = id - (outi-2)*(outi-1)/2 + 1;


                //                double test = startId + 0.99*maxChecks4optDivide ;
                //                if(startId > 0&& id > test)//350631671)
                //                    printf("largeID %f, local_id %f \n", id, local_id);
                //                else if (id <5)
                //                    printf("SmallID %f, local_id %f, outi %f, outj %f \n", id, local_id, outi, outj);
                //                else if (id <0)
                //                    printf("SmallID < 0 %f, local_id %f , outi %f, outj %f \n", id, local_id, outi, outj);
                if(id == (startId + maxChecks4optDivide-1))
                    printf("bound id  %f, local_id %f , outi %f, outj %f \n", id, local_id, outi, outj);


                if(outi < maxChecks2opt && outj < maxChecks2opt)
                {

                    double sqrtOutIK = 8.0 * (double )outi + 1.0;
                    int k = int(3 + sqrt(sqrtOutIK)) / 2 ;
                    int p = outi - (k-2)*(k-1)/2 + 1;

                    double sqrtOutJk = 8.0 * (double)outj + 1.0;
                    int j = int(3 + sqrt(sqrtOutJk)) / 2 ;
                    int w = outj - (j-2)*(j-1)/2 + 1;


                    if( k > p && p> j&& j>w&& k< width && p<width && j<width && w<width &&  k > 0 && p > 0 && j > 0 && w > 0 && p+1!=k && w+1!=j && j+1!=p)
                    {

                        if(w > 9144)
                            printf(" maximum 4-opt id = %f, outi= %f, outj=%f, inner k,p,j,w =(%d, %d, %d, %d), \n", id, outi, outj, k, p, j, w);


                        bool existingCandidate = 0;
                        if(nn_source.minRadiusMap[0][w-1] == 1 || nn_source.minRadiusMap[0][j-1] == 1 ||nn_source.minRadiusMap[0][p-1] == 1 ||nn_source.minRadiusMap[0][k-1] == 1)
                            existingCandidate = 1;

                        if(existingCandidate == 0)
                        {

                            float oldLength = dist(w-1, w, arrayTSP[0]) + dist(j-1, j, arrayTSP[0]) + dist(p-1, p, arrayTSP[0])+ dist(k-1, k, arrayTSP[0]);
                            float newLength;//25 is fixed for 4-opt
                            int array[8];
                            array[0] = w-1;
                            array[1] = w;
                            array[2] = j-1;
                            array[3] = j;
                            array[4] = p-1;
                            array[5] = p;
                            array[6] = k-1;
                            array[7] = k;

                            int finalSelect = -1;
                            float optimiz = -INFINITY;

                            for(int opt = 0; opt < 200; opt +=8) //  4 edges 8 nodes
                            {
                                int nd1 = nn_source.nodeParentMap[0][opt] -1;
                                int nd2 = nn_source.nodeParentMap[0][opt+1] -1;
                                int nd3 = nn_source.nodeParentMap[0][opt+2] -1;
                                int nd4 = nn_source.nodeParentMap[0][opt+3] -1;
                                int nd5 = nn_source.nodeParentMap[0][opt+4] -1;
                                int nd6 = nn_source.nodeParentMap[0][opt+5] -1;
                                int nd7 = nn_source.nodeParentMap[0][opt+6] -1;
                                int nd8 = nn_source.nodeParentMap[0][opt+7] -1;


                                int optCandi = opt / 8;
                                //printf("GPU search nd1-8 %d, %d, %d, %d, %d, %d, %d, %d; optCandi=%d \n", nd1, nd2, nd3, nd4, nd5, nd6, nd7, nd8, optCandi);
                                newLength= dist(array[nd1],array[nd2], arrayTSP[0]) + dist(array[nd3],array[nd4], arrayTSP[0]) + dist(array[nd5],array[nd6], arrayTSP[0])+ dist(array[nd7],array[nd8], arrayTSP[0]);

                                float opti = oldLength - newLength;
                                if(opti > 0 && opti > optimiz)
                                {
                                    finalSelect = optCandi;
                                    optimiz = opti;

                                    atomicExch(&(nn_source.minRadiusMap[0][w-1]), 1);
                                    atomicExch(&(nn_source.minRadiusMap[0][j-1]), 1);
                                    atomicExch(&(nn_source.minRadiusMap[0][p-1]), 1);
                                    atomicExch(&(nn_source.minRadiusMap[0][k-1]), 1);

                                }
                            }


                            if(finalSelect >= 0)
                            {

                                unsigned int node1 = (int)arrayTSP[0][w-1].current;
                                unsigned int node3 = (int)arrayTSP[0][j-1].current;
                                unsigned int node5 = (int)arrayTSP[0][p-1].current;
                                unsigned int node7 = (int)arrayTSP[0][k-1].current;

                                //                            float localMinChange = nn_source.minRadiusMap[0][node1];

                                //                            if(optimiz > localMinChange)
                                {

                                    unsigned long long result = 0;
                                    result = result | node3;
                                    result = result << 16;
                                    result = result | node5;
                                    result = result << 16;
                                    result = result | node7;

                                    float codekopt = finalSelect; //WB.Q 2024 densityMap only mark the k value of k-opt, and mark which mode of k-exchange
                                    codekopt = finalSelect * 100 + 4;

                                    //  printf("GPU search, node1, node3, node5, node7, %d, %d, %d, %d; order(%d,%d,%d,%d), optvalue %lld, codekopt %f \n",
                                    //           node1, node3, node5, node7, nn_source.grayValueMap[0][node1], nn_source.grayValueMap[0][node3], nn_source.grayValueMap[0][node5] , nn_source.grayValueMap[0][node7], result, codekopt);
                                    atomicExch(&(nn_source.optCandidateMap[0][node1]), result); // WB.Q 2024 find a solution to judge non-interacted 23456-opt
                                    atomicExch(&(nn_source.densityMap[0][node1]), codekopt); // WB.Q 2024 densityMap only mark the k value of k-opt, and mark which mode of k-exchange
                                    //                                atomicExch(&(nn_source.minRadiusMap[0][node1]), optimiz);
                                }
                            }

                        }

                    }//end if k<p
                }

            }
        }

    }
    __syncthreads();
}// end K_4optOneThreadOne4opt




/*!
 * \brief 2024 QWB: add parallel 4-opt
 */
template <template<typename, typename> class NeuralNetLinks, class BufferDimension, class Point>
KERNEL void K_4opt_oneThreadOne4opt_qiaoIterStride_Best(NeuralNetLinks<BufferDimension, Point> nn_source,
                                                        Grid<doubleLinkedEdgeForTSP> arrayTSP,
                                                        double maxChecks2opt, double maxChecks4opt, double maxChecks4optDivide,
                                                        double iter, double istride)
{

    double local_id = threadIdx.x + blockIdx.x * blockDim.x;// + gridDim * blockDim.x;
    int width =  nn_source.adaptiveMap.width; // each thread has this register

    if(local_id < maxChecks4opt)
    {
        double startId = maxChecks4optDivide * (istride);

        if(local_id == 0 )
            printf("StartID %f, local_id %f \n", startId, local_id);

        for(double id = local_id*iter ; id < (local_id+1)*(iter); id++)
        {
            id = id + startId;

            if(id > 0 && id < maxChecks4opt)
            {

                double  outi, outj;
                double sqrtOuti = 8.0 * (double)id + 1.0;
                //                outi = int(3 + sqrt(sqrtOuti)) / 2 ;
                outi = (3 + sqrt(sqrtOuti)) / 2 ;
                outi = trunc(outi);
                outj = id - (outi-2)*(outi-1)/2 + 1;


                //                double test = startId + 0.99*maxChecks4optDivide ;
                //                if(startId > 0&& id > test)//350631671)
                //                    printf("largeID %f, local_id %f \n", id, local_id);
                //                else if (id <5)
                //                    printf("SmallID %f, local_id %f, outi %f, outj %f \n", id, local_id, outi, outj);
                //                else if (id <0)
                //                    printf("SmallID < 0 %f, local_id %f , outi %f, outj %f \n", id, local_id, outi, outj);
                //                if(id == (startId + maxChecks4optDivide-1))
                //                    printf("bound id  %f, local_id %f , outi %f, outj %f \n", id, local_id, outi, outj);


                if(outi < maxChecks2opt && outj < maxChecks2opt)
                {

                    double sqrtOutIK = 8.0 * (double )outi + 1.0;
                    int k = int(3 + sqrt(sqrtOutIK)) / 2 ;
                    int p = outi - (k-2)*(k-1)/2 + 1;

                    double sqrtOutJk = 8.0 * (double)outj + 1.0;
                    int j = int(3 + sqrt(sqrtOutJk)) / 2 ;
                    int w = outj - (j-2)*(j-1)/2 + 1;


                    if( k > p && p> j&& j>w&& k< width && p<width && j<width && w<width &&  k > 0 && p > 0 && j > 0 && w > 0 && p+1!=k && w+1!=j && j+1!=p)
                    {

                        if(w > 9144)
                            printf(" maximum 4-opt id = %f, outi= %f, outj=%f, inner k,p,j,w =(%d, %d, %d, %d), \n", id, outi, outj, k, p, j, w);


                        //                        bool existingCandidate = 0;
                        //                        if(nn_source.minRadiusMap[0][w-1] == 1 || nn_source.minRadiusMap[0][j-1] == 1 ||nn_source.minRadiusMap[0][p-1] == 1 ||nn_source.minRadiusMap[0][k-1] == 1)
                        //                            existingCandidate = 1;

                        //                        if(existingCandidate == 0)
                        {

                            float oldLength = dist(w-1, w, arrayTSP[0]) + dist(j-1, j, arrayTSP[0]) + dist(p-1, p, arrayTSP[0])+ dist(k-1, k, arrayTSP[0]);
                            float newLength;//25 is fixed for 4-opt
                            int array[8];
                            array[0] = w-1;
                            array[1] = w;
                            array[2] = j-1;
                            array[3] = j;
                            array[4] = p-1;
                            array[5] = p;
                            array[6] = k-1;
                            array[7] = k;

                            int finalSelect = -1;
                            float optimiz = -INFINITY;

                            for(int opt = 0; opt < 200; opt +=8) //  4 edges 8 nodes
                            {
                                int nd1 = nn_source.nodeParentMap[0][opt] -1;
                                int nd2 = nn_source.nodeParentMap[0][opt+1] -1;
                                int nd3 = nn_source.nodeParentMap[0][opt+2] -1;
                                int nd4 = nn_source.nodeParentMap[0][opt+3] -1;
                                int nd5 = nn_source.nodeParentMap[0][opt+4] -1;
                                int nd6 = nn_source.nodeParentMap[0][opt+5] -1;
                                int nd7 = nn_source.nodeParentMap[0][opt+6] -1;
                                int nd8 = nn_source.nodeParentMap[0][opt+7] -1;


                                int optCandi = opt / 8;
                                //printf("GPU search nd1-8 %d, %d, %d, %d, %d, %d, %d, %d; optCandi=%d \n", nd1, nd2, nd3, nd4, nd5, nd6, nd7, nd8, optCandi);
                                newLength= dist(array[nd1],array[nd2], arrayTSP[0]) + dist(array[nd3],array[nd4], arrayTSP[0]) + dist(array[nd5],array[nd6], arrayTSP[0])+ dist(array[nd7],array[nd8], arrayTSP[0]);

                                float opti = oldLength - newLength;
                                if(opti > 0 && opti > optimiz)
                                {
                                    finalSelect = optCandi;
                                    optimiz = opti;

                                    //                                    atomicExch(&(nn_source.minRadiusMap[0][w-1]), 1);
                                    //                                    atomicExch(&(nn_source.minRadiusMap[0][j-1]), 1);
                                    //                                    atomicExch(&(nn_source.minRadiusMap[0][p-1]), 1);
                                    //                                    atomicExch(&(nn_source.minRadiusMap[0][k-1]), 1);

                                }
                            }


                            if(finalSelect >= 0)
                            {

                                unsigned int node1 = (int)arrayTSP[0][w-1].current;
                                unsigned int node3 = (int)arrayTSP[0][j-1].current;
                                unsigned int node5 = (int)arrayTSP[0][p-1].current;
                                unsigned int node7 = (int)arrayTSP[0][k-1].current;

                                float localMinChange = nn_source.minRadiusMap[0][node1];

                                if(optimiz > localMinChange)
                                {

                                    unsigned long long result = 0;
                                    result = result | node3;
                                    result = result << 16;
                                    result = result | node5;
                                    result = result << 16;
                                    result = result | node7;

                                    float codekopt = finalSelect; //WB.Q 2024 densityMap only mark the k value of k-opt, and mark which mode of k-exchange
                                    codekopt = finalSelect * 100 + 4;

                                    //  printf("GPU search, node1, node3, node5, node7, %d, %d, %d, %d; order(%d,%d,%d,%d), optvalue %lld, codekopt %f \n",
                                    //           node1, node3, node5, node7, nn_source.grayValueMap[0][node1], nn_source.grayValueMap[0][node3], nn_source.grayValueMap[0][node5] , nn_source.grayValueMap[0][node7], result, codekopt);
                                    atomicExch(&(nn_source.optCandidateMap[0][node1]), result); // WB.Q 2024 find a solution to judge non-interacted 23456-opt
                                    atomicExch(&(nn_source.densityMap[0][node1]), codekopt); // WB.Q 2024 densityMap only mark the k value of k-opt, and mark which mode of k-exchange
                                    atomicExch(&(nn_source.minRadiusMap[0][node1]), optimiz);
                                }
                            }

                        }

                    }//end if k<p
                }

            }
        }

    }
    __syncthreads();
}// end K_4optOneThreadOne4opt




/*!
 * \brief 2024 QWB: add parallel 4-opt
 */
template <template<typename, typename> class NeuralNetLinks, class BufferDimension, class Point>
KERNEL void K_4opt_oneThreadOne4opt_qiaoIterStride_Best_shared(NeuralNetLinks<BufferDimension, Point> nn_source,
                                                               Grid<doubleLinkedEdgeForTSP> arrayTSP,
                                                               double maxChecks2opt, double maxChecks4opt, double maxChecks4optDivide,
                                                               double iter, double istride)
{

    double local_id = threadIdx.x + blockIdx.x * blockDim.x;// + gridDim * blockDim.x;
    int width =  nn_source.adaptiveMap.width; // each thread has this register




    __shared__ doubleLinkedEdgeForTSP sharedArrayTSP[SHAREDMAXCITIES];
    //        __shared__ float sharedArrayOccupied[SHAREDMAXCITIES];
    __shared__ float optPossibilities[OPTPOSSIBILITES4OPT];
    float iterShared = (float)width / (float)BLOCKSIZE;
    for(int opt = 0; opt < iterShared; opt++)
    {
        int m = threadIdx.x + opt*BLOCKSIZE;
        if(m < width)
        {
            sharedArrayTSP[m].current = arrayTSP[0][m].current;
            sharedArrayTSP[m].currentCoord[0] = arrayTSP[0][m].currentCoord[0];
            sharedArrayTSP[m].currentCoord[1] = arrayTSP[0][m].currentCoord[1];

        }
        __syncthreads();
    }

    if(threadIdx.x < OPTPOSSIBILITES4OPT)
        optPossibilities[threadIdx.x] = nn_source.nodeParentMap[0][threadIdx.x ];

    __syncthreads();

    if(local_id < maxChecks4opt)
    {
        double startId = maxChecks4optDivide * (istride);

        if(local_id == 0 )
            printf("StartID %f, local_id %f \n", startId, local_id);


        for(double id = local_id*iter ; id < (local_id+1)*(iter); id++)
        {
            id = id + startId;

            if(id > 0 && id < maxChecks4opt)
            {

                double  outi, outj;
                double sqrtOuti = 8.0 * (double)id + 1.0;
                //                outi = int(3 + sqrt(sqrtOuti)) / 2 ;
                outi = (3 + sqrt(sqrtOuti)) / 2 ;
                outi = trunc(outi);
                outj = id - (outi-2)*(outi-1)/2 + 1;


                //                double test = startId + 0.99*maxChecks4optDivide ;
                //                if(startId > 0&& id > test)//350631671)
                //                    printf("largeID %f, local_id %f \n", id, local_id);
                //                else if (id <5)
                //                    printf("SmallID %f, local_id %f, outi %f, outj %f \n", id, local_id, outi, outj);
                //                else if (id <0)
                //                    printf("SmallID < 0 %f, local_id %f , outi %f, outj %f \n", id, local_id, outi, outj);
                //                if(id == (startId + maxChecks4optDivide-1))
                //                    printf("bound id  %f, local_id %f , outi %f, outj %f \n", id, local_id, outi, outj);


                if(outi < maxChecks2opt && outj < maxChecks2opt)
                {

                    double sqrtOutIK = 8.0 * (double )outi + 1.0;
                    int k = int(3 + sqrt(sqrtOutIK)) / 2 ;
                    int p = outi - (k-2)*(k-1)/2 + 1;

                    double sqrtOutJk = 8.0 * (double)outj + 1.0;
                    int j = int(3 + sqrt(sqrtOutJk)) / 2 ;
                    int w = outj - (j-2)*(j-1)/2 + 1;


                    if( k > p && p> j&& j>w&& k< width && p<width && j<width && w<width &&  k > 0 && p > 0 && j > 0 && w > 0 && p+1!=k && w+1!=j && j+1!=p)
                    {

                        if(w > 9144)
                            printf(" maximum 4-opt id = %f, outi= %f, outj=%f, inner k,p,j,w =(%d, %d, %d, %d), \n", id, outi, outj, k, p, j, w);


                        //                        bool existingCandidate = 0;
                        //                        if(nn_source.minRadiusMap[0][w-1] == 1 || nn_source.minRadiusMap[0][j-1] == 1 ||nn_source.minRadiusMap[0][p-1] == 1 ||nn_source.minRadiusMap[0][k-1] == 1)
                        //                            existingCandidate = 1;

                        //                        if(existingCandidate == 0)
                        {

                            //                            float oldLength = dist(w-1, w, arrayTSP[0]) + dist(j-1, j, arrayTSP[0]) + dist(p-1, p, arrayTSP[0])+ dist(k-1, k, arrayTSP[0]);
                            float oldLength = dist(w-1, w, sharedArrayTSP) + dist(j-1, j, sharedArrayTSP) + dist(p-1, p, sharedArrayTSP)+ dist(k-1, k, sharedArrayTSP);

                            float newLength;//25 is fixed for 4-opt
                            int array[8];
                            array[0] = w-1;
                            array[1] = w;
                            array[2] = j-1;
                            array[3] = j;
                            array[4] = p-1;
                            array[5] = p;
                            array[6] = k-1;
                            array[7] = k;

                            int finalSelect = -1;
                            float optimiz = -INFINITY;

                            for(int opt = 0; opt < 200; opt +=8) //  4 edges 8 nodes
                            {
                                //                                int nd1 = nn_source.nodeParentMap[0][opt] -1;
                                //                                int nd2 = nn_source.nodeParentMap[0][opt+1] -1;
                                //                                int nd3 = nn_source.nodeParentMap[0][opt+2] -1;
                                //                                int nd4 = nn_source.nodeParentMap[0][opt+3] -1;
                                //                                int nd5 = nn_source.nodeParentMap[0][opt+4] -1;
                                //                                int nd6 = nn_source.nodeParentMap[0][opt+5] -1;
                                //                                int nd7 = nn_source.nodeParentMap[0][opt+6] -1;
                                //                                int nd8 = nn_source.nodeParentMap[0][opt+7] -1;


                                int nd1 = optPossibilities[opt] -1;
                                int nd2 = optPossibilities[opt+1] -1;
                                int nd3 = optPossibilities[opt+2] -1;
                                int nd4 = optPossibilities[opt+3] -1;
                                int nd5 = optPossibilities[opt+4] -1;
                                int nd6 = optPossibilities[opt+5] -1;
                                int nd7 = optPossibilities[opt+6] -1;
                                int nd8 = optPossibilities[opt+7] -1;


                                int optCandi = opt / 8;
                                //printf("GPU search nd1-8 %d, %d, %d, %d, %d, %d, %d, %d; optCandi=%d \n", nd1, nd2, nd3, nd4, nd5, nd6, nd7, nd8, optCandi);
                                //                                newLength= dist(array[nd1],array[nd2], arrayTSP[0]) + dist(array[nd3],array[nd4], arrayTSP[0]) + dist(array[nd5],array[nd6], arrayTSP[0])+ dist(array[nd7],array[nd8], arrayTSP[0]);
                                newLength= dist(array[nd1],array[nd2], sharedArrayTSP) + dist(array[nd3],array[nd4], sharedArrayTSP) + dist(array[nd5],array[nd6], sharedArrayTSP)+ dist(array[nd7],array[nd8], sharedArrayTSP);

                                float opti = oldLength - newLength;
                                if(opti > 0 && opti > optimiz)
                                {
                                    finalSelect = optCandi;
                                    optimiz = opti;

                                    //                                    atomicExch(&(nn_source.minRadiusMap[0][w-1]), 1);
                                    //                                    atomicExch(&(nn_source.minRadiusMap[0][j-1]), 1);
                                    //                                    atomicExch(&(nn_source.minRadiusMap[0][p-1]), 1);
                                    //                                    atomicExch(&(nn_source.minRadiusMap[0][k-1]), 1);

                                }
                            }


                            if(finalSelect >= 0)
                            {

                                //                                unsigned int node1 = (int)arrayTSP[0][w-1].current;
                                //                                unsigned int node3 = (int)arrayTSP[0][j-1].current;
                                //                                unsigned int node5 = (int)arrayTSP[0][p-1].current;
                                //                                unsigned int node7 = (int)arrayTSP[0][k-1].current;
                                unsigned int node1 = (int)sharedArrayTSP[w-1].current;
                                unsigned int node3 = (int)sharedArrayTSP[j-1].current;
                                unsigned int node5 = (int)sharedArrayTSP[p-1].current;
                                unsigned int node7 = (int)sharedArrayTSP[k-1].current;


                                float localMinChange = nn_source.minRadiusMap[0][node1];

                                if(optimiz > localMinChange)
                                {

                                    unsigned long long result = 0;
                                    result = result | node3;
                                    result = result << 16;
                                    result = result | node5;
                                    result = result << 16;
                                    result = result | node7;

                                    float codekopt = finalSelect; //WB.Q 2024 densityMap only mark the k value of k-opt, and mark which mode of k-exchange
                                    codekopt = finalSelect * 100 + 4;

                                    //  printf("GPU search, node1, node3, node5, node7, %d, %d, %d, %d; order(%d,%d,%d,%d), optvalue %lld, codekopt %f \n",
                                    //           node1, node3, node5, node7, nn_source.grayValueMap[0][node1], nn_source.grayValueMap[0][node3], nn_source.grayValueMap[0][node5] , nn_source.grayValueMap[0][node7], result, codekopt);
                                    atomicExch(&(nn_source.optCandidateMap[0][node1]), result); // WB.Q 2024 find a solution to judge non-interacted 23456-opt
                                    atomicExch(&(nn_source.densityMap[0][node1]), codekopt); // WB.Q 2024 densityMap only mark the k value of k-opt, and mark which mode of k-exchange
                                    atomicExch(&(nn_source.minRadiusMap[0][node1]), optimiz);
                                }
                            }

                        }

                    }//end if k<p
                }

            }
        }

    }
    __syncthreads();
}// end K_4optOneThreadOne4opt




/*!
 * \brief 2024 QWB: add parallel 4-opt
 */
//epecially for small size, copy all cities into shared memory
template <template<typename, typename> class NeuralNetLinks, class BufferDimension, class Point>
KERNEL void K_4opt_oneThreadOne4opt_qiaoIterStride_shared(NeuralNetLinks<BufferDimension, Point> nn_source,
                                                          Grid<doubleLinkedEdgeForTSP> arrayTSP,
                                                          double maxChecks2opt, double maxChecks4opt, double maxChecks4optDivide,
                                                          double iter, double istride)
{

    double local_id = threadIdx.x + blockIdx.x * blockDim.x;// + gridDim * blockDim.x;
    int width =  nn_source.adaptiveMap.width; // each thread has this register



    __shared__ doubleLinkedEdgeForTSP sharedArrayTSP[SHAREDMAXCITIES];
    __shared__ float sharedArrayOccupied[SHAREDMAXCITIES];
    __shared__ float optPossibilities[OPTPOSSIBILITES4OPT];
    float iterShared = (float)width / (float)BLOCKSIZE;
    for(int opt = 0; opt < iterShared; opt++)
    {
        int m = threadIdx.x + opt*BLOCKSIZE;
        if(m < width)
        {
            sharedArrayTSP[m].current = arrayTSP[0][m].current;
            sharedArrayTSP[m].currentCoord[0] = arrayTSP[0][m].currentCoord[0];
            sharedArrayTSP[m].currentCoord[1] = arrayTSP[0][m].currentCoord[1];

        }
        __syncthreads();
    }

    if(threadIdx.x < OPTPOSSIBILITES4OPT)
        optPossibilities[threadIdx.x] = nn_source.nodeParentMap[0][threadIdx.x ];

    __syncthreads();


    if(local_id < maxChecks4opt)
    {
        double startId = maxChecks4optDivide * (istride);

        if(local_id == 0 )
            printf("StartID %f, local_id %f \n", startId, local_id);


        for(double id = local_id*iter ; id < (local_id+1)*(iter); id++)
        {
            id = id + startId;

            if(id > 0 && id < maxChecks4opt)
            {

                double  outi, outj;
                double sqrtOuti = 8.0 * (double)id + 1.0;
                //                outi = int(3 + sqrt(sqrtOuti)) / 2 ;
                outi = (3 + sqrt(sqrtOuti)) / 2 ;
                outi = trunc(outi);
                outj = id - (outi-2)*(outi-1)/2 + 1;


                //                double test = startId + 0.99*maxChecks4optDivide ;
                //                if(startId > 0&& id > test)//350631671)
                //                    printf("largeID %f, local_id %f \n", id, local_id);
                //                else if (id <5)
                //                    printf("SmallID %f, local_id %f, outi %f, outj %f \n", id, local_id, outi, outj);
                //                else if (id <0)
                //                    printf("SmallID < 0 %f, local_id %f , outi %f, outj %f \n", id, local_id, outi, outj);
                //                if(id == (startId + maxChecks4optDivide-1))
                //                    printf("bound id  %f, local_id %f , outi %f, outj %f \n", id, local_id, outi, outj);


                if(outi < maxChecks2opt && outj < maxChecks2opt)
                {

                    double sqrtOutIK = 8.0 * (double )outi + 1.0;
                    int k = int(3 + sqrt(sqrtOutIK)) / 2 ;
                    int p = outi - (k-2)*(k-1)/2 + 1;

                    double sqrtOutJk = 8.0 * (double)outj + 1.0;
                    int j = int(3 + sqrt(sqrtOutJk)) / 2 ;
                    int w = outj - (j-2)*(j-1)/2 + 1;


                    if( k > p && p> j&& j>w&& k< width && p<width && j<width && w<width &&  k > 0 && p > 0 && j > 0 && w > 0 && p+1!=k && w+1!=j && j+1!=p)
                    {

                        if(w > 9144)
                            printf(" maximum 4-opt id = %f, outi= %f, outj=%f, inner k,p,j,w =(%d, %d, %d, %d), \n", id, outi, outj, k, p, j, w);


                        bool existingCandidate = 0;
                        //                        if(nn_source.minRadiusMap[0][w-1] == 1 || nn_source.minRadiusMap[0][j-1] == 1 ||nn_source.minRadiusMap[0][p-1] == 1 ||nn_source.minRadiusMap[0][k-1] == 1)
                        //                            existingCandidate = 1;
                        if(sharedArrayOccupied[w-1] == 1 || sharedArrayOccupied[j-1] == 1 ||sharedArrayOccupied[p-1] == 1 ||sharedArrayOccupied[k-1] == 1)
                            existingCandidate = 1;

                        if(existingCandidate == 0)
                        {

                            float oldLength = dist(w-1, w, sharedArrayTSP) + dist(j-1, j, sharedArrayTSP) + dist(p-1, p, sharedArrayTSP)+ dist(k-1, k, sharedArrayTSP);
                            float newLength;//25 is fixed for 4-opt
                            int array[8];
                            array[0] = w-1;
                            array[1] = w;
                            array[2] = j-1;
                            array[3] = j;
                            array[4] = p-1;
                            array[5] = p;
                            array[6] = k-1;
                            array[7] = k;

                            int finalSelect = -1;
                            float optimiz = -INFINITY;

                            for(int opt = 0; opt < 200; opt +=8) //  4 edges 8 nodes
                            {
                                //                                int nd1 = nn_source.nodeParentMap[0][opt] -1;
                                //                                int nd2 = nn_source.nodeParentMap[0][opt+1] -1;
                                //                                int nd3 = nn_source.nodeParentMap[0][opt+2] -1;
                                //                                int nd4 = nn_source.nodeParentMap[0][opt+3] -1;
                                //                                int nd5 = nn_source.nodeParentMap[0][opt+4] -1;
                                //                                int nd6 = nn_source.nodeParentMap[0][opt+5] -1;
                                //                                int nd7 = nn_source.nodeParentMap[0][opt+6] -1;
                                //                                int nd8 = nn_source.nodeParentMap[0][opt+7] -1;

                                int nd1 = optPossibilities[opt] -1;
                                int nd2 = optPossibilities[opt+1] -1;
                                int nd3 = optPossibilities[opt+2] -1;
                                int nd4 = optPossibilities[opt+3] -1;
                                int nd5 = optPossibilities[opt+4] -1;
                                int nd6 = optPossibilities[opt+5] -1;
                                int nd7 = optPossibilities[opt+6] -1;
                                int nd8 = optPossibilities[opt+7] -1;



                                int optCandi = opt / 8;
                                //printf("GPU search nd1-8 %d, %d, %d, %d, %d, %d, %d, %d; optCandi=%d \n", nd1, nd2, nd3, nd4, nd5, nd6, nd7, nd8, optCandi);
                                newLength= dist(array[nd1],array[nd2], sharedArrayTSP) + dist(array[nd3],array[nd4], sharedArrayTSP) + dist(array[nd5],array[nd6], sharedArrayTSP)+ dist(array[nd7],array[nd8], sharedArrayTSP);

                                float opti = oldLength - newLength;
                                if(opti > 0 && opti > optimiz)
                                {
                                    finalSelect = optCandi;
                                    optimiz = opti;

                                    //                                    atomicExch(&(nn_source.minRadiusMap[0][w-1]), 1);
                                    //                                    atomicExch(&(nn_source.minRadiusMap[0][j-1]), 1);
                                    //                                    atomicExch(&(nn_source.minRadiusMap[0][p-1]), 1);
                                    //                                    atomicExch(&(nn_source.minRadiusMap[0][k-1]), 1);

                                    //                                    atomicExch(&(sharedArrayOccupied[w-1]), 1);
                                    //                                    atomicExch(&(sharedArrayOccupied[j-1]), 1);
                                    //                                    atomicExch(&(sharedArrayOccupied[p-1]), 1);
                                    //                                    atomicExch(&(sharedArrayOccupied[k-1]), 1);

                                    sharedArrayOccupied[w-1]= 1;
                                    sharedArrayOccupied[j-1]= 1;
                                    sharedArrayOccupied[p-1]= 1;
                                    sharedArrayOccupied[k-1]=1;

                                }
                            }


                            if(finalSelect >= 0)
                            {

                                unsigned int node1 = (int)sharedArrayTSP[w-1].current;
                                unsigned int node3 = (int)sharedArrayTSP[j-1].current;
                                unsigned int node5 = (int)sharedArrayTSP[p-1].current;
                                unsigned int node7 = (int)sharedArrayTSP[k-1].current;

                                //                            float localMinChange = nn_source.minRadiusMap[0][node1];

                                //                            if(optimiz > localMinChange)
                                {

                                    unsigned long long result = 0;
                                    result = result | node3;
                                    result = result << 16;
                                    result = result | node5;
                                    result = result << 16;
                                    result = result | node7;

                                    float codekopt = finalSelect; //WB.Q 2024 densityMap only mark the k value of k-opt, and mark which mode of k-exchange
                                    codekopt = finalSelect * 100 + 4;

                                    //  printf("GPU search, node1, node3, node5, node7, %d, %d, %d, %d; order(%d,%d,%d,%d), optvalue %lld, codekopt %f \n",
                                    //           node1, node3, node5, node7, nn_source.grayValueMap[0][node1], nn_source.grayValueMap[0][node3], nn_source.grayValueMap[0][node5] , nn_source.grayValueMap[0][node7], result, codekopt);
                                    atomicExch(&(nn_source.optCandidateMap[0][node1]), result); // WB.Q 2024 find a solution to judge non-interacted 23456-opt
                                    atomicExch(&(nn_source.densityMap[0][node1]), codekopt); // WB.Q 2024 densityMap only mark the k value of k-opt, and mark which mode of k-exchange
                                    //                                atomicExch(&(nn_source.minRadiusMap[0][node1]), optimiz);
                                }
                            }

                        }

                    }//end if k<p
                }

            }
        }

    }
    __syncthreads();
}// end K_4optOneThreadOne4opt



/*!
 * \brief 2024 QWB: add parallel 4-opt
 */
//epecially for small size, copy all cities into shared memory
template <template<typename, typename> class NeuralNetLinks, class BufferDimension, class Point>
KERNEL void K_4opt_oneThreadOne4opt_qiaoIterStride_shared_noOccupy(NeuralNetLinks<BufferDimension, Point> nn_source,
                                                                   Grid<doubleLinkedEdgeForTSP> arrayTSP,
                                                                   double maxChecks2opt, double maxChecks4opt, double maxChecks4optDivide,
                                                                   double iter, double istride)
{

    double local_id = threadIdx.x + blockIdx.x * blockDim.x;// + gridDim * blockDim.x;
    int width =  nn_source.adaptiveMap.width; // each thread has this register



    __shared__ doubleLinkedEdgeForTSP sharedArrayTSP[SHAREDMAXCITIES];
    //    __shared__ float sharedArrayOccupied[SHAREDMAXCITIES];
    __shared__ float optPossibilities[OPTPOSSIBILITES4OPT];
    float iterShared = (float)width / (float)BLOCKSIZE;
    for(int opt = 0; opt < iterShared; opt++)
    {
        int m = threadIdx.x + opt*BLOCKSIZE;
        if(m < width)
        {
            sharedArrayTSP[m].current = arrayTSP[0][m].current;
            sharedArrayTSP[m].currentCoord[0] = arrayTSP[0][m].currentCoord[0];
            sharedArrayTSP[m].currentCoord[1] = arrayTSP[0][m].currentCoord[1];

        }
        __syncthreads();
    }

    if(threadIdx.x < OPTPOSSIBILITES4OPT)
        optPossibilities[threadIdx.x] = nn_source.nodeParentMap[0][threadIdx.x ];

    __syncthreads();


    if(local_id < maxChecks4opt)
    {
        double startId = maxChecks4optDivide * (istride);

        if(local_id == 0 )
            printf("StartID %f, local_id %f \n", startId, local_id);


        for(double id = local_id*iter ; id < (local_id+1)*(iter); id++)
        {
            id = id + startId;

            if(id > 0 && id < maxChecks4opt)
            {

                double  outi, outj;
                double sqrtOuti = 8.0 * (double)id + 1.0;
                //                outi = int(3 + sqrt(sqrtOuti)) / 2 ;
                outi = (3 + sqrt(sqrtOuti)) / 2 ;
                outi = trunc(outi);
                outj = id - (outi-2)*(outi-1)/2 + 1;


                //                double test = startId + 0.99*maxChecks4optDivide ;
                //                if(startId > 0&& id > test)//350631671)
                //                    printf("largeID %f, local_id %f \n", id, local_id);
                //                else if (id <5)
                //                    printf("SmallID %f, local_id %f, outi %f, outj %f \n", id, local_id, outi, outj);
                //                else if (id <0)
                //                    printf("SmallID < 0 %f, local_id %f , outi %f, outj %f \n", id, local_id, outi, outj);
                //                if(id == (startId + maxChecks4optDivide-1))
                //                    printf("bound id  %f, local_id %f , outi %f, outj %f \n", id, local_id, outi, outj);


                if(outi < maxChecks2opt && outj < maxChecks2opt)
                {

                    double sqrtOutIK = 8.0 * (double )outi + 1.0;
                    int k = int(3 + sqrt(sqrtOutIK)) / 2 ;
                    int p = outi - (k-2)*(k-1)/2 + 1;

                    double sqrtOutJk = 8.0 * (double)outj + 1.0;
                    int j = int(3 + sqrt(sqrtOutJk)) / 2 ;
                    int w = outj - (j-2)*(j-1)/2 + 1;


                    if( k > p && p> j&& j>w&& k< width && p<width && j<width && w<width &&  k > 0 && p > 0 && j > 0 && w > 0 && p+1!=k && w+1!=j && j+1!=p)
                    {

                        if(w > 9144)
                            printf(" maximum 4-opt id = %f, outi= %f, outj=%f, inner k,p,j,w =(%d, %d, %d, %d), \n", id, outi, outj, k, p, j, w);


                        bool existingCandidate = 0;
                        if(nn_source.minRadiusMap[0][w-1] == 1 || nn_source.minRadiusMap[0][j-1] == 1 ||nn_source.minRadiusMap[0][p-1] == 1 ||nn_source.minRadiusMap[0][k-1] == 1)
                            existingCandidate = 1;
                        //                        if(sharedArrayOccupied[w-1] == 1 || sharedArrayOccupied[j-1] == 1 ||sharedArrayOccupied[p-1] == 1 ||sharedArrayOccupied[k-1] == 1)
                        //                            existingCandidate = 1;

                        if(existingCandidate == 0)
                        {

                            float oldLength = dist(w-1, w, sharedArrayTSP) + dist(j-1, j, sharedArrayTSP) + dist(p-1, p, sharedArrayTSP)+ dist(k-1, k, sharedArrayTSP);
                            float newLength;//25 is fixed for 4-opt
                            int array[8];
                            array[0] = w-1;
                            array[1] = w;
                            array[2] = j-1;
                            array[3] = j;
                            array[4] = p-1;
                            array[5] = p;
                            array[6] = k-1;
                            array[7] = k;

                            int finalSelect = -1;
                            float optimiz = -INFINITY;

                            for(int opt = 0; opt < 200; opt +=8) //  4 edges 8 nodes
                            {
                                //                                int nd1 = nn_source.nodeParentMap[0][opt] -1;
                                //                                int nd2 = nn_source.nodeParentMap[0][opt+1] -1;
                                //                                int nd3 = nn_source.nodeParentMap[0][opt+2] -1;
                                //                                int nd4 = nn_source.nodeParentMap[0][opt+3] -1;
                                //                                int nd5 = nn_source.nodeParentMap[0][opt+4] -1;
                                //                                int nd6 = nn_source.nodeParentMap[0][opt+5] -1;
                                //                                int nd7 = nn_source.nodeParentMap[0][opt+6] -1;
                                //                                int nd8 = nn_source.nodeParentMap[0][opt+7] -1;

                                int nd1 = optPossibilities[opt] -1;
                                int nd2 = optPossibilities[opt+1] -1;
                                int nd3 = optPossibilities[opt+2] -1;
                                int nd4 = optPossibilities[opt+3] -1;
                                int nd5 = optPossibilities[opt+4] -1;
                                int nd6 = optPossibilities[opt+5] -1;
                                int nd7 = optPossibilities[opt+6] -1;
                                int nd8 = optPossibilities[opt+7] -1;



                                int optCandi = opt / 8;
                                //printf("GPU search nd1-8 %d, %d, %d, %d, %d, %d, %d, %d; optCandi=%d \n", nd1, nd2, nd3, nd4, nd5, nd6, nd7, nd8, optCandi);
                                newLength= dist(array[nd1],array[nd2], sharedArrayTSP) + dist(array[nd3],array[nd4], sharedArrayTSP) + dist(array[nd5],array[nd6], sharedArrayTSP)+ dist(array[nd7],array[nd8], sharedArrayTSP);

                                float opti = oldLength - newLength;
                                if(opti > 0 && opti > optimiz)
                                {
                                    finalSelect = optCandi;
                                    optimiz = opti;

                                    atomicExch(&(nn_source.minRadiusMap[0][w-1]), 1);
                                    atomicExch(&(nn_source.minRadiusMap[0][j-1]), 1);
                                    atomicExch(&(nn_source.minRadiusMap[0][p-1]), 1);
                                    atomicExch(&(nn_source.minRadiusMap[0][k-1]), 1);

                                    //                                    atomicExch(&(sharedArrayOccupied[w-1]), 1);
                                    //                                    atomicExch(&(sharedArrayOccupied[j-1]), 1);
                                    //                                    atomicExch(&(sharedArrayOccupied[p-1]), 1);
                                    //                                    atomicExch(&(sharedArrayOccupied[k-1]), 1);

                                    //                                    sharedArrayOccupied[w-1]= 1;
                                    //                                    sharedArrayOccupied[j-1]= 1;
                                    //                                    sharedArrayOccupied[p-1]= 1;
                                    //                                    sharedArrayOccupied[k-1]=1;

                                }
                            }


                            if(finalSelect >= 0)
                            {

                                unsigned int node1 = (int)sharedArrayTSP[w-1].current;
                                unsigned int node3 = (int)sharedArrayTSP[j-1].current;
                                unsigned int node5 = (int)sharedArrayTSP[p-1].current;
                                unsigned int node7 = (int)sharedArrayTSP[k-1].current;

                                //                            float localMinChange = nn_source.minRadiusMap[0][node1];

                                //                            if(optimiz > localMinChange)
                                {

                                    unsigned long long result = 0;
                                    result = result | node3;
                                    result = result << 16;
                                    result = result | node5;
                                    result = result << 16;
                                    result = result | node7;

                                    float codekopt = finalSelect; //WB.Q 2024 densityMap only mark the k value of k-opt, and mark which mode of k-exchange
                                    codekopt = finalSelect * 100 + 4;

                                    //  printf("GPU search, node1, node3, node5, node7, %d, %d, %d, %d; order(%d,%d,%d,%d), optvalue %lld, codekopt %f \n",
                                    //           node1, node3, node5, node7, nn_source.grayValueMap[0][node1], nn_source.grayValueMap[0][node3], nn_source.grayValueMap[0][node5] , nn_source.grayValueMap[0][node7], result, codekopt);
                                    atomicExch(&(nn_source.optCandidateMap[0][node1]), result); // WB.Q 2024 find a solution to judge non-interacted 23456-opt
                                    atomicExch(&(nn_source.densityMap[0][node1]), codekopt); // WB.Q 2024 densityMap only mark the k value of k-opt, and mark which mode of k-exchange
                                    //                                atomicExch(&(nn_source.minRadiusMap[0][node1]), optimiz);
                                }
                            }

                        }

                    }//end if k<p
                }

            }
        }

    }
    __syncthreads();
}// end K_4optOneThreadOne4opt






/*!
 * \brief 191116 QWB: add parallel 2-opt with rocki's method
 */
//epecially for small size, copy all cities into shared memory
template <template<typename, typename> class NeuralNetLinks, class BufferDimension, class Point>
KERNEL void K_5opt_oneThreadOne5opt_rockiSmall(NeuralNetLinks<BufferDimension, Point> nn_source,
                                               Grid<doubleLinkedEdgeForTSP> arrayTSP,
                                               int idRow5th, double maxChecks4opt, double maxChecks2opt,
                                               unsigned int iter)
{

    double id = threadIdx.x + blockIdx.x * blockDim.x;
    int width =  nn_source.adaptiveMap.width; // each thread has this register

    if(id < maxChecks4opt)
    {

        double outi, outj;

        //WB.Q this way will produce i = j
        outi = int(3 + sqrt(8.0f * (double)id + 1.0f)) / 2 ;
        outj = id - (outi-2)*(outi-1)/2 + 1;

        if(outi < maxChecks2opt && outj < maxChecks2opt)
        {
            int k = int(3 + sqrt(8.0f * (double)outi + 1.0f)) / 2 ;
            int p = outi - (k-2)*(k-1)/2 + 1;

            int j = int(3 + sqrt(8.0f * (double)outj + 1.0f)) / 2 ;
            int w = outj - (j-2)*(j-1)/2 + 1;

            if(id == maxChecks4opt-2)
                printf("maximum 50opt id = %d, outi= %d, outj=%d, inner row, k,p,j,w =(%d, %d, %d, %d, %d), \n", id, outi, outj, idRow5th, k, p, j, w);

            if( idRow5th > k+1 && k > p && p> j&& j>w&& k< width && p<width && j<width && w<width &&  k > 0 && p > 0 && j > 0 && w > 0 && p+1!=k && w+1!=j && j+1!=p)
            {

                //                        if(idRow5th > width -2 && k > width - 2)
                //                            printf(" maximum 4-opt id = %d, outi= %d, outj=%d, inner k,p,j,w =(%d, %d, %d, %d, %d), \n", id, outi, outj, k, p, j, w);

                bool existingCandidate = 0;
                if(nn_source.minRadiusMap[0][w-1] == 1 || nn_source.minRadiusMap[0][j-1] == 1 ||nn_source.minRadiusMap[0][p-1] == 1
                        ||nn_source.minRadiusMap[0][k-1] == 1 ||nn_source.minRadiusMap[0][idRow5th-1] == 1)
                    existingCandidate = 1;

                if(existingCandidate == 0)
                {

                    float oldLength = dist(w-1, w, arrayTSP[0]) + dist(j-1, j, arrayTSP[0]) + dist(p-1, p, arrayTSP[0])+ dist(k-1, k, arrayTSP[0]) + dist(idRow5th-1, idRow5th, arrayTSP[0]);

                    float newLength;
                    int array[10];
                    array[0] = w-1;
                    array[1] = w;
                    array[2] = j-1;
                    array[3] = j;
                    array[4] = p-1;
                    array[5] = p;
                    array[6] = k-1;
                    array[7] = k;
                    array[8] = idRow5th-1;
                    array[9] = idRow5th;

                    int finalSelect = -1;
                    float optimiz = -INFINITY;

                    for(int opt = 0; opt < 2080; opt +=10) //  4 edges 8 nodes
                    {
                        int nd1 = nn_source.nodeParentMap[0][opt] -1;
                        int nd2 = nn_source.nodeParentMap[0][opt+1] -1;
                        int nd3 = nn_source.nodeParentMap[0][opt+2] -1;
                        int nd4 = nn_source.nodeParentMap[0][opt+3] -1;
                        int nd5 = nn_source.nodeParentMap[0][opt+4] -1;
                        int nd6 = nn_source.nodeParentMap[0][opt+5] -1;
                        int nd7 = nn_source.nodeParentMap[0][opt+6] -1;
                        int nd8 = nn_source.nodeParentMap[0][opt+7] -1;
                        int nd9 = nn_source.nodeParentMap[0][opt+8] -1;
                        int nd10 = nn_source.nodeParentMap[0][opt+9] -1;


                        int optCandi = opt / 10;
                        // printf("GPU search nd1-8 %d, %d, %d, %d, %d, %d, %d, %d; optCandi=%d \n", nd1, nd2, nd3, nd4, nd5, nd6, nd7, nd8, optCandi);
                        newLength = dist(array[nd1],array[nd2], arrayTSP[0]) + dist(array[nd3],array[nd4], arrayTSP[0])
                                + dist(array[nd5],array[nd6], arrayTSP[0])+ dist(array[nd7],array[nd8], arrayTSP[0]) + dist(array[nd9],array[nd10], arrayTSP[0]);

                        float opti = oldLength - newLength;
                        if(opti > 0 && opti > optimiz)
                        {
                            finalSelect = optCandi;
                            optimiz = opti;

                            atomicExch(&(nn_source.minRadiusMap[0][w-1]), 1);
                            atomicExch(&(nn_source.minRadiusMap[0][j-1]), 1);
                            atomicExch(&(nn_source.minRadiusMap[0][p-1]), 1);
                            atomicExch(&(nn_source.minRadiusMap[0][k-1]), 1);
                            atomicExch(&(nn_source.minRadiusMap[0][idRow5th-1]), 1);

                        }
                    }

                    if(finalSelect >= 0)
                    {

                        unsigned int node1 = (int)arrayTSP[0][w-1].current;
                        unsigned int node3 = (int)arrayTSP[0][j-1].current;
                        unsigned int node5 = (int)arrayTSP[0][p-1].current;
                        unsigned int node7 = (int)arrayTSP[0][k-1].current;
                        unsigned int node9 = (int)arrayTSP[0][idRow5th-1].current;

                        //                            float localMinChange = nn_source.minRadiusMap[0][node1];

                        //                            if(optimiz > localMinChange)
                        {

                            unsigned long long result = 0;
                            result = result | node3;
                            result = result << 16;
                            result = result | node5;
                            result = result << 16;
                            result = result | node7;
                            result = result << 16;
                            result = result | node9;

                            float codekopt = finalSelect; //WB.Q 2024 densityMap only mark the k value of k-opt, and mark which mode of k-exchange
                            codekopt = finalSelect * 100 + 5;

                            //                                printf("GPU search, node1, node3, node5, node7, %d, %d, %d, %d; order(%d,%d,%d,%d), optvalue %lld, codekopt %f \n",
                            //                                       node1, node3, node5, node7, nn_source.grayValueMap[0][node1], nn_source.grayValueMap[0][node3], nn_source.grayValueMap[0][node5] , nn_source.grayValueMap[0][node7], result, codekopt);
                            atomicExch(&(nn_source.optCandidateMap[0][node1]), result); // WB.Q 2024 find a solution to judge non-interacted 23456-opt
                            atomicExch(&(nn_source.densityMap[0][node1]), codekopt); // WB.Q 2024 densityMap only mark the k value of k-opt, and mark which mode of k-exchange
                            //                              atomicExch(&(nn_source.minRadiusMap[0][node1]), optimiz);
                        }
                    }
                }

            }
        }



    }
    __syncthreads();
}// end K_2optOneThreadOne2opt




/*!
 * \brief 191116 QWB: add parallel 2-opt with rocki's method
 */
//epecially for small size, copy all cities into shared memory
template <template<typename, typename> class NeuralNetLinks, class BufferDimension, class Point>
KERNEL void K_5opt_oneThreadOne5opt_rockiSmall_iter(NeuralNetLinks<BufferDimension, Point> nn_source,
                                                    Grid<doubleLinkedEdgeForTSP> arrayTSP,
                                                    int idRow5th, double maxChecks4opt,
                                                    double maxChecks2opt,
                                                    unsigned int iter)
{

    double local_id = threadIdx.x + blockIdx.x * blockDim.x;
    int width =  nn_source.adaptiveMap.width; // each thread has this register

    if(local_id < maxChecks4opt)
    {

        int packSize = blockDim.x * gridDim.x;

        for(int nu = 0; nu <= iter; nu++)
        {

            double id = local_id + nu * packSize;

            if(id < maxChecks4opt)
            {

                double outi, outj;

                //WB.Q this way will produce i = j
                outi = int(3 + sqrt(8.0f * (double)id + 1.0f)) / 2 ;
                outj = id - (outi-2)*(outi-1)/2 + 1;

                if(outi < maxChecks2opt && outj < maxChecks2opt)
                {
                    int k = int(3 + sqrt(8.0f * (double)outi + 1.0f)) / 2 ;
                    int p = outi - (k-2)*(k-1)/2 + 1;

                    int j = int(3 + sqrt(8.0f * (double)outj + 1.0f)) / 2 ;
                    int w = outj - (j-2)*(j-1)/2 + 1;

                    if(id == maxChecks4opt-2)
                        printf("maximum 50opt id = %d, outi= %d, outj=%d, inner row, k,p,j,w =(%d, %d, %d, %d, %d), \n", id, outi, outj, idRow5th, k, p, j, w);

                    if( idRow5th > k+1 && k > p && p> j&& j>w&& k< width && p<width && j<width && w<width &&  k > 0 && p > 0 && j > 0 && w > 0 && p+1!=k && w+1!=j && j+1!=p)
                    {

                        //                        if(idRow5th > width -2 && k > width - 2)
                        //                            printf(" maximum 4-opt id = %d, outi= %d, outj=%d, inner k,p,j,w =(%d, %d, %d, %d, %d), \n", id, outi, outj, k, p, j, w);

                        bool existingCandidate = 0;
                        if(nn_source.minRadiusMap[0][w-1] == 1 || nn_source.minRadiusMap[0][j-1] == 1 ||nn_source.minRadiusMap[0][p-1] == 1
                                ||nn_source.minRadiusMap[0][k-1] == 1 ||nn_source.minRadiusMap[0][idRow5th-1] == 1)
                            existingCandidate = 1;

                        if(existingCandidate == 0)
                        {

                            float oldLength = dist(w-1, w, arrayTSP[0]) + dist(j-1, j, arrayTSP[0]) + dist(p-1, p, arrayTSP[0])+ dist(k-1, k, arrayTSP[0]) + dist(idRow5th-1, idRow5th, arrayTSP[0]);

                            float newLength;
                            int array[10];
                            array[0] = w-1;
                            array[1] = w;
                            array[2] = j-1;
                            array[3] = j;
                            array[4] = p-1;
                            array[5] = p;
                            array[6] = k-1;
                            array[7] = k;
                            array[8] = idRow5th-1;
                            array[9] = idRow5th;

                            int finalSelect = -1;
                            float optimiz = -INFINITY;

                            for(int opt = 0; opt < 2080; opt +=10) //  4 edges 8 nodes
                            {
                                int nd1 = nn_source.nodeParentMap[0][opt] -1;
                                int nd2 = nn_source.nodeParentMap[0][opt+1] -1;
                                int nd3 = nn_source.nodeParentMap[0][opt+2] -1;
                                int nd4 = nn_source.nodeParentMap[0][opt+3] -1;
                                int nd5 = nn_source.nodeParentMap[0][opt+4] -1;
                                int nd6 = nn_source.nodeParentMap[0][opt+5] -1;
                                int nd7 = nn_source.nodeParentMap[0][opt+6] -1;
                                int nd8 = nn_source.nodeParentMap[0][opt+7] -1;
                                int nd9 = nn_source.nodeParentMap[0][opt+8] -1;
                                int nd10 = nn_source.nodeParentMap[0][opt+9] -1;


                                int optCandi = opt / 10;
                                // printf("GPU search nd1-8 %d, %d, %d, %d, %d, %d, %d, %d; optCandi=%d \n", nd1, nd2, nd3, nd4, nd5, nd6, nd7, nd8, optCandi);
                                newLength = dist(array[nd1],array[nd2], arrayTSP[0]) + dist(array[nd3],array[nd4], arrayTSP[0])
                                        + dist(array[nd5],array[nd6], arrayTSP[0])+ dist(array[nd7],array[nd8], arrayTSP[0]) + dist(array[nd9],array[nd10], arrayTSP[0]);

                                float opti = oldLength - newLength;
                                if(opti > 0 && opti > optimiz)
                                {
                                    finalSelect = optCandi;
                                    optimiz = opti;

                                    atomicExch(&(nn_source.minRadiusMap[0][w-1]), 1);
                                    atomicExch(&(nn_source.minRadiusMap[0][j-1]), 1);
                                    atomicExch(&(nn_source.minRadiusMap[0][p-1]), 1);
                                    atomicExch(&(nn_source.minRadiusMap[0][k-1]), 1);
                                    atomicExch(&(nn_source.minRadiusMap[0][idRow5th-1]), 1);

                                }
                            }

                            if(finalSelect >= 0)
                            {

                                unsigned int node1 = (int)arrayTSP[0][w-1].current;
                                unsigned int node3 = (int)arrayTSP[0][j-1].current;
                                unsigned int node5 = (int)arrayTSP[0][p-1].current;
                                unsigned int node7 = (int)arrayTSP[0][k-1].current;
                                unsigned int node9 = (int)arrayTSP[0][idRow5th-1].current;

                                //                            float localMinChange = nn_source.minRadiusMap[0][node1];

                                //                            if(optimiz > localMinChange)
                                {

                                    unsigned long long result = 0;
                                    result = result | node3;
                                    result = result << 16;
                                    result = result | node5;
                                    result = result << 16;
                                    result = result | node7;
                                    result = result << 16;
                                    result = result | node9;

                                    float codekopt = finalSelect; //WB.Q 2024 densityMap only mark the k value of k-opt, and mark which mode of k-exchange
                                    codekopt = finalSelect * 100 + 5;

                                    //                                printf("GPU search, node1, node3, node5, node7, %d, %d, %d, %d; order(%d,%d,%d,%d), optvalue %lld, codekopt %f \n",
                                    //                                       node1, node3, node5, node7, nn_source.grayValueMap[0][node1], nn_source.grayValueMap[0][node3], nn_source.grayValueMap[0][node5] , nn_source.grayValueMap[0][node7], result, codekopt);
                                    atomicExch(&(nn_source.optCandidateMap[0][node1]), result); // WB.Q 2024 find a solution to judge non-interacted 23456-opt
                                    atomicExch(&(nn_source.densityMap[0][node1]), codekopt); // WB.Q 2024 densityMap only mark the k value of k-opt, and mark which mode of k-exchange
                                    //                              atomicExch(&(nn_source.minRadiusMap[0][node1]), optimiz);
                                }
                            }
                        }

                    }
                }


            }
        }
    }
    __syncthreads();
}// end K_2optOneThreadOne2opt



/*!
 * \brief 2409 QWB: add parallel 5opt with rocki's method
 */
//epecially for small size, copy all cities into shared memory
template <template<typename, typename> class NeuralNetLinks, class BufferDimension, class Point>
KERNEL void K_5opt_oneThreadOne5opt_qiao_stride_iter(NeuralNetLinks<BufferDimension, Point> nn_source,
                                                     Grid<doubleLinkedEdgeForTSP> arrayTSP,
                                                     int idRow5th, double maxChecks2opt, double maxChecks4opt, double maxChecks4optDivide,
                                                     double iter, double istride)
{

    double local_id = threadIdx.x + blockIdx.x * blockDim.x;
    int width =  nn_source.adaptiveMap.width; // each thread has this register

    if(local_id < maxChecks4opt)
    {

        double startId = maxChecks4optDivide * (istride);

        if(local_id == 0)
            printf("StartID %f, local_id %f \n", startId, local_id);


        for(double id = local_id*iter ; id < (local_id+1)*(iter); id++)
        {

            id = id + startId;

            if(id > 0 && id < maxChecks4opt)
            {

                double outi, outj;
                double sqrtOuti = 8.0 * (double)id + 1.0;
                outi = int(3 + sqrt(sqrtOuti)) / 2 ;
                outj = id - (outi-2)*(outi-1)/2 + 1;

                if(outi < maxChecks2opt && outj < maxChecks2opt)
                {
                    double sqrtOutIK = 8.0 * (double )outi + 1.0;
                    int k = int(3 + sqrt(sqrtOutIK)) / 2 ;
                    int p = outi - (k-2)*(k-1)/2 + 1;

                    double sqrtOutJk = 8.0 * (double)outj + 1.0;
                    int j = int(3 + sqrt(sqrtOutJk)) / 2 ;
                    int w = outj - (j-2)*(j-1)/2 + 1;

                    if(id > maxChecks4opt-2)
                        printf("maximum 50opt id = %d, outi= %d, outj=%d, inner row, k,p,j,w =(%d, %d, %d, %d, %d), \n", id, outi, outj, idRow5th, k, p, j, w);

                    if( idRow5th > k+1 && k > p && p> j&& j>w&& k< width && p<width && j<width && w<width &&  k > 0 && p > 0 && j > 0 && w > 0 && p+1!=k && w+1!=j && j+1!=p)
                    {

                        //                        if(w > 9144)
                        //                        printf(" maximum 5-opt id = %f, outi= %f, outj=%f, inner k,p,j,w =(%d, %d, %d, %d, %d), \n", id, outi, outj, idRow5th, k, p, j, w);


                        bool existingCandidate = 0;
                        if(nn_source.minRadiusMap[0][w-1] == 1 || nn_source.minRadiusMap[0][j-1] == 1 ||nn_source.minRadiusMap[0][p-1] == 1
                                ||nn_source.minRadiusMap[0][k-1] == 1 ||nn_source.minRadiusMap[0][idRow5th-1] == 1)
                            existingCandidate = 1;

                        if(existingCandidate == 0)
                        {

                            float oldLength = dist(w-1, w, arrayTSP[0]) + dist(j-1, j, arrayTSP[0]) + dist(p-1, p, arrayTSP[0])+ dist(k-1, k, arrayTSP[0]) + dist(idRow5th-1, idRow5th, arrayTSP[0]);

                            float newLength;
                            int array[10];
                            array[0] = w-1;
                            array[1] = w;
                            array[2] = j-1;
                            array[3] = j;
                            array[4] = p-1;
                            array[5] = p;
                            array[6] = k-1;
                            array[7] = k;
                            array[8] = idRow5th-1;
                            array[9] = idRow5th;

                            int finalSelect = -1;
                            float optimiz = -INFINITY;

                            for(int opt = 0; opt < 2080; opt +=10) //  4 edges 8 nodes
                            {
                                int nd1 = nn_source.nodeParentMap[0][opt] -1;
                                int nd2 = nn_source.nodeParentMap[0][opt+1] -1;
                                int nd3 = nn_source.nodeParentMap[0][opt+2] -1;
                                int nd4 = nn_source.nodeParentMap[0][opt+3] -1;
                                int nd5 = nn_source.nodeParentMap[0][opt+4] -1;
                                int nd6 = nn_source.nodeParentMap[0][opt+5] -1;
                                int nd7 = nn_source.nodeParentMap[0][opt+6] -1;
                                int nd8 = nn_source.nodeParentMap[0][opt+7] -1;
                                int nd9 = nn_source.nodeParentMap[0][opt+8] -1;
                                int nd10 = nn_source.nodeParentMap[0][opt+9] -1;


                                int optCandi = opt / 10;
                                // printf("GPU search nd1-8 %d, %d, %d, %d, %d, %d, %d, %d; optCandi=%d \n", nd1, nd2, nd3, nd4, nd5, nd6, nd7, nd8, optCandi);
                                newLength = dist(array[nd1],array[nd2], arrayTSP[0]) + dist(array[nd3],array[nd4], arrayTSP[0])
                                        + dist(array[nd5],array[nd6], arrayTSP[0])+ dist(array[nd7],array[nd8], arrayTSP[0]) + dist(array[nd9],array[nd10], arrayTSP[0]);

                                float opti = oldLength - newLength;
                                if(opti > 0 && opti > optimiz)
                                {
                                    finalSelect = optCandi;
                                    optimiz = opti;

                                    atomicExch(&(nn_source.minRadiusMap[0][w-1]), 1);
                                    atomicExch(&(nn_source.minRadiusMap[0][j-1]), 1);
                                    atomicExch(&(nn_source.minRadiusMap[0][p-1]), 1);
                                    atomicExch(&(nn_source.minRadiusMap[0][k-1]), 1);
                                    atomicExch(&(nn_source.minRadiusMap[0][idRow5th-1]), 1);

                                }
                            }

                            if(finalSelect >= 0)
                            {

                                unsigned int node1 = (int)arrayTSP[0][w-1].current;
                                unsigned int node3 = (int)arrayTSP[0][j-1].current;
                                unsigned int node5 = (int)arrayTSP[0][p-1].current;
                                unsigned int node7 = (int)arrayTSP[0][k-1].current;
                                unsigned int node9 = (int)arrayTSP[0][idRow5th-1].current;

                                //                            float localMinChange = nn_source.minRadiusMap[0][node1];

                                //                            if(optimiz > localMinChange)
                                {

                                    unsigned long long result = 0;
                                    result = result | node3;
                                    result = result << 16;
                                    result = result | node5;
                                    result = result << 16;
                                    result = result | node7;
                                    result = result << 16;
                                    result = result | node9;

                                    float codekopt = finalSelect; //WB.Q 2024 densityMap only mark the k value of k-opt, and mark which mode of k-exchange
                                    codekopt = finalSelect * 100 + 5;

                                    //                                printf("GPU search, node1, node3, node5, node7, %d, %d, %d, %d; order(%d,%d,%d,%d), optvalue %lld, codekopt %f \n",
                                    //                                       node1, node3, node5, node7, nn_source.grayValueMap[0][node1], nn_source.grayValueMap[0][node3], nn_source.grayValueMap[0][node5] , nn_source.grayValueMap[0][node7], result, codekopt);
                                    atomicExch(&(nn_source.optCandidateMap[0][node1]), result); // WB.Q 2024 find a solution to judge non-interacted 23456-opt
                                    atomicExch(&(nn_source.densityMap[0][node1]), codekopt); // WB.Q 2024 densityMap only mark the k value of k-opt, and mark which mode of k-exchange
                                    //                              atomicExch(&(nn_source.minRadiusMap[0][node1]), optimiz);
                                }
                            }
                        }

                    }
                }


            }
        }
    }
    __syncthreads();
}// end K_5optOneThreadOne5opt



/*!
 * \brief 2409 QWB: add parallel 5opt with rocki's method
 */
//epecially for small size, copy all cities into shared memory
template <template<typename, typename> class NeuralNetLinks, class BufferDimension, class Point>
KERNEL void K_5opt_oneThreadOne5opt_qiao_stride_iter_shared(NeuralNetLinks<BufferDimension, Point> nn_source,
                                                            Grid<doubleLinkedEdgeForTSP> arrayTSP,
                                                            int idRow5th, double maxChecks2opt, double maxChecks4opt, double maxChecks4optDivide,
                                                            double iter, double istride)
{

    double local_id = threadIdx.x + blockIdx.x * blockDim.x;
    int width =  nn_source.adaptiveMap.width; // each thread has this register


    __shared__ doubleLinkedEdgeForTSP sharedArrayTSP[SHAREDMAXCITIES];
    __shared__ float sharedArrayOccupied[SHAREDMAXCITIES];
    __shared__ float optPossibilities[OPTPOSSIBILITES5OPT];
    float iterShared = (float)width / (float)BLOCKSIZE;
    for(int opt = 0; opt < iterShared; opt++)
    {
        int m = threadIdx.x + opt*BLOCKSIZE;
        if(m < width)
        {
            sharedArrayTSP[m].current = arrayTSP[0][m].current;
            sharedArrayTSP[m].currentCoord[0] = arrayTSP[0][m].currentCoord[0];
            sharedArrayTSP[m].currentCoord[1] = arrayTSP[0][m].currentCoord[1];

        }
        __syncthreads();
    }

    float iterSharedPossble =  (float)OPTPOSSIBILITES5OPT / (float)BLOCKSIZE;
    for(int opt = 0; opt < iterSharedPossble; opt++)
    {
        int m = threadIdx.x + opt*BLOCKSIZE;

        if(m < OPTPOSSIBILITES5OPT)
            optPossibilities[m] = nn_source.nodeParentMap[0][m];

        __syncthreads();

    }


    if(local_id < maxChecks4opt)
    {

        double startId = maxChecks4optDivide * (istride);

        if(local_id == 0)
            printf("StartID %f, local_id %f \n", startId, local_id);


        for(double id = local_id*iter ; id < (local_id+1)*(iter); id++)
        {

            id = id + startId;

            if(id > 0 && id < maxChecks4opt)
            {

                double outi, outj;
                double sqrtOuti = 8.0 * (double)id + 1.0;
                outi = int(3 + sqrt(sqrtOuti)) / 2 ;
                outj = id - (outi-2)*(outi-1)/2 + 1;

                if(outi < maxChecks2opt && outj < maxChecks2opt)
                {
                    double sqrtOutIK = 8.0 * (double )outi + 1.0;
                    int k = int(3 + sqrt(sqrtOutIK)) / 2 ;
                    int p = outi - (k-2)*(k-1)/2 + 1;

                    double sqrtOutJk = 8.0 * (double)outj + 1.0;
                    int j = int(3 + sqrt(sqrtOutJk)) / 2 ;
                    int w = outj - (j-2)*(j-1)/2 + 1;

                    if(id > maxChecks4opt-2)
                        printf("maximum 50opt id = %d, outi= %d, outj=%d, inner row, k,p,j,w =(%d, %d, %d, %d, %d), \n", id, outi, outj, idRow5th, k, p, j, w);

                    if( idRow5th > k+1 && k > p && p> j&& j>w&& k< width && p<width && j<width && w<width &&  k > 0 && p > 0 && j > 0 && w > 0 && p+1!=k && w+1!=j && j+1!=p)
                    {

                        //                        if(w > 9144)
                        //                        printf(" maximum 5-opt id = %f, outi= %f, outj=%f, inner k,p,j,w =(%d, %d, %d, %d, %d), \n", id, outi, outj, idRow5th, k, p, j, w);


                        bool existingCandidate = 0;

                        //                        if(nn_source.minRadiusMap[0][w-1] == 1 || nn_source.minRadiusMap[0][j-1] == 1 ||nn_source.minRadiusMap[0][p-1] == 1
                        //                                ||nn_source.minRadiusMap[0][k-1] == 1 ||nn_source.minRadiusMap[0][idRow5th-1] == 1)
                        //                            existingCandidate = 1;

                        if(sharedArrayOccupied[w-1] == 1 || sharedArrayOccupied[j-1] == 1 ||sharedArrayOccupied[p-1] == 1
                                ||sharedArrayOccupied[k-1] == 1 ||sharedArrayOccupied[idRow5th-1] == 1)
                            existingCandidate = 1;

                        if(existingCandidate == 0)
                        {

                            float oldLength = dist(w-1, w, sharedArrayTSP) + dist(j-1, j, sharedArrayTSP) + dist(p-1, p, sharedArrayTSP)+ dist(k-1, k, sharedArrayTSP) + dist(idRow5th-1, idRow5th, sharedArrayTSP);

                            float newLength;
                            int array[10];
                            array[0] = w-1;
                            array[1] = w;
                            array[2] = j-1;
                            array[3] = j;
                            array[4] = p-1;
                            array[5] = p;
                            array[6] = k-1;
                            array[7] = k;
                            array[8] = idRow5th-1;
                            array[9] = idRow5th;

                            int finalSelect = -1;
                            float optimiz = -INFINITY;

                            for(int opt = 0; opt < 2080; opt +=10) //  4 edges 8 nodes
                            {
                                //                                int nd1 = nn_source.nodeParentMap[0][opt] -1;
                                //                                int nd2 = nn_source.nodeParentMap[0][opt+1] -1;
                                //                                int nd3 = nn_source.nodeParentMap[0][opt+2] -1;
                                //                                int nd4 = nn_source.nodeParentMap[0][opt+3] -1;
                                //                                int nd5 = nn_source.nodeParentMap[0][opt+4] -1;
                                //                                int nd6 = nn_source.nodeParentMap[0][opt+5] -1;
                                //                                int nd7 = nn_source.nodeParentMap[0][opt+6] -1;
                                //                                int nd8 = nn_source.nodeParentMap[0][opt+7] -1;
                                //                                int nd9 = nn_source.nodeParentMap[0][opt+8] -1;
                                //                                int nd10 = nn_source.nodeParentMap[0][opt+9] -1;

                                int nd1 = optPossibilities[opt] -1;
                                int nd2 = optPossibilities[opt+1] -1;
                                int nd3 = optPossibilities[opt+2] -1;
                                int nd4 = optPossibilities[opt+3] -1;
                                int nd5 = optPossibilities[opt+4] -1;
                                int nd6 = optPossibilities[opt+5] -1;
                                int nd7 = optPossibilities[opt+6] -1;
                                int nd8 = optPossibilities[opt+7] -1;
                                int nd9 = optPossibilities[opt+8] -1;
                                int nd10 = optPossibilities[opt+9] -1;


                                int optCandi = opt / 10;
                                // printf("GPU search nd1-8 %d, %d, %d, %d, %d, %d, %d, %d; optCandi=%d \n", nd1, nd2, nd3, nd4, nd5, nd6, nd7, nd8, optCandi);
                                newLength = dist(array[nd1],array[nd2], sharedArrayTSP) + dist(array[nd3],array[nd4], sharedArrayTSP)
                                        + dist(array[nd5],array[nd6], sharedArrayTSP)+ dist(array[nd7],array[nd8], sharedArrayTSP) + dist(array[nd9],array[nd10], sharedArrayTSP);

                                float opti = oldLength - newLength;
                                if(opti > 0 && opti > optimiz)
                                {
                                    finalSelect = optCandi;
                                    optimiz = opti;

                                    //                                    atomicExch(&(nn_source.minRadiusMap[0][w-1]), 1);
                                    //                                    atomicExch(&(nn_source.minRadiusMap[0][j-1]), 1);
                                    //                                    atomicExch(&(nn_source.minRadiusMap[0][p-1]), 1);
                                    //                                    atomicExch(&(nn_source.minRadiusMap[0][k-1]), 1);
                                    //                                    atomicExch(&(nn_source.minRadiusMap[0][idRow5th-1]), 1);

                                    atomicExch(&(sharedArrayOccupied[w-1]), 1);
                                    atomicExch(&(sharedArrayOccupied[j-1]), 1);
                                    atomicExch(&(sharedArrayOccupied[p-1]), 1);
                                    atomicExch(&(sharedArrayOccupied[k-1]), 1);
                                    atomicExch(&(sharedArrayOccupied[idRow5th-1]), 1);

                                }
                            }

                            if(finalSelect >= 0)
                            {

                                unsigned int node1 = (int)sharedArrayTSP[w-1].current;
                                unsigned int node3 = (int)sharedArrayTSP[j-1].current;
                                unsigned int node5 = (int)sharedArrayTSP[p-1].current;
                                unsigned int node7 = (int)sharedArrayTSP[k-1].current;
                                unsigned int node9 = (int)sharedArrayTSP[idRow5th-1].current;

                                //                            float localMinChange = nn_source.minRadiusMap[0][node1];

                                //                            if(optimiz > localMinChange)
                                {

                                    unsigned long long result = 0;
                                    result = result | node3;
                                    result = result << 16;
                                    result = result | node5;
                                    result = result << 16;
                                    result = result | node7;
                                    result = result << 16;
                                    result = result | node9;

                                    float codekopt = finalSelect; //WB.Q 2024 densityMap only mark the k value of k-opt, and mark which mode of k-exchange
                                    codekopt = finalSelect * 100 + 5;

                                    //                                printf("GPU search, node1, node3, node5, node7, %d, %d, %d, %d; order(%d,%d,%d,%d), optvalue %lld, codekopt %f \n",
                                    //                                       node1, node3, node5, node7, nn_source.grayValueMap[0][node1], nn_source.grayValueMap[0][node3], nn_source.grayValueMap[0][node5] , nn_source.grayValueMap[0][node7], result, codekopt);
                                    atomicExch(&(nn_source.optCandidateMap[0][node1]), result); // WB.Q 2024 find a solution to judge non-interacted 23456-opt
                                    atomicExch(&(nn_source.densityMap[0][node1]), codekopt); // WB.Q 2024 densityMap only mark the k value of k-opt, and mark which mode of k-exchange
                                    //                              atomicExch(&(nn_source.minRadiusMap[0][node1]), optimiz);
                                }
                            }
                        }

                    }
                }


            }
        }
    }
    __syncthreads();
}// end K_5optOneThreadOne5opt




/*!
 * \brief 2409 QWB: add parallel 5opt with rocki's method
 */
//epecially for small size, copy all cities into shared memory totally 32068bytes shared mem,for 1979 cities and 5-opt possibilities
template <template<typename, typename> class NeuralNetLinks, class BufferDimension, class Point>
KERNEL void K_5opt_oneThreadOne5opt_qiao_stride_iter_shared_noOccupy(NeuralNetLinks<BufferDimension, Point> nn_source,
                                                                     Grid<doubleLinkedEdgeForTSP> arrayTSP,
                                                                     int idRow5th, double maxChecks2opt, double maxChecks4opt, double maxChecks4optDivide,
                                                                     double iter, double istride)
{

    double local_id = threadIdx.x + blockIdx.x * blockDim.x;
    int width =  nn_source.adaptiveMap.width; // each thread has this register


    __shared__ doubleLinkedEdgeForTSP sharedArrayTSP[SHAREDMAXCITIES]; // 23748bytes = 3*1979*float
    //    __shared__ float sharedArrayOccupied[SHAREDMAXCITIES];
    __shared__ float optPossibilities[OPTPOSSIBILITES5OPT]; //8320bytes = 2080floats
    float iterShared = (float)width / (float)BLOCKSIZE;
    for(int opt = 0; opt < iterShared; opt++)
    {
        int m = threadIdx.x + opt*BLOCKSIZE;
        if(m < width)
        {
            sharedArrayTSP[m].current = arrayTSP[0][m].current;
            sharedArrayTSP[m].currentCoord[0] = arrayTSP[0][m].currentCoord[0];
            sharedArrayTSP[m].currentCoord[1] = arrayTSP[0][m].currentCoord[1];

        }
        __syncthreads();
    }

    float iterSharedPossble =  (float)OPTPOSSIBILITES5OPT / (float)BLOCKSIZE;
    for(int opt = 0; opt < iterSharedPossble; opt++)
    {
        int m = threadIdx.x + opt*BLOCKSIZE;

        if(m < OPTPOSSIBILITES5OPT)
            optPossibilities[m] = nn_source.nodeParentMap[0][m];

        __syncthreads();

    }


    if(local_id < maxChecks4opt)
    {

        double startId = maxChecks4optDivide * (istride);

        if(local_id == 0)
            printf("StartID %f, local_id %f \n", startId, local_id);


        for(double id = local_id*iter ; id < (local_id+1)*(iter); id++)
        {

            id = id + startId;

            if(id > 0 && id < maxChecks4opt)
            {

                double outi, outj;
                double sqrtOuti = 8.0 * (double)id + 1.0;
                outi = int(3 + sqrt(sqrtOuti)) / 2 ;
                outj = id - (outi-2)*(outi-1)/2 + 1;

                if(outi < maxChecks2opt && outj < maxChecks2opt)
                {
                    double sqrtOutIK = 8.0 * (double )outi + 1.0;
                    int k = int(3 + sqrt(sqrtOutIK)) / 2 ;
                    int p = outi - (k-2)*(k-1)/2 + 1;

                    double sqrtOutJk = 8.0 * (double)outj + 1.0;
                    int j = int(3 + sqrt(sqrtOutJk)) / 2 ;
                    int w = outj - (j-2)*(j-1)/2 + 1;

                    if(id > maxChecks4opt-2)
                        printf("maximum 50opt id = %d, outi= %d, outj=%d, inner row, k,p,j,w =(%d, %d, %d, %d, %d), \n", id, outi, outj, idRow5th, k, p, j, w);

                    if( idRow5th > k+1 && k > p && p> j&& j>w&& k< width && p<width && j<width && w<width &&  k > 0 && p > 0 && j > 0 && w > 0 && p+1!=k && w+1!=j && j+1!=p)
                    {

                        //                        if(w > 9144)
                        //                        printf(" maximum 5-opt id = %f, outi= %f, outj=%f, inner k,p,j,w =(%d, %d, %d, %d, %d), \n", id, outi, outj, idRow5th, k, p, j, w);


                        bool existingCandidate = 0;

                        if(nn_source.minRadiusMap[0][w-1] == 1 || nn_source.minRadiusMap[0][j-1] == 1 ||nn_source.minRadiusMap[0][p-1] == 1
                                ||nn_source.minRadiusMap[0][k-1] == 1 ||nn_source.minRadiusMap[0][idRow5th-1] == 1)
                            existingCandidate = 1;

                        if(existingCandidate == 0)
                        {

                            float oldLength = dist(w-1, w, sharedArrayTSP) + dist(j-1, j, sharedArrayTSP) + dist(p-1, p, sharedArrayTSP)+ dist(k-1, k, sharedArrayTSP) + dist(idRow5th-1, idRow5th, sharedArrayTSP);

                            float newLength;
                            int array[10];
                            array[0] = w-1;
                            array[1] = w;
                            array[2] = j-1;
                            array[3] = j;
                            array[4] = p-1;
                            array[5] = p;
                            array[6] = k-1;
                            array[7] = k;
                            array[8] = idRow5th-1;
                            array[9] = idRow5th;

                            int finalSelect = -1;
                            float optimiz = -INFINITY;

                            for(int opt = 0; opt < 2080; opt +=10) //  4 edges 8 nodes
                            {
                                //                                int nd1 = nn_source.nodeParentMap[0][opt] -1;
                                //                                int nd2 = nn_source.nodeParentMap[0][opt+1] -1;
                                //                                int nd3 = nn_source.nodeParentMap[0][opt+2] -1;
                                //                                int nd4 = nn_source.nodeParentMap[0][opt+3] -1;
                                //                                int nd5 = nn_source.nodeParentMap[0][opt+4] -1;
                                //                                int nd6 = nn_source.nodeParentMap[0][opt+5] -1;
                                //                                int nd7 = nn_source.nodeParentMap[0][opt+6] -1;
                                //                                int nd8 = nn_source.nodeParentMap[0][opt+7] -1;
                                //                                int nd9 = nn_source.nodeParentMap[0][opt+8] -1;
                                //                                int nd10 = nn_source.nodeParentMap[0][opt+9] -1;

                                int nd1 = optPossibilities[opt] -1;
                                int nd2 = optPossibilities[opt+1] -1;
                                int nd3 = optPossibilities[opt+2] -1;
                                int nd4 = optPossibilities[opt+3] -1;
                                int nd5 = optPossibilities[opt+4] -1;
                                int nd6 = optPossibilities[opt+5] -1;
                                int nd7 = optPossibilities[opt+6] -1;
                                int nd8 = optPossibilities[opt+7] -1;
                                int nd9 = optPossibilities[opt+8] -1;
                                int nd10 = optPossibilities[opt+9] -1;


                                int optCandi = opt / 10;
                                // printf("GPU search nd1-8 %d, %d, %d, %d, %d, %d, %d, %d; optCandi=%d \n", nd1, nd2, nd3, nd4, nd5, nd6, nd7, nd8, optCandi);
                                newLength = dist(array[nd1],array[nd2], sharedArrayTSP) + dist(array[nd3],array[nd4], sharedArrayTSP)
                                        + dist(array[nd5],array[nd6], sharedArrayTSP)+ dist(array[nd7],array[nd8], sharedArrayTSP) + dist(array[nd9],array[nd10], sharedArrayTSP);

                                float opti = oldLength - newLength;
                                if(opti > 0 && opti > optimiz)
                                {
                                    finalSelect = optCandi;
                                    optimiz = opti;

                                    atomicExch(&(nn_source.minRadiusMap[0][w-1]), 1);
                                    atomicExch(&(nn_source.minRadiusMap[0][j-1]), 1);
                                    atomicExch(&(nn_source.minRadiusMap[0][p-1]), 1);
                                    atomicExch(&(nn_source.minRadiusMap[0][k-1]), 1);
                                    atomicExch(&(nn_source.minRadiusMap[0][idRow5th-1]), 1);

                                }
                            }

                            if(finalSelect >= 0)
                            {

                                unsigned int node1 = (int)sharedArrayTSP[w-1].current;
                                unsigned int node3 = (int)sharedArrayTSP[j-1].current;
                                unsigned int node5 = (int)sharedArrayTSP[p-1].current;
                                unsigned int node7 = (int)sharedArrayTSP[k-1].current;
                                unsigned int node9 = (int)sharedArrayTSP[idRow5th-1].current;

                                //                            float localMinChange = nn_source.minRadiusMap[0][node1];

                                //                            if(optimiz > localMinChange)
                                {

                                    unsigned long long result = 0;
                                    result = result | node3;
                                    result = result << 16;
                                    result = result | node5;
                                    result = result << 16;
                                    result = result | node7;
                                    result = result << 16;
                                    result = result | node9;

                                    float codekopt = finalSelect; //WB.Q 2024 densityMap only mark the k value of k-opt, and mark which mode of k-exchange
                                    codekopt = finalSelect * 100 + 5;

                                    //                                printf("GPU search, node1, node3, node5, node7, %d, %d, %d, %d; order(%d,%d,%d,%d), optvalue %lld, codekopt %f \n",
                                    //                                       node1, node3, node5, node7, nn_source.grayValueMap[0][node1], nn_source.grayValueMap[0][node3], nn_source.grayValueMap[0][node5] , nn_source.grayValueMap[0][node7], result, codekopt);
                                    atomicExch(&(nn_source.optCandidateMap[0][node1]), result); // WB.Q 2024 find a solution to judge non-interacted 23456-opt
                                    atomicExch(&(nn_source.densityMap[0][node1]), codekopt); // WB.Q 2024 densityMap only mark the k value of k-opt, and mark which mode of k-exchange
                                    //                              atomicExch(&(nn_source.minRadiusMap[0][node1]), optimiz);
                                }
                            }
                        }

                    }
                }


            }
        }
    }
    __syncthreads();
}// end K_5optOneThreadOne5opt



/*!
 * \brief 2409 QWB: add parallel 5opt with rocki's method
 */
//epecially for small size, copy all cities into shared memory
template <template<typename, typename> class NeuralNetLinks, class BufferDimension, class Point>
KERNEL void K_5opt_oneThreadOne5opt_qiao_stride_iter_shared_onlyPossibility(NeuralNetLinks<BufferDimension, Point> nn_source,
                                                                            Grid<doubleLinkedEdgeForTSP> arrayTSP,
                                                                            int idRow5th, double maxChecks2opt, double maxChecks4opt, double maxChecks4optDivide,
                                                                            double iter, double istride)
{

    double local_id = threadIdx.x + blockIdx.x * blockDim.x;
    int width =  nn_source.adaptiveMap.width; // each thread has this register


    //    __shared__ doubleLinkedEdgeForTSP sharedArrayTSP[SHAREDMAXCITIES];
    //    __shared__ float sharedArrayOccupied[SHAREDMAXCITIES];
    __shared__ float optPossibilities[OPTPOSSIBILITES5OPT];
    //    float iterShared = (float)width / (float)BLOCKSIZE;
    //    for(int opt = 0; opt < iterShared; opt++)
    //    {
    //        int m = threadIdx.x + opt*BLOCKSIZE;
    //        if(m < width)
    //        {
    //            sharedArrayTSP[m].current = arrayTSP[0][m].current;
    //            sharedArrayTSP[m].currentCoord[0] = arrayTSP[0][m].currentCoord[0];
    //            sharedArrayTSP[m].currentCoord[1] = arrayTSP[0][m].currentCoord[1];

    //        }
    //        __syncthreads();
    //    }

    float iterSharedPossble =  (float)OPTPOSSIBILITES5OPT / (float)BLOCKSIZE;
    for(int opt = 0; opt < iterSharedPossble; opt++)
    {
        int m = threadIdx.x + opt*BLOCKSIZE;

        if(m < OPTPOSSIBILITES5OPT)
            optPossibilities[m] = nn_source.nodeParentMap[0][m];

        __syncthreads();

    }


    if(local_id < maxChecks4opt)
    {

        double startId = maxChecks4optDivide * (istride);

        if(local_id == 0)
            printf("StartID %f, local_id %f \n", startId, local_id);


        for(double id = local_id*iter ; id < (local_id+1)*(iter); id++)
        {

            id = id + startId;

            if(id > 0 && id < maxChecks4opt)
            {

                double outi, outj;
                double sqrtOuti = 8.0 * (double)id + 1.0;
                outi = int(3 + sqrt(sqrtOuti)) / 2 ;
                outj = id - (outi-2)*(outi-1)/2 + 1;

                if(outi < maxChecks2opt && outj < maxChecks2opt)
                {
                    double sqrtOutIK = 8.0 * (double )outi + 1.0;
                    int k = int(3 + sqrt(sqrtOutIK)) / 2 ;
                    int p = outi - (k-2)*(k-1)/2 + 1;

                    double sqrtOutJk = 8.0 * (double)outj + 1.0;
                    int j = int(3 + sqrt(sqrtOutJk)) / 2 ;
                    int w = outj - (j-2)*(j-1)/2 + 1;

                    if(id > maxChecks4opt-2)
                        printf("maximum 50opt id = %d, outi= %d, outj=%d, inner row, k,p,j,w =(%d, %d, %d, %d, %d), \n", id, outi, outj, idRow5th, k, p, j, w);

                    if( idRow5th > k+1 && k > p && p> j&& j>w&& k< width && p<width && j<width && w<width &&  k > 0 && p > 0 && j > 0 && w > 0 && p+1!=k && w+1!=j && j+1!=p)
                    {

                        if(w > 9144)
                            printf(" maximum 5-opt id = %f, outi= %f, outj=%f, inner k,p,j,w =(%d, %d, %d, %d, %d), \n", id, outi, outj, idRow5th, k, p, j, w);


                        bool existingCandidate = 0;

                        if(nn_source.minRadiusMap[0][w-1] == 1 || nn_source.minRadiusMap[0][j-1] == 1 ||nn_source.minRadiusMap[0][p-1] == 1
                                ||nn_source.minRadiusMap[0][k-1] == 1 ||nn_source.minRadiusMap[0][idRow5th-1] == 1)
                            existingCandidate = 1;

                        if(existingCandidate == 0)
                        {

                            float oldLength = dist(w-1, w, arrayTSP[0]) + dist(j-1, j, arrayTSP[0]) + dist(p-1, p, arrayTSP[0])+ dist(k-1, k, arrayTSP[0]) + dist(idRow5th-1, idRow5th, arrayTSP[0]);

                            float newLength;
                            int array[10];
                            array[0] = w-1;
                            array[1] = w;
                            array[2] = j-1;
                            array[3] = j;
                            array[4] = p-1;
                            array[5] = p;
                            array[6] = k-1;
                            array[7] = k;
                            array[8] = idRow5th-1;
                            array[9] = idRow5th;

                            int finalSelect = -1;
                            float optimiz = -INFINITY;

                            for(int opt = 0; opt < 2080; opt +=10) //  4 edges 8 nodes
                            {

                                int nd1 = optPossibilities[opt] -1;
                                int nd2 = optPossibilities[opt+1] -1;
                                int nd3 = optPossibilities[opt+2] -1;
                                int nd4 = optPossibilities[opt+3] -1;
                                int nd5 = optPossibilities[opt+4] -1;
                                int nd6 = optPossibilities[opt+5] -1;
                                int nd7 = optPossibilities[opt+6] -1;
                                int nd8 = optPossibilities[opt+7] -1;
                                int nd9 = optPossibilities[opt+8] -1;
                                int nd10 = optPossibilities[opt+9] -1;


                                int optCandi = opt / 10;
                                // printf("GPU search nd1-8 %d, %d, %d, %d, %d, %d, %d, %d; optCandi=%d \n", nd1, nd2, nd3, nd4, nd5, nd6, nd7, nd8, optCandi);
                                newLength = dist(array[nd1],array[nd2], arrayTSP[0]) + dist(array[nd3],array[nd4], arrayTSP[0])
                                        + dist(array[nd5],array[nd6], arrayTSP[0])+ dist(array[nd7],array[nd8], arrayTSP[0]) + dist(array[nd9],array[nd10], arrayTSP[0]);

                                float opti = oldLength - newLength;
                                if(opti > 0 && opti > optimiz)
                                {
                                    finalSelect = optCandi;
                                    optimiz = opti;

                                    atomicExch(&(nn_source.minRadiusMap[0][w-1]), 1);
                                    atomicExch(&(nn_source.minRadiusMap[0][j-1]), 1);
                                    atomicExch(&(nn_source.minRadiusMap[0][p-1]), 1);
                                    atomicExch(&(nn_source.minRadiusMap[0][k-1]), 1);
                                    atomicExch(&(nn_source.minRadiusMap[0][idRow5th-1]), 1);

                                }
                            }

                            if(finalSelect >= 0)
                            {

                                unsigned int node1 = (int)arrayTSP[0][w-1].current;
                                unsigned int node3 = (int)arrayTSP[0][j-1].current;
                                unsigned int node5 = (int)arrayTSP[0][p-1].current;
                                unsigned int node7 = (int)arrayTSP[0][k-1].current;
                                unsigned int node9 = (int)arrayTSP[0][idRow5th-1].current;

                                //                            float localMinChange = nn_source.minRadiusMap[0][node1];

                                //                            if(optimiz > localMinChange)
                                {

                                    unsigned long long result = 0;
                                    result = result | node3;
                                    result = result << 16;
                                    result = result | node5;
                                    result = result << 16;
                                    result = result | node7;
                                    result = result << 16;
                                    result = result | node9;

                                    float codekopt = finalSelect; //WB.Q 2024 densityMap only mark the k value of k-opt, and mark which mode of k-exchange
                                    codekopt = finalSelect * 100 + 5;

                                    //                                printf("GPU search, node1, node3, node5, node7, %d, %d, %d, %d; order(%d,%d,%d,%d), optvalue %lld, codekopt %f \n",
                                    //                                       node1, node3, node5, node7, nn_source.grayValueMap[0][node1], nn_source.grayValueMap[0][node3], nn_source.grayValueMap[0][node5] , nn_source.grayValueMap[0][node7], result, codekopt);
                                    atomicExch(&(nn_source.optCandidateMap[0][node1]), result); // WB.Q 2024 find a solution to judge non-interacted 23456-opt
                                    atomicExch(&(nn_source.densityMap[0][node1]), codekopt); // WB.Q 2024 densityMap only mark the k value of k-opt, and mark which mode of k-exchange
                                    //                              atomicExch(&(nn_source.minRadiusMap[0][node1]), optimiz);
                                }
                            }
                        }

                    }
                }


            }
        }
    }
    __syncthreads();
}// end K_5optOneThreadOne5opt





/*!
 * \brief 202408 QWB: add parallel 3-opt with rocki's method find the best from one node
 */
template <template<typename, typename> class NeuralNetLinks, class BufferDimension, class Point>
KERNEL void K_3opt_oneThreadOne3opt_rockiSmall_findBest(NeuralNetLinks<BufferDimension, Point> nn_source,
                                                        Grid<doubleLinkedEdgeForTSP> arrayTSP,
                                                        double maxChecks3opt,
                                                        unsigned int iter)
{

    double id = threadIdx.x + blockIdx.x * blockDim.x;

    int width =  nn_source.adaptiveMap.width; // each thread has this register

    //    if(id == 0 )
    //        printf(" 0 maxium  %lld , maxim blockId = %lld , maxim blockId = %d, blockDim.x = %d \n ", maxChecks3opt, maxChecks3opt / 1024,  blockIdx.x , blockDim.x);

    //    if(blockIdx.x > maxChecks3opt / 1024 - 4 )
    //        printf(" block maxium  %lld ,  blockId = %d, blockDim.x = %d \n ", maxChecks3opt,  blockIdx.x , blockDim.x);

    if(id < maxChecks3opt)
    {

        //        int packSize = blockDim.x * gridDim.x;
        int row, i, j;
        double subtriplicate = (1.0)/3;
        //        for(int nu = 0; nu <= iter; nu++)
        {

            //             id = local_id;  + nu * packSize;

            //            if(id < maxChecks3opt)
            {

                //WB.Q this way will produce i = j
                double idid = 9*id*id;
                double idMul3 = 3*id;
                double rowN0 = idMul3 + sqrt(idid - (1.0)/9);
                double rowN1 = pow(rowN0, subtriplicate);
                double rowN2 = idMul3 - sqrt(idid - (1.0)/9);
                float rowN3 = pow(rowN2, subtriplicate);
                float rowN4 = rowN1 + rowN3 + 1;

                row = int(rowN4);// check which one works


                //qiao only for test
                if(id >maxChecks3opt - 2)
                    printf("3-opt maxmum id %d,  row %d, i %d, j %d \n", id, row, i,j);


                if(row <= width)
                {
                    double id2opt = (row-1)*(row)*(row+1)/6 - id;

                    //WB.Q this way will produce i = j
                    i = int(3 + sqrt(8.0 * (double)id2opt + 1.0)) / 2 ;
                    j = id2opt - (i-2)*(i-1)/2 + 1;


                    if(i<row && i!=row &&i+1!= row &&j > 0 && j < i && j-1 >= 0 && j <= width && j+1 != i && j+ width != i+1 && i-1 >= 0 && i < width-1)
                    {

                        //qiao only for test
                        if(row > width - 2)
                            printf("3-opt maxmum row id %d, row %d, i %d, j %d \n", id, row, i,j);

                        double newLength[4];

                        double oldLength = dist(j-1, j, arrayTSP[0]) + dist(i-1, i, arrayTSP[0]) + dist(row-1, row, arrayTSP[0]);
                        newLength[0] = dist(j-1, i, arrayTSP[0]) + dist(row-1, i-1, arrayTSP[0]) + dist(row, j, arrayTSP[0]);
                        newLength[1] = dist(j-1, row-1,arrayTSP[0]) + dist(row, i-1, arrayTSP[0]) + dist(i,j,arrayTSP[0]);
                        newLength[2] = dist(j-1, i-1, arrayTSP[0]) + dist(row-1, j, arrayTSP[0]) + dist(row, i,arrayTSP[0]);
                        newLength[3] = dist(j-1, i, arrayTSP[0]) + dist(row-1, j,arrayTSP[0]) + dist(row,i-1,arrayTSP[0]);

                        int finalSelect = -1;
                        double optimiz = -INFINITY;
                        for(int i = 0; i < 4; i++)
                        {
                            float opti = oldLength - newLength[i];
                            if(opti > 0 && opti > optimiz)
                            {
                                finalSelect = i;
                                optimiz = opti;

                                //                               if(blockIdx.x == 124698010  ||blockIdx.x == 124698013   || blockIdx.x == 0)
                                //                                printf("3opt GPU blockIdx.x=%d, selec %d, order %d, %d, %d, oldLength %f, newi %f, opti %f, optimiz %f ; node135 %d,%d,%d\n",blockIdx.x, finalSelect,
                                //                                       nn_source.grayValueMap[0][j-1], nn_source.grayValueMap[0][i-1], nn_source.grayValueMap[0][row-1]
                                //                                        , oldLength, newLength[i], opti, optimiz, j-1, i-1, row-1);


                            }
                        }

                        if(finalSelect >= 0)
                        {
                            float optimization = oldLength - newLength[finalSelect];
                            // here automic operation is necessary

                            unsigned int node1 = (int)arrayTSP[0][j-1].current;
                            unsigned int node3 = (int)arrayTSP[0][i-1].current;
                            unsigned int node5 = (int)arrayTSP[0][row-1].current;

                            //                            printf("3opt GPU mode %d, order %d, %d, %d, oldLength %f, new1 %f, new2 %f, new3 %f, new4 %f; node135 %d,%d,%d \n", finalSelect, nn_source.grayValueMap[0][node1], nn_source.grayValueMap[0][node3], nn_source.grayValueMap[0][node5]
                            //                                    , oldLength, newLength[0],newLength[1], newLength[2], newLength[3], node1, node3, node5);

                            float localMinChange = nn_source.minRadiusMap[0][node1];

                            if(optimization > localMinChange)
                            {

                                unsigned long long result = 0;
                                result = result | node3;
                                result = result << 16;
                                result = result | node5;

                                float codekopt = finalSelect; //WB.Q 2024 densityMap only mark the k value of k-opt, and mark which mode of k-exchange
                                codekopt = finalSelect * 100 + 3;

                                atomicExch(&(nn_source.optCandidateMap[0][node1]), result); // WB.Q 2024 find a solution to judge non-interacted 23456-opt
                                atomicExch(&(nn_source.densityMap[0][node1]), codekopt); // WB.Q 2024 densityMap only mark the k value of k-opt, and mark which mode of k-exchange
                                atomicExch(&(nn_source.minRadiusMap[0][node1]), optimization);
                            }
                        }
                    }//end if i j

                }//end if row <= width

            }
        }


    }
    __syncthreads();
}// end K_2optOneThreadOne3opt





/*!
 * \brief 202408 QWB: add parallel 3-opt with rocki's method one node only participates one candidates
 */
//epecially for small size, copy all cities into shared memory
template <template<typename, typename> class NeuralNetLinks, class BufferDimension, class Point>
KERNEL void K_3opt_oneThreadOne3opt_rockiSmall_iterBest(NeuralNetLinks<BufferDimension, Point> nn_source,
                                                        Grid<doubleLinkedEdgeForTSP> arrayTSP,
                                                        double maxChecks3opt,
                                                        double iter)
{

    double local_id = threadIdx.x + blockIdx.x * blockDim.x;

    int width =  nn_source.adaptiveMap.width; // each thread has this register

    if(local_id < maxChecks3opt)
    {
        //qiao only for test
        if(local_id == 0)
            printf("local_id %f \n", local_id);


        for(double id = local_id*iter ; id < (local_id+1)*(iter); id++)
        {


            if(id > 0 && id < maxChecks3opt)
            {

                int row, i, j;
                double subtriplicate = (1.0)/3;


                //                if(id > 350631671)
                {

                    //WB.Q this way will produce i = j
                    double idid = 9*id*id - (1.0)/9 ;
                    double idMul3 = 3*id;
                    double rowN0 = idMul3 + sqrt(idid );
                    double rowN1 = pow(rowN0, subtriplicate);
                    double rowN2 = idMul3 - sqrt(idid);
                    double rowN3 = pow(rowN2, subtriplicate);
                    double rowN4 = rowN1 + rowN3 + 1;


                    if(row <= width)
                    {

                        double tempRowRow = (double)(row-1) / 6;
                        double tempRowRowRow = tempRowRow*(row)*(row-2);
                        double id2opt = fabs( id - tempRowRowRow );
                        //WB.Q this way will produce i = j
                        double sqrtTemp = 8.0 * (double)id2opt + 1.0;
                        i = int(3 + sqrt(sqrtTemp)) / 2 ;
                        j = id2opt - (i-2)*(i-1)/2 + 1;


                        row = int(rowN4);// check which one works


                        if(i<row && i!=row &&i+1!= row && j > 0 && j < i && j-1 >= 0 && j <= width && j+1 != i && j+ width != i+1 && i-1 >= 0 && i < width-1)
                        {

                            if(j> 9145)//350631671)
                                printf("largeRow %f, local_id %f, idid %f \n idMul3 %f, rowN0 %f , rowN3 %f, rowN4 %f , i %d, j %d, id2opt %f, sqrtTemp %f \n", id, local_id, idid, idMul3, rowN0, rowN3, rowN4, i, j, id2opt, sqrtTemp);

                            double newLength[4];

                            double oldLength = dist(j-1, j, arrayTSP[0]) + dist(i-1, i, arrayTSP[0]) + dist(row-1, row, arrayTSP[0]);
                            newLength[0] = dist(j-1, i, arrayTSP[0]) + dist(row-1, i-1, arrayTSP[0]) + dist(row, j, arrayTSP[0]);
                            newLength[1] = dist(j-1, row-1,arrayTSP[0]) + dist(row, i-1, arrayTSP[0]) + dist(i,j,arrayTSP[0]);
                            newLength[2] = dist(j-1, i-1, arrayTSP[0]) + dist(row-1, j, arrayTSP[0]) + dist(row, i,arrayTSP[0]);
                            newLength[3] = dist(j-1, i, arrayTSP[0]) + dist(row-1, j,arrayTSP[0]) + dist(row,i-1,arrayTSP[0]);

                            int finalSelect = -1;
                            double optimiz = -INFINITY;
                            for(int i = 0; i < 4; i++)
                            {
                                float opti = oldLength - newLength[i];
                                if(opti > 0 && opti > optimiz)
                                {
                                    finalSelect = i;
                                    optimiz = opti;

                                }
                            }


                            if(finalSelect >= 0)
                            {

                                unsigned int node1 = (int)arrayTSP[0][j-1].current;
                                unsigned int node3 = (int)arrayTSP[0][i-1].current;
                                unsigned int node5 = (int)arrayTSP[0][row-1].current;


                                double localMinChange = nn_source.minRadiusMap[0][node1];
                                double optimization = oldLength - newLength[finalSelect];

                                if(optimization > localMinChange)
                                {

                                    unsigned long long result = 0;
                                    result = result | node3;
                                    result = result << 16;
                                    result = result | node5;

                                    float codekopt = finalSelect; //WB.Q 2024 densityMap only mark the k value of k-opt, and mark which mode of k-exchange
                                    codekopt = finalSelect * 100 + 3;

                                    atomicExch(&(nn_source.optCandidateMap[0][node1]), result); // WB.Q 2024 find a solution to judge non-interacted 23456-opt
                                    atomicExch(&(nn_source.densityMap[0][node1]), codekopt); // WB.Q 2024 densityMap only mark the k value of k-opt, and mark which mode of k-exchange
                                    atomicExch(&(nn_source.minRadiusMap[0][node1]), optimization);
                                }
                            }



                        }//end if i j

                    }//end if row <= width

                }

            }


        }


    }
    __syncthreads();
}// end K_2optOneThreadOne3opt


/*!
 * \brief 202408 QWB: add parallel 3-opt with rocki's method one node only participates one candidates work correctly final version
 */
//epecially for small size, copy all cities into shared memory
template <template<typename, typename> class NeuralNetLinks, class BufferDimension, class Point>
KERNEL void K_3opt_oneThreadOne3opt_rockiSmall_iter(NeuralNetLinks<BufferDimension, Point> nn_source,
                                                    Grid<doubleLinkedEdgeForTSP> arrayTSP,
                                                    double maxChecks3opt,
                                                    double iter)
{

    double local_id = threadIdx.x + blockIdx.x * blockDim.x;

    int width =  nn_source.adaptiveMap.width; // each thread has this register

    if(local_id < maxChecks3opt)
    {


        for(double id = local_id*iter ; id < (local_id+1)*(iter); id++)
        {


            if(id > 0 && id < maxChecks3opt)
            {

                int row, i, j;
                double subtriplicate = (1.0)/3;


                //                if(id > 350631671)
                {

                    //WB.Q this way will produce i = j
                    double idid = 9*id*id - (1.0)/9 ;
                    double idMul3 = 3*id;
                    double rowN0 = idMul3 + sqrt(idid );
                    double rowN1 = pow(rowN0, subtriplicate);
                    double rowN2 = idMul3 - sqrt(idid);
                    double rowN3 = pow(rowN2, subtriplicate);
                    double rowN4 = rowN1 + rowN3 + 1;


                    if(row <= width)
                    {

                        double tempRowRow = (double)(row-1) / 6;
                        double tempRowRowRow = tempRowRow*(row)*(row-2);
                        double id2opt = fabs( id - tempRowRowRow );
                        //WB.Q this way will produce i = j
                        double sqrtTemp = 8.0 * (double)id2opt + 1.0;
                        i = int(3 + sqrt(sqrtTemp)) / 2 ;
                        j = id2opt - (i-2)*(i-1)/2 + 1;


                        row = int(rowN4);// check which one works


                        if(i<row && i!=row &&i+1!= row && j > 0 && j < i && j-1 >= 0 && j <= width && j+1 != i && j+ width != i+1 && i-1 >= 0 && i < width-1)
                        {

                            bool existingCandidate = 0;
                            if(nn_source.minRadiusMap[0][row-1] == 1 || nn_source.minRadiusMap[0][j-1] == 1 ||nn_source.minRadiusMap[0][i-1] == 1)
                                existingCandidate = 1;


                            if(j> 9145)//350631671)
                                printf("largeRow %f, local_id %f, idid %f \n idMul3 %f, rowN0 %f , rowN3 %f, rowN4 %f , i %d, j %d, id2opt %f, sqrtTemp %f \n", id, local_id, idid, idMul3, rowN0, rowN3, rowN4, i, j, id2opt, sqrtTemp);



                            if(existingCandidate == 0)
                            {

                                double newLength[4];

                                double oldLength = dist(j-1, j, arrayTSP[0]) + dist(i-1, i, arrayTSP[0]) + dist(row-1, row, arrayTSP[0]);
                                newLength[0] = dist(j-1, i, arrayTSP[0]) + dist(row-1, i-1, arrayTSP[0]) + dist(row, j, arrayTSP[0]);
                                newLength[1] = dist(j-1, row-1,arrayTSP[0]) + dist(row, i-1, arrayTSP[0]) + dist(i,j,arrayTSP[0]);
                                newLength[2] = dist(j-1, i-1, arrayTSP[0]) + dist(row-1, j, arrayTSP[0]) + dist(row, i,arrayTSP[0]);
                                newLength[3] = dist(j-1, i, arrayTSP[0]) + dist(row-1, j,arrayTSP[0]) + dist(row,i-1,arrayTSP[0]);

                                int finalSelect = -1;
                                double optimiz = -INFINITY;
                                for(int i = 0; i < 4; i++)
                                {
                                    float opti = oldLength - newLength[i];
                                    if(opti > 0 && opti > optimiz)
                                    {
                                        finalSelect = i;
                                        optimiz = opti;

                                        atomicExch(&(nn_source.minRadiusMap[0][i-1]), 1);
                                        atomicExch(&(nn_source.minRadiusMap[0][j-1]), 1);
                                        atomicExch(&(nn_source.minRadiusMap[0][row-1]), 1);

                                    }
                                }


                                if(finalSelect >= 0)
                                {

                                    unsigned int node1 = (int)arrayTSP[0][j-1].current;
                                    unsigned int node3 = (int)arrayTSP[0][i-1].current;
                                    unsigned int node5 = (int)arrayTSP[0][row-1].current;

                                    {

                                        unsigned long long result = 0;
                                        result = result | node3;
                                        result = result << 16;
                                        result = result | node5;

                                        float codekopt = finalSelect; //WB.Q 2024 densityMap only mark the k value of k-opt, and mark which mode of k-exchange
                                        codekopt = finalSelect * 100 + 3;

                                        atomicExch(&(nn_source.optCandidateMap[0][node1]), result); // WB.Q 2024 find a solution to judge non-interacted 23456-opt
                                        atomicExch(&(nn_source.densityMap[0][node1]), codekopt); // WB.Q 2024 densityMap only mark the k value of k-opt, and mark which mode of k-exchange

                                    }
                                }

                            }

                        }//end if i j

                    }//end if row <= width

                }

            }


        }


    }
    __syncthreads();
}// end K_2optOneThreadOne3opt



/*!
 * \brief 202408 QWB: add parallel 3-opt with rocki's method one node only participates one candidates work correctly final version
 */
//epecially for small size, copy all cities into shared memory
template <template<typename, typename> class NeuralNetLinks, class BufferDimension, class Point>
KERNEL void K_3opt_oneThreadOne3opt_rockiSmall_iterStride(NeuralNetLinks<BufferDimension, Point> nn_source,
                                                          Grid<doubleLinkedEdgeForTSP> arrayTSP,
                                                          double maxChecks3opt,  double maxChecksoptDivide,
                                                          double iter, double istride)
{

    double local_id = threadIdx.x + blockIdx.x * blockDim.x;

    int width =  nn_source.adaptiveMap.width; // each thread has this register

    if(local_id < maxChecks3opt)
    {

        double startId = maxChecksoptDivide * (istride);

        if(local_id == 0)
            printf("StartID %f, local_id %f \n", startId, local_id);


        for(double id = local_id*iter ; id < (local_id+1)*(iter); id++)
        {

            id = id + startId;

            if(id > 0 && id < maxChecks3opt)
            {

                //                if(id > 350631671)
                //                    printf("id %f, local_id %f \n", id, local_id);

                int row, i, j;
                double subtriplicate = (1.0)/3;

                //                if(id > 350631671)
                {

                    //WB.Q this way will produce i = j
                    double idid = 9*id*id - (1.0)/9 ;
                    double idMul3 = 3*id;
                    double rowN0 = idMul3 + sqrt(idid );
                    double rowN1 = pow(rowN0, subtriplicate);
                    double rowN2 = idMul3 - sqrt(idid);
                    double rowN3 = pow(rowN2, subtriplicate);
                    double rowN4 = rowN1 + rowN3 + 1;


                    if(row <= width)
                    {

                        double tempRowRow = (double)(row-1) / 6;
                        double tempRowRowRow = tempRowRow*(row)*(row-2);
                        //                        double tempRowRow;
                        //                        double tempRowRowRow;

                        //                        if(iter > 1)
                        //                        {
                        //                            tempRowRow = (double)(row-1) / 6;
                        //                            tempRowRowRow = tempRowRow*(row)*(row-2);
                        //                        }
                        //                        else
                        //                        {
                        //                            tempRowRow = (double)(row-1) / 6;
                        //                            tempRowRowRow = tempRowRow*(row)*(row+1);

                        //                        }

                        double id2opt = fabs( id - tempRowRowRow );
                        //WB.Q this way will produce i = j
                        double sqrtTemp = 8.0 * (double)id2opt + 1.0;
                        i = int(3 + sqrt(sqrtTemp)) / 2 ;
                        j = id2opt - (i-2)*(i-1)/2 + 1;


                        row = int(rowN4);// check which one works


                        if(i<row && i!=row &&i+1!= row && j > 0 && j < i && j-1 >= 0 && j <= width && j+1 != i && j+ width != i+1 && i-1 >= 0 && i < width-1)
                        {

                            bool existingCandidate = 0;
                            if(nn_source.minRadiusMap[0][row-1] == 1 || nn_source.minRadiusMap[0][j-1] == 1 ||nn_source.minRadiusMap[0][i-1] == 1)
                                existingCandidate = 1;


                            //                            if(j> 9145)//350631671)
                            //                                printf("largeRow %f, local_id %f, idid %f \n idMul3 %f, rowN0 %f , rowN3 %f, rowN4 %f , i %d, j %d, id2opt %f, sqrtTemp %f \n", id, local_id, idid, idMul3, rowN0, rowN3, rowN4, i, j, id2opt, sqrtTemp);



                            if(existingCandidate == 0)
                            {

                                double newLength[4];
                                double oldLength = dist(j-1, j, arrayTSP[0]) + dist(i-1, i, arrayTSP[0]) + dist(row-1, row, arrayTSP[0]);
                                newLength[0] = dist(j-1, i, arrayTSP[0]) + dist(row-1, i-1, arrayTSP[0]) + dist(row, j, arrayTSP[0]);
                                newLength[1] = dist(j-1, row-1,arrayTSP[0]) + dist(row, i-1, arrayTSP[0]) + dist(i,j,arrayTSP[0]);
                                newLength[2] = dist(j-1, i-1, arrayTSP[0]) + dist(row-1, j, arrayTSP[0]) + dist(row, i,arrayTSP[0]);
                                newLength[3] = dist(j-1, i, arrayTSP[0]) + dist(row-1, j,arrayTSP[0]) + dist(row,i-1,arrayTSP[0]);

                                int finalSelect = -1;
                                double optimiz = -INFINITY;
                                for(int i = 0; i < 4; i++)
                                {
                                    float opti = oldLength - newLength[i];
                                    if(opti > 0 && opti > optimiz)
                                    {
                                        finalSelect = i;
                                        optimiz = opti;

                                        atomicExch(&(nn_source.minRadiusMap[0][i-1]), 1);
                                        atomicExch(&(nn_source.minRadiusMap[0][j-1]), 1);
                                        atomicExch(&(nn_source.minRadiusMap[0][row-1]), 1);

                                    }
                                }


                                if(finalSelect >= 0)
                                {

                                    unsigned int node1 = (int)arrayTSP[0][j-1].current;
                                    unsigned int node3 = (int)arrayTSP[0][i-1].current;
                                    unsigned int node5 = (int)arrayTSP[0][row-1].current;

                                    {

                                        unsigned long long result = 0;
                                        result = result | node3;
                                        result = result << 16;
                                        result = result | node5;

                                        float codekopt = finalSelect; //WB.Q 2024 densityMap only mark the k value of k-opt, and mark which mode of k-exchange
                                        codekopt = finalSelect * 100 + 3;

                                        atomicExch(&(nn_source.optCandidateMap[0][node1]), result); // WB.Q 2024 find a solution to judge non-interacted 23456-opt
                                        atomicExch(&(nn_source.densityMap[0][node1]), codekopt); // WB.Q 2024 densityMap only mark the k value of k-opt, and mark which mode of k-exchange

                                    }
                                }

                            }

                        }//end if i j

                    }//end if row <= width

                }

            }


        }


    }
    __syncthreads();
}// end K_2optOneThreadOne3opt


/*!
 * \brief 202408 QWB: add parallel 3-opt with rocki's method, using sharedArrayOccupied produces small quantity of opts
 */
//epecially for small size, copy all cities into shared memory
template <template<typename, typename> class NeuralNetLinks, class BufferDimension, class Point>
KERNEL void K_3opt_oneThreadOne3opt_rockiSmall_iterStride_shared(NeuralNetLinks<BufferDimension, Point> nn_source,
                                                                 Grid<doubleLinkedEdgeForTSP> arrayTSP,
                                                                 double maxChecks3opt,  double maxChecksoptDivide,
                                                                 double iter, double istride)
{

    double local_id = threadIdx.x + blockIdx.x * blockDim.x;

    int width =  nn_source.adaptiveMap.width; // each thread has this register


    __shared__ doubleLinkedEdgeForTSP sharedArrayTSP[SHAREDMAXCITIES];

    float iterShared = (float)width / (float)BLOCKSIZE;

    for(int opt = 0; opt < iterShared; opt++)
    {
        int m = threadIdx.x + opt*BLOCKSIZE;
        if(m < width)
        {
            sharedArrayTSP[m].current = arrayTSP[0][m].current;
            sharedArrayTSP[m].currentCoord[0] = arrayTSP[0][m].currentCoord[0];
            sharedArrayTSP[m].currentCoord[1] = arrayTSP[0][m].currentCoord[1];
        }
        __syncthreads();
    }


    if(local_id < maxChecks3opt)
    {

        double startId = maxChecksoptDivide * (istride);

        if(local_id == 0)
            printf("StartID %f, local_id %f \n", startId, local_id);


        for(double id = local_id*iter ; id < (local_id+1)*(iter); id++)
        {

            id = id + startId;

            if(id > 0 && id < maxChecks3opt)
            {

                int row, i, j;
                double subtriplicate = (1.0)/3;

                // if(id > 350631671)
                {

                    //WB.Q this way will produce i = j
                    double idid = 9*id*id - (1.0)/9 ;
                    double idMul3 = 3*id;
                    double rowN0 = idMul3 + sqrt(idid );
                    double rowN1 = pow(rowN0, subtriplicate);
                    double rowN2 = idMul3 - sqrt(idid);
                    double rowN3 = pow(rowN2, subtriplicate);
                    double rowN4 = rowN1 + rowN3 + 1;


                    if(row <= width)
                    {

                        double tempRowRow = (double)(row-1) / 6;
                        double tempRowRowRow = tempRowRow*(row)*(row-2);
                        //                        double tempRowRow;
                        //                        double tempRowRowRow;

                        //                        if(iter > 1)
                        //                        {
                        //                            tempRowRow = (double)(row-1) / 6;
                        //                            tempRowRowRow = tempRowRow*(row)*(row-2);
                        //                        }
                        //                        else
                        //                        {
                        //                            tempRowRow = (double)(row-1) / 6;
                        //                            tempRowRowRow = tempRowRow*(row)*(row+1);

                        //                        }

                        double id2opt = fabs( id - tempRowRowRow );
                        //WB.Q this way will produce i = j
                        double sqrtTemp = 8.0 * (double)id2opt + 1.0;
                        i = int(3 + sqrt(sqrtTemp)) / 2 ;
                        j = id2opt - (i-2)*(i-1)/2 + 1;


                        row = int(rowN4);// check which one works


                        if(i<row && i!=row &&i+1!= row && j > 0 && j < i && j-1 >= 0 && j <= width && j+1 != i && j+ width != i+1 && i-1 >= 0 && i < width-1)
                        {

                            bool existingCandidate = 0;

                            if(nn_source.minRadiusMap[0][row-1] == 1 || nn_source.minRadiusMap[0][j-1] == 1 ||nn_source.minRadiusMap[0][i-1] == 1)
                                existingCandidate = 1;

                            //                            if(sharedArrayTSP[row-1].occupied == 1 || sharedArrayTSP[j-1].occupied == 1 || sharedArrayTSP[i-1].occupied == 1)
                            //                                existingCandidate = 1;


                            //  if(j> 9145)//350631671)
                            //    printf("largeRow %f, local_id %f, idid %f \n idMul3 %f, rowN0 %f , rowN3 %f, rowN4 %f , i %d, j %d, id2opt %f, sqrtTemp %f \n", id, local_id, idid, idMul3, rowN0, rowN3, rowN4, i, j, id2opt, sqrtTemp);



                            if(existingCandidate == 0)
                            {

                                double newLength[4];
                                double oldLength = dist(j-1, j, sharedArrayTSP) + dist(i-1, i, sharedArrayTSP) + dist(row-1, row, sharedArrayTSP);
                                newLength[0] = dist(j-1, i, sharedArrayTSP) + dist(row-1, i-1, sharedArrayTSP) + dist(row, j, sharedArrayTSP);
                                newLength[1] = dist(j-1, row-1,sharedArrayTSP) + dist(row, i-1, sharedArrayTSP) + dist(i,j,sharedArrayTSP);
                                newLength[2] = dist(j-1, i-1, sharedArrayTSP) + dist(row-1, j, sharedArrayTSP) + dist(row, i,sharedArrayTSP);
                                newLength[3] = dist(j-1, i, sharedArrayTSP) + dist(row-1, j,sharedArrayTSP) + dist(row,i-1,sharedArrayTSP);

                                //                                printf("largeRow %f, local_id %f, idid %f \n idMul3 %f, rowN0 %f , rowN3 %f, rowN4 %f , i %d, j %d, id2opt %f, sqrtTemp %f , shared %d, shareOcuu %f \n",
                                //                                       id, local_id, idid, idMul3, rowN0, rowN3, rowN4, i, j, id2opt, sqrtTemp, sharedArrayTSP[j-1].current, sharedArrayTSP[row-1].occupied);



                                int finalSelect = -1;
                                double optimiz = -INFINITY;
                                for(int i = 0; i < 4; i++)
                                {
                                    float opti = oldLength - newLength[i];
                                    if(opti > 0 && opti > optimiz)
                                    {
                                        finalSelect = i;
                                        optimiz = opti;

                                        atomicExch(&(nn_source.minRadiusMap[0][i-1]), 1);
                                        atomicExch(&(nn_source.minRadiusMap[0][j-1]), 1);
                                        atomicExch(&(nn_source.minRadiusMap[0][row-1]), 1);


                                    }
                                }


                                if(finalSelect >= 0)
                                {

                                    unsigned int node1 = (int)sharedArrayTSP[j-1].current;
                                    unsigned int node3 = (int)sharedArrayTSP[i-1].current;
                                    unsigned int node5 = (int)sharedArrayTSP[row-1].current;

                                    {

                                        unsigned long long result = 0;
                                        result = result | node3;
                                        result = result << 16;
                                        result = result | node5;

                                        float codekopt = finalSelect; //WB.Q 2024 densityMap only mark the k value of k-opt, and mark which mode of k-exchange
                                        codekopt = finalSelect * 100 + 3;

                                        atomicExch(&(nn_source.optCandidateMap[0][node1]), result); // WB.Q 2024 find a solution to judge non-interacted 23456-opt
                                        atomicExch(&(nn_source.densityMap[0][node1]), codekopt); // WB.Q 2024 densityMap only mark the k value of k-opt, and mark which mode of k-exchange

                                    }
                                }

                            }

                        }//end if i j

                    }//end if row <= width

                }

            }


        }

    }
    __syncthreads();
}// end K_2optOneThreadOne3opt



/*!
 * \brief 202408 QWB: add parallel 3-opt with rocki's method, using sharedArrayOccupied produces small quantity of opts
 */
//epecially for small size, copy all cities into shared memory
template <template<typename, typename> class NeuralNetLinks, class BufferDimension, class Point>
KERNEL void K_3opt_oneThreadOne3opt_rockiSmall_iterStride_sharedOccupy(NeuralNetLinks<BufferDimension, Point> nn_source,
                                                                       Grid<doubleLinkedEdgeForTSP> arrayTSP,
                                                                       double maxChecks3opt,  double maxChecksoptDivide,
                                                                       double iter, double istride)
{

    double local_id = threadIdx.x + blockIdx.x * blockDim.x;

    int width =  nn_source.adaptiveMap.width; // each thread has this register


    __shared__ doubleLinkedEdgeForTSP sharedArrayTSP[SHAREDMAXCITIES];
    __shared__ float sharedArrayOccupied[SHAREDMAXCITIES];

    float iterShared = (float)width / (float)BLOCKSIZE;

    for(int opt = 0; opt < iterShared; opt++)
    {
        int m = threadIdx.x + opt*BLOCKSIZE;
        if(m < width)
        {
            sharedArrayTSP[m].current = arrayTSP[0][m].current;
            sharedArrayTSP[m].currentCoord[0] = arrayTSP[0][m].currentCoord[0];
            sharedArrayTSP[m].currentCoord[1] = arrayTSP[0][m].currentCoord[1];
        }
        __syncthreads();
    }


    if(local_id < maxChecks3opt)
    {

        double startId = maxChecksoptDivide * (istride);

        if(local_id == 0)
            printf("StartID %f, local_id %f \n", startId, local_id);


        for(double id = local_id*iter ; id < (local_id+1)*(iter); id++)
        {

            id = id + startId;

            if(id > 0 && id < maxChecks3opt)
            {

                //                if(id > 350631671)
                //                    printf("id %f, local_id %f \n", id, local_id);

                int row, i, j;
                double subtriplicate = (1.0)/3;

                {

                    //WB.Q this way will produce i = j
                    double idid = 9*id*id - (1.0)/9 ;
                    double idMul3 = 3*id;
                    double rowN0 = idMul3 + sqrt(idid );
                    double rowN1 = pow(rowN0, subtriplicate);
                    double rowN2 = idMul3 - sqrt(idid);
                    double rowN3 = pow(rowN2, subtriplicate);
                    double rowN4 = rowN1 + rowN3 + 1;


                    if(row <= width)
                    {

                        double tempRowRow = (double)(row-1) / 6;
                        double tempRowRowRow = tempRowRow*(row)*(row-2);
                        //                        double tempRowRow;
                        //                        double tempRowRowRow;

                        //                        if(iter > 1)
                        //                        {
                        //                            tempRowRow = (double)(row-1) / 6;
                        //                            tempRowRowRow = tempRowRow*(row)*(row-2);
                        //                        }
                        //                        else
                        //                        {
                        //                            tempRowRow = (double)(row-1) / 6;
                        //                            tempRowRowRow = tempRowRow*(row)*(row+1);

                        //                        }

                        double id2opt = fabs( id - tempRowRowRow );
                        //WB.Q this way will produce i = j
                        double sqrtTemp = 8.0 * (double)id2opt + 1.0;
                        i = int(3 + sqrt(sqrtTemp)) / 2 ;
                        j = id2opt - (i-2)*(i-1)/2 + 1;


                        row = int(rowN4);// check which one works


                        if(i<row && i!=row &&i+1!= row && j > 0 && j < i && j-1 >= 0 && j <= width && j+1 != i && j+ width != i+1 && i-1 >= 0 && i < width-1)
                        {

                            bool existingCandidate = 0;


                            if(sharedArrayOccupied[row-1] == 1 || sharedArrayOccupied[j-1] == 1 || sharedArrayOccupied[i-1] == 1)
                                existingCandidate = 1;

                            //                            if(sharedArrayTSP[row-1].occupied == 1 || sharedArrayTSP[j-1].occupied == 1 || sharedArrayTSP[i-1].occupied == 1)
                            //                                existingCandidate = 1;


                            //                              if(j> 1960)//350631671)
                            //                                printf("largeRow %f, local_id %f, idid %f \n idMul3 %f, rowN0 %f , rowN3 %f, rowN4 %f , i %d, j %d, id2opt %f, sqrtTemp %f \n", id, local_id, idid, idMul3, rowN0, rowN3, rowN4, i, j, id2opt, sqrtTemp);



                            if(existingCandidate == 0)
                            {

                                double newLength[4];
                                double oldLength = dist(j-1, j, sharedArrayTSP) + dist(i-1, i, sharedArrayTSP) + dist(row-1, row, sharedArrayTSP);
                                newLength[0] = dist(j-1, i, sharedArrayTSP) + dist(row-1, i-1, sharedArrayTSP) + dist(row, j, sharedArrayTSP);
                                newLength[1] = dist(j-1, row-1,sharedArrayTSP) + dist(row, i-1, sharedArrayTSP) + dist(i,j,sharedArrayTSP);
                                newLength[2] = dist(j-1, i-1, sharedArrayTSP) + dist(row-1, j, sharedArrayTSP) + dist(row, i,sharedArrayTSP);
                                newLength[3] = dist(j-1, i, sharedArrayTSP) + dist(row-1, j,sharedArrayTSP) + dist(row,i-1,sharedArrayTSP);

                                if(j> 1970)//350631671)
                                    printf("largeRow %f, local_id %f, idid %f \n idMul3 %f, rowN0 %f , rowN3 %f, rowN4 %f , i %d, j %d, id2opt %f, sqrtTemp %f , shared %d\n",
                                           id, local_id, idid, idMul3, rowN0, rowN3, rowN4, i, j, id2opt, sqrtTemp, sharedArrayTSP[j-1].current);



                                int finalSelect = -1;
                                double optimiz = -INFINITY;
                                for(int i = 0; i < 4; i++)
                                {
                                    float opti = oldLength - newLength[i];
                                    if(opti > 0 && opti > optimiz)
                                    {
                                        finalSelect = i;
                                        optimiz = opti;

                                        sharedArrayOccupied[i-1] = 1;
                                        sharedArrayOccupied[j-1] =1;
                                        sharedArrayOccupied[row-1]= 1;


                                    }
                                }


                                if(finalSelect >= 0)
                                {

                                    unsigned int node1 = (int)sharedArrayTSP[j-1].current;
                                    unsigned int node3 = (int)sharedArrayTSP[i-1].current;
                                    unsigned int node5 = (int)sharedArrayTSP[row-1].current;

                                    {

                                        unsigned long long result = 0;
                                        result = result | node3;
                                        result = result << 16;
                                        result = result | node5;

                                        float codekopt = finalSelect; //WB.Q 2024 densityMap only mark the k value of k-opt, and mark which mode of k-exchange
                                        codekopt = finalSelect * 100 + 3;

                                        atomicExch(&(nn_source.optCandidateMap[0][node1]), result); // WB.Q 2024 find a solution to judge non-interacted 23456-opt
                                        atomicExch(&(nn_source.densityMap[0][node1]), codekopt); // WB.Q 2024 densityMap only mark the k value of k-opt, and mark which mode of k-exchange


                                    }
                                }

                            }

                        }//end if i j

                    }//end if row <= width

                }

            }


        }

    }
    __syncthreads();
}// end K_2optOneThreadOne3opt






/*!
 * \brief 202408 QWB: add parallel 3-opt with rocki's method one node only participates one candidates work correctly final version
 */
//epecially for small size, copy all cities into shared memory
template <template<typename, typename> class NeuralNetLinks, class BufferDimension, class Point>
KERNEL void K_3opt_oneThreadOne3opt_rockiSmall_iterStride_onlySharedOccupy(NeuralNetLinks<BufferDimension, Point> nn_source,
                                                                           Grid<doubleLinkedEdgeForTSP> arrayTSP,
                                                                           double maxChecks3opt,  double maxChecksoptDivide,
                                                                           double iter, double istride)
{

    double local_id = threadIdx.x + blockIdx.x * blockDim.x;

    int width =  nn_source.adaptiveMap.width; // each thread has this register


    //    __shared__ doubleLinkedEdgeForTSP sharedArrayTSP[SHAREDMAXCITIES];
    __shared__ float sharedArrayOccupied[SHAREDMAXCITIES];


    //    printf("local_id %f, shared %d, shareOcuu %f \n",
    //           local_id,  sharedArrayTSP[1023].current, sharedArrayTSP[1023].occupied);



    if(local_id < maxChecks3opt)
    {

        double startId = maxChecksoptDivide * (istride);

        if(local_id == 0)
            printf("StartID %f, local_id %f \n", startId, local_id);


        //        __shared__ doubleLinkedEdgeForTSP sharedArrayTSP[SHAREDMAXCITIES];

        //        float iterShared = (float)width / (float)BLOCKSIZE;

        //        for(int opt = 0; opt < iterShared; opt++)

        //        {
        //            int m = threadIdx.x + opt*BLOCKSIZE;
        //            if(m < width)
        //            {
        //                sharedArrayTSP[m].current = arrayTSP[0][m].current;
        //                sharedArrayTSP[m].currentCoord[0] = arrayTSP[0][m].currentCoord[0];
        //                sharedArrayTSP[m].currentCoord[1] = arrayTSP[0][m].currentCoord[1];
        //            }

        //            __syncthreads();
        //        }



        for(double id = local_id*iter ; id < (local_id+1)*(iter); id++)
        {

            id = id + startId;

            if(id > 0 && id < maxChecks3opt)
            {

                //                if(id > 350631671)
                //                    printf("id %f, local_id %f \n", id, local_id);

                int row, i, j;
                double subtriplicate = (1.0)/3;

                // if(id > 350631671)
                {

                    //WB.Q this way will produce i = j
                    double idid = 9*id*id - (1.0)/9 ;
                    double idMul3 = 3*id;
                    double rowN0 = idMul3 + sqrt(idid );
                    double rowN1 = pow(rowN0, subtriplicate);
                    double rowN2 = idMul3 - sqrt(idid);
                    double rowN3 = pow(rowN2, subtriplicate);
                    double rowN4 = rowN1 + rowN3 + 1;


                    if(row <= width)
                    {

                        double tempRowRow = (double)(row-1) / 6;
                        double tempRowRowRow = tempRowRow*(row)*(row-2);
                        //                        double tempRowRow;
                        //                        double tempRowRowRow;

                        //                        if(iter > 1)
                        //                        {
                        //                            tempRowRow = (double)(row-1) / 6;
                        //                            tempRowRowRow = tempRowRow*(row)*(row-2);
                        //                        }
                        //                        else
                        //                        {
                        //                            tempRowRow = (double)(row-1) / 6;
                        //                            tempRowRowRow = tempRowRow*(row)*(row+1);

                        //                        }

                        double id2opt = fabs( id - tempRowRowRow );
                        //WB.Q this way will produce i = j
                        double sqrtTemp = 8.0 * (double)id2opt + 1.0;
                        i = int(3 + sqrt(sqrtTemp)) / 2 ;
                        j = id2opt - (i-2)*(i-1)/2 + 1;


                        row = int(rowN4);// check which one works


                        if(i<row && i!=row &&i+1!= row && j > 0 && j < i && j-1 >= 0 && j <= width && j+1 != i && j+ width != i+1 && i-1 >= 0 && i < width-1)
                        {

                            bool existingCandidate = 0;

                            //                            if(nn_source.minRadiusMap[0][row-1] == 1 || nn_source.minRadiusMap[0][j-1] == 1 ||nn_source.minRadiusMap[0][i-1] == 1)
                            //                                existingCandidate = 1;

                            if(sharedArrayOccupied[row-1] == 1 || sharedArrayOccupied[j-1] == 1 || sharedArrayOccupied[i-1] == 1)
                                existingCandidate = 1;

                            //                            if(sharedArrayTSP[row-1].occupied == 1 || sharedArrayTSP[j-1].occupied == 1 || sharedArrayTSP[i-1].occupied == 1)
                            //                                existingCandidate = 1;


                            //  if(j> 9145)//350631671)
                            //    printf("largeRow %f, local_id %f, idid %f \n idMul3 %f, rowN0 %f , rowN3 %f, rowN4 %f , i %d, j %d, id2opt %f, sqrtTemp %f \n", id, local_id, idid, idMul3, rowN0, rowN3, rowN4, i, j, id2opt, sqrtTemp);



                            if(existingCandidate == 0)
                            {

                                double newLength[4];
                                double oldLength = dist(j-1, j, arrayTSP[0]) + dist(i-1, i, arrayTSP[0]) + dist(row-1, row, arrayTSP[0]);
                                newLength[0] = dist(j-1, i, arrayTSP[0]) + dist(row-1, i-1, arrayTSP[0]) + dist(row, j, arrayTSP[0]);
                                newLength[1] = dist(j-1, row-1,arrayTSP[0]) + dist(row, i-1, arrayTSP[0]) + dist(i,j,arrayTSP[0]);
                                newLength[2] = dist(j-1, i-1, arrayTSP[0]) + dist(row-1, j, arrayTSP[0]) + dist(row, i,arrayTSP[0]);
                                newLength[3] = dist(j-1, i, arrayTSP[0]) + dist(row-1, j,arrayTSP[0]) + dist(row,i-1,arrayTSP[0]);

                                //  printf("largeRow %f, local_id %f, idid %f \n idMul3 %f, rowN0 %f , rowN3 %f, rowN4 %f , i %d, j %d, id2opt %f, sqrtTemp %f , shared %d, shareOcuu %f \n",
                                //      id, local_id, idid, idMul3, rowN0, rowN3, rowN4, i, j, id2opt, sqrtTemp, sharedArrayTSP[j-1].current, sharedArrayTSP[row-1].occupied);



                                int finalSelect = -1;
                                double optimiz = -INFINITY;
                                for(int i = 0; i < 4; i++)
                                {
                                    float opti = oldLength - newLength[i];
                                    if(opti > 0 && opti > optimiz)
                                    {
                                        finalSelect = i;
                                        optimiz = opti;

                                        //   atomicExch(&(nn_source.minRadiusMap[0][i-1]), 1);
                                        //   atomicExch(&(nn_source.minRadiusMap[0][j-1]), 1);
                                        //   atomicExch(&(nn_source.minRadiusMap[0][row-1]), 1);


                                        atomicExch(&(sharedArrayOccupied[i-1]), 1);
                                        atomicExch(&(sharedArrayOccupied[j-1]), 1);
                                        atomicExch(&(sharedArrayOccupied[row-1]), 1);


                                        //                                           __syncthreads();

                                        //                                        sharedArrayOccupied[i-1] = 1;
                                        //                                        sharedArrayOccupied[j-1] =1;
                                        //                                        sharedArrayOccupied[row-1]= 1;

                                        //  sharedArrayTSP[i-1].occupied = 1;
                                        //  sharedArrayTSP[j-1].occupied =1;
                                        //  sharedArrayTSP[row-1].occupied= 1;

                                    }
                                }


                                if(finalSelect >= 0)
                                {

                                    unsigned int node1 = (int)arrayTSP[0][j-1].current;
                                    unsigned int node3 = (int)arrayTSP[0][i-1].current;
                                    unsigned int node5 = (int)arrayTSP[0][row-1].current;

                                    {

                                        unsigned long long result = 0;
                                        result = result | node3;
                                        result = result << 16;
                                        result = result | node5;

                                        float codekopt = finalSelect; //WB.Q 2024 densityMap only mark the k value of k-opt, and mark which mode of k-exchange
                                        codekopt = finalSelect * 100 + 3;

                                        atomicExch(&(nn_source.optCandidateMap[0][node1]), result); // WB.Q 2024 find a solution to judge non-interacted 23456-opt
                                        atomicExch(&(nn_source.densityMap[0][node1]), codekopt); // WB.Q 2024 densityMap only mark the k value of k-opt, and mark which mode of k-exchange

                                    }
                                }

                            }

                        }//end if i j

                    }//end if row <= width

                }

            }


        }

    }
    __syncthreads();
}// end K_2optOneThreadOne3opt



/*!
 * \brief 202408 QWB: add parallel 3-opt with rocki's method one node only participates one candidates work correctly final version
 */
//epecially for small size, copy all cities into shared memory
template <template<typename, typename> class NeuralNetLinks, class BufferDimension, class Point>
KERNEL void K_3opt_oneThreadOne3opt_rockiSmall_iterStrideBest(NeuralNetLinks<BufferDimension, Point> nn_source,
                                                              Grid<doubleLinkedEdgeForTSP> arrayTSP,
                                                              double maxChecks3opt,  double maxChecksoptDivide,
                                                              double iter, double istride)
{

    double local_id = threadIdx.x + blockIdx.x * blockDim.x;

    int width =  nn_source.adaptiveMap.width; // each thread has this register

    if(local_id < maxChecks3opt)
    {

        double startId = maxChecksoptDivide * (istride);

        if(local_id == 0)
            printf("StartID %f, local_id %f \n", startId, local_id);


        for(double id = local_id*iter ; id < (local_id+1)*(iter); id++)
        {

            id = id + startId;

            if(id > 0 && id < maxChecks3opt)
            {

                int row, i, j;
                double subtriplicate = (1.0)/3;

                {

                    //WB.Q this way will produce i = j
                    double idid = 9*id*id - (1.0)/9 ;
                    double idMul3 = 3*id;
                    double rowN0 = idMul3 + sqrt(idid );
                    double rowN1 = pow(rowN0, subtriplicate);
                    double rowN2 = idMul3 - sqrt(idid);
                    double rowN3 = pow(rowN2, subtriplicate);
                    double rowN4 = rowN1 + rowN3 + 1;


                    if(row <= width)
                    {
                        double tempRowRow = (double)(row-1) / 6;
                        double tempRowRowRow = tempRowRow*(row)*(row-2);
                        double id2opt = fabs( id - tempRowRowRow );
                        //WB.Q this way will produce i = j
                        double sqrtTemp = 8.0 * (double)id2opt + 1.0;
                        i = int(3 + sqrt(sqrtTemp)) / 2 ;
                        j = id2opt - (i-2)*(i-1)/2 + 1;


                        row = int(rowN4);// check which one works
                        //  printf("largeRow %f, local_id %f, idid %f \n idMul3 %f, rowN0 %f , rowN3 %f, rowN4 %f , i %d, j %d, id2opt %f, sqrtTemp %f \n", id, local_id, idid, idMul3, rowN0, rowN3, rowN4, i, j, id2opt, sqrtTemp);


                        if(i<row && i!=row &&i+1!= row && j > 0 && j < i && j-1 >= 0 && j <= width && j+1 != i && j+ width != i+1 && i-1 >= 0 && i < width-1)
                        {

                            double newLength[4];

                            double oldLength = dist(j-1, j, arrayTSP[0]) + dist(i-1, i, arrayTSP[0]) + dist(row-1, row, arrayTSP[0]);
                            newLength[0] = dist(j-1, i, arrayTSP[0]) + dist(row-1, i-1, arrayTSP[0]) + dist(row, j, arrayTSP[0]);
                            newLength[1] = dist(j-1, row-1,arrayTSP[0]) + dist(row, i-1, arrayTSP[0]) + dist(i,j,arrayTSP[0]);
                            newLength[2] = dist(j-1, i-1, arrayTSP[0]) + dist(row-1, j, arrayTSP[0]) + dist(row, i,arrayTSP[0]);
                            newLength[3] = dist(j-1, i, arrayTSP[0]) + dist(row-1, j,arrayTSP[0]) + dist(row,i-1,arrayTSP[0]);

                            int finalSelect = -1;
                            double optimiz = -INFINITY;
                            for(int i = 0; i < 4; i++)
                            {
                                float opti = oldLength - newLength[i];
                                if(opti > 0 && opti > optimiz)
                                {
                                    finalSelect = i;
                                    optimiz = opti;
                                }
                            }


                            if(finalSelect >= 0)
                            {

                                unsigned int node1 = (int)arrayTSP[0][j-1].current;
                                unsigned int node3 = (int)arrayTSP[0][i-1].current;
                                unsigned int node5 = (int)arrayTSP[0][row-1].current;

                                double localMinChange = nn_source.minRadiusMap[0][node1];
                                double optimization = oldLength - newLength[finalSelect];

                                if(optimization > localMinChange)
                                {

                                    unsigned long long result = 0;
                                    result = result | node3;
                                    result = result << 16;
                                    result = result | node5;

                                    float codekopt = finalSelect; //WB.Q 2024 densityMap only mark the k value of k-opt, and mark which mode of k-exchange
                                    codekopt = finalSelect * 100 + 3;

                                    atomicExch(&(nn_source.optCandidateMap[0][node1]), result); // WB.Q 2024 find a solution to judge non-interacted 23456-opt
                                    atomicExch(&(nn_source.densityMap[0][node1]), codekopt); // WB.Q 2024 densityMap only mark the k value of k-opt, and mark which mode of k-exchange
                                    atomicExch(&(nn_source.minRadiusMap[0][node1]), optimization);

                                }
                            }

                        }//end if i j

                    }//end if row <= width

                }

            }

        }

    }
    __syncthreads();
}// end K_2optOneThreadOne3opt



/*!
 * \brief 202408 QWB: add parallel 3-opt with rocki's method one node only participates one candidates work correctly final version
 */
//epecially for small size, copy all cities into shared memory
template <template<typename, typename> class NeuralNetLinks, class BufferDimension, class Point>
KERNEL void K_3opt_oneThreadOne3opt_rockiSmall_iterStrideBest_shared(NeuralNetLinks<BufferDimension, Point> nn_source,
                                                                     Grid<doubleLinkedEdgeForTSP> arrayTSP,
                                                                     double maxChecks3opt,  double maxChecksoptDivide,
                                                                     double iter, double istride)
{

    double local_id = threadIdx.x + blockIdx.x * blockDim.x;

    int width =  nn_source.adaptiveMap.width; // each thread has this register

    __shared__ doubleLinkedEdgeForTSP sharedArrayTSP[SHAREDMAXCITIES];
    //    __shared__ float sharedArrayOccupied[SHAREDMAXCITIES];

    float iterShared = (float)width / (float)BLOCKSIZE;

    for(int opt = 0; opt < iterShared; opt++)
    {
        int m = threadIdx.x + opt*BLOCKSIZE;
        if(m < width)
        {
            sharedArrayTSP[m].current = arrayTSP[0][m].current;
            sharedArrayTSP[m].currentCoord[0] = arrayTSP[0][m].currentCoord[0];
            sharedArrayTSP[m].currentCoord[1] = arrayTSP[0][m].currentCoord[1];
            //            sharedArrayOccupied[m] = nn_source.minRadiusMap[0][m];
        }
        __syncthreads();
    }




    if(local_id < maxChecks3opt)
    {

        double startId = maxChecksoptDivide * (istride);

        if(local_id == 0)
            printf("StartID %f, local_id %f \n", startId, local_id);


        for(double id = local_id*iter ; id < (local_id+1)*(iter); id++)
        {

            id = id + startId;

            if(id > 0 && id < maxChecks3opt)
            {

                int row, i, j;
                double subtriplicate = (1.0)/3;

                {

                    //WB.Q this way will produce i = j
                    double idid = 9*id*id - (1.0)/9 ;
                    double idMul3 = 3*id;
                    double rowN0 = idMul3 + sqrt(idid );
                    double rowN1 = pow(rowN0, subtriplicate);
                    double rowN2 = idMul3 - sqrt(idid);
                    double rowN3 = pow(rowN2, subtriplicate);
                    double rowN4 = rowN1 + rowN3 + 1;


                    if(row <= width)
                    {
                        double tempRowRow = (double)(row-1) / 6;
                        double tempRowRowRow = tempRowRow*(row)*(row-2);
                        double id2opt = fabs( id - tempRowRowRow );
                        //WB.Q this way will produce i = j
                        double sqrtTemp = 8.0 * (double)id2opt + 1.0;
                        i = int(3 + sqrt(sqrtTemp)) / 2 ;
                        j = id2opt - (i-2)*(i-1)/2 + 1;


                        row = int(rowN4);// check which one works
                        //  printf("largeRow %f, local_id %f, idid %f \n idMul3 %f, rowN0 %f , rowN3 %f, rowN4 %f , i %d, j %d, id2opt %f, sqrtTemp %f \n", id, local_id, idid, idMul3, rowN0, rowN3, rowN4, i, j, id2opt, sqrtTemp);


                        if(i<row && i!=row &&i+1!= row && j > 0 && j < i && j-1 >= 0 && j <= width && j+1 != i && j+ width != i+1 && i-1 >= 0 && i < width-1)
                        {

                            double newLength[4];

                            double oldLength = dist(j-1, j, sharedArrayTSP) + dist(i-1, i, sharedArrayTSP) + dist(row-1, row, sharedArrayTSP);
                            newLength[0] = dist(j-1, i, sharedArrayTSP) + dist(row-1, i-1, sharedArrayTSP) + dist(row, j, sharedArrayTSP);
                            newLength[1] = dist(j-1, row-1,sharedArrayTSP) + dist(row, i-1, sharedArrayTSP) + dist(i,j,sharedArrayTSP);
                            newLength[2] = dist(j-1, i-1, sharedArrayTSP) + dist(row-1, j, sharedArrayTSP) + dist(row, i,sharedArrayTSP);
                            newLength[3] = dist(j-1, i, sharedArrayTSP) + dist(row-1, j,sharedArrayTSP) + dist(row,i-1,sharedArrayTSP);

                            int finalSelect = -1;
                            double optimiz = -INFINITY;
                            for(int i = 0; i < 4; i++)
                            {
                                float opti = oldLength - newLength[i];
                                if(opti > 0 && opti > optimiz)
                                {
                                    finalSelect = i;
                                    optimiz = opti;
                                }
                            }


                            if(finalSelect >= 0)
                            {

                                unsigned int node1 = (int)sharedArrayTSP[j-1].current;
                                unsigned int node3 = (int)sharedArrayTSP[i-1].current;
                                unsigned int node5 = (int)sharedArrayTSP[row-1].current;

                                double localMinChange = nn_source.minRadiusMap[0][node1];
                                double optimization = oldLength - newLength[finalSelect];

                                if(optimization > localMinChange)
                                {

                                    unsigned long long result = 0;
                                    result = result | node3;
                                    result = result << 16;
                                    result = result | node5;

                                    float codekopt = finalSelect; //WB.Q 2024 densityMap only mark the k value of k-opt, and mark which mode of k-exchange
                                    codekopt = finalSelect * 100 + 3;

                                    atomicExch(&(nn_source.optCandidateMap[0][node1]), result); // WB.Q 2024 find a solution to judge non-interacted 23456-opt
                                    atomicExch(&(nn_source.densityMap[0][node1]), codekopt); // WB.Q 2024 densityMap only mark the k value of k-opt, and mark which mode of k-exchange
                                    atomicExch(&(nn_source.minRadiusMap[0][node1]), optimization);

                                }
                            }

                        }//end if i j

                    }//end if row <= width

                }

            }

        }

    }
    __syncthreads();
}// end K_2optOneThreadOne3opt

/*!
 * \brief 202408 QWB: add parallel 3-opt with rocki's method one node only participates one candidates
 */
//epecially for small size, copy all cities into shared memory
template <template<typename, typename> class NeuralNetLinks, class BufferDimension, class Point>
KERNEL void K_3opt_oneThreadOne3opt_rockiSmall_iter_double(NeuralNetLinks<BufferDimension, Point> nn_source,
                                                           Grid<doubleLinkedEdgeForTSP> arrayTSP,
                                                           double maxChecks3opt,
                                                           double iter)
{

    double local_id = threadIdx.x + blockIdx.x * blockDim.x;

    int width =  nn_source.adaptiveMap.width; // each thread has this register

    if(local_id < maxChecks3opt)
    {

        int packSize = blockDim.x * gridDim.x;
        int row, i, j;
        double subtriplicate = (1.0)/3;

        for(int nu = 0; nu <= iter; nu++)
        {

            double id = local_id + nu * packSize;

            if(id < maxChecks3opt)
            {

                //WB.Q this way will produce i = j
                double idid = 9*id*id;
                double idMul3 = 3*id;
                double rowN0 = idMul3 + sqrt(idid - (1.0)/9);
                double rowN1 = pow(rowN0, subtriplicate);
                double rowN2 = idMul3 - sqrt(idid - (1.0)/9);
                float rowN3 = pow(rowN2, subtriplicate);
                float rowN4 = rowN1 + rowN3 + 1;

                row = int(rowN4);// check which one works

                if(row <= width)
                {

                    //                    double rowrowrow = (double)(row-1) / (double )6;
                    //                    double id2opt = 8.0 * (rowrowrow *(row)*(row+1) - id );
                    //                    //WB.Q this way will produce i = j
                    //                    i = int(3 + sqrt(id2opt + 1.0) ) / 2 ;
                    //                    j = int(id2opt - (i-2)*(i-1)/2 + 1);


                    double id2opt = (row-1)*(row)*(row+1)/6 - id;
                    //WB.Q this way will produce i = j
                    i = int(3 + sqrt(8.0 * (double)id2opt + 1.0)) / 2 ;
                    j = id2opt - (i-2)*(i-1)/2 + 1;


                    //                    //qiao only for test
                    //                    if(id == maxChecks3opt - 10)
                    //                        printf("3-opt maxmum row j= %d, i %d, row %d, id % , localid %f, nu %d \n", j, i, row, id, local_id, nu);


                    if(i<row && i!=row &&i+1!= row &&j > 0 && j < i && j-1 >= 0 && j <= width && j+1 != i && j+ width != i+1 && i-1 >= 0 && i < width-1)
                    {

                        //                        //qiao only for test
                        //                        if(nn_source.grayValueMap[0][row-1] > 5000 || nn_source.grayValueMap[0][i-1] > 5000 || nn_source.grayValueMap[0][j-1] > 5000)
                        //                            printf("3-opt maxmum row id %d, row %d, i %d, j %d \n", id, row, i,j);

                        //                        //qiao only for test
                        //                        if(row == width - 10)
                        //                            printf("3-opt maxmum j %d, i %d, row %d,  \n", j, i, row);


                        bool existingCandidate = 0;
                        if(nn_source.minRadiusMap[0][row-1] == 1 || nn_source.minRadiusMap[0][j-1] == 1 ||nn_source.minRadiusMap[0][i-1] == 1)
                            existingCandidate = 1;


                        if(existingCandidate == 0)
                        {

                            double newLength[4];

                            double oldLength = dist(j-1, j, arrayTSP[0]) + dist(i-1, i, arrayTSP[0]) + dist(row-1, row, arrayTSP[0]);
                            newLength[0] = dist(j-1, i, arrayTSP[0]) + dist(row-1, i-1, arrayTSP[0]) + dist(row, j, arrayTSP[0]);
                            newLength[1] = dist(j-1, row-1,arrayTSP[0]) + dist(row, i-1, arrayTSP[0]) + dist(i,j,arrayTSP[0]);
                            newLength[2] = dist(j-1, i-1, arrayTSP[0]) + dist(row-1, j, arrayTSP[0]) + dist(row, i,arrayTSP[0]);
                            newLength[3] = dist(j-1, i, arrayTSP[0]) + dist(row-1, j,arrayTSP[0]) + dist(row,i-1,arrayTSP[0]);

                            int finalSelect = -1;
                            double optimiz = -INFINITY;
                            for(int i = 0; i < 4; i++)
                            {
                                float opti = oldLength - newLength[i];
                                if(opti > 0 && opti > optimiz)
                                {
                                    finalSelect = i;
                                    optimiz = opti;

                                    //                               if(blockIdx.x == 124698010  ||blockIdx.x == 124698013   || blockIdx.x == 0)
                                    //                                printf("3opt GPU blockIdx.x=%d, selec %d, order %d, %d, %d, oldLength %f, newi %f, opti %f, optimiz %f ; node135 %d,%d,%d\n",blockIdx.x, finalSelect,
                                    //                                       nn_source.grayValueMap[0][j-1], nn_source.grayValueMap[0][i-1], nn_source.grayValueMap[0][row-1]
                                    //                                        , oldLength, newLength[i], opti, optimiz, j-1, i-1, row-1);

                                    atomicExch(&(nn_source.minRadiusMap[0][i-1]), 1);
                                    atomicExch(&(nn_source.minRadiusMap[0][j-1]), 1);
                                    atomicExch(&(nn_source.minRadiusMap[0][row-1]), 1);

                                }
                            }


                            if(finalSelect >= 0)
                            {
                                //                            float optimization = oldLength - newLength[finalSelect];
                                // here automic operation is necessary

                                unsigned int node1 = (int)arrayTSP[0][j-1].current;
                                unsigned int node3 = (int)arrayTSP[0][i-1].current;
                                unsigned int node5 = (int)arrayTSP[0][row-1].current;

                                //                            printf("3opt GPU mode %d, order %d, %d, %d, oldLength %f, new1 %f, new2 %f, new3 %f, new4 %f; node135 %d,%d,%d \n", finalSelect, nn_source.grayValueMap[0][node1], nn_source.grayValueMap[0][node3], nn_source.grayValueMap[0][node5]
                                //                                    , oldLength, newLength[0],newLength[1], newLength[2], newLength[3], node1, node3, node5);

                                //                            float localMinChange = nn_source.minRadiusMap[0][node1];

                                //                            if(optimization > localMinChange)
                                {

                                    unsigned long long result = 0;
                                    result = result | node3;
                                    result = result << 16;
                                    result = result | node5;

                                    float codekopt = finalSelect; //WB.Q 2024 densityMap only mark the k value of k-opt, and mark which mode of k-exchange
                                    codekopt = finalSelect * 100 + 3;

                                    atomicExch(&(nn_source.optCandidateMap[0][node1]), result); // WB.Q 2024 find a solution to judge non-interacted 23456-opt
                                    atomicExch(&(nn_source.densityMap[0][node1]), codekopt); // WB.Q 2024 densityMap only mark the k value of k-opt, and mark which mode of k-exchange
                                    //                                atomicExch(&(nn_source.minRadiusMap[0][node1]), optimization);
                                }
                            }

                        }



                    }//end if i j

                }//end if row <= width

            }
        }


    }
    __syncthreads();
}// end K_2optOneThreadOne3opt



/*!
 * \brief 202408 QWB: add parallel 3-opt with rocki's method one node only participates one candidates
 */
//epecially for small size, copy all cities into shared memory
template <template<typename, typename> class NeuralNetLinks, class BufferDimension, class Point>
KERNEL void K_3opt_oneThreadOne3opt_rockiSmall(NeuralNetLinks<BufferDimension, Point> nn_source,
                                               Grid<doubleLinkedEdgeForTSP> arrayTSP,
                                               double maxChecks3opt,
                                               double iter)
{

    double id = threadIdx.x + blockIdx.x * blockDim.x;

    int width =  nn_source.adaptiveMap.width; // each thread has this register

    int row, i, j;
    double subtriplicate = (1.0)/3;


    if(id > maxChecks3opt-5)//350631671)
        printf("largeID id %f \n",  id);
    else if (id <5)
        printf("SmallID id %f \n", id);



    if(id < maxChecks3opt)
    {

        //WB.Q this way will produce i = j
        double idid = 9*id*id;
        double idMul3 = 3*id;
        double rowN0 = idMul3 + sqrt(idid - (1.0)/9);
        double rowN1 = pow(rowN0, subtriplicate);
        double rowN2 = idMul3 - sqrt(idid - (1.0)/9);
        double rowN3 = pow(rowN2, subtriplicate);
        float rowN4 = rowN1 + rowN3 + 1;

        row = int(rowN4);// check which one works

        if(row <= width)
        {
            //            double rowrowrow = (row-1)*(row)*(row+1)/6  - id;
            //            double id2opt = 8.0 * (rowrowrow) + 1.0;
            //            double sqrtId2opt = 3 + sqrt(id2opt) ;

            //            //WB.Q this way will produce i = j
            //            i = (sqrtId2opt) / 2 ; // i = ((sqrtId2opt) / 2) ;// i = (int)((sqrtId2opt) / 2) ; error
            //            j = id2opt - (i-2)*(i-1)/2 + 1;


            double id2opt = (row-1)*(row)*(row+1)/6 - id;
            double sqrtTemp = 8.0 * (double)id2opt + 1.0;
            double sqrtId2opt = 3 + sqrt(sqrtTemp);
            i = (sqrtId2opt) / 2 ;
            j = id2opt - (i-2)*(i-1)/2 + 1;


            //            double id2opt = (row-1)*(row)*(row+1)/6 - id;
            //            //WB.Q this way will produce i = j
            //            i = int(3 + sqrt(8.0 * (double)id2opt + 1.0)) / 2 ;
            //            j = id2opt - (i-2)*(i-1)/2 + 1;




            //            //qiao only for test
            //            if(row > width -10  ||id > maxChecks3opt - 10)
            //                printf("3-opt maxmum  row %d, id %lld \n", row, id);

            if(i<row && i!=row &&i+1!= row &&j > 0 && j < i && j-1 >= 0 && j <= width && j+1 != i && j+ width != i+1 && i-1 >= 0 && i < width-1)
            {

                //                        //qiao only for test
                //                        if(nn_source.grayValueMap[0][row-1] > 5000 || nn_source.grayValueMap[0][i-1] > 5000 || nn_source.grayValueMap[0][j-1] > 5000)
                //                            printf("3-opt maxmum row id %d, row %d, i %d, j %d \n", id, row, i,j);

                //                        //qiao only for test
                //                        if(row > width -10 ||  j > width-10 || i > width -10 ||id > maxChecks3opt - 10)
                //                            printf("3-opt maxmum j %d, i %d, row %d, id %lld \n", j, i, row, id);
                printf("3-opt maxmum row j= %d, i %d, row %d, id %f \n", j, i, row, id);



                bool existingCandidate = 0;
                if(nn_source.minRadiusMap[0][row-1] == 1 || nn_source.minRadiusMap[0][j-1] == 1 ||nn_source.minRadiusMap[0][i-1] == 1)
                    existingCandidate = 1;

                if(existingCandidate == 0)
                {

                    double newLength[4];

                    double oldLength = dist(j-1, j, arrayTSP[0]) + dist(i-1, i, arrayTSP[0]) + dist(row-1, row, arrayTSP[0]);
                    newLength[0] = dist(j-1, i, arrayTSP[0]) + dist(row-1, i-1, arrayTSP[0]) + dist(row, j, arrayTSP[0]);
                    newLength[1] = dist(j-1, row-1,arrayTSP[0]) + dist(row, i-1, arrayTSP[0]) + dist(i,j,arrayTSP[0]);
                    newLength[2] = dist(j-1, i-1, arrayTSP[0]) + dist(row-1, j, arrayTSP[0]) + dist(row, i,arrayTSP[0]);
                    newLength[3] = dist(j-1, i, arrayTSP[0]) + dist(row-1, j,arrayTSP[0]) + dist(row,i-1,arrayTSP[0]);

                    int finalSelect = -1;
                    double optimiz = -INFINITY;
                    for(int i = 0; i < 4; i++)
                    {
                        float opti = oldLength - newLength[i];
                        if(opti > 0 && opti > optimiz)
                        {
                            finalSelect = i;
                            optimiz = opti;

                            //                               if(blockIdx.x == 124698010  ||blockIdx.x == 124698013   || blockIdx.x == 0)
                            //                                printf("3opt GPU blockIdx.x=%d, selec %d, order %d, %d, %d, oldLength %f, newi %f, opti %f, optimiz %f ; node135 %d,%d,%d\n",blockIdx.x, finalSelect,
                            //                                       nn_source.grayValueMap[0][j-1], nn_source.grayValueMap[0][i-1], nn_source.grayValueMap[0][row-1]
                            //                                        , oldLength, newLength[i], opti, optimiz, j-1, i-1, row-1);

                            atomicExch(&(nn_source.minRadiusMap[0][i-1]), 1);
                            atomicExch(&(nn_source.minRadiusMap[0][j-1]), 1);
                            atomicExch(&(nn_source.minRadiusMap[0][row-1]), 1);

                        }
                    }


                    if(finalSelect >= 0)
                    {

                        unsigned int node1 = (int)arrayTSP[0][j-1].current;
                        unsigned int node3 = (int)arrayTSP[0][i-1].current;
                        unsigned int node5 = (int)arrayTSP[0][row-1].current;

                        unsigned long long result = 0;
                        result = result | node3;
                        result = result << 16;
                        result = result | node5;

                        float codekopt = finalSelect; //WB.Q 2024 densityMap only mark the k value of k-opt, and mark which mode of k-exchange
                        codekopt = finalSelect * 100 + 3;

                        atomicExch(&(nn_source.optCandidateMap[0][node1]), result); // WB.Q 2024 find a solution to judge non-interacted 23456-opt
                        atomicExch(&(nn_source.densityMap[0][node1]), codekopt); // WB.Q 2024 densityMap only mark the k value of k-opt, and mark which mode of k-exchange

                    }

                }

            }//end if i j

        }//end if row <= width

    }

    __syncthreads();
}// end K_2optOneThreadOne3opt



/*!
 * \brief 2408 QWB: add parallel 6-opt
 */
template <template<typename, typename> class NeuralNetLinks, class BufferDimension, class Point>
KERNEL void K_6opt_oneThreadOne6opt_rockiSmall(NeuralNetLinks<BufferDimension, Point> nn_source,
                                               Grid<doubleLinkedEdgeForTSP> arrayTSP,
                                               double maxChecks6opt, double maxChecks3opt, unsigned int iter)
{

    double id = threadIdx.x + blockIdx.x * blockDim.x;

    int width =  nn_source.adaptiveMap.width; // each thread has this register

    if(id < maxChecks6opt)
    {

        //        int packSize = blockDim.x * gridDim.x;
        double  outi, outj;
        int row,row_1, i, j, i_1, j_1;
        double subtriplicate = (1.0)/3;

        //WB.Q this way will produce i = j
        outi = (3 + sqrt(8.0f * (double )id + 1.0f)) / 2 ;
        outj = id - (outi-2)*(outi-1)/2 + 1;

        //WB.Q this way will produce i = j
        double rowN0 = 3*outi + sqrt(9*outi*outi - (1.0)/9);
        double rowN1 = pow(rowN0, subtriplicate);
        double rowN2 = 3*outi - sqrt(9*outi*outi - (1.0)/9);
        double rowN3 = pow(rowN2, subtriplicate);
        double rowN4 = rowN1 + rowN3 + 1;

        //WB.Q this way will produce i = j
        double rowN0_1 = 3*outj + sqrt(9*outj*outj - (1.0)/9);
        double rowN1_1 = pow(rowN0_1, subtriplicate);
        double rowN2_1 = 3*outj - sqrt(9*outj*outj - (1.0)/9);
        double rowN3_1 = pow(rowN2_1, subtriplicate);
        double rowN4_1 = rowN1_1 + rowN3_1 + 1;

        row = int(rowN4);// check which one works
        row_1 = int(rowN4_1);

        if(row < width && row_1 < width)
        {
            double id2opt = (row-1)*(row)*(row+1)/6 - outi;

            //WB.Q this way will produce i = j
            i = int(3 + sqrt(8.0f * (double)id2opt + 1.0f)) / 2 ;
            j = id2opt - (i-2)*(i-1)/2 + 1;

            double id2opt_1 = (row_1-1)*(row_1)*(row_1+1)/6 - outj;

            //WB.Q this way will produce i = j
            i_1 = int(3 + sqrt(8.0f * (double)id2opt_1 + 1.0f)) / 2 ;
            j_1 = id2opt_1 - (i_1-2)*(i_1-1)/2 + 1;


            //                    //qiao only for test
            //                    if(id == maxChecks6opt - 2)
            //                        printf("6-opt maxmim id %d, outi,outj:(%d,%d), row,i,j,row1,i1,j1:(%d,%d,%d,%d,%d,%d) \n",id, outi,outj, row, i,j, row_1,i_1,j_1);


            if(i<row && row_1 < j-1 && i!=row && i+1!= row && j > 0 && j < i && j-1 >= 0 && j < width && j+1 != i && j + width != i+1 && i-1 >= 0 && i < width-1
                    && i_1<row_1 && i_1!=row_1 && i_1+1!= row_1 && j_1 > 0 && j_1 < i_1 && j_1-1 >= 0 && j_1 < width && j_1+1 != i_1 && j_1 + width != i_1+1 && i_1-1 >= 0 && i_1 < width-1
                    && row_1 < row && row_1+1!=row && i > i_1 && i_1+1!=i && j> j_1 && j!=j_1+1 && i!=i_1 && j!=j_1 && i!=j_1 && j!=i_1 && i != row_1 && j!= row_1 && i_1!=row &&j_1!=row
                    )
            {

                //                        if(row > width -2)
                //                            printf("6-opt maxmim id %d, outi,outj:(%d,%d), row,i,j,row1,i1,j1:(%d,%d,%d,%d,%d,%d) \n",id, outi,outj, row, i,j, row_1,i_1,j_1);


                bool existingCandidate = 0;
                if(nn_source.minRadiusMap[0][j_1-1] == 1 || nn_source.minRadiusMap[0][i_1-1] == 1 ||
                        nn_source.minRadiusMap[0][row_1-1] == 1 ||nn_source.minRadiusMap[0][j-1] == 1 ||
                        nn_source.minRadiusMap[0][i-1] == 1 ||nn_source.minRadiusMap[0][row-1] == 1)
                    existingCandidate = 1;

                if(existingCandidate == 0)
                {

                    float oldLength = dist(j_1-1, j_1, arrayTSP[0]) + dist(i_1-1, i_1, arrayTSP[0]) + dist(row_1-1, row_1, arrayTSP[0])
                            + dist(j-1, j, arrayTSP[0]) + dist(i-1, i, arrayTSP[0]) + dist(row-1, row, arrayTSP[0])    ;

                    float newLength;
                    int array[12];
                    array[0] = j_1-1;
                    array[1] = j_1;
                    array[2] = i_1-1;
                    array[3] = i_1;
                    array[4] = row_1-1;
                    array[5] = row_1;
                    array[6] = j-1;
                    array[7] = j;
                    array[8] = i-1;
                    array[9] = i;
                    array[10] = row-1;
                    array[11] = row;

                    int finalSelect = -1;
                    float optimiz = -INFINITY;


                    for(int opt = 0; opt < 23220; opt +=12) //  6 edges 12 nodes 1935 sets 1935*12=23220 nodes
                    {
                        int nd1 = nn_source.nodeParentMap[0][opt] -1;
                        int nd2 = nn_source.nodeParentMap[0][opt+1] -1;
                        int nd3 = nn_source.nodeParentMap[0][opt+2] -1;
                        int nd4 = nn_source.nodeParentMap[0][opt+3] -1;
                        int nd5 = nn_source.nodeParentMap[0][opt+4] -1;
                        int nd6 = nn_source.nodeParentMap[0][opt+5] -1;
                        int nd7 = nn_source.nodeParentMap[0][opt+6] -1;
                        int nd8 = nn_source.nodeParentMap[0][opt+7] -1;
                        int nd9 = nn_source.nodeParentMap[0][opt+8] -1;
                        int nd10 = nn_source.nodeParentMap[0][opt+9] -1;
                        int nd11 = nn_source.nodeParentMap[0][opt+10] -1;
                        int nd12 = nn_source.nodeParentMap[0][opt+11] -1;

                        int optCandi = opt / 12;

                        newLength = dist(array[nd1],array[nd2], arrayTSP[0]) + dist(array[nd3],array[nd4], arrayTSP[0])
                                + dist(array[nd5],array[nd6], arrayTSP[0])+ dist(array[nd7],array[nd8], arrayTSP[0])
                                + dist(array[nd9],array[nd10], arrayTSP[0]) + dist(array[nd11],array[nd12], arrayTSP[0] );

                        float opti = oldLength - newLength;
                        if(opti > 0 && opti > optimiz)
                        {
                            finalSelect = optCandi;
                            optimiz = opti;


                            atomicExch(&(nn_source.minRadiusMap[0][j_1-1]), 1);
                            atomicExch(&(nn_source.minRadiusMap[0][i_1-1]), 1);
                            atomicExch(&(nn_source.minRadiusMap[0][row_1-1]), 1);
                            atomicExch(&(nn_source.minRadiusMap[0][j-1]), 1);
                            atomicExch(&(nn_source.minRadiusMap[0][i-1]), 1);
                            atomicExch(&(nn_source.minRadiusMap[0][row-1]), 1);
                        }

                    }

                    if(finalSelect >= 0)
                    {

                        int node1 = (int)arrayTSP[0][j_1-1].current;
                        int node3 = (int)arrayTSP[0][i_1-1].current;
                        int node5 = (int)arrayTSP[0][row_1-1].current;
                        int node7 = (int)arrayTSP[0][j-1].current;
                        int node9 = (int)arrayTSP[0][i-1].current;
                        int node11 = (int)arrayTSP[0][row-1].current;

                        //                            float localMinChange = nn_source.minRadiusMap[0][node1];

                        //                            if(optimiz > localMinChange)
                        {

                            //12 is restricted by 64 bit and five numbers
                            unsigned long long result = 0;
                            result = result | node3;
                            result = result << 12;
                            result = result | node5;
                            result = result << 12;
                            result = result | node7;
                            result = result << 12;
                            result = result | node9;
                            result = result << 12;
                            result = result | node11;

                            float codekopt = finalSelect; //WB.Q 2024 densityMap only mark the k value of k-opt, and mark which mode of k-exchange
                            codekopt = finalSelect * 100 + 6;

                            //                                printf("GPU search, node1, node3, node5, node7, %d, %d, %d, %d; order(%d,%d,%d,%d), optvalue %lld, codekopt %f \n",
                            //                                       node1, node3, node5, node7, nn_source.grayValueMap[0][node1], nn_source.grayValueMap[0][node3], nn_source.grayValueMap[0][node5] , nn_source.grayValueMap[0][node7], result, codekopt);
                            atomicExch(&(nn_source.optCandidateMap[0][node1]), result); // WB.Q 2024 find a solution to judge non-interacted 23456-opt
                            atomicExch(&(nn_source.densityMap[0][node1]), codekopt); // WB.Q 2024 densityMap only mark the k value of k-opt, and mark which mode of k-exchange
                            //                                atomicExch(&(nn_source.minRadiusMap[0][node1]), optimiz);
                        }
                    }
                }

            }//end if i j
        }//end if row <= width

    }
    __syncthreads();
}// end K_2optOneThreadOne3opt



/*!
 * \brief 2408 QWB: add parallel 6-opt
 */
template <template<typename, typename> class NeuralNetLinks, class BufferDimension, class Point>
KERNEL void K_6opt_oneThreadOne6opt_rockiSmall_iter(NeuralNetLinks<BufferDimension, Point> nn_source,
                                                    Grid<doubleLinkedEdgeForTSP> arrayTSP,
                                                    double maxChecks6opt, double maxChecks3opt, unsigned int iter)
{

    double local_id = threadIdx.x + blockIdx.x * blockDim.x;

    int width =  nn_source.adaptiveMap.width; // each thread has this register

    if(local_id < maxChecks6opt)
    {

        int packSize = blockDim.x * gridDim.x;

        for(int nu = 0; nu <= iter; nu++)
        {

            double id = local_id + nu * packSize;

            if(id < maxChecks6opt)
            {

                double  outi, outj;
                int row,row_1, i, j, i_1, j_1;
                double subtriplicate = (1.0)/3;

                //WB.Q this way will produce i = j
                outi = (3 + sqrt(8.0f * (double )id + 1.0f)) / 2 ;
                outj = id - (outi-2)*(outi-1)/2 + 1;

                //WB.Q this way will produce i = j
                double rowN0 = 3*outi + sqrt(9*outi*outi - (1.0)/9);
                double rowN1 = pow(rowN0, subtriplicate);
                double rowN2 = 3*outi - sqrt(9*outi*outi - (1.0)/9);
                double rowN3 = pow(rowN2, subtriplicate);
                double rowN4 = rowN1 + rowN3 + 1;

                //WB.Q this way will produce i = j
                double rowN0_1 = 3*outj + sqrt(9*outj*outj - (1.0)/9);
                double rowN1_1 = pow(rowN0_1, subtriplicate);
                double rowN2_1 = 3*outj - sqrt(9*outj*outj - (1.0)/9);
                double rowN3_1 = pow(rowN2_1, subtriplicate);
                double rowN4_1 = rowN1_1 + rowN3_1 + 1;

                row = int(rowN4);// check which one works
                row_1 = int(rowN4_1);

                if(row < width && row_1 < width)
                {
                    double id2opt = (row-1)*(row)*(row+1)/6 - outi;

                    //WB.Q this way will produce i = j
                    i = int(3 + sqrt(8.0f * (double)id2opt + 1.0f)) / 2 ;
                    j = id2opt - (i-2)*(i-1)/2 + 1;

                    double id2opt_1 = (row_1-1)*(row_1)*(row_1+1)/6 - outj;

                    //WB.Q this way will produce i = j
                    i_1 = int(3 + sqrt(8.0f * (double)id2opt_1 + 1.0f)) / 2 ;
                    j_1 = id2opt_1 - (i_1-2)*(i_1-1)/2 + 1;


                    //                    //qiao only for test
                    //                    if(id == maxChecks6opt - 2)
                    //                        printf("6-opt maxmim id %d, outi,outj:(%d,%d), row,i,j,row1,i1,j1:(%d,%d,%d,%d,%d,%d) \n",id, outi,outj, row, i,j, row_1,i_1,j_1);


                    if(i<row && row_1 < j-1 && i!=row && i+1!= row && j > 0 && j < i && j-1 >= 0 && j < width && j+1 != i && j + width != i+1 && i-1 >= 0 && i < width-1
                            && i_1<row_1 && i_1!=row_1 && i_1+1!= row_1 && j_1 > 0 && j_1 < i_1 && j_1-1 >= 0 && j_1 < width && j_1+1 != i_1 && j_1 + width != i_1+1 && i_1-1 >= 0 && i_1 < width-1
                            && row_1 < row && row_1+1!=row && i > i_1 && i_1+1!=i && j> j_1 && j!=j_1+1 && i!=i_1 && j!=j_1 && i!=j_1 && j!=i_1 && i != row_1 && j!= row_1 && i_1!=row &&j_1!=row
                            )
                    {

                        //                        if(row > width -2)
                        //                            printf("6-opt maxmim id %d, outi,outj:(%d,%d), row,i,j,row1,i1,j1:(%d,%d,%d,%d,%d,%d) \n",id, outi,outj, row, i,j, row_1,i_1,j_1);


                        bool existingCandidate = 0;
                        if(nn_source.minRadiusMap[0][j_1-1] == 1 || nn_source.minRadiusMap[0][i_1-1] == 1 ||
                                nn_source.minRadiusMap[0][row_1-1] == 1 ||nn_source.minRadiusMap[0][j-1] == 1 ||
                                nn_source.minRadiusMap[0][i-1] == 1 ||nn_source.minRadiusMap[0][row-1] == 1)
                            existingCandidate = 1;

                        if(existingCandidate == 0)
                        {

                            float oldLength = dist(j_1-1, j_1, arrayTSP[0]) + dist(i_1-1, i_1, arrayTSP[0]) + dist(row_1-1, row_1, arrayTSP[0])
                                    + dist(j-1, j, arrayTSP[0]) + dist(i-1, i, arrayTSP[0]) + dist(row-1, row, arrayTSP[0])    ;

                            float newLength;
                            int array[12];
                            array[0] = j_1-1;
                            array[1] = j_1;
                            array[2] = i_1-1;
                            array[3] = i_1;
                            array[4] = row_1-1;
                            array[5] = row_1;
                            array[6] = j-1;
                            array[7] = j;
                            array[8] = i-1;
                            array[9] = i;
                            array[10] = row-1;
                            array[11] = row;

                            int finalSelect = -1;
                            float optimiz = -INFINITY;


                            for(int opt = 0; opt < 23220; opt +=12) //  6 edges 12 nodes 1935 sets 1935*12=23220 nodes
                            {
                                int nd1 = nn_source.nodeParentMap[0][opt] -1;
                                int nd2 = nn_source.nodeParentMap[0][opt+1] -1;
                                int nd3 = nn_source.nodeParentMap[0][opt+2] -1;
                                int nd4 = nn_source.nodeParentMap[0][opt+3] -1;
                                int nd5 = nn_source.nodeParentMap[0][opt+4] -1;
                                int nd6 = nn_source.nodeParentMap[0][opt+5] -1;
                                int nd7 = nn_source.nodeParentMap[0][opt+6] -1;
                                int nd8 = nn_source.nodeParentMap[0][opt+7] -1;
                                int nd9 = nn_source.nodeParentMap[0][opt+8] -1;
                                int nd10 = nn_source.nodeParentMap[0][opt+9] -1;
                                int nd11 = nn_source.nodeParentMap[0][opt+10] -1;
                                int nd12 = nn_source.nodeParentMap[0][opt+11] -1;

                                int optCandi = opt / 12;

                                newLength = dist(array[nd1],array[nd2], arrayTSP[0]) + dist(array[nd3],array[nd4], arrayTSP[0])
                                        + dist(array[nd5],array[nd6], arrayTSP[0])+ dist(array[nd7],array[nd8], arrayTSP[0])
                                        + dist(array[nd9],array[nd10], arrayTSP[0]) + dist(array[nd11],array[nd12], arrayTSP[0] );

                                float opti = oldLength - newLength;
                                if(opti > 0 && opti > optimiz)
                                {
                                    finalSelect = optCandi;
                                    optimiz = opti;

                                    atomicExch(&(nn_source.minRadiusMap[0][j_1-1]), 1);
                                    atomicExch(&(nn_source.minRadiusMap[0][i_1-1]), 1);
                                    atomicExch(&(nn_source.minRadiusMap[0][row_1-1]), 1);
                                    atomicExch(&(nn_source.minRadiusMap[0][j-1]), 1);
                                    atomicExch(&(nn_source.minRadiusMap[0][i-1]), 1);
                                    atomicExch(&(nn_source.minRadiusMap[0][row-1]), 1);

                                }

                            }

                            if(finalSelect >= 0)
                            {

                                int node1 = (int)arrayTSP[0][j_1-1].current;
                                int node3 = (int)arrayTSP[0][i_1-1].current;
                                int node5 = (int)arrayTSP[0][row_1-1].current;
                                int node7 = (int)arrayTSP[0][j-1].current;
                                int node9 = (int)arrayTSP[0][i-1].current;
                                int node11 = (int)arrayTSP[0][row-1].current;

                                //                            float localMinChange = nn_source.minRadiusMap[0][node1];

                                //                            if(optimiz > localMinChange)
                                {

                                    //12 is restricted by 64 bit and five numbers
                                    unsigned long long result = 0;
                                    result = result | node3;
                                    result = result << 12;
                                    result = result | node5;
                                    result = result << 12;
                                    result = result | node7;
                                    result = result << 12;
                                    result = result | node9;
                                    result = result << 12;
                                    result = result | node11;

                                    float codekopt = finalSelect; //WB.Q 2024 densityMap only mark the k value of k-opt, and mark which mode of k-exchange
                                    codekopt = finalSelect * 100 + 6;

                                    //                                printf("GPU search, node1, node3, node5, node7, %d, %d, %d, %d; order(%d,%d,%d,%d), optvalue %lld, codekopt %f \n",
                                    //                                       node1, node3, node5, node7, nn_source.grayValueMap[0][node1], nn_source.grayValueMap[0][node3], nn_source.grayValueMap[0][node5] , nn_source.grayValueMap[0][node7], result, codekopt);
                                    atomicExch(&(nn_source.optCandidateMap[0][node1]), result); // WB.Q 2024 find a solution to judge non-interacted 23456-opt
                                    atomicExch(&(nn_source.densityMap[0][node1]), codekopt); // WB.Q 2024 densityMap only mark the k value of k-opt, and mark which mode of k-exchange
                                    //                                atomicExch(&(nn_source.minRadiusMap[0][node1]), optimiz);
                                }
                            }
                        }

                    }//end if i j
                }//end if row <= width

            }
        }
    }
    __syncthreads();
}// end K_6optOneThreadOne6opt



/*!
 * \brief 2408 QWB: add parallel 6-opt
 */
template <template<typename, typename> class NeuralNetLinks, class BufferDimension, class Point>
KERNEL void K_6opt_oneThreadOne6opt_qiao_stride_iter(NeuralNetLinks<BufferDimension, Point> nn_source,
                                                     Grid<doubleLinkedEdgeForTSP> arrayTSP,
                                                     double maxChecks6opt, double maxChecks3opt, double maxChecksoptDivide,
                                                     double iter, double istride)
{

    double local_id = threadIdx.x + blockIdx.x * blockDim.x;

    int width =  nn_source.adaptiveMap.width; // each thread has this register

    if(local_id < maxChecks6opt)
    {

        double startId = maxChecksoptDivide * (istride);

        if(local_id == 0)
            printf("StartID %f, local_id %f \n", startId, local_id);


        for(double id = local_id*iter ; id < (local_id+1)*(iter); id++)
        {

            id = id + startId;

            if(id < maxChecks6opt)
            {

                double  outi, outj;
                int row,row_1, i, j, i_1, j_1;
                double subtriplicate = (1.0)/3;

                //WB.Q this way will produce i = j
                double sqrtOuti = 8.0 * (double )id + 1.0;
                outi = (3 + sqrt(sqrtOuti)) / 2 ;
                outi = trunc(outi);
                outj = id - (outi-2)*(outi-1)/2 + 1;


                if(outi < maxChecks3opt && outj <maxChecks3opt)

                {

                    //WB.Q this way will produce i = j
                    double idid = 9*outi*outi - (1.0)/9;
                    double idMuli = 3*outi;
                    double rowN0 = idMuli + sqrt(idid);
                    double rowN1 = pow(rowN0, subtriplicate);
                    double rowN2 = idMuli - sqrt(idid);
                    double rowN3 = pow(rowN2, subtriplicate);
                    double rowN4 = rowN1 + rowN3 + 1;

                    //WB.Q this way will produce i = j
                    double ididj= 9*outj*outj - (1.0)/9;
                    double idMulj = 3*outj;
                    double rowN0_1 = idMulj + sqrt(ididj);
                    double rowN1_1 = pow(rowN0_1, subtriplicate);
                    double rowN2_1 = idMulj - sqrt(ididj);
                    double rowN3_1 = pow(rowN2_1, subtriplicate);
                    double rowN4_1 = rowN1_1 + rowN3_1 + 1;

                    row = int(rowN4);// check which one works
                    row_1 = int(rowN4_1);

                    if(row < width && row_1 < width)
                    {
                        double tempRowRow = (double)(row-1) / 6;
                        double tempRowRowRow = tempRowRow * row *(row-2);
                        //                    double id2opt = (row-1)*(row)*(row+1)/6 - outi;
                        double id2opt = fabs(outi - tempRowRowRow);

                        //WB.Q this way will produce i = j
                        double sqrtTemp = 8.0 * (double)id2opt + 1.0;
                        i = int(3 + sqrt(sqrtTemp)) / 2 ;
                        j = id2opt - (i-2)*(i-1)/2 + 1;

                        //                    double id2opt_1 = (row_1-1)*(row_1)*(row_1+1)/6 - outj;
                        double tempRowRowj = (double)(row_1-1) / 6;
                        double tempRowRowRowj = tempRowRowj * row_1 * (row_1 -2);
                        double id2opt_1 = fabs(outj - tempRowRowRowj);

                        //WB.Q this way will produce i = j
                        double sqrtTempj = 8.0 * (double)id2opt_1 + 1.0;
                        i_1 = int(3 + sqrt(sqrtTempj)) / 2 ;
                        j_1 = id2opt_1 - (i_1-2)*(i_1-1)/2 + 1;


                        //                    //qiao only for test
                        //                    if(id == maxChecks6opt - 2)
                        //                        printf("6-opt maxmim id %d, outi,outj:(%d,%d), row,i,j,row1,i1,j1:(%d,%d,%d,%d,%d,%d) \n",id, outi,outj, row, i,j, row_1,i_1,j_1);


                        if(i<row && row_1 < j-1 && i!=row && i+1!= row && j > 0 && j < i && j-1 >= 0 && j < width && j+1 != i && j + width != i+1 && i-1 >= 0 && i < width-1
                                && i_1<row_1 && i_1!=row_1 && i_1+1!= row_1 && j_1 > 0 && j_1 < i_1 && j_1-1 >= 0 && j_1 < width && j_1+1 != i_1 && j_1 + width != i_1+1 && i_1-1 >= 0 && i_1 < width-1
                                && row_1 < row && row_1+1!=row && i > i_1 && i_1+1!=i && j> j_1 && j!=j_1+1 && i!=i_1 && j!=j_1 && i!=j_1 && j!=i_1 && i != row_1 && j!= row_1 && i_1!=row &&j_1!=row
                                )
                        {

                            if(row > width -2)
                                printf("6-opt maxmim id %f, outi,outj:(%f,%f), row,i,j,row1,i1,j1:(%d,%d,%d,%d,%d,%d) \n",id, outi,outj, row, i,j, row_1,i_1,j_1);


                            bool existingCandidate = 0;
                            if(nn_source.minRadiusMap[0][j_1-1] == 1 || nn_source.minRadiusMap[0][i_1-1] == 1 ||
                                    nn_source.minRadiusMap[0][row_1-1] == 1 ||nn_source.minRadiusMap[0][j-1] == 1 ||
                                    nn_source.minRadiusMap[0][i-1] == 1 ||nn_source.minRadiusMap[0][row-1] == 1)
                                existingCandidate = 1;

                            if(existingCandidate == 0)
                            {

                                float oldLength = dist(j_1-1, j_1, arrayTSP[0]) + dist(i_1-1, i_1, arrayTSP[0]) + dist(row_1-1, row_1, arrayTSP[0])
                                        + dist(j-1, j, arrayTSP[0]) + dist(i-1, i, arrayTSP[0]) + dist(row-1, row, arrayTSP[0])    ;

                                float newLength;
                                int array[12];
                                array[0] = j_1-1;
                                array[1] = j_1;
                                array[2] = i_1-1;
                                array[3] = i_1;
                                array[4] = row_1-1;
                                array[5] = row_1;
                                array[6] = j-1;
                                array[7] = j;
                                array[8] = i-1;
                                array[9] = i;
                                array[10] = row-1;
                                array[11] = row;

                                int finalSelect = -1;
                                float optimiz = -INFINITY;


                                for(int opt = 0; opt < 23220; opt +=12) //  6 edges 12 nodes 1935 sets 1935*12=23220 nodes
                                {
                                    int nd1 = nn_source.nodeParentMap[0][opt] -1;
                                    int nd2 = nn_source.nodeParentMap[0][opt+1] -1;
                                    int nd3 = nn_source.nodeParentMap[0][opt+2] -1;
                                    int nd4 = nn_source.nodeParentMap[0][opt+3] -1;
                                    int nd5 = nn_source.nodeParentMap[0][opt+4] -1;
                                    int nd6 = nn_source.nodeParentMap[0][opt+5] -1;
                                    int nd7 = nn_source.nodeParentMap[0][opt+6] -1;
                                    int nd8 = nn_source.nodeParentMap[0][opt+7] -1;
                                    int nd9 = nn_source.nodeParentMap[0][opt+8] -1;
                                    int nd10 = nn_source.nodeParentMap[0][opt+9] -1;
                                    int nd11 = nn_source.nodeParentMap[0][opt+10] -1;
                                    int nd12 = nn_source.nodeParentMap[0][opt+11] -1;

                                    int optCandi = opt / 12;

                                    newLength = dist(array[nd1],array[nd2], arrayTSP[0]) + dist(array[nd3],array[nd4], arrayTSP[0])
                                            + dist(array[nd5],array[nd6], arrayTSP[0])+ dist(array[nd7],array[nd8], arrayTSP[0])
                                            + dist(array[nd9],array[nd10], arrayTSP[0]) + dist(array[nd11],array[nd12], arrayTSP[0] );

                                    float opti = oldLength - newLength;
                                    if(opti > 0 && opti > optimiz)
                                    {
                                        finalSelect = optCandi;
                                        optimiz = opti;

                                        atomicExch(&(nn_source.minRadiusMap[0][j_1-1]), 1);
                                        atomicExch(&(nn_source.minRadiusMap[0][i_1-1]), 1);
                                        atomicExch(&(nn_source.minRadiusMap[0][row_1-1]), 1);
                                        atomicExch(&(nn_source.minRadiusMap[0][j-1]), 1);
                                        atomicExch(&(nn_source.minRadiusMap[0][i-1]), 1);
                                        atomicExch(&(nn_source.minRadiusMap[0][row-1]), 1);

                                    }

                                }

                                if(finalSelect >= 0)
                                {

                                    int node1 = (int)arrayTSP[0][j_1-1].current;
                                    int node3 = (int)arrayTSP[0][i_1-1].current;
                                    int node5 = (int)arrayTSP[0][row_1-1].current;
                                    int node7 = (int)arrayTSP[0][j-1].current;
                                    int node9 = (int)arrayTSP[0][i-1].current;
                                    int node11 = (int)arrayTSP[0][row-1].current;

                                    //                            float localMinChange = nn_source.minRadiusMap[0][node1];

                                    //                            if(optimiz > localMinChange)
                                    {

                                        //12 is restricted by 64 bit and five numbers
                                        unsigned long long result = 0;
                                        result = result | node3;
                                        result = result << 12;
                                        result = result | node5;
                                        result = result << 12;
                                        result = result | node7;
                                        result = result << 12;
                                        result = result | node9;
                                        result = result << 12;
                                        result = result | node11;

                                        float codekopt = finalSelect; //WB.Q 2024 densityMap only mark the k value of k-opt, and mark which mode of k-exchange
                                        codekopt = finalSelect * 100 + 6;

                                        //                                printf("GPU search, node1, node3, node5, node7, %d, %d, %d, %d; order(%d,%d,%d,%d), optvalue %lld, codekopt %f \n",
                                        //                                       node1, node3, node5, node7, nn_source.grayValueMap[0][node1], nn_source.grayValueMap[0][node3], nn_source.grayValueMap[0][node5] , nn_source.grayValueMap[0][node7], result, codekopt);
                                        atomicExch(&(nn_source.optCandidateMap[0][node1]), result); // WB.Q 2024 find a solution to judge non-interacted 23456-opt
                                        atomicExch(&(nn_source.densityMap[0][node1]), codekopt); // WB.Q 2024 densityMap only mark the k value of k-opt, and mark which mode of k-exchange
                                        //                                atomicExch(&(nn_source.minRadiusMap[0][node1]), optimiz);
                                    }
                                }
                            }

                        }//end if i j
                    }//end if row <= width

                }


            }
        }
    }
    __syncthreads();
}// end K_6optOneThreadOne6opt


/*!
 * \brief 2408 QWB: add parallel 6-opt
 */
template <template<typename, typename> class NeuralNetLinks, class BufferDimension, class Point>
KERNEL void K_6opt_oneThreadOne6opt_qiao_stride_iter_onlySharePossibility(NeuralNetLinks<BufferDimension, Point> nn_source,
                                                                          Grid<doubleLinkedEdgeForTSP> arrayTSP,
                                                                          double maxChecks6opt, double maxChecks3opt, double maxChecksoptDivide,
                                                                          double iter, double istride)
{

    double local_id = threadIdx.x + blockIdx.x * blockDim.x;

    int width =  nn_source.adaptiveMap.width; // each thread has this register


    __shared__ float optPossibilities[OPTPOSSIBILITES6OPT];

    float iterSharedPossble =  (float)OPTPOSSIBILITES6OPT / (float)BLOCKSIZE;
    for(int opt = 0; opt < iterSharedPossble; opt++)
    {
        int m = threadIdx.x + opt*BLOCKSIZE;

        if(m < OPTPOSSIBILITES6OPT)
            optPossibilities[m] = nn_source.nodeParentMap[0][m];

        __syncthreads();

    }

    if(local_id < maxChecks6opt)
    {

        double startId = maxChecksoptDivide * (istride);

        if(local_id == 0)
            printf("StartID %f, local_id %f \n", startId, local_id);


        for(double id = local_id*iter ; id < (local_id+1)*(iter); id++)
        {

            id = id + startId;

            if(id < maxChecks6opt)
            {

                double  outi, outj;
                int row,row_1, i, j, i_1, j_1;
                double subtriplicate = (1.0)/3;

                //WB.Q this way will produce i = j
                double sqrtOuti = 8.0 * (double )id + 1.0;
                outi = (3 + sqrt(sqrtOuti)) / 2 ;
                outi = trunc(outi);
                outj = id - (outi-2)*(outi-1)/2 + 1;


                if(outi < maxChecks3opt && outj <maxChecks3opt)

                {

                    //WB.Q this way will produce i = j
                    double idid = 9*outi*outi - (1.0)/9;
                    double idMuli = 3*outi;
                    double rowN0 = idMuli + sqrt(idid);
                    double rowN1 = pow(rowN0, subtriplicate);
                    double rowN2 = idMuli - sqrt(idid);
                    double rowN3 = pow(rowN2, subtriplicate);
                    double rowN4 = rowN1 + rowN3 + 1;

                    //WB.Q this way will produce i = j
                    double ididj= 9*outj*outj - (1.0)/9;
                    double idMulj = 3*outj;
                    double rowN0_1 = idMulj + sqrt(ididj);
                    double rowN1_1 = pow(rowN0_1, subtriplicate);
                    double rowN2_1 = idMulj - sqrt(ididj);
                    double rowN3_1 = pow(rowN2_1, subtriplicate);
                    double rowN4_1 = rowN1_1 + rowN3_1 + 1;

                    row = int(rowN4);// check which one works
                    row_1 = int(rowN4_1);

                    if(row < width && row_1 < width)
                    {
                        double tempRowRow = (double)(row-1) / 6;
                        double tempRowRowRow = tempRowRow * row *(row-2);
                        //                    double id2opt = (row-1)*(row)*(row+1)/6 - outi;
                        double id2opt = fabs(outi - tempRowRowRow);

                        //WB.Q this way will produce i = j
                        double sqrtTemp = 8.0 * (double)id2opt + 1.0;
                        i = int(3 + sqrt(sqrtTemp)) / 2 ;
                        j = id2opt - (i-2)*(i-1)/2 + 1;

                        //                    double id2opt_1 = (row_1-1)*(row_1)*(row_1+1)/6 - outj;
                        double tempRowRowj = (double)(row_1-1) / 6;
                        double tempRowRowRowj = tempRowRowj * row_1 * (row_1 -2);
                        double id2opt_1 = fabs(outj - tempRowRowRowj);

                        //WB.Q this way will produce i = j
                        double sqrtTempj = 8.0 * (double)id2opt_1 + 1.0;
                        i_1 = int(3 + sqrt(sqrtTempj)) / 2 ;
                        j_1 = id2opt_1 - (i_1-2)*(i_1-1)/2 + 1;


                        //                    //qiao only for test
                        //                    if(id == maxChecks6opt - 2)
                        //                        printf("6-opt maxmim id %d, outi,outj:(%d,%d), row,i,j,row1,i1,j1:(%d,%d,%d,%d,%d,%d) \n",id, outi,outj, row, i,j, row_1,i_1,j_1);


                        if(i<row && row_1 < j-1 && i!=row && i+1!= row && j > 0 && j < i && j-1 >= 0 && j < width && j+1 != i && j + width != i+1 && i-1 >= 0 && i < width-1
                                && i_1<row_1 && i_1!=row_1 && i_1+1!= row_1 && j_1 > 0 && j_1 < i_1 && j_1-1 >= 0 && j_1 < width && j_1+1 != i_1 && j_1 + width != i_1+1 && i_1-1 >= 0 && i_1 < width-1
                                && row_1 < row && row_1+1!=row && i > i_1 && i_1+1!=i && j> j_1 && j!=j_1+1 && i!=i_1 && j!=j_1 && i!=j_1 && j!=i_1 && i != row_1 && j!= row_1 && i_1!=row &&j_1!=row
                                )
                        {

                            if(row > width -2)
                                printf("6-opt maxmim id %f, outi,outj:(%f,%f), row,i,j,row1,i1,j1:(%d,%d,%d,%d,%d,%d) \n",id, outi,outj, row, i,j, row_1,i_1,j_1);


                            bool existingCandidate = 0;
                            if(nn_source.minRadiusMap[0][j_1-1] == 1 || nn_source.minRadiusMap[0][i_1-1] == 1 ||
                                    nn_source.minRadiusMap[0][row_1-1] == 1 ||nn_source.minRadiusMap[0][j-1] == 1 ||
                                    nn_source.minRadiusMap[0][i-1] == 1 ||nn_source.minRadiusMap[0][row-1] == 1)
                                existingCandidate = 1;

                            if(existingCandidate == 0)
                            {

                                float oldLength = dist(j_1-1, j_1, arrayTSP[0]) + dist(i_1-1, i_1, arrayTSP[0]) + dist(row_1-1, row_1, arrayTSP[0])
                                        + dist(j-1, j, arrayTSP[0]) + dist(i-1, i, arrayTSP[0]) + dist(row-1, row, arrayTSP[0])    ;

                                float newLength;
                                int array[12];
                                array[0] = j_1-1;
                                array[1] = j_1;
                                array[2] = i_1-1;
                                array[3] = i_1;
                                array[4] = row_1-1;
                                array[5] = row_1;
                                array[6] = j-1;
                                array[7] = j;
                                array[8] = i-1;
                                array[9] = i;
                                array[10] = row-1;
                                array[11] = row;

                                int finalSelect = -1;
                                float optimiz = -INFINITY;


                                for(int opt = 0; opt < 23220; opt +=12) //  6 edges 12 nodes 1935 sets 1935*12=23220 nodes
                                {
                                    int nd1 = optPossibilities[opt] -1;
                                    int nd2 = optPossibilities[opt+1] -1;
                                    int nd3 = optPossibilities[opt+2] -1;
                                    int nd4 = optPossibilities[opt+3] -1;
                                    int nd5 = optPossibilities[opt+4] -1;
                                    int nd6 = optPossibilities[opt+5] -1;
                                    int nd7 = optPossibilities[opt+6] -1;
                                    int nd8 = optPossibilities[opt+7] -1;
                                    int nd9 = optPossibilities[opt+8] -1;
                                    int nd10 = optPossibilities[opt+9] -1;
                                    int nd11 = optPossibilities[opt+10] -1;
                                    int nd12 = optPossibilities[opt+11] -1;

                                    int optCandi = opt / 12;

                                    newLength = dist(array[nd1],array[nd2], arrayTSP[0]) + dist(array[nd3],array[nd4], arrayTSP[0])
                                            + dist(array[nd5],array[nd6], arrayTSP[0])+ dist(array[nd7],array[nd8], arrayTSP[0])
                                            + dist(array[nd9],array[nd10], arrayTSP[0]) + dist(array[nd11],array[nd12], arrayTSP[0] );

                                    float opti = oldLength - newLength;
                                    if(opti > 0 && opti > optimiz)
                                    {
                                        finalSelect = optCandi;
                                        optimiz = opti;

                                        atomicExch(&(nn_source.minRadiusMap[0][j_1-1]), 1);
                                        atomicExch(&(nn_source.minRadiusMap[0][i_1-1]), 1);
                                        atomicExch(&(nn_source.minRadiusMap[0][row_1-1]), 1);
                                        atomicExch(&(nn_source.minRadiusMap[0][j-1]), 1);
                                        atomicExch(&(nn_source.minRadiusMap[0][i-1]), 1);
                                        atomicExch(&(nn_source.minRadiusMap[0][row-1]), 1);

                                    }

                                }

                                if(finalSelect >= 0)
                                {

                                    int node1 = (int)arrayTSP[0][j_1-1].current;
                                    int node3 = (int)arrayTSP[0][i_1-1].current;
                                    int node5 = (int)arrayTSP[0][row_1-1].current;
                                    int node7 = (int)arrayTSP[0][j-1].current;
                                    int node9 = (int)arrayTSP[0][i-1].current;
                                    int node11 = (int)arrayTSP[0][row-1].current;

                                    //                            float localMinChange = nn_source.minRadiusMap[0][node1];

                                    //                            if(optimiz > localMinChange)
                                    {

                                        //12 is restricted by 64 bit and five numbers
                                        unsigned long long result = 0;
                                        result = result | node3;
                                        result = result << 12;
                                        result = result | node5;
                                        result = result << 12;
                                        result = result | node7;
                                        result = result << 12;
                                        result = result | node9;
                                        result = result << 12;
                                        result = result | node11;

                                        float codekopt = finalSelect; //WB.Q 2024 densityMap only mark the k value of k-opt, and mark which mode of k-exchange
                                        codekopt = finalSelect * 100 + 6;

                                        //                                printf("GPU search, node1, node3, node5, node7, %d, %d, %d, %d; order(%d,%d,%d,%d), optvalue %lld, codekopt %f \n",
                                        //                                       node1, node3, node5, node7, nn_source.grayValueMap[0][node1], nn_source.grayValueMap[0][node3], nn_source.grayValueMap[0][node5] , nn_source.grayValueMap[0][node7], result, codekopt);
                                        atomicExch(&(nn_source.optCandidateMap[0][node1]), result); // WB.Q 2024 find a solution to judge non-interacted 23456-opt
                                        atomicExch(&(nn_source.densityMap[0][node1]), codekopt); // WB.Q 2024 densityMap only mark the k value of k-opt, and mark which mode of k-exchange
                                        //                                atomicExch(&(nn_source.minRadiusMap[0][node1]), optimiz);
                                    }
                                }
                            }

                        }//end if i j
                    }//end if row <= width

                }


            }
        }
    }
    __syncthreads();
}// end K_6optOneThreadOne6opt


//! 0617 QWB: add parallel 2opt one thread one 2-opt with Rocki
template <template<typename, typename> class NeuralNetLinks, class BufferDimension, class Point
          >
GLOBAL inline void K_oneThreadOne2opt_RockiSmall( NeuralNetLinks<BufferDimension, Point>& nn_source,
                                                  Grid<doubleLinkedEdgeForTSP>& linkCoordTourGpu,
                                                  unsigned long maxChecks,
                                                  unsigned int iter
                                                  ) {

    //    KER_CALL_THREAD_BLOCK_1D_fix(b, t,
    //                                 BLOCKSIZE,
    //                                 16,
    //                                 GRIDSIZE, //for rocki large global
    //                                 // maxChecks/BLOCKSIZE + 1, // for rocki large global
    //                                 nn_source.adaptiveMap.width);



    KER_CALL_THREAD_BLOCK(b, t,BLOCKSIZE,1, maxChecks, 1);

    //                K_2opt_oneThreadOne2opt_rockiSmall_shared _KER_CALL_(b, t) (nn_source.densityMap, nn_source.minRadiusMap, linkCoordTourGpu, maxChecks, iter);
    K_2opt_oneThreadOne2opt_rockiSmall _KER_CALL_(b, t) (nn_source, linkCoordTourGpu, maxChecks, iter);// global

    cudaChk(cudaPeekAtLastError());
}



//! 0617 QWB: add parallel 2opt one thread one 2-opt with Rocki
template <template<typename, typename> class NeuralNetLinks, class BufferDimension, class Point
          >
GLOBAL inline void K_oneThreadOne2opt_Rocki_iterStride( NeuralNetLinks<BufferDimension, Point>& nn_source,
                                                        Grid<doubleLinkedEdgeForTSP>& linkCoordTourGpu,
                                                        double max2optChecks, double maxChecksoOptDivide,
                                                        double iter, double istride
                                                        ) {

    KER_CALL_THREAD_BLOCK_1D_fix(b, t,
                                 BLOCKSIZE,
                                 16,
                                 GRIDSIZE, //for rocki large global
                                 // maxChecks/BLOCKSIZE + 1, // for rocki large global
                                 nn_source.adaptiveMap.width);

    //  K_2opt_oneThreadOne2opt_qiaoIterStride_best_shared _KER_CALL_(b, t) (nn_source, linkCoordTourGpu, max2optChecks, maxChecksoOptDivide, iter, istride);
    //      K_2opt_oneThreadOne2opt_qiaoIterStride _KER_CALL_(b, t) (nn_source, linkCoordTourGpu, max2optChecks, maxChecksoOptDivide, iter, istride);// global
    K_2opt_oneThreadOne2opt_qiaoIterStride_best _KER_CALL_(b, t) (nn_source, linkCoordTourGpu, max2optChecks, maxChecksoOptDivide, iter, istride);// global the faster way to converage that iter one node select the first 2-opt

    cudaChk(cudaPeekAtLastError());
}

//! 0617 QWB: add parallel 2opt one thread one 2-opt with Rocki small
template <template<typename, typename> class NeuralNetLinks, class BufferDimension, class Point
          >
GLOBAL inline void K_oneThreadOne4opt_qiao_iterStride( NeuralNetLinks<BufferDimension, Point>& nn_source,
                                                       Grid<doubleLinkedEdgeForTSP>& linkCoordTourGpu,
                                                       double  maxChecks2opt, double maxChecks4opt, double maxChecks4optDivide,
                                                       double iter, double istride
                                                       ) {

    //qiao here does not run correctly
    KER_CALL_THREAD_BLOCK_1D_fix(b, t,
                                 BLOCKSIZE, 16,
                                 GRIDSIZE, //for rocki large global
                                 //                                 maxChecks/BLOCKSIZE + 1, // for rocki large global
                                 nn_source.adaptiveMap.width);
    cout << "grid blocks : " << b.x << ", b.y " << b.y << ", thread t.x " << t.x << ", t.y " << t.y << " , INT_MAX="
         << INT_MAX  << ", DBL_MAX= " << DBL_MAX << ", LLONG_MAX=" << LLONG_MAX <<endl;

    //correct  70110
    //    K_4opt_oneThreadOne4opt_qiaoIterStride_Best  _KER_CALL_(b, t) (nn_source, linkCoordTourGpu,  maxChecks2opt, maxChecks4opt, maxChecks4optDivide, iter, istride);// global


    //correct 66359
    //    K_4opt_oneThreadOne4opt_qiaoIterStride_Best_shared _KER_CALL_(b, t) (nn_source, linkCoordTourGpu,  maxChecks2opt, maxChecks4opt, maxChecks4optDivide, iter, istride);// global


    //correct 62940ms 62926ms 71opts
    //    K_4opt_oneThreadOne4opt_qiaoIterStride _KER_CALL_(b, t) (nn_source, linkCoordTourGpu,  maxChecks2opt, maxChecks4opt, maxChecks4optDivide, iter, istride);// global

    //correct the least time consuming over above three version 55899, produce 33 opts
    //    K_4opt_oneThreadOne4opt_qiaoIterStride_shared _KER_CALL_(b, t) (nn_source, linkCoordTourGpu,  maxChecks2opt, maxChecks4opt, maxChecks4optDivide, iter, istride);// global


    //correct 55890ms 68opts
    K_4opt_oneThreadOne4opt_qiaoIterStride_shared_noOccupy _KER_CALL_(b, t) (nn_source, linkCoordTourGpu,  maxChecks2opt, maxChecks4opt, maxChecks4optDivide, iter, istride);// global


    //        KER_CALL_THREAD_BLOCK(b, t,BLOCKSIZE,1, maxChecks4opt, 1);
    //        double blocks =  (maxChecks4opt + BLOCKSIZE - 1) / BLOCKSIZE ;
    //        cout << "thread block : " << b.x << ", b.y " << b.y << ", t.x " << t.x << ", t.y " << t.y << " blocks=" << blocks <<  ", INT_MAX="
    //             << INT_MAX  << ", DBL_MAX= " << DBL_MAX << ", LLONG_MAX=" << LLONG_MAX <<endl;
    //        K_4opt_oneThreadOne4opt_rockiSmall _KER_CALL_(b, t) (nn_source, linkCoordTourGpu,  maxChecks2opt, maxChecks4opt, iter);// global

    cudaChk(cudaPeekAtLastError());
}

//! 0617 QWB: add parallel 2opt one thread one 2-opt with Rocki small
template <template<typename, typename> class NeuralNetLinks, class BufferDimension, class Point
          >
GLOBAL inline void K_oneThreadOne5opt_RockiSmall( NeuralNetLinks<BufferDimension, Point>& nn_source,
                                                  Grid<doubleLinkedEdgeForTSP>& linkCoordTourGpu,
                                                  int n, double maxChecks4opt,double maxChecks2opt,
                                                  unsigned int iter
                                                  ) {

    KER_CALL_THREAD_BLOCK_1D_fix(b, t,
                                 BLOCKSIZE, 16,
                                 GRIDSIZE, //for rocki large global
                                 //                                 maxChecks/BLOCKSIZE + 1, // for rocki large global
                                 nn_source.adaptiveMap.width);
    //        double blocks =  (maxChecks4opt + BLOCKSIZE - 1) / BLOCKSIZE ;
    //        cout << "thread block : " << b.x << ", b.y " << b.y << ", t.x " << t.x << ", t.y " << t.y << " blocks=" << blocks <<  ", INT_MAX="
    //             << INT_MAX  << ", DBL_MAX= " << DBL_MAX << ", LLONG_MAX=" << LLONG_MAX <<endl;
    K_5opt_oneThreadOne5opt_rockiSmall_iter _KER_CALL_(b, t) (nn_source, linkCoordTourGpu, n ,maxChecks4opt, maxChecks2opt,iter);// global




    //    KER_CALL_THREAD_BLOCK(b, t,BLOCKSIZE,1, maxChecks4opt, 1);
    //    double blocks =  (maxChecks4opt + BLOCKSIZE - 1) / BLOCKSIZE ;
    //    cout << "thread block : " << b.x << ", b.y " << b.y << ", t.x " << t.x << ", t.y " << t.y << " blocks=" << blocks <<  ", INT_MAX="
    //         << INT_MAX  << ", DBL_MAX= " << DBL_MAX << ", LLONG_MAX=" << LLONG_MAX <<endl;
    //    K_5opt_oneThreadOne5opt_rockiSmall _KER_CALL_(b, t) (nn_source, linkCoordTourGpu, n ,maxChecks4opt, maxChecks2opt,iter);// global

    cudaChk(cudaPeekAtLastError());
}




//! 0617 QWB: add parallel 5opt
template <template<typename, typename> class NeuralNetLinks, class BufferDimension, class Point
          >
GLOBAL inline void K_oneThreadOne5opt_qiao_StrideIter( NeuralNetLinks<BufferDimension, Point>& nn_source,
                                                       Grid<doubleLinkedEdgeForTSP>& linkCoordTourGpu,
                                                       int n, double  maxChecks2opt, double maxChecks4opt, double maxChecks4optDivide,
                                                       double iter, double istride
                                                       ) {

    KER_CALL_THREAD_BLOCK_1D_fix(b, t,
                                 BLOCKSIZE, 16,
                                 GRIDSIZE, //for rocki large global
                                 //                                 maxChecks/BLOCKSIZE + 1, // for rocki large global
                                 nn_source.adaptiveMap.width);
    cout << "grid blocks : " << b.x << ", b.y " << b.y << ", thread t.x " << t.x << ", t.y " << t.y << " , INT_MAX="
         << INT_MAX  << ", DBL_MAX= " << DBL_MAX << ", LLONG_MAX=" << LLONG_MAX <<endl;

            K_5opt_oneThreadOne5opt_qiao_stride_iter _KER_CALL_(b, t) (nn_source, linkCoordTourGpu, n , maxChecks2opt, maxChecks4opt, maxChecks4optDivide, iter, istride);// global

    //    K_5opt_oneThreadOne5opt_qiao_stride_iter_shared _KER_CALL_(b, t) (nn_source, linkCoordTourGpu, n , maxChecks2opt, maxChecks4opt, maxChecks4optDivide, iter, istride);// global

    //    K_5opt_oneThreadOne5opt_qiao_stride_iter_shared_noOccupy _KER_CALL_(b, t) (nn_source, linkCoordTourGpu, n , maxChecks2opt, maxChecks4opt, maxChecks4optDivide, iter, istride);// global

//    K_5opt_oneThreadOne5opt_qiao_stride_iter_shared_onlyPossibility _KER_CALL_(b, t) (nn_source, linkCoordTourGpu, n , maxChecks2opt, maxChecks4opt, maxChecks4optDivide, iter, istride);// global


    cudaChk(cudaPeekAtLastError());
}



//! 0624 QWB: add parallel 3opt one thread one 3-opt with Rocki small
template <template<typename, typename> class NeuralNetLinks, class BufferDimension, class Point
          >
GLOBAL inline void K_oneThreadOne3opt_RockiSmall( NeuralNetLinks<BufferDimension, Point>& nn_source,
                                                  Grid<doubleLinkedEdgeForTSP>& linkCoordTourGpu,
                                                  double maxChecks3opt,
                                                  double iter
                                                  ) {


    KER_CALL_THREAD_BLOCK_1D_fix(b, t,
                                 BLOCKSIZE,
                                 16,
                                 GRIDSIZE, //for rocki large global
                                 // maxChecks/BLOCKSIZE + 1, // for rocki large global
                                 nn_source.adaptiveMap.width);


    cout << "grid blocks : " << b.x << ", b.y " << b.y << ", thread t.x " << t.x << ", t.y " << t.y << " blocks="  <<  ", INT_MAX="
         << INT_MAX  << ", DBL_MAX= " << DBL_MAX << ", LLONG_MAX=" << LLONG_MAX <<endl;


    K_3opt_oneThreadOne3opt_rockiSmall_iter _KER_CALL_(b, t) (nn_source, linkCoordTourGpu, maxChecks3opt, iter);// global
    K_3opt_oneThreadOne3opt_rockiSmall_iterBest _KER_CALL_(b, t) (nn_source, linkCoordTourGpu, maxChecks3opt, iter);// global



    //        KER_CALL_THREAD_BLOCK(b, t,BLOCKSIZE,1, maxChecks3opt, 1);

    //        double blocks =  (maxChecks3opt + BLOCKSIZE - 1) / BLOCKSIZE ;
    //        cout << "thread block : " << b.x << ", b.y " << b.y << ", t.x " << t.x << ", t.y " << t.y << " blocks=" << blocks <<  ", INT_MAX="
    //             << INT_MAX  << ", DBL_MAX= " << DBL_MAX << ", LLONG_MAX=" << LLONG_MAX <<endl;

    //        if(blocks > 2147483647)
    //        { cout << "Error Grid size bigger than maximum >>>>>>>>>>>>>>>> " << endl;
    //            return;
    //        }

    //        K_3opt_oneThreadOne3opt_rockiSmall _KER_CALL_(b, t) (nn_source, linkCoordTourGpu, maxChecks3opt, iter);// global


    cudaChk(cudaPeekAtLastError());
}




//! 0624 QWB: add parallel 3opt one thread one 3-opt with Rocki small work correctly donot change any number
template <template<typename, typename> class NeuralNetLinks, class BufferDimension, class Point
          >
GLOBAL inline void K_oneThreadOne3opt_qiao_stride( NeuralNetLinks<BufferDimension, Point>& nn_source,
                                                   Grid<doubleLinkedEdgeForTSP>& linkCoordTourGpu,
                                                   double maxChecks3opt, double maxChecksoOptDivide,
                                                   double iter, double istride
                                                   ) {


    KER_CALL_THREAD_BLOCK_1D_fix(b, t,
                                 BLOCKSIZE,
                                 16,
                                 GRIDSIZE, //for rocki large global
                                 // maxChecks/BLOCKSIZE + 1, // for rocki large global
                                 nn_source.adaptiveMap.width);


    cout << "grid blocks : " << b.x << ", b.y " << b.y << ", thread t.x " << t.x << ", t.y " << t.y << " blocks="  <<  ", INT_MAX="
         << INT_MAX  << ", DBL_MAX= " << DBL_MAX << ", LLONG_MAX=" << LLONG_MAX <<endl;

    //correct
    // K_3opt_oneThreadOne3opt_rockiSmall_iterStrideBest _KER_CALL_(b, t) (nn_source, linkCoordTourGpu, maxChecks3opt,maxChecksoOptDivide, iter, istride);// global

    //correct
    // K_3opt_oneThreadOne3opt_rockiSmall_iterStrideBest_shared  _KER_CALL_(b, t) (nn_source, linkCoordTourGpu, maxChecks3opt,maxChecksoOptDivide, iter, istride);// global


    //correct
    // K_3opt_oneThreadOne3opt_rockiSmall_iterStride _KER_CALL_(b, t) (nn_source, linkCoordTourGpu, maxChecks3opt,maxChecksoOptDivide, iter, istride);// global

    //correct 93opts  6650.7ms
    //    K_3opt_oneThreadOne3opt_rockiSmall_iterStride_shared _KER_CALL_(b, t) (nn_source, linkCoordTourGpu, maxChecks3opt,maxChecksoOptDivide, iter, istride);// global

    //correct produce half quantity of opts than do not use sharedOccupy 6670ms
    K_3opt_oneThreadOne3opt_rockiSmall_iterStride_sharedOccupy  _KER_CALL_(b, t) (nn_source, linkCoordTourGpu, maxChecks3opt,maxChecksoOptDivide, iter, istride);// global

    //error does not work to build sharedMem while write it on another place
    //K_3opt_oneThreadOne3opt_rockiSmall_iterStride_onlySharedOccupy _KER_CALL_(b, t) (nn_source, linkCoordTourGpu, maxChecks3opt,maxChecksoOptDivide, iter, istride);// global


    cudaChk(cudaPeekAtLastError());
}


//! 0624 QWB: add parallel 3opt one thread one 3-opt with Rocki small
template <template<typename, typename> class NeuralNetLinks, class BufferDimension, class Point
          >
GLOBAL inline void K_oneThreadOne6opt_RockiSmall( NeuralNetLinks<BufferDimension, Point>& nn_source,
                                                  Grid<doubleLinkedEdgeForTSP>& linkCoordTourGpu,
                                                  double maxChecks6opt, double maxChecks3opt,
                                                  unsigned int iter
                                                  ) {

    KER_CALL_THREAD_BLOCK_1D_fix(b, t,
                                 BLOCKSIZE, 16,
                                 GRIDSIZE, //for rocki large global
                                 // maxChecks/BLOCKSIZE + 1, // for rocki large global
                                 nn_source.adaptiveMap.width);

    double blocks =  (maxChecks6opt + BLOCKSIZE - 1) / BLOCKSIZE ;
    cout << "thread block : " << b.x << ", b.y " << b.y << ", t.x " << t.x << ", t.y " << t.y << " blocks=" << blocks <<  ", INT_MAX="
         << INT_MAX  << ", DBL_MAX= " << DBL_MAX << ", LLONG_MAX=" << LLONG_MAX <<endl;

    K_6opt_oneThreadOne6opt_rockiSmall_iter _KER_CALL_(b, t) (nn_source, linkCoordTourGpu, maxChecks6opt,maxChecks3opt, iter);// global






    //    KER_CALL_THREAD_BLOCK(b, t,BLOCKSIZE,1, maxChecks6opt, 1);
    //    double blocks =  (maxChecks6opt + BLOCKSIZE - 1) / BLOCKSIZE ;
    //    cout << "thread block : " << b.x << ", b.y " << b.y << ", t.x " << t.x << ", t.y " << t.y << " blocks=" << blocks <<  ", INT_MAX="
    //         << INT_MAX  << ", DBL_MAX= " << DBL_MAX << ", LLONG_MAX=" << LLONG_MAX <<endl;
    //    K_6opt_oneThreadOne6opt_rockiSmall _KER_CALL_(b, t) (nn_source, linkCoordTourGpu, maxChecks6opt,maxChecks3opt, iter);// global

    cudaChk(cudaPeekAtLastError());
}


//! 0624 QWB: add parallel 3opt one thread one 3-opt with Rocki small
template <template<typename, typename> class NeuralNetLinks, class BufferDimension, class Point
          >
GLOBAL inline void K_oneThreadOne6opt_qiao_iterStride( NeuralNetLinks<BufferDimension, Point>& nn_source,
                                                       Grid<doubleLinkedEdgeForTSP>& linkCoordTourGpu,
                                                       double maxChecks6opt, double maxChecks3opt, double maxChecksoOptDivide,
                                                       double iter, double iStride
                                                       ) {

    KER_CALL_THREAD_BLOCK_1D_fix(b, t,
                                 BLOCKSIZE, 16,
                                 GRIDSIZE, //for rocki large global
                                 // maxChecks/BLOCKSIZE + 1, // for rocki large global
                                 nn_source.adaptiveMap.width);

    double blocks =  (maxChecks6opt + BLOCKSIZE - 1) / BLOCKSIZE ;
    cout << "thread block : " << b.x << ", b.y " << b.y << ", t.x " << t.x << ", t.y " << t.y << " blocks=" << blocks <<  ", INT_MAX="
         << INT_MAX  << ", DBL_MAX= " << DBL_MAX << ", LLONG_MAX=" << LLONG_MAX <<endl;

        K_6opt_oneThreadOne6opt_qiao_stride_iter _KER_CALL_(b, t) (nn_source, linkCoordTourGpu, maxChecks6opt,maxChecks3opt, maxChecksoOptDivide, iter, iStride);// global

//    K_6opt_oneThreadOne6opt_qiao_stride_iter_onlySharePossibility _KER_CALL_(b, t) (nn_source, linkCoordTourGpu, maxChecks6opt,maxChecks3opt, maxChecksoOptDivide, iter, iStride);// global


    cudaChk(cudaPeekAtLastError());
}


//! WB.Q add to returenChangeLinks from one node
template <template<typename, typename> class NeuralNetLinks, class BufferDimension, class Point>
__device__ void K_returnChangeLinks(NeuralNetLinks<BufferDimension, Point> nn_source,
                                    PointCoord node1, PointCoord& node2, int& changeLink1, int& changeLink2)
{
    int N = nn_source.adaptiveMap.width;

    PointCoord node2_(0, 0);
    nn_source.networkLinks[0][node1[0]].get(0, node2_);
    node2[0] = node2_[0];
    node2[1] = node2_[1];

    // make sure node2 is in right direction of node1
    if(nn_source.grayValueMap[node1[1]][node1[0]] == N-1 && nn_source.grayValueMap[node2[1]][node2[0]] != 0 )
    {
        nn_source.networkLinks[node1[1]][node1[0]].get(1, node2_);
        node2[0] = node2_[0];
        node2[1] = node2_[1];
        changeLink1 = 1;
    }
    else if((nn_source.grayValueMap[node1[1]][node1[0]] != N-1) && nn_source.grayValueMap[node2[1]][node2[0]] - 1 != nn_source.grayValueMap[0][node1[0]] )
    {
        nn_source.networkLinks[node1[1]][node1[0]].get(1, node2_);
        node2[0] = node2_[0];
        node2[1] = node2_[1];
        changeLink1 = 1;
    }
    else
        changeLink1 = 0;

    PointCoord node1_(0, 0);
    nn_source.networkLinks[0][node2[0]].get(0, node1_);
    if((int)node1_[0] != node1[0] || (int)node1_[1] != node1[1])
    {
        changeLink2 = 1;
    }
    else
        changeLink2 = 0;
}


//! WB.Q add to execute non-iteracted 2opt only with node3
template <template<typename, typename> class NeuralNetLinks, class BufferDimension, class Point
          >
KERNEL void K_2opt_executeNonItera2optOnlyWithNode3(NeuralNetLinks<BufferDimension, Point> nn_source)
{
    KER_SCHED(nn_source.adaptiveMap.width, nn_source.adaptiveMap.height)

            if (_x < nn_source.adaptiveMap.width && _y < nn_source.adaptiveMap.height)
    {
        if(nn_source.activeMap[0][_x]){
            // execute non interact 2-opt without checking
            PointCoord node1(_x, 0);

            int node3_int = nn_source.densityMap[0][_x];
            PointCoord node3(node3_int, 0);

            // node2
            PointCoord node2(0, 0);
            PointCoord node4(0, 0);

            int changeLink1 = 0;
            int changeLink2 = 0;
            int changeLink3 = 0;
            int changeLink4 = 0;

            K_returnChangeLinks(nn_source, node1, node2, changeLink1, changeLink2);
            K_returnChangeLinks(nn_source, node3, node4, changeLink3, changeLink4);

            nn_source.networkLinks[0][node1[0]].bCell[changeLink1] = node3;
            nn_source.networkLinks[0][node3_int].bCell[changeLink3] = node1;
            nn_source.networkLinks[0][node2[0]].bCell[changeLink2] = node4;
            nn_source.networkLinks[0][node4[0]].bCell[changeLink4] = node2;


        }

    }

    END_KER_SCHED
}



//! QWB: execute non-interacted 2-exchanges only with node3
template <template<typename, typename> class NeuralNetLinks, class BufferDimension, class Point
          >
GLOBAL inline void K_executeNonItera2ExchangeOnlyWithNode3( NeuralNetLinks<BufferDimension, Point>& nn_source) {

    KER_CALL_THREAD_BLOCK_1D(b, t,
                             BLOCKSIZE, 16,
                             nn_source.adaptiveMap.width,
                             nn_source.adaptiveMap.height);
    K_2opt_executeNonItera2optOnlyWithNode3 _KER_CALL_(b, t) (nn_source);

}




//wb.Q compute odds nodes
template <typename Grid, typename Grid2 >
KERNEL inline void K_Christ_Compute_MiniCostMatching(Grid g_oddsNodesMap, Grid2 g_linkMap) {

    KER_SCHED_3D(g_linkMap.getWidth(), g_linkMap.getHeight(), g_linkMap.getDepth())

            typename Grid::index_type ps(_x, _y, _z);
    if (g_linkMap.valideIndex(ps))
    {
        typedef typename Grid::index_type index_type;
        typedef typename Grid2::point_type ss_node_type;

    }
    END_KER_SCHED_3D

            SYNCTHREADS
}



/**********************************************************************
 * K_omputeEdgeEulerTour compute each edge's euler tour next
 *********************************************************************/
template <typename Grid, typename Grid2, typename Grid3>
KERNEL inline void K_Christ_ComputeEdgeEulerTour(Grid g_odds_links,
                                                 Grid2 g_emst_links,
                                                 Grid3 g_euler_links,
                                                 GLfloat usingChristMatching) {

    KER_SCHED_3D(g_point.getWidth(), g_point.getHeight(), g_point.getDepth())

            typedef typename Grid2::index_type index_type;

    index_type ps(_x, _y, _z);
    if (g_emst_links.valideIndex(ps)) // wb.Q only because the cm is spiciallized for all odds nodes, so not use this: && g_links(ps).numLinks %2 != 0)
    {
        //for each link of ps, get edge,
        GLint numLinks = g_emst_links(ps).numLinks;

        for(int i = 0; i < numLinks; i++){

            // get ps,pco
            index_type pco = g_emst_links(ps).bCell[i];

            // access pco,ps
            GLint numPcoLinks = g_emst_links(pco).numLinks;

            GLint flag = 0;

            for(int j = 0; j < numPcoLinks-1; j++){ //wb.Q here need to consider odds matching

                index_type pcopco = g_emst_links(pco).bCell[j];

                if(pcopco == ps){ // find pco,ps
                    g_euler_links(ps).bCell[i] = g_emst_links(pco).bCell[j+1];
                    flag = 1;
                }
            }

            if(flag == 0 && numPcoLinks % 2 == 1 && usingChristMatching == 1)
            {
                g_euler_links(ps).bCell[i] = g_odds_links(pco).bCell[0];
            }
            else if(flag == 0)
            {
                g_euler_links(ps).bCell[i] = g_emst_links(pco).bCell[0];
            }
        }

        if(numLinks % 2 == 1 && usingChristMatching == 1){

            g_euler_links(ps).bCell[numLinks] = g_emst_links(ps).bCell[0];
        }



        //        usingChristMatching = 2;
        //        printf("using %f ", usingChristMatching);

        printf("\n");
    }
    END_KER_SCHED_3D

            SYNCTHREADS
}




/**********************************************************************
 * K_Christ_ShortCutSmallTspTour computes each small euler tour's TSP tour
 *********************************************************************/
template <typename Grid, typename Grid2, typename Grid3, typename Grid4, typename Grid5, typename Grid6>
KERNEL inline void K_Christ_ShortCutSmallTspTour(Grid g_odds_links,
                                                 Grid2 g_emst_links,
                                                 Grid3 g_euler_links,
                                                 Grid4 g_fixedMap,
                                                 Grid5 g_flagMap,
                                                 Grid6 g_tspResultMap,
                                                 GLfloat usingChristMatching) {

    KER_SCHED_3D(g_point.getWidth(), g_point.getHeight(), g_point.getDepth())

            typedef typename Grid2::index_type index_type;

    index_type ps(_x, _y, _z);
    if (g_emst_links.valideIndex(ps) && g_fixedMap(ps) == 1) // wb.Q start from
    {

        //start from one root of emst, traverse each euler tour,
        //use flagMap to mark which node has been occupied, if size smaller than size of component,
        //add usingChristMatching or not



        //        usingChristMatching = 2;
        //        printf("using %f ", usingChristMatching);

        printf("\n");
    }
    END_KER_SCHED_3D

            SYNCTHREADS
}

/**********************************************************************
 * K_FindNextClosestPoint for each point in Component
 *********************************************************************/
template <typename Grid, typename Grid2, typename Grid3, typename Grid4, typename Grid5, typename Grid6, typename Grid71, typename Grid7 >
KERNEL inline void K_Christ_ComputeOddsNodesNNS(Grid cm,
                                                Grid2 g_dstree,
                                                Grid3 g_point,
                                                Grid4 g_dist,
                                                Grid5 g_corr,
                                                Grid6 g_ss,
                                                Grid71 g_odds_links,
                                                Grid7 g_emst_links) {

    KER_SCHED_3D(g_point.getWidth(), g_point.getHeight(), g_point.getDepth())

            typedef typename Grid::index_type index_type_cm;
    typedef typename Grid3::index_type index_type;
    typedef typename Grid3::point_type point_type;

    index_type ps(_x, _y, _z);
    if (g_ss.valideIndex(ps) && g_emst_links(ps).numLinks % 2 != 0  && g_odds_links(ps).numLinks == 0 ) // wb.Q only because the cm is spiciallized for all odds nodes, so not use this: && g_links(ps).numLinks %2 != 0)
    {
        if (g_ss(ps).pc[0] != -1)
            g_ss(ps).search(cm, g_dstree, g_point, g_dist, g_corr, g_odds_links);
        else
            printf("erreur search\n");
    }
    END_KER_SCHED_3D

            SYNCTHREADS
}


/**********************************************************************
 * K_Christ_ComputeNodesTspAttribute for each point in oddsmap and emstmap, check their attribute to be a TSP tour
 *********************************************************************/
template <typename Grid, typename Grid2, typename Grid3, typename Grid4, typename Grid5, typename Grid6, typename Grid71, typename Grid7 >
KERNEL inline void K_Christ_ComputeNodesTspAttribute(Grid cm,
                                                     Grid2 g_dstree,
                                                     Grid3 g_point,
                                                     Grid4 g_dist,
                                                     Grid5 g_corr,
                                                     Grid6 g_ss,
                                                     Grid71 g_odds_links,
                                                     Grid7 g_emst_links) {

    KER_SCHED_3D(g_point.getWidth(), g_point.getHeight(), g_point.getDepth())

            typedef typename Grid::index_type index_type_cm;
    typedef typename Grid3::index_type index_type;
    typedef typename Grid3::point_type point_type;

    index_type ps(_x, _y, _z);

    if (g_ss.valideIndex(ps) && g_emst_links(ps).numLinks == CHRISTTHREE)
    {
        // get odds link of ps
        index_type pco = g_odds_links(ps).bCell[0];
        GLint numOddPco = g_emst_links(pco).numLinks; // 3 or 1

        //case 1
        if(numOddPco == 1)
        {
            index_type pcoPco = g_emst_links(pco).bCell[0];

            if(ps == pcoPco) // pco has only one emst node and it is ps
            {

            }
            else // pco has only one emst node, and it is not ps
            {

            }

        }
        else  // pco has three emst nodes
        {
            bool psIsPco = 0;
            for (int i = 0; i < CHRISTTHREE; i++)
            {
                index_type pcoPco = g_emst_links(pco).bCell[i];
                if(ps == pcoPco)
                {
                    psIsPco = 1;
                }
            }

            if(psIsPco == 1) // pco has three emst node, and one is ps
            {

            }
            else // pco has three emst node, but none of them is ps
            {

            }

        }


    }
    if (g_ss.valideIndex(ps) && g_emst_links(ps).numLinks == CHRISTFOUR) // wb.Q only because the cm is spiciallized for all odds nodes, so not use this: && g_links(ps).numLinks %2 != 0)
    {

        // for each link of ps compute

        // endNode1  endNode2  centerNode==ps, backNode1, backNode2

        // if centerNode sumDegree == 2, do nothing, todo: only compute sumDegree once at somewhere

        // if centerNode sumDegree == 4, it has following cases

        // for each link_i of sumDegree: count link_i NumNode3 NumNode2 NumNode4,
        // if link_i sumDegree == 2  numNode2++;  if sumDegree==4 numNode4++;  if psOddsLink==link_i emstOdd=1;
        // then
        // if numNode2==2 !=oddsLink oddLinkSumDegree4/2==1 result(ps).oneLink = psOddsLink;  result(ps).anotherLink= anyOneNum2; result(ps).backNode1=anotherNode2; result(ps).backEndNode=psOddsLink
        //(when connection, for each sumDegree==2, check its one of two link is sumLink==4 or not, if ==4,check if ps==nodeSum4.links, or ps==backNode1 ;
        // when connection, for each sumDegree==2, if emstLink==oddsLink, result(ps).oneLink=oddsLink result(ps).anotherLink=result(oddLink).backNode1 )

    }

    END_KER_SCHED_3D

            SYNCTHREADS
}



/**********************************************************************
 *
 *********************************************************************/
template <typename Grid, typename Grid2>
KERNEL inline void K_Christ_MinMatchOddsNodesCorres(Grid g_link,
                                                    Grid2 g_corres) {

    KER_SCHED_3D(g_link.getWidth(), g_link.getHeight(), g_link.getDepth())

            typedef typename Grid::index_type index_type;

    index_type ps(_x, _y, _z);
    if (g_link.valideIndex(ps) && g_link(ps).numLinks == 0 ) //  wb.Q only because the cm is spiciallized for all odds nodes, so not use this: && g_links(ps).numLinks %2 != 0)
    {
        //wb.Q check whether pco's link == ps
        index_type nnsPs(-1);
        nnsPs = g_corres(ps);

        index_type nnsPco(-1);
        nnsPco = g_corres(nnsPs);

        if(nnsPs[0] == -1 || nnsPco[0] == -1);
        //            printf("cuda side link == -1 \n");
        else if (nnsPs != ps && nnsPco == ps)
            g_link(ps).insert(nnsPs);

    }
    END_KER_SCHED_3D

            SYNCTHREADS
}





/*******************************************************
 *  Spiral Search initialisation
 *******************************************************/
//#ifdef CELLULAR_ADAPTIVE
#if CELLULAR_ADAPTIVE

//template <typename CellularMatrix, typename Grid, typename Grid2 >
//KERNEL inline void K_EMST_initializeSpiralSearch(CellularMatrix cm, Grid g_point, Grid2 g_ss) {

//    KER_SCHED_3D(g_point.getWidth(), g_point.getHeight(), g_point.getDepth())

//            typename Grid::index_type ps(_x, _y, _z);
//    if (g_point.valideIndex(ps))
//    {
//        typedef typename CellularMatrix::index_type index_type_cm;
//        typedef typename Grid::index_type index_type;
//        typedef typename Grid2::point_type ss_node_type;

//        index_type_cm pc = cm.vgd.findCell(g_point(ps)); // pc: x, y, or z.

//        GLint pcDll = pc[0] * cm.getWidth() + pc[1] * cm.getHeight() + pc[2] * cm.getDepth()
//                + g_point.getWidth() * g_point.getHeight() * g_point.getDepth() ;

////        typename Grid::index_type pcDllc(pcDll, _y, _z);


//        if (cm.valideAndPositiveIndex(pc)) {

//            GLint mine = cm.g_dll.compute_offset(ps); // ps 2D/3D, g_dll 1D
////            GLint prev = cm.g_dll.compute_offset(pcDllc);
//            GLint prev = pcDll;

//            GLint old;
//            GLint link = cm.g_dll(pcDll);

//            do {
//                old = link;
//                cm.g_dll(mine) = old;
//                link = atomicCAS(&cm.g_dll(prev), link, mine);
//            } while (link != old);

//        }

//        else {
//            g_ss(ps) = ss_node_type( index_type_cm(-1),
//                                     index_type(-1),
//                                     0,
//                                     -1);
//        }
//    }
//    END_KER_SCHED_3D

//            SYNCTHREADS
//}
// wb.Q Juin 2019 Adaptive size cellular partition using dynamic linked list, do not need static buffer.
// CellularMatrix<int> instead of CellularMatrix<BufferIndex>
//  on insère l'index en tête, et on recherche en parcourant la liste,
//  comme avec une dll, donc un seul <int> par cellule, et une seule grille dll pour tous les points,
// So, there is no need to define cellular<buffer>, change buffer to <int>
// each cell has three elements, one is the start node (or root), one is index of the end node, the last is the size
template <typename CellularMatrix, typename Grid, typename Grid2 >
KERNEL inline void K_EMST_initializeSpiralSearch(CellularMatrix cm, Grid g_point, Grid2 g_ss) {

    KER_SCHED_3D(g_point.getWidth(), g_point.getHeight(), g_point.getDepth())

            typename Grid::index_type ps(_x, _y, _z);
    if (g_point.valideIndex(ps))
    {
        typedef typename CellularMatrix::index_type index_type_cm;
        typedef typename Grid::index_type index_type;
        typedef typename Grid2::point_type ss_node_type;

        index_type_cm pc = cm.vgd.findCell(g_point(ps));  // pc: x, y, or z.

        if (cm.valideAndPositiveIndex(pc)) {
            GLint mine = cm.g_dll.compute_offset(ps);
            GLint old;
            GLint link = cm.g_cellular(pc);

            do {
                old = link;
                cm.g_dll(mine) = old;
                link = atomicCAS(&cm.g_cellular(pc), link, mine);
            } while (link != old);

            g_ss(ps).init(pc, ps, 0, MAX(cm.vgd.getWidthDual(),MAX(cm.vgd.getHeightDual(),cm.vgd.getDepthDual())));
            //            g_ss(ps).init(pc, ps, 0, cm.vgd.getWidthDual() + cm.vgd.getHeightDual() + cm.vgd.getDepthDual());
        }
        else {
            g_ss(ps) = ss_node_type( index_type_cm(-1),
                                     index_type(-1),
                                     0,
                                     -1);
        }

        //        printf("g_cm %d \n", cm.g_cellular(pc));

    }
    END_KER_SCHED_3D

            SYNCTHREADS
}
#else
template <typename CellularMatrix, typename Grid, typename Grid2 >
KERNEL inline void K_EMST_initializeSpiralSearch(CellularMatrix cm, Grid g_point, Grid2 g_ss) {

    KER_SCHED_3D(g_point.getWidth(), g_point.getHeight(), g_point.getDepth())

            typename Grid::index_type ps(_x, _y, _z);
    if (g_point.valideIndex(ps))
    {
        typedef typename CellularMatrix::index_type index_type_cm;
        typedef typename Grid::index_type index_type;
        typedef typename Grid2::point_type ss_node_type;

        index_type_cm pc = cm.vgd.findCell(g_point(ps));

        if (cm.valideAndPositiveIndex(pc)) {
            //            printf("%d , %d , %d \n",ps[0], ps[1], ps[2]);
            cm(pc).insert(ps);
            g_ss(ps).init(pc, ps, 0, MAX(cm.vgd.getWidthDual(),MAX(cm.vgd.getHeightDual(),cm.vgd.getDepthDual())));
            //            g_ss(ps).init(pc, ps, 0, cm.vgd.getWidthDual() + cm.vgd.getHeightDual() + cm.vgd.getDepthDual());
        }
        else {
            g_ss(ps) = ss_node_type( index_type_cm(-1),
                                     index_type(-1),
                                     0,
                                     -1);
        }
    }
    END_KER_SCHED_3D

            SYNCTHREADS
}
#endif




/**********************************************************************
 * K_FindNextClosestPoint for each point in Component
 *********************************************************************/
template <typename Grid, typename Grid2, typename Grid3, typename Grid4, typename Grid5, typename Grid6 >
KERNEL inline void K_EMST_computeOctants(Grid cm,
                                         Grid2 g_dstree,
                                         Grid3 g_point,
                                         Grid4 g_dist,
                                         Grid5 g_corr,
                                         Grid6 g_ss) {

    KER_SCHED_3D(g_link.getWidth(), g_link.getHeight(), g_link.getDepth())

            typedef typename Grid::index_type index_type_cm;
    typedef typename Grid2::index_type index_type;
    typedef typename Grid2::point_type point_type;

    index_type ps(_x, _y, _z);
    if (g_dstree.valideIndex(ps))
    {
        if (g_ss(ps).pc[0] != -1)
            g_ss(ps).computeOctants(cm, g_dstree, g_point, g_dist, g_corr);
        else
            printf("erreur search\n");
    }
    END_KER_SCHED_3D

            SYNCTHREADS
}

template <typename Grid, typename Grid2, typename Grid3, typename Grid4, typename Grid5, typename Grid6 >
KERNEL inline void K_EMST_computeOctant(Grid cm,
                                        Grid2 g_dstree,
                                        Grid3 g_point,
                                        Grid4 g_dist,
                                        Grid5 g_corr,
                                        Grid6 g_ss) {

    KER_SCHED_3D(g_link.getWidth(), g_link.getHeight(), g_link.getDepth())

            typedef typename Grid::index_type index_type_cm;
    typedef typename Grid2::index_type index_type;
    typedef typename Grid2::point_type point_type;

    index_type ps(_x, _y, _z);
    if (g_dstree.valideIndex(ps))
    {
        if (g_ss(ps).pc[0] != -1)
            g_ss(ps).computeOctant(cm, g_dstree, g_point, g_dist, g_corr);
        else
            printf("erreur search\n");
    }
    END_KER_SCHED_3D

            SYNCTHREADS
}

template <typename Grid, typename Grid2, typename Grid3, typename Grid4, typename Grid5, typename Grid6 >
KERNEL inline void K_EMST_findNextClosestPoint(Grid cm,
                                               Grid2 g_dstree,
                                               Grid3 g_point,
                                               Grid4 g_dist,
                                               Grid5 g_corr,
                                               Grid6 g_ss) {

    KER_SCHED_3D(g_link.getWidth(), g_link.getHeight(), g_link.getDepth())

            typedef typename Grid::index_type index_type_cm;
    typedef typename Grid2::index_type index_type;
    typedef typename Grid2::point_type point_type;

    index_type ps(_x, _y, _z);
    if (g_dstree.valideIndex(ps))
    {
        if (g_ss(ps).pc[0] != -1)
            g_ss(ps).search(cm, g_dstree, g_point, g_dist, g_corr);
        else
            printf("erreur search\n");
    }
    END_KER_SCHED_3D

            SYNCTHREADS
}

/*******************************************************
 *  Cellular Matrix management
 *******************************************************/
template <typename CellularMatrix, typename Grid >
KERNEL inline void K_EMST_refreshCell(CellularMatrix cm, Grid g) {

    KER_SCHED_3D(g.getWidth(), g.getHeight(), g.getDepth())

            typename Grid::index_type ps(_x, _y, _z);
    if (g.valideIndex(ps))
    {
        typename CellularMatrix::index_type pc(-1);

        pc = cm.vgd.findCell(g(ps));

        if (cm.valideAndPositiveIndex(pc))
            cm(pc).insert(ps);
        else
            printf("erreur index %d %d %d \n", pc[0], pc[1], pc[2]);
    }
    END_KER_SCHED_3D

            SYNCTHREADS
}

template <typename Grid, typename Grid2, typename Grid3, typename Grid4, typename Grid5 >
KERNEL inline void K_EMST_FindNextClosestPoint(Grid cm,
                                               Grid2 g_dstree,
                                               Grid3 g_point,
                                               Grid4 g_dist,
                                               Grid5 g_corr) {

    KER_SCHED_3D(g_link.getWidth(), g_link.getHeight(), g_link.getDepth())

            typedef typename Grid::index_type index_type_cm;
    typedef typename Grid2::index_type index_type;
    typedef typename Grid2::point_type point_type;

    index_type ps(_x, _y, _z);
    if (g_dstree.valideIndex(ps))
    {
        index_type_cm pc;

        // Find cell coordinate of ps
        pc = cm.vgd.findCell(g_point(ps));

        if (cm.valideAndPositiveIndex(pc)) {

            NodeSpiralSearch<index_type_cm, index_type> nss(
                        pc,
                        ps,
                        0,
                        cm.vgd.getWidthDual()+cm.vgd.getHeightDual()
                        );

            nss.search(cm, g_dstree, g_point, g_dist, g_corr);
        }
    }
    END_KER_SCHED_3D

            SYNCTHREADS
}

/*****************************************************
 * Evaluation
 *****************************************************/
template <typename Grid1, typename Grid2, typename Grid3 >
KERNEL inline void K_EMST_evaluate_ST(Grid1 g_link, Grid2 g_point, Grid3 g_obj) {

    KER_SCHED_3D(g_link.getWidth(), g_link.getHeight(), g_link.getDepth())

            typedef typename Grid1::index_type index_type;
    typedef typename Grid2::point_type point_type;

    index_type ps(_x, _y, _z);
    if (g_link.valideIndex(ps))
    {
        g_obj(ps)[obj_distr] = g_link(ps).numLinks;
        g_obj(ps)[obj_length] = 0;

        // Compute sum of outgoing lengths
        for (int i = 0; i < g_link(ps).numLinks; ++i){

            index_type pco(-1);
            g_link(ps).get(i, pco);

            g_obj(ps)[obj_length] +=
                    components::DistanceEuclidean<point_type>()(
                        g_point(ps),
                        g_point(pco)
                        );
        }
    }
    END_KER_SCHED_3D

            SYNCTHREADS
}

template <typename Grid1, typename Grid2, typename Grid3, typename Dist >
KERNEL inline void K_EMST_length_ST(Grid1 g_link, Grid2 g_point, Grid3 g_obj, Dist dist) {

    KER_SCHED_3D(g_link.getWidth(), g_link.getHeight(), g_link.getDepth())

            typedef typename Grid1::index_type index_type;
    typedef typename Grid2::point_type point_type;

    index_type ps(_x, _y, _z);

    if (g_link.valideIndex(ps))
    {
        g_obj(ps)[obj_length] = 0;

        // Compute sum of outgoing lengths
        for (int i = 0; i < g_link(ps).numLinks; i ++){

            index_type pco(-1);
            g_link(ps).get(i, pco);

            g_obj(ps)[obj_length] +=
                    dist(
                        g_point(ps),
                        g_point(pco)
                        );
        }

    }
    END_KER_SCHED_3D

            SYNCTHREADS
}

/*******************************************************
 * Disjoint Set Tree Initialisation
 ******************************************************/
template <typename Grid >
KERNEL inline void K_EMST_initDisjointSet(Grid g) {

    KER_SCHED_3D(g.getWidth(), g.getHeight(), g.getDepth())

            typename Grid::index_type idx(_x, _y, _z);
    if (g.valideIndex(idx))
    {
        g(idx) = g.compute_offset(idx);// / sizeof(Grid::point_type);
    }
    END_KER_SCHED_3D

            SYNCTHREADS
}

template <class CellularMatrix, class Grid >
GLOBAL inline void K_EMST_refreshCell_cpu(CellularMatrix& cm, Grid& g) {

    for (int _z = 0; _z < (g.getDepth()); ++_z) {
        for (int _y = 0; _y < (g.getHeight()); ++_y) {
            for (int _x = 0; _x < (g.getWidth()); ++_x) {

                if (_x < g.getWidth() && _y < g.getHeight() && _z < g.getDepth())
                {
                    //Grid::index_type
                    PointCoord ps(_x, _y, _z);
                    //CellularMatrix::index_type
                    PointCoord minP(-1);

                    minP = cm.vgd.findCell(g[ps[1]][ps[0]]);
                    //printf("index( %d , %d ) : p( %f , %f ) : minP( %d , %d ) \n", ps[0], ps[1], g[ps[1]][ps[0]][0], g[ps[1]][ps[0]][1], minP[0], minP[1]);

                    if (minP[0] >= 0 && minP[0] < cm.getWidth() && minP[1] >= 0 && minP[1] < cm.getHeight())
                        cm[minP[1]][minP[0]].insert_cpu(ps);
                }

            }}}

}

/*******************************************************
 *  Clear Links
 *******************************************************/

template <typename Grid >
KERNEL inline void K_EMST_clearLinks(Grid g) {

    KER_SCHED_3D(g.getWidth(), g.getHeight(), g.getDepth())

            typename Grid::index_type ps(_x, _y, _z);
    if (g.valideIndex(ps))
    {
        g(ps).clearLinks();
    }
    END_KER_SCHED_3D

            SYNCTHREADS
}

/**********************************************************************
 * Flatten Disjoint Set Tree
 *********************************************************************/
template <typename Grid >
KERNEL inline void K_EMST_flatten_DST(Grid g_dstree) {

    KER_SCHED_3D(g_dstree.getWidth(), g_dstree.getHeight(), g_dstree.getDepth())

            typedef typename Grid::index_type index_type;

    index_type ps(_x, _y, _z);
    if (g_dstree.valideIndex(ps))
    {
        // find root and set as parent
        //g_dstree(ps) = g_dstree.findRoot(g_dstree, g_dstree.compute_offset(ps));
        GLint r = g_dstree.findRoot(g_dstree, g_dstree.compute_offset(ps));
        atomicExch(&(g_dstree(ps)), r);
    }
    END_KER_SCHED_3D

            SYNCTHREADS
}

template <typename Grid, typename Grid2 >
KERNEL inline void K_EMST_flatten_DST_1(Grid g_dstree, Grid2 g_parent) {

    KER_SCHED_3D(g_dstree.getWidth(), g_dstree.getHeight(), g_dstree.getDepth())

            typedef typename Grid::index_type index_type;

    index_type ps(_x, _y, _z);
    if (g_dstree.valideIndex(ps))
    {
        GLint r = g_dstree.compute_offset(ps);
        //GLint i = 0;
        while(r != g_dstree(r) /*&& ++i <= 10*/)
        {
            r = g_dstree(r); // r is the root
        }
        g_parent(ps) = r;
    }
    END_KER_SCHED_3D

            SYNCTHREADS
}
template <typename Grid, typename Grid2 >
KERNEL inline void K_EMST_flatten_DST_2(Grid g_dstree, Grid2 g_parent) {

    KER_SCHED_3D(g_dstree.getWidth(), g_dstree.getHeight(), g_dstree.getDepth())

            typedef typename Grid::index_type index_type;

    index_type ps(_x, _y, _z);
    if (g_dstree.valideIndex(ps))
    {
        g_dstree(ps) = g_parent(ps);
    }
    END_KER_SCHED_3D

            SYNCTHREADS
}

/**********************************************************************
 * Find Min of Component
 *********************************************************************/
template <class Grid1,
          class Grid2,
          class Grid3,
          class Grid4,
          class Grid5,
          class Grid6,
          class Grid7,
          class Grid8,
          class Grid9,
          class Grid10,
          class Grid11
          >
KERNEL void K_EMST_findMinInComponentDB(Grid1 g_link,
                                        Grid2 g_dstree,
                                        Grid3 g_fix,
                                        Grid4 g_corr,
                                        Grid5 g_dist,
                                        Grid6 g_evt,
                                        Grid7 g_vnum,
                                        Grid8 g_parent,
                                        Grid9 g_win,
                                        Grid10 g_dest,
                                        Grid11 g_minD,
                                        int state,
                                        int final_state
                                        ) {

    KER_SCHED_3D(g_link.getWidth(), g_link.getHeight(), g_link.getDepth())

            typedef typename Grid1::index_type index_type;

    index_type ps(_x, _y, _z);
    if (g_link.valideIndex(ps))
    {
        index_type win_node(-1);
        index_type dest_node(-1);
        GLfloat minDist = INFINITY;

        if (g_dstree(ps) == g_dstree.compute_offset(ps)) {
            atomicAdd(&(g_evt(ps)), 1);
        }

        //int state = -2;
        while (state != final_state) {
            if (state == -1) {// visited

                if (atomicAdd(&(g_vnum(ps)), 0) == 0) {

                    state++;

                    index_type p2 = g_corr(ps);
                    if (p2 != index_type(-1)) {
                        win_node = ps;
                        dest_node = p2;
                        minDist = g_dist(ps);
                    }

                    // Compute sum of outgoing lengths
                    for (int i = 0; i < g_link(ps).numLinks; ++i){

                        index_type pco(-1);
                        g_link(ps).get(i, pco);

                        if (g_parent.compute_offset(pco) != g_parent(ps)) {

                            index_type w_node = pco.atomicAddition(&(g_win(pco)), 0);
                            index_type d_node = pco.atomicAddition(&(g_dest(pco)), 0);
                            GLfloat minD = atomicAdd(&(g_minD(pco)), 0);

                            if (w_node != index_type(-1) && p2 != index_type(-1)) {
                                if (EMST_isInf(
                                            g_link.compute_offset(w_node),
                                            g_link.compute_offset(d_node),
                                            minD,
                                            g_link.compute_offset(win_node),
                                            g_link.compute_offset(dest_node),
                                            minDist)) {
                                    win_node = w_node;
                                    dest_node = d_node;
                                    minDist = minD;
                                }
                            }
                            else if (p2 == index_type(-1)) {
                                win_node = w_node;
                                dest_node = d_node;
                                minDist = minD;
                            }
                        }//not parent
                    }
                    // Memorize Winner node
                    ps.atomicExchange(&(g_win(ps)), win_node);
                    ps.atomicExchange(&(g_dest(ps)), dest_node);
                    atomicExch(&(g_minD(ps)), minDist);

                    if (g_dstree(ps) == g_dstree.compute_offset(ps)) {
                        if (win_node[0] != -1)//index_type(-1))
                            g_fix(win_node) = true;
                    }
                    else
                        atomicAdd(&(g_vnum(g_parent(ps))), -1);
                }
                //                else
                //                    state++;
            }
            else
                if (state == -2){

                    if (atomicAdd(&(g_evt(ps)), 0)) {
                        state++;
                        int numLinks = g_link(ps).numLinks;
                        if (g_dstree(ps) != g_dstree.compute_offset(ps)) {
                            numLinks -= 1;
                        }
                        atomicExch(&(g_vnum(ps)), numLinks);
                        for (int i = 0; i < g_link(ps).numLinks; ++i){
                            index_type pco(-1);
                            g_link(ps).get(i, pco);

                            // test to avoid parent
                            if (g_parent.compute_offset(pco)
                                    != g_parent(ps)) {
                                atomicExch(&(g_parent(pco)), g_parent.compute_offset(ps));
                                atomicAdd(&(g_evt(pco)), 2);
                            }
                        }
                    }
                }
        }
    }
    END_KER_SCHED_3D

            SYNCTHREADS
}

template <class Grid1,
          class Grid2,
          class Grid6,
          class Grid7,
          class Grid8
          >
KERNEL void K_EMST_findMinDB_1(Grid1 g_link,
                               Grid2 g_dstree,
                               Grid6 g_evt,
                               Grid7 g_vnum,
                               Grid8 g_parent
                               ) {

    KER_SCHED_3D(g_link.getWidth(), g_link.getHeight(), g_link.getDepth())

            typedef typename Grid1::index_type index_type;

    index_type ps(_x, _y, _z);
    if (g_link.valideIndex(ps))
    {

        if (g_dstree(ps) == g_dstree.compute_offset(ps)) {
            atomicAdd(&(g_evt(ps)), 1);
        }

        int state = -2;
        while (state != -1) {
            if (state == -2){

                if (atomicAdd(&(g_evt(ps)), 0)) {
                    state++;
                    int numLinks = g_link(ps).numLinks;
                    if (g_dstree(ps) != g_dstree.compute_offset(ps)) {
                        numLinks -= 1;
                    }
                    atomicExch(&(g_vnum(ps)), numLinks);
                    for (int i = 0; i < g_link(ps).numLinks; ++i){
                        index_type pco(-1);
                        g_link(ps).get(i, pco);

                        // test to avoid parent
                        if (g_parent.compute_offset(pco)
                                != g_parent(ps)) {
                            atomicExch(&(g_parent(pco)), g_parent.compute_offset(ps));
                            atomicAdd(&(g_evt(pco)), 2);
                        }
                    }
                }
            }
        }
    }
    END_KER_SCHED_3D

            SYNCTHREADS
}

template <class Grid1,
          class Grid2,
          class Grid3,
          class Grid4,
          class Grid5,
          class Grid6,
          class Grid7,
          class Grid8,
          class Grid9,
          class Grid10,
          class Grid11
          >
KERNEL void K_EMST_findMinDB_2(Grid1 g_link,
                               Grid2 g_dstree,
                               Grid3 g_fix,
                               Grid4 g_corr,
                               Grid5 g_dist,
                               Grid6 g_evt,
                               Grid7 g_vnum,
                               Grid8 g_parent,
                               Grid9 g_win,
                               Grid10 g_dest,
                               Grid11 g_minD
                               ) {

    KER_SCHED_3D(g_link.getWidth(), g_link.getHeight(), g_link.getDepth())

            typedef typename Grid1::index_type index_type;

    index_type ps(_x, _y, _z);
    if (g_link.valideIndex(ps))
    {
        index_type win_node(-1);
        index_type dest_node(-1);
        GLfloat minDist = INFINITY;

        int state = -1;
        while (state != 0) {
            if (state == -1) {// visited

                if (atomicAdd(&(g_vnum(ps)), 0) == 0) {

                    state++;

                    index_type p2 = g_corr(ps);
                    if (p2 != index_type(-1)) {
                        win_node = ps;
                        dest_node = p2;
                        minDist = g_dist(ps);
                    }

                    // Compute sum of outgoing lengths
                    for (int i = 0; i < g_link(ps).numLinks; ++i){

                        index_type pco(-1);
                        g_link(ps).get(i, pco);

                        if (g_parent.compute_offset(pco) != g_parent(ps)) {

                            index_type w_node = pco.atomicAddition(&(g_win(pco)), 0);
                            index_type d_node = pco.atomicAddition(&(g_dest(pco)), 0);
                            GLfloat minD = atomicAdd(&(g_minD(pco)), 0);

                            if (w_node != index_type(-1) && p2 != index_type(-1)) {
                                if (EMST_isInf(
                                            g_link.compute_offset(w_node),
                                            g_link.compute_offset(d_node),
                                            minD,
                                            g_link.compute_offset(win_node),
                                            g_link.compute_offset(dest_node),
                                            minDist)) {
                                    win_node = w_node;
                                    dest_node = d_node;
                                    minDist = minD;
                                }
                            }
                            else if (p2 == index_type(-1)) {
                                win_node = w_node;
                                dest_node = d_node;
                                minDist = minD;
                            }
                        }//not parent
                    }
                    // Memorize Winner node
                    ps.atomicExchange(&(g_win(ps)), win_node);
                    ps.atomicExchange(&(g_dest(ps)), dest_node);
                    atomicExch(&(g_minD(ps)), minDist);

                    if (g_dstree(ps) == g_dstree.compute_offset(ps)) {
                        if (win_node[0] != -1)//index_type(-1))
                            g_fix(win_node) = true;
                    }
                    else
                        atomicAdd(&(g_vnum(g_parent(ps))), -1);
                }
                //                else
                //                    state++;
            }
        }
    }
    END_KER_SCHED_3D

            SYNCTHREADS
}

template <class Grid1,
          class Grid2,
          class Grid6,
          class Grid7,
          class Grid8,
          class Grid12
          >
KERNEL void K_EMST_findMinDBActivate_1(Grid1 g_link,
                                       Grid2 g_dstree,
                                       Grid6 g_evt,
                                       Grid7 g_vnum,
                                       Grid8 g_parent,
                                       Grid12 g_state
                                       ) {

    KER_SCHED_3D(g_link.getWidth(), g_link.getHeight(), g_link.getDepth())

            typedef typename Grid1::index_type index_type;

    index_type ps(_x, _y, _z);
    if (g_link.valideIndex(ps))
    {

        //        if (g_dstree(ps) == g_dstree.compute_offset(ps)) {
        //            atomicAdd(&(g_evt(ps)), 1);
        //        }

        if (g_state(ps) == -2){

            if (atomicAdd(&(g_evt(ps)), 0)||(g_dstree(ps) == g_dstree.compute_offset(ps))) {
                g_state(ps) += 1;
                int numLinks = g_link(ps).numLinks;
                if (g_dstree(ps) != g_dstree.compute_offset(ps)) {
                    numLinks -= 1;
                }
                atomicExch(&(g_vnum(ps)), numLinks);
                for (int i = 0; i < g_link(ps).numLinks; ++i){
                    index_type pco(-1);
                    g_link(ps).get(i, pco);

                    // test to avoid parent
                    if (g_parent.compute_offset(pco)
                            != g_parent(ps)) {
                        atomicExch(&(g_parent(pco)), g_parent.compute_offset(ps));
                        atomicAdd(&(g_evt(pco)), 2);
                    }
                }
            }
        }
    }
    END_KER_SCHED_3D

            SYNCTHREADS
}

template <class Grid1,
          class Grid2,
          class Grid3,
          class Grid4,
          class Grid5,
          class Grid6,
          class Grid7,
          class Grid8,
          class Grid9,
          class Grid10,
          class Grid11,
          class Grid12
          >
KERNEL void K_EMST_findMinDBActivate_2(Grid1 g_link,
                                       Grid2 g_dstree,
                                       Grid3 g_fix,
                                       Grid4 g_corr,
                                       Grid5 g_dist,
                                       Grid6 g_evt,
                                       Grid7 g_vnum,
                                       Grid8 g_parent,
                                       Grid9 g_win,
                                       Grid10 g_dest,
                                       Grid11 g_minD,
                                       Grid12 g_state
                                       ) {

    KER_SCHED_3D(g_link.getWidth(), g_link.getHeight(), g_link.getDepth())

            typedef typename Grid1::index_type index_type;

    index_type ps(_x, _y, _z);
    if (g_link.valideIndex(ps))
    {
        index_type win_node(-1);
        index_type dest_node(-1);
        GLdouble minDist = HUGE_VAL;

        if (g_state(ps) == -1) {// visited

            if (g_vnum(ps) == 0) {

                g_state(ps) += 1;

                index_type p2 = g_corr(ps);
                if (p2 != index_type(-1)) {
                    win_node = ps;
                    dest_node = p2;
                    minDist = g_dist(ps);
                }

                // Compute sum of outgoing lengths
                for (int i = 0; i < g_link(ps).numLinks; ++i){

                    index_type pco(-1);
                    g_link(ps).get(i, pco);

                    if (g_parent.compute_offset(pco) != g_parent(ps)) {

                        index_type w_node = g_win(pco);
                        index_type d_node = g_dest(pco);
                        GLdouble minD = g_minD(pco);

                        if (d_node != index_type(-1) && dest_node != index_type(-1)) {

                            GLint w_nodef = g_dstree.compute_offset(w_node);
                            GLint d_nodef = g_dstree.compute_offset(d_node);
                            GLint win_nodef = g_dstree.compute_offset(win_node);
                            GLint dest_nodef = g_dstree.compute_offset(dest_node);

                            GLint id1x = MIN(w_nodef,d_nodef);
                            GLint id2x = MAX(w_nodef,d_nodef);
                            GLint idd1x = MIN(win_nodef,dest_nodef);
                            GLint idd2x = MAX(win_nodef,dest_nodef);

                            if (EMST_isInf(
                                        id1x,
                                        id2x,
                                        minD,
                                        idd1x,
                                        idd2x,
                                        minDist)) {
                                win_node = w_node;
                                dest_node = d_node;
                                minDist = minD;
                            }
                        }
                        else if (dest_node == index_type(-1)) {
                            win_node = w_node;
                            dest_node = d_node;
                            minDist = minD;
                        }
                    }//not parent
                }
                // Memorize Winner node
                ps.atomicExchange(&(g_win(ps)), win_node);
                ps.atomicExchange(&(g_dest(ps)), dest_node);
                atomicExch((GLfloat*)&(g_minD(ps)), minDist);

                if (g_dstree(ps) == g_dstree.compute_offset(ps)) {
                    if (win_node[0] != -1)//index_type(-1))
                        g_fix(win_node) = true;
                }
                else
                    atomicAdd(&(g_vnum(g_parent(ps))), -1);
            }
        }
    }
    END_KER_SCHED_3D

            SYNCTHREADS
}

template <class Grid1,
          class Grid2,
          class Grid6,
          class Grid7,
          class Grid8,
          class Grid12
          >
KERNEL void K_EMST_diffusateDetectCycle(Grid1 g_link,
                                        Grid2 g_dstree,
                                        Grid6 g_evt,
                                        Grid7 g_vnum,
                                        Grid8 g_parent,
                                        Grid12 g_state
                                        ) {

    KER_SCHED_3D(g_link.getWidth(), g_link.getHeight(), g_link.getDepth())

            typedef typename Grid1::index_type index_type;

    index_type ps(_x, _y, _z);
    if (g_link.valideIndex(ps))
    {

        //        if (g_dstree(ps) == g_dstree.compute_offset(ps)) {
        //            atomicAdd(&(g_evt(ps)), 1);
        //        }

        if (g_state(ps) == -2){

            bool isRoot = (g_dstree(ps) == g_dstree.compute_offset(ps));

            if (g_evt(ps) || isRoot) {
                g_state(ps) += 1;
                int numLinks = g_link(ps).numLinks;
                if (!isRoot) {
                    numLinks -= 1;
                }
                else
                    atomicAdd(&(g_evt(ps)), 1);
                atomicExch(&(g_vnum(ps)), numLinks);
                for (int i = 0; i < g_link(ps).numLinks; ++i){
                    index_type pco(-1);
                    g_link(ps).get(i, pco);

                    // test to avoid parent
                    if (g_parent.compute_offset(pco)
                            != g_parent(ps)) {
                        if (!g_evt(pco)) {
                            atomicExch(&(g_parent(pco)), g_parent.compute_offset(ps));
                            atomicAdd(&(g_evt(pco)), 2);
                        }
                        else
                            atomicAdd(&(g_evt(pco)), 50);
                    }
                }
            }
        }
    }
    END_KER_SCHED_3D

            SYNCTHREADS
}

template <class Grid1,
          class Grid2,
          class Grid6,
          class Grid7,
          class Grid8,
          class Grid12
          >
KERNEL void K_EMST_eliminateCycle(Grid1 g_link,
                                  Grid2 g_dstree,
                                  Grid6 g_evt,
                                  Grid7 g_vnum,
                                  Grid8 g_parent,
                                  Grid12 g_state
                                  ) {

    KER_SCHED_3D(g_link.getWidth(), g_link.getHeight(), g_link.getDepth())

            typedef typename Grid1::index_type index_type;

    index_type ps(_x, _y, _z);
    if (g_link.valideIndex(ps))
    {
        if (g_state(ps) == -2){

            g_dstree(ps) = g_dstree.compute_offset(ps);
            g_link(ps).clearLinks();

        }
    }
    END_KER_SCHED_3D

            SYNCTHREADS
}

template <class Grid1,
          class Grid2,
          class Grid6,
          class Grid7,
          class Grid8
          >
KERNEL void K_EMST_diffusateDetectCycle_2(Grid1 g_link,
                                          Grid2 g_dstree,
                                          Grid6 g_evt,
                                          Grid7 g_vnum,
                                          Grid8 g_parent
                                          ) {

    KER_SCHED_3D(g_link.getWidth(), g_link.getHeight(), g_link.getDepth())

            typedef typename Grid1::index_type index_type;

    index_type ps(_x, _y, _z);
    if (g_link.valideIndex(ps))
    {

        if (g_dstree(ps) == g_dstree.compute_offset(ps)) {
            atomicAdd(&(g_evt(ps)), 1);
        }

        int state = -2;
        while (state != -1) {
            if (state == -2){

                if (atomicAdd(&(g_evt(ps)), 0)) {
                    state++;
                    int numLinks = g_link(ps).numLinks;
                    if (g_dstree(ps) != g_dstree.compute_offset(ps)) {
                        numLinks -= 1;
                    }
                    atomicExch(&(g_vnum(ps)), numLinks);
                    for (int i = 0; i < g_link(ps).numLinks; ++i){
                        index_type pco(-1);
                        g_link(ps).get(i, pco);

                        // test to avoid parent
                        if (g_parent.compute_offset(pco)
                                != g_parent(ps)) {
                            atomicExch(&(g_parent(pco)), g_parent.compute_offset(ps));
                            atomicAdd(&(g_evt(pco)), 2);
                        }
                    }
                }
            }
        }
    }
    END_KER_SCHED_3D

            SYNCTHREADS
}

template <typename Index, typename Grid1, typename Grid2, typename Grid3 >
DEVICE_HOST void EMST_findMinOutgoingEdge(Index parent, Index p1, Index& win_node, Index& dest_node, GLfloat& minDist, Grid1& g_link, Grid2& g_corr, Grid3& g_dist) {

    typedef typename Grid1::index_type index_type;

    index_type p2 = g_corr(p1);
    if (p2[0] != -1/*index_type(-1)*/) {
        win_node = p1;
        dest_node = p2;
        minDist = g_dist(p1);
    }

    // Compute sum of outgoing lengths
    for (int i = 0; i < g_link(p1).numLinks; ++i){

        index_type pco(-1);
        g_link(p1).get(i, pco);

        if (pco != parent) {

            index_type w_node(-1);
            index_type d_node(-1);
            GLfloat minD = HUGE_VAL;

            //        index_type ppco = g_corr(pco);
            //        if (ppco[0] != -1/*index_type(-1)*/) {
            //            w_node = pco;
            //            d_node = ppco;
            //            minD = g_dens(pco);
            //        }
            if (g_dist(p1) >= 0.0f)
                EMST_findMinOutgoingEdge(p1, pco, w_node, d_node, minD, g_link, g_corr, g_dist);

            if (w_node[0] != -1/*index_type(-1)*/ && p2[0] != -1/*index_type(-1)*/) {
                if (EMST_isInf(
                            g_link.compute_offset(w_node),
                            g_link.compute_offset(d_node),
                            minD,
                            g_link.compute_offset(win_node),
                            g_link.compute_offset(dest_node),
                            minDist)) {
                    win_node = w_node;
                    dest_node = d_node;
                    minDist = minD;
                }
            }
            else if (p2[0] == -1/*index_type(-1)*/) {
                win_node = w_node;
                dest_node = d_node;
                minDist = minD;
            }

            if (g_dist(p1) >= 0.0f)
                g_dist(pco) = -1.0f;
            else if (g_dist(p1) <= -1.0f)
                g_dist(pco) = -2.0f;

        }//not parent
    }
}

template <typename Grid1, typename Grid2, typename Grid3, typename Grid4, typename Grid5 >
KERNEL void K_EMST_findMinInComponent(Grid1 g_link, Grid2 g_dstree, Grid3 g_fix, Grid4 g_corr, Grid5 g_dist) {

    KER_SCHED_3D(g_link.getWidth(), g_link.getHeight(), g_link.getDepth())

            typedef typename Grid1::index_type index_type;

    index_type ps(_x, _y, _z);
    if (g_link.valideIndex(ps))
    {
        if (g_dstree.findRoot(g_dstree, g_dstree.compute_offset(ps)) == g_dstree.compute_offset(ps)) {
            index_type win_node(-1);
            index_type dest_node(-1);
            GLfloat minDist = HUGE_VAL;

            // Search recursively
            EMST_findMinOutgoingEdge(index_type(-1), ps, win_node, dest_node, minDist, g_link, g_corr, g_dist);

            if (win_node[0] != -1/*index_type(-1)*/)
                g_fix(win_node) = true;
        }
    }
    END_KER_SCHED_3D

            SYNCTHREADS
}

template <typename Grid1, typename Grid2, typename Grid3, typename Grid4, typename Grid5 >
void K_EMST_findMinInComponent_cpu(Grid1 g_link, Grid2 g_dstree, Grid3 g_fix, Grid4 g_corr, Grid5 g_dist) {

    typedef typename Grid1::index_type index_type;

    for (int _z = 0; _z < (g_link.getDepth()); ++_z) {
        for (int _y = 0; _y < (g_link.getHeight()); ++_y) {
            for (int _x = 0; _x < (g_link.getWidth()); ++_x) {

                index_type ps(_x, _y, _z);
                if (g_link.valideIndex(ps))
                {
                    if (g_dstree.findRoot(g_dstree, g_dstree.compute_offset(ps)) == g_dstree.compute_offset(ps)) {
                        //if (g_dstree(ps) == g_dstree.compute_offset(ps)) {
                        index_type win_node(-1);
                        index_type dest_node(-1);
                        GLfloat minDist = INFINITY;

                        // Search recursively
                        EMST_findMinOutgoingEdge(index_type(-1), ps, win_node, dest_node, minDist, g_link, g_corr, g_dist);

                        if (win_node != index_type(-1))
                            g_fix(win_node) = true;
                    }
                }

            }}}
}

/**********************************************************************
 * Connect graph and Union of disjoint set components
 *********************************************************************/
template <typename Grid1, typename Grid2, typename Grid3, typename Grid4 >
KERNEL inline void K_EMST_connectComponentAndUnion(Grid1 g_link, Grid2 g_dstree, Grid3 g_fix, Grid4 g_corr) {

    KER_SCHED_3D(g_link.getWidth(), g_link.getHeight(), g_link.getDepth())

            typedef typename Grid1::index_type index_type;

    index_type ps(_x, _y, _z);
    if (g_link.valideIndex(ps))
    {
        if (g_fix(ps)){

            index_type pcorr = g_corr(ps);

            g_link(ps).insert(pcorr);

            size_t rootPS = g_dstree.findRoot(g_dstree, g_dstree.compute_offset(ps));
            size_t rootPCorr = g_dstree.findRoot(g_dstree, g_dstree.compute_offset(pcorr));

            // test to avoid double insertion
            if (!g_fix(pcorr)
                    || (g_fix(pcorr) && (g_corr(pcorr) != ps))) {
                g_link(pcorr).insert(ps);
                //g_dstree(rootPS) = rootPCorr;
                atomicExch(&(g_dstree(rootPS)), rootPCorr);
            }
            else if (g_link.compute_offset(ps) < g_link.compute_offset(pcorr)) {
                // Set new disjoint set tree root of new set (component)
                //g_dstree(rootPS) = rootPCorr;
                atomicExch(&(g_dstree(rootPCorr)), rootPS);
            }
        }// end if winner node
    }
    END_KER_SCHED_3D

            SYNCTHREADS
}


template <typename Grid1, typename Grid2, typename Grid3, typename Grid4, typename Grid5, typename Grid6 >
KERNEL inline void K_EMST_connectComponentAndUnion_2(Grid1 g_link, Grid2 g_dstree, Grid3 g_fix, Grid4 g_corr, Grid5 g_parent, Grid6 g_fixedMap) {

    KER_SCHED_3D(g_link.getWidth(), g_link.getHeight(), g_link.getDepth())

            typedef typename Grid1::index_type index_type;

    index_type ps(_x, _y, _z);
    if (g_link.valideIndex(ps))
    {
        if (g_fix(ps)){

            index_type pcorr = g_corr(ps);

            g_link(ps).insert(pcorr);

            GLint rootPS = g_parent(ps);
            GLint rootPCorr = g_parent(pcorr);

            // reduce global mem access
            bool g_fi = g_fix(pcorr);

            // test to avoid double insertion
            if (!g_fi
                    || (g_fi && (g_corr(pcorr) != ps))) {
                g_link(pcorr).insert(ps);
                g_dstree(rootPS) = rootPCorr;
            }
            else if (g_link.compute_offset(ps) < g_link.compute_offset(pcorr)) {
                g_dstree(rootPCorr) = rootPS;

                //here is the final component id where no change cased by connect-union, for christ
                g_fixedMap(rootPS) = 1;
            }
        }// end if winner node
    }
    END_KER_SCHED_3D

            SYNCTHREADS
}

// wb.q one link network not sure, just for difference in the behave of warp threads
template <typename Grid1, typename Grid2, typename Grid3, typename Grid4, typename Grid5 >
KERNEL inline void K_EMST_connectComponentAndUnion_2_forFutureOneDirection(Grid1 g_link, Grid2 g_dstree, Grid3 g_fix, Grid4 g_corr, Grid5 g_parent) {

    KER_SCHED_3D(g_link.getWidth(), g_link.getHeight(), g_link.getDepth())

            typedef typename Grid1::index_type index_type;

    index_type ps(_x, _y, _z);
    if (g_link.valideIndex(ps))
    {
        if (g_fix(ps)){

            index_type pcorr = g_corr(ps);

            GLint rootPS = g_parent(ps);
            GLint rootPCorr = g_parent(pcorr);

            // reduce global mem access
            bool g_fi = g_fix(pcorr);

            // test to avoid double insertion
            if (!g_fi
                    || (g_fi && (g_corr(pcorr) != ps))) {
                g_link(ps).insert(pcorr);
                g_link(pcorr).insert(ps);
                g_dstree(rootPS) = rootPCorr;
            }
            else if (g_link.compute_offset(ps) < g_link.compute_offset(pcorr)) {
                g_link(ps).insert(pcorr);
                g_dstree(rootPCorr) = rootPS;
            }
        }// end if winner node
    }
    END_KER_SCHED_3D

            SYNCTHREADS
}







/**
 * \brief Class EMSTOperators
 * Classe that group operators for EMST Building
 **/
template <class CellularMatrixR,
          class CellularMatrixD,
          class NetLink,
          class CellR,
          class CellD,
          class NIter,
          class NIterDual,
          class ViewG,
          class BufferDimension
          >
class EMSTOperators
{

public:
    DEVICE_HOST explicit EMSTOperators(){}

    // Initialize the values in GPU grids EMST
    GLOBAL void gpuResetValue(NetLink& nnLGpu){

        //        nnLGpu.distanceMap.gpuResetValue(infinity);// on cpu, infinity turn out to be 1
        nnLGpu.densityMap.gpuResetValue(infinity);// on cpu, infinity turn out to be 1
        nnLGpu.disjointSetMap.gpuResetValue(infinity);
        nnLGpu.activeMap.gpuResetValue(1);// QWB 241016 only for mst, specially initialize activeMap = 1, to mark the thread running shortest outgoing
        nnLGpu.fixedMap.gpuResetValue(1);// QWB 241016, only for mst, specially initialize fixedmap = 1 to mark initial winner nodes
        nnLGpu.minRadiusMap.gpuResetValue(0);
        nnLGpu.sizeOfComponentMap.gpuResetValue(1);
        PointCoord pInitial(-1);
        nnLGpu.correspondenceMap.gpuResetValue(pInitial);
    }

    // wb.Q: initialize values in CPU grids before EMST
    GLOBAL void cpuResetValue(NetLink& nnLCpu){

        //        nnLCpu.distanceMap.resetValue(infinity);
        nnLCpu.densityMap.resetValue(infinity);
        nnLCpu.disjointSetMap.resetValue(infinity);
        nnLCpu.activeMap.resetValue(1);// QWB 241016 only for mst, specially initialize activeMap = 1, to mark the thread running shortest outgoing
        nnLCpu.fixedMap.resetValue(1);// QWB 241016, only for mst, specially initialize fixedmap = 1 to mark initial winner nodes
        nnLCpu.minRadiusMap.resetValue(0);
        nnLCpu.sizeOfComponentMap.resetValue(1);
        PointCoord pInitial(-1);
        nnLCpu.correspondenceMap.resetValue(pInitial);
    }

    /**
     * KERNEL CALLS
     */

    template <class CellularMatrix, class Grid, class Grid2 >
    GLOBAL inline void K_initializeSpiralSearch(CellularMatrix& cm, Grid& g_point, Grid2& g_ss) {

        KER_CALL_THREAD_BLOCK_3D(b, t,
                                 EMST_BLOCK_SIZE, 1, 1,
                                 g_point.getWidth(),
                                 g_point.getHeight(),
                                 g_point.getDepth());

        K_EMST_initializeSpiralSearch _KER_CALL_(b, t)(cm, g_point, g_ss);
    }

    template <class CellularMatrix, class Grid >
    GLOBAL inline void K_refreshCell(CellularMatrix& cm, Grid& g) {

        KER_CALL_THREAD_BLOCK_3D(b, t,
                                 EMST_BLOCK_SIZE, 1, 1,
                                 g.getWidth(),
                                 g.getHeight(),
                                 g.getDepth());

        K_EMST_refreshCell _KER_CALL_(b, t)(cm, g);
    }

    template <class CellularMatrix, class Grid >
    GLOBAL inline void K_refreshCell_cpu(CellularMatrix& cm, Grid& g) {

        K_EMST_refreshCell_cpu (cm, g);
    }

    template <class Grid, class Grid2, class Grid3, class Grid4, class Grid5, class Grid6 >
    GLOBAL void K_computeOctants(Grid& cm, Grid2& g_dstree, Grid3& g_point, Grid4& g_dist, Grid5& g_corr, Grid6& g_ss) {

        KER_CALL_THREAD_BLOCK_3D(b, t,
                                 EMST_BLOCK_SIZE, 1, 1,
                                 g_dstree.getWidth(),
                                 g_dstree.getHeight(),
                                 g_dstree.getDepth());

        K_EMST_computeOctants _KER_CALL_(b, t)(cm, g_dstree, g_point, g_dist, g_corr, g_ss);
    }

    template <class Grid, class Grid2, class Grid3, class Grid4, class Grid5, class Grid6 >
    GLOBAL void K_computeOctant(Grid& cm, Grid2& g_dstree, Grid3& g_point, Grid4& g_dist, Grid5& g_corr, Grid6& g_ss) {

        KER_CALL_THREAD_BLOCK_3D(b, t,
                                 EMST_BLOCK_SIZE, 1, 1,
                                 g_dstree.getWidth(),
                                 g_dstree.getHeight(),
                                 g_dstree.getDepth());

        K_EMST_computeOctant _KER_CALL_(b, t)(cm, g_dstree, g_point, g_dist, g_corr, g_ss);
    }

    template <class Grid, class Grid2, class Grid3, class Grid4, class Grid5, class Grid6 >
    GLOBAL void K_findNextClosestPoint(Grid& cm, Grid2& g_dstree, Grid3& g_point, Grid4& g_dist, Grid5& g_corr, Grid6& g_ss) {

        KER_CALL_THREAD_BLOCK_3D(b, t,
                                 EMST_BLOCK_SIZE, 1, 1,
                                 g_dstree.getWidth(),
                                 g_dstree.getHeight(),
                                 g_dstree.getDepth());

        K_EMST_findNextClosestPoint _KER_CALL_(b, t)(cm, g_dstree, g_point, g_dist, g_corr, g_ss);
    }

    template <class Grid, class Grid2, class Grid3, class Grid4, class Grid5 >
    GLOBAL void K_FindNextClosestPoint(Grid& cm, Grid2& g_dstree, Grid3& g_point, Grid4& g_dist, Grid5& g_corr) {

        KER_CALL_THREAD_BLOCK_3D(b, t,
                                 EMST_BLOCK_SIZE, 1, 1,
                                 g_dstree.getWidth(),
                                 g_dstree.getHeight(),
                                 g_dstree.getDepth());

        K_EMST_FindNextClosestPoint _KER_CALL_(b, t)(cm, g_dstree, g_point, g_dist, g_corr);
    }

    /**
     * Termination test
     */
    template<class Grid>
    bool testTermination(Grid& testGrid){

        typedef typename Grid::index_type index_type;

        //GLint numTest = testGrid(index_type(0));
        GLint numTest = testGrid.findRoot(testGrid, testGrid.compute_offset(index_type(0)));
        int nComp = 0;
        for (int j = 0; j < testGrid.height; j++ )
            for (int i = 0; i < testGrid.width; i++)
            {
                index_type ps(i, j);
                GLint r = testGrid(ps);
                //GLint r = testGrid.findRoot(testGrid, testGrid.compute_offset(ps));
                if (numTest != r)
                    nComp += 1;// testGrid[j][i];
            }
        return nComp;

    }

    template <class Grid1, class Grid2, class Grid3 >
    GLOBAL void K_evaluate_ST(Grid1& g_link, Grid2& g_point, Grid3& g_obj) {

        KER_CALL_THREAD_BLOCK_3D(b, t,
                                 EMST_BLOCK_SIZE, 1, 1,
                                 g_link.getWidth(),
                                 g_link.getHeight(),
                                 g_link.getDepth());

        K_EMST_evaluate_ST _KER_CALL_(b, t)(g_link, g_point, g_obj);
    }

    template <class Grid1, class Grid2, class Grid3 >
    GLOBAL void K_length_ST(Grid1& g_link, Grid2& g_point, Grid3& g_obj) {

        KER_CALL_THREAD_BLOCK_3D(b, t,
                                 EMST_BLOCK_SIZE, 1, 1,
                                 g_link.getWidth(),
                                 g_link.getHeight(),
                                 g_link.getDepth());

        K_EMST_length_ST _KER_CALL_(b, t)(g_link, g_point, g_obj);
    }

    template <class Grid >
    GLOBAL void K_initDisjointSet(Grid& g) {

        KER_CALL_THREAD_BLOCK_3D(b, t,
                                 EMST_BLOCK_SIZE, 1, 1,
                                 g.getWidth(),
                                 g.getHeight(),
                                 g.getDepth());

        K_EMST_initDisjointSet _KER_CALL_(b, t)(g);
    }

    template <class Grid >
    GLOBAL void K_flatten_DST_0(Grid& g_dstree) {

        KER_CALL_THREAD_BLOCK_3D(b, t,
                                 EMST_BLOCK_SIZE, 1, 1,
                                 g_dstree.getWidth(),
                                 g_dstree.getHeight(),
                                 g_dstree.getDepth());

        K_EMST_flatten_DST _KER_CALL_(b, t)(g_dstree);
    }

    template <class Grid, class Grid2 >
    GLOBAL void K_flatten_DST(Grid& g_dstree, Grid2& g_parent) {

        g_parent.gpuResetValue(-1);

        KER_CALL_THREAD_BLOCK_3D(b, t,
                                 EMST_BLOCK_SIZE, 1, 1,
                                 g_dstree.getWidth(),
                                 g_dstree.getHeight(),
                                 g_dstree.getDepth());

        K_EMST_flatten_DST_1 _KER_CALL_(b, t)(g_dstree, g_parent);
        g_parent.gpuCopyDeviceToDevice(g_dstree);
    }

    template <class Grid >
    GLOBAL void K_clearLinks(Grid& g) {

        KER_CALL_THREAD_BLOCK_3D(b, t,
                                 EMST_BLOCK_SIZE, 1, 1,
                                 g.getWidth(),
                                 g.getHeight(),
                                 g.getDepth());

        K_EMST_clearLinks _KER_CALL_(b, t) (g);
    }

    template <class Grid1,
              class Grid2,
              class Grid3,
              class Grid4,
              class Grid5,
              class Grid6,
              class Grid7,
              class Grid8,
              class Grid9,
              class Grid10,
              class Grid11,
              class Grid12
              >
    GLOBAL void K_findMinInComponentDB(Grid1& g_link,
                                       Grid2& g_dstree,
                                       Grid3& g_fix,
                                       Grid4& g_corr,
                                       Grid5& g_dist,
                                       Grid6 g_evt,
                                       Grid7 g_vnum,
                                       Grid8 g_parent,
                                       Grid9 g_win,
                                       Grid10 g_dest,
                                       Grid11 g_minD,
                                       Grid12 g_state,
                                       Grid12 g_state_cpu
                                       ) {

        KER_CALL_THREAD_BLOCK_3D(b, t,
                                 EMST_BLOCK_SIZE, 1, 1,
                                 g_link.getWidth(),
                                 g_link.getHeight(),
                                 g_link.getDepth());

        g_state.gpuResetValue(-2);
        while (true) {

            K_EMST_findMinDBActivate_1 _KER_CALL_(b, t)(g_link,
                                                        g_dstree,
                                                        g_evt,
                                                        g_vnum,
                                                        g_parent,
                                                        g_state
                                                        );
            g_state_cpu.gpuCopyDeviceToHost(g_state);

            BOp op;
            int result = 0;
            op.K_sumReduction(g_state_cpu, result);
            if (result == -g_link.getWidth())
                break;
        }
        while (true) {

            K_EMST_findMinDBActivate_2 _KER_CALL_(b, t)(g_link,
                                                        g_dstree,
                                                        g_fix,
                                                        g_corr,
                                                        g_dist,
                                                        g_evt,
                                                        g_vnum,
                                                        g_parent,
                                                        g_win,
                                                        g_dest,
                                                        g_minD,
                                                        g_state
                                                        );
            g_state_cpu.gpuCopyDeviceToHost(g_state);

            BOp op;
            int result = 0;
            op.K_sumReduction(g_state_cpu, result);
            if (!result)
                break;
        }
    }

    template <class Grid1,
              class Grid2,
              class Grid6,
              class Grid7,
              class Grid8,
              class Grid12
              >
    GLOBAL void K_diffusateDetectCycleDB(Grid1& g_link,
                                         Grid2& g_dstree,
                                         Grid6 g_evt,
                                         Grid7 g_vnum,
                                         Grid8 g_parent,
                                         Grid12 g_state,
                                         Grid12 g_state_cpu
                                         ) {

        KER_CALL_THREAD_BLOCK_3D(b, t,
                                 EMST_BLOCK_SIZE, 1, 1,
                                 g_link.getWidth(),
                                 g_link.getHeight(),
                                 g_link.getDepth());

        int i = 71009;
        int result = 0;
        while (--i) {

            K_EMST_diffusateDetectCycle _KER_CALL_(b, t)(g_link,
                                                         g_dstree,
                                                         g_evt,
                                                         g_vnum,
                                                         g_parent,
                                                         g_state
                                                         );
            g_state_cpu.gpuCopyDeviceToHost(g_state);

            BOp op;
            result = 0;
            op.K_sumReduction(g_state_cpu, result);
            if (result == -g_link.getWidth())
                break;
        }
        //        K_EMST_eliminateCycle _KER_CALL_(b, t) (g_link,
        //                              g_dstree,
        //                              g_evt,
        //                              g_vnum,
        //                              g_parent,
        //                              g_state
        //                              );
        cout << "DIFFUSION !!!!!!!!!!!!!!!! " << result << endl;
    }

    template <class Grid1,
              class Grid2,
              class Grid6,
              class Grid7,
              class Grid8,
              class Grid12
              >
    GLOBAL void K_diffusateDetectCycleDB_2(Grid1& g_link,
                                           Grid2& g_dstree,
                                           Grid6 g_evt,
                                           Grid7 g_vnum,
                                           Grid8 g_parent,
                                           Grid12 g_state,
                                           Grid12 g_state_cpu
                                           ) {

        KER_CALL_THREAD_BLOCK_3D(b, t,
                                 EMST_BLOCK_SIZE, 1, 1,
                                 g_link.getWidth(),
                                 g_link.getHeight(),
                                 g_link.getDepth());


        K_EMST_diffusateDetectCycle_2 _KER_CALL_(b, t)(g_link,
                                                       g_dstree,
                                                       g_evt,
                                                       g_vnum,
                                                       g_parent
                                                       );
    }

    template <class Grid1, class Grid2, class Grid3, class Grid4, class Grid5 >
    GLOBAL void K_FindMinInComponent(Grid1& g_link, Grid2& g_dstree, Grid3& g_fix, Grid4& g_corr, Grid5& g_dist) {

        KER_CALL_THREAD_BLOCK_3D(b, t,
                                 EMST_BLOCK_SIZE, 1, 1,
                                 g_link.getWidth(),
                                 g_link.getHeight(),
                                 g_link.getDepth());

        K_EMST_findMinInComponent _KER_CALL_(b, t)(g_link, g_dstree, g_fix, g_corr, g_dist);
    }

    template <class Grid1, class Grid2, class Grid3, class Grid4, class Grid5 >
    GLOBAL void K_FindMinInComponent_cpu(Grid1& g_link, Grid2& g_dstree, Grid3& g_fix, Grid4& g_corr, Grid5& g_dist) {

        K_EMST_findMinInComponent_cpu(g_link, g_dstree, g_fix, g_corr, g_dist);
    }

    template <class Grid1, class Grid2, class Grid3, class Grid4 >
    GLOBAL void K_connectComponentAndUnion(Grid1& g_link, Grid2& g_dstree, Grid3& g_fix, Grid4& g_corr) {

        KER_CALL_THREAD_BLOCK_3D(b, t,
                                 EMST_BLOCK_SIZE, 1, 1,
                                 g_link.getWidth(),
                                 g_link.getHeight(),
                                 g_link.getDepth());

        K_EMST_connectComponentAndUnion _KER_CALL_(b, t)(g_link, g_dstree, g_fix, g_corr);
    }

    template <class Grid1, class Grid2, class Grid3, class Grid4, class Grid5, class Grid6 >
    GLOBAL void K_connectComponentAndUnion(Grid1& g_link, Grid2& g_dstree, Grid3& g_fix, Grid4& g_corr, Grid5& g_parent, Grid6& g_fixedMap) {

        g_dstree.gpuCopyDeviceToDevice(g_parent);

        KER_CALL_THREAD_BLOCK_3D(b, t,
                                 EMST_BLOCK_SIZE, 1, 1,
                                 g_link.getWidth(),
                                 g_link.getHeight(),
                                 g_link.getDepth());

        K_EMST_connectComponentAndUnion_2 _KER_CALL_(b, t)(g_link, g_dstree, g_fix, g_corr, g_parent, g_fixedMap);
    }

    template <typename Grid, typename Grid2 >
    GLOBAL void K_createComponentList(Grid& g_link, Grid& g_dstree, Grid2& g_corr) {

        // Init to -1

        KER_CALL_THREAD_BLOCK_3D(b, t,
                                 EMST_BLOCK_SIZE, 1, 1,
                                 g_dstree.getWidth(),
                                 g_dstree.getHeight(),
                                 g_dstree.getDepth());

        K_NEMST_createComponentList _KER_CALL_(b, t)(g_link, g_dstree, g_corr);
    }

    template <class Grid1,
              class Grid2,
              class Grid3,
              class Grid4,
              class Grid5
              >
    GLOBAL void K_findMinPair(Grid1& g_link,
                              Grid2& g_dstree,
                              Grid3& g_fix,
                              Grid4& g_corr,
                              Grid5& g_dist
                              )
    {

        KER_CALL_THREAD_BLOCK_3D(b, t,
                                 EMST_BLOCK_SIZE, 1, 1,
                                 g_link.getWidth(),
                                 g_link.getHeight(),
                                 g_link.getDepth());

        K_NEMST_findMinPair _KER_CALL_(b, t)(g_link,
                                             g_dstree,
                                             g_fix,
                                             g_corr,
                                             g_dist
                                             );
    }
};

}//namespace operators

#endif // EMST_OPERATORS_H
