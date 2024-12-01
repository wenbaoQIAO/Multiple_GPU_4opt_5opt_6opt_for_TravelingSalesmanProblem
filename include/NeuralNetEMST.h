#ifndef NEURALNET_EMST_H
#define NEURALNET_EMST_H
/*
 ***************************************************************************
 *
 * Author : Wenbao. Qiao, J.C. Créput
 * Creation date : Apil. 2016
 * This file contains the standard distributed graph representation with independent adjacency list
 * assigned to each vertex of the graph, namely an adjacency list where each vertex only possesses
 * a collection of its adjacency neighboring vertices.
 * It naturally follows that the graph is doubly linked since each node of a given edge has a link
 * in the node's adjacency list towards to the connected node. We call it as doubly linked vertex list (DLVL)
 * in order to distinguish DLVL from doubly linked list (DLL) or doubly connected edge list (DCEL).
 * For self-organizing irregular network applications, it is also called as "Neural Network Links".
 *
 * If you use this source codes, please reference our publication:
 * Qiao, Wen-bao, and Jean-charles Créput.
 * "Massive Parallel Self-organizing Map and 2-Opt on GPU to Large Scale TSP."
 * International Work-Conference on Artificial Neural Networks. Springer, Cham, 2017.
 *
 ***************************************************************************
 */
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <vector>
#include <cstddef>

#include "macros_cuda.h"
#include "Node.h"
#include "NeuralNet.h"
#include "Objectives.h"
#include "GridOfNodes.h"
#include "distances_matching.h"

//! WB.Q add
#include "BufferLink.h"
#define useDerive 1 //1 class nnLinks derive from nn
#define infinity 999999
#define codeBit 65535
#define codeBit12 4095
#define decodePiK 1
#define initialPrepareValue -1
#define initialPrepareValueLL 0
#define Viewer 0 //1 do not use viewer, use CellularMatrix,  0 use viewer, need to verify other two places in NeuralNet.h and NetLinks.h

using namespace std;
using namespace components;

// wenbao.Qiao add read/write networkLinks
#define NNG_SFX_NETWORK_LINKS  ".links"
#define NNG_SFX_CELLULARMATRIX_LINK_IMAGE  ".cmLinks"
#define NN_SFX_ADAPTIVE_ORIGINALMAP_TSP  ".pointsOri"

namespace components
{

template <class BufferLink, class Point>
class NeuralNetEMST : public NeuralNet<Point, GLfloatP>
{
public:
    typedef Point point_type;
    // Disjoint set map as offset values
    Grid<GLint> disjointSetMap;
    Grid<BufferLink > networkLinks;
    Grid<GLint> fixedLinks;
    Grid<PointCoord> correspondenceMap;
    Grid<GLfloat> minRadiusMap;
    Grid<Point> adaptiveMapOri;
    Grid<GLfloat> sizeOfComponentMap;

    // Working grids
    Grid<GLint> evtMap; //6-opt possiblities
    Grid<GLint> nVisitedMap; // 5-opt possibilites
    Grid<GLint> nodeParentMap;//4-opt possibilities
    Grid<PointCoord> nodeWinMap;
    Grid<PointCoord> nodeDestMap;

    Grid<unsigned long long> optCandidateMap;




public:
    DEVICE_HOST NeuralNetEMST() {}

    //    DEVICE_HOST NeuralNetEMST(int nnW, int nnH):
    //        networkLinks(nnW, nnH),
    //        fixedLinks(nnW, nnH),
    //        correspondenceMap(nnW, nnH),
    //        minRadiusMap(nnW, nnH),
    //        adaptiveMapOri(nnW, nnH),
    //        sizeOfComponentMap(nnW, nnH),
    //        NeuralNet<Point, GLfloat>(nnW, nnH)
    //    { }

    void resize(int w, int h){
        //        distanceMap.resize(w, h);
        disjointSetMap.resize(w, h);

        networkLinks.resize(w, h);
        fixedLinks.resize(w, h);
        correspondenceMap.resize(w, h);
        minRadiusMap.resize(w, h);
        adaptiveMapOri.resize(w, h);
        sizeOfComponentMap.resize(w, h);

        NeuralNet<Point, GLfloatP>::resize(w, h);
    }

    void freeMem(){
        //        distanceMap.freeMem();
        disjointSetMap.freeMem();
        networkLinks.freeMem();
        fixedLinks.freeMem();
        correspondenceMap.freeMem();
        minRadiusMap.freeMem();
        adaptiveMapOri.freeMem();
        sizeOfComponentMap.freeMem();
        NeuralNet<Point, GLfloatP>::freeMem();
    }


    void gpuResize(int w, int h){
        //        distanceMap.gpuResize(w, h);
        disjointSetMap.gpuResize(w, h);
        networkLinks.gpuResize(w, h);
        fixedLinks.gpuResize(w, h);
        correspondenceMap.gpuResize(w, h);
        minRadiusMap.gpuResize(w, h);
        adaptiveMapOri.gpuResize(w, h);
        sizeOfComponentMap.gpuResize(w, h);
        NeuralNet<Point, GLfloatP>::gpuResize(w, h);
    }

    void clone(NeuralNetEMST& nn) {
        //        distanceMap.clone(nn.distanceMap);
        disjointSetMap.clone(nn.disjointSetMap);
        networkLinks.clone(nn.networkLinks);
        fixedLinks.clone(nn.fixedLinks);
        correspondenceMap.clone(nn.correspondenceMap);
        minRadiusMap.clone(nn.minRadiusMap);
        adaptiveMapOri.clone(nn.adaptiveMapOri);
        sizeOfComponentMap.clone(nn.sizeOfComponentMap);
        NeuralNet<Point, GLfloatP>::clone(nn);
    }

    void gpuClone(NeuralNetEMST& nn) {
        //        distanceMap.gpuClone(nn.distanceMap);
        disjointSetMap.gpuClone(nn.disjointSetMap);
        networkLinks.gpuClone(nn.networkLinks);
        fixedLinks.gpuClone(nn.fixedLinks);
        correspondenceMap.gpuClone(nn.correspondenceMap);
        minRadiusMap.gpuClone(nn.minRadiusMap);
        adaptiveMapOri.gpuClone(nn.adaptiveMapOri);
        sizeOfComponentMap.gpuClone(nn.sizeOfComponentMap);
        NeuralNet<Point, GLfloatP>::gpuClone(nn);
    }
    void setIdentical(NeuralNetEMST& nn) {
        //        distanceMap.setIdentical(nn.distanceMap);
        disjointSetMap.setIdentical(nn.disjointSetMap);
        networkLinks.setIdentical(nn.networkLinks);
        fixedLinks.setIdentical(nn.fixedLinks);
        correspondenceMap.setIdentical(nn.correspondenceMap);
        minRadiusMap.setIdentical(nn.minRadiusMap);
        adaptiveMapOri.setIdentical(nn.adaptiveMapOri);
        sizeOfComponentMap.setIdentical(nn.sizeOfComponentMap);
        NeuralNet<Point, GLfloatP>::setIdentical(nn);
    }

    void gpuSetIdentical(NeuralNetEMST& nn) {
        //        distanceMap.gpuSetIdentical(nn.distanceMap);
        disjointSetMap.gpuSetIdentical(nn.disjointSetMap);
        networkLinks.gpuSetIdentical(nn.networkLinks);
        fixedLinks.gpuSetIdentical(nn.fixedLinks);
        correspondenceMap.gpuSetIdentical(nn.correspondenceMap);
        minRadiusMap.gpuSetIdentical(nn.minRadiusMap);
        adaptiveMapOri.gpuSetIdentical(nn.adaptiveMapOri);
        sizeOfComponentMap.gpuSetIdentical(nn.sizeOfComponentMap);
        NeuralNet<Point, GLfloatP>::gpuSetIdentical(nn);
    }

    void gpuCopyHostToDevice(NeuralNetEMST & gpuNeuralNetLinks){
        //        this->distanceMap.gpuCopyHostToDevice(gpuNeuralNetLinks.distanceMap);
        this->disjointSetMap.gpuCopyHostToDevice(gpuNeuralNetLinks.disjointSetMap);
        this->networkLinks.gpuCopyHostToDevice(gpuNeuralNetLinks.networkLinks);
        this->fixedLinks.gpuCopyHostToDevice(gpuNeuralNetLinks.fixedLinks);
        this->correspondenceMap.gpuCopyHostToDevice(gpuNeuralNetLinks.correspondenceMap);
        this->minRadiusMap.gpuCopyHostToDevice(gpuNeuralNetLinks.minRadiusMap);
        this->adaptiveMapOri.gpuCopyHostToDevice(gpuNeuralNetLinks.adaptiveMapOri);
        this->sizeOfComponentMap.gpuCopyHostToDevice(gpuNeuralNetLinks.sizeOfComponentMap);
        this->objectivesMap.gpuCopyHostToDevice(gpuNeuralNetLinks.objectivesMap);
        this->adaptiveMap.gpuCopyHostToDevice(gpuNeuralNetLinks.adaptiveMap);
        this->activeMap.gpuCopyHostToDevice(gpuNeuralNetLinks.activeMap);
        this->fixedMap.gpuCopyHostToDevice(gpuNeuralNetLinks.fixedMap);
        this->colorMap.gpuCopyHostToDevice(gpuNeuralNetLinks.colorMap);
        this->grayValueMap.gpuCopyHostToDevice(gpuNeuralNetLinks.grayValueMap);
        this->densityMap.gpuCopyHostToDevice(gpuNeuralNetLinks.densityMap);
    }

    void gpuCopyDeviceToHost(NeuralNetEMST & gpuNeuralNetLinks){
        //        this->distanceMap.gpuCopyDeviceToHost(gpuNeuralNetLinks.distanceMap);
        this->disjointSetMap.gpuCopyDeviceToHost(gpuNeuralNetLinks.disjointSetMap);
        this->networkLinks.gpuCopyDeviceToHost(gpuNeuralNetLinks.networkLinks);
        this->fixedLinks.gpuCopyDeviceToHost(gpuNeuralNetLinks.fixedLinks);
        this->correspondenceMap.gpuCopyDeviceToHost(gpuNeuralNetLinks.correspondenceMap);
        this->minRadiusMap.gpuCopyDeviceToHost(gpuNeuralNetLinks.minRadiusMap);
        this->adaptiveMapOri.gpuCopyDeviceToHost(gpuNeuralNetLinks.adaptiveMapOri);
        this->sizeOfComponentMap.gpuCopyDeviceToHost(gpuNeuralNetLinks.sizeOfComponentMap);
        this->objectivesMap.gpuCopyDeviceToHost(gpuNeuralNetLinks.objectivesMap);
        this->adaptiveMap.gpuCopyDeviceToHost(gpuNeuralNetLinks.adaptiveMap);
        this->activeMap.gpuCopyDeviceToHost(gpuNeuralNetLinks.activeMap);
        this->fixedMap.gpuCopyDeviceToHost(gpuNeuralNetLinks.fixedMap);
        this->colorMap.gpuCopyDeviceToHost(gpuNeuralNetLinks.colorMap);
        this->grayValueMap.gpuCopyDeviceToHost(gpuNeuralNetLinks.grayValueMap);
        this->densityMap.gpuCopyDeviceToHost(gpuNeuralNetLinks.densityMap);
    }

    void gpuCopyDeviceToDevice(NeuralNetEMST & gpuNeuralNetLinks){
        //        this->distanceMap.gpuCopyDeviceToDevice(gpuNeuralNetLinks.distanceMap);
        this->disjointSetMap.gpuCopyDeviceToDevice(gpuNeuralNetLinks.disjointSetMap);
        this->networkLinks.gpuCopyDeviceToDevice(gpuNeuralNetLinks.networkLinks);
        this->fixedLinks.gpuCopyDeviceToDevice(gpuNeuralNetLinks.fixedLinks);
        this->correspondenceMap.gpuCopyDeviceToDevice(gpuNeuralNetLinks.correspondenceMap);
        this->minRadiusMap.gpuCopyDeviceToDevice(gpuNeuralNetLinks.minRadiusMap);
        this->adaptiveMapOri.gpuCopyDeviceToDevice(gpuNeuralNetLinks.adaptiveMapOri);
        this->sizeOfComponentMap.gpuCopyDeviceToDevice(gpuNeuralNetLinks.sizeOfComponentMap);
        this->objectivesMap.gpuCopyDeviceToDevice(gpuNeuralNetLinks.objectivesMap);
        this->adaptiveMap.gpuCopyDeviceToDevice(gpuNeuralNetLinks.adaptiveMap);
        this->activeMap.gpuCopyDeviceToDevice(gpuNeuralNetLinks.activeMap);
        this->fixedMap.gpuCopyDeviceToDevice(gpuNeuralNetLinks.fixedMap);
        this->colorMap.gpuCopyDeviceToDevice(gpuNeuralNetLinks.colorMap);
        this->grayValueMap.gpuCopyDeviceToDevice(gpuNeuralNetLinks.grayValueMap);
        this->densityMap.gpuCopyDeviceToDevice(gpuNeuralNetLinks.densityMap);
        this->disjointSetMap.gpuCopyDeviceToDevice(gpuNeuralNetLinks.disjointSetMap);
    }
    //!wbQ: read just netLinks
    void readNetLinks(string str){

        int pos = this->getPos(str);
        ifstream fi;
        string str_sub = str.substr(0, pos);
        str_sub.append(NNG_SFX_NETWORK_LINKS);
        fi.open(str_sub.c_str() );
        if (!fi) {
            std::cout << "erreur read: not exits "<< str_sub << endl; }
        else
        {
            std::cout << "read netLinks from: "<< str_sub << endl;
            int _w = 0;
            int _h = 0;
            fi >> str >> str >> _w;
            fi >> str >> str >> _h;
            networkLinks.resize(_w, _h);
            fixedLinks.resize(_w, _h);
            NeuralNet<Point, GLfloat>::resize(_w, _h);
        }

        char strLink[256];
        while(fi >> strLink){
            int y = 0;
            int x = 0;
            fi  >> strLink >> x >> y;
            if (y > networkLinks.height || x > networkLinks.width){
                cout << "error: fail read network links, links over range." << endl;
                cout << "over range y = " << y << " , over range x = " << x << endl;
                fi.close();
            }
            else
            {
                fi >> networkLinks[y][x];
                this->fixedLinks[y][x] = 1;
            }
        } ;
        fi.close();
    }


    //! 290416 QWB add to just write netLinks, use "write" to write otherMaps
    void writeLinks(string str, string strFileInput){

        cout << "Enter write links and adaptiveMap " << endl;

        int pos= this->getPos(str);
        string str_sub;

        int pos_input = getPos(strFileInput);
        string inputName;
        inputName = strFileInput.substr(0, pos_input);

        ofstream fo;

        if(networkLinks.width != 0 && networkLinks.height != 0){
            str_sub = str.substr(0, pos);
            str_sub.append("_");
            str_sub.append(inputName);
            str_sub.append(NNG_SFX_NETWORK_LINKS);
            fo.open(str_sub.c_str() );
            if (!fo) {
                std::cout << "erreur write " << str_sub << endl; }
            else
            {
                cout << "write netWorkLinks to: " << str_sub << endl;
                fo << "Width = " << networkLinks.width << " ";
                fo << "Height = " << networkLinks.height << " " << endl;
            }
            for (int y = 0; y < networkLinks.height; y ++)
                for (int x = 0; x < networkLinks.width; x++)
                {
                    if (this->networkLinks[y][x].numLinks >= 0){
                        fo << endl << "NodeP = " << x << " " << y;
                        fo << networkLinks[y][x];
                    }
                }
            fo.close();
        }
        else
            cout << "Error writeLinks: this NN does not have netLinks." << endl;

        //! write adaptiveMap;
        if(this->adaptiveMap.width != 0 && this->adaptiveMap.height != 0) {
            str_sub = str.substr(0, pos);
            str_sub.append("_");
            str_sub.append(inputName);

            str_sub.append(NN_SFX_ADAPTIVE_MAP_IMAGE);
            fo.open(str_sub.c_str());
            if (!fo) {
                std::cout << "erreur  write " << str_sub << endl; }
            fo << this->adaptiveMap;
            fo.close();
        }

        //        if(correspondenceMap.width!=0 || correspondenceMap.height!=0){
        //            str_sub = str.substr(0, pos);
        //            str_sub.append(".correspondenceMap");
        //            fo.open(str_sub.c_str() );
        //            if (!fo) {
        //                std::cout << "erreur write "<< str_sub << endl; }
        //            fo << correspondenceMap;
        //            fo.close();
        //        }

        //        if(disjointSetMap.width!=0 || disjointSetMap.height!=0){
        //            str_sub = str.substr(0, pos);
        //            str_sub.append(".disjointSetMap");
        //            fo.open(str_sub.c_str() );
        //            if (!fo) {
        //                std::cout << "erreur write "<< str_sub << endl; }
        //            fo << disjointSetMap;
        //            fo.close();
        //        }

    }

    //! 290416 QWB add to just write netLinks, use "write" to write otherMaps
    void writeLinks(string str){

        int pos= this->getPos(str);
        string str_sub;
        ofstream fo;

        if(networkLinks.width != 0 && networkLinks.height != 0){
            str_sub = str.substr(0, pos);
            str_sub.append(NNG_SFX_NETWORK_LINKS);
            fo.open(str_sub.c_str() );
            if (!fo) {
                std::cout << "erreur write " << str_sub << endl; }
            else
            {
                cout << "write netWorkLinks to: " << str_sub << endl;
                fo << "Width = " << networkLinks.width << " ";
                fo << "Height = " << networkLinks.height << " " << endl;
            }
            for (int y = 0; y < networkLinks.height; y ++)
                for (int x = 0; x < networkLinks.width; x++)
                {
                    if (this->networkLinks[y][x].numLinks >= 0){
                        fo << endl << "NodeP = " << x << " " << y;
                        fo << networkLinks[y][x];
                    }
                }
            fo.close();
        }
        else
            cout << "Error writeLinks: this NN does not have netLinks." << endl;

        if(correspondenceMap.width!=0 || correspondenceMap.height!=0){
            str_sub = str.substr(0, pos);
            str_sub.append(".correspondenceMap");
            fo.open(str_sub.c_str() );
            if (!fo) {
                std::cout << "erreur write "<< str_sub << endl; }
            fo << correspondenceMap;
            fo.close();
        }

        if(disjointSetMap.width!=0 || disjointSetMap.height!=0){
            str_sub = str.substr(0, pos);
            str_sub.append(".disjointSetMap");
            fo.open(str_sub.c_str() );
            if (!fo) {
                std::cout << "erreur write "<< str_sub << endl; }
            fo << disjointSetMap;
            fo.close();
        }

    }

    //! WB.Q 101216 add to read original map for TSP
    void readOriginalMap(string str) {

        int pos = this->getPos(str);
        ifstream fi;
        //! read adaptiveMapOri
        string str_sub = str.substr(0, pos);
        str_sub.append(NN_SFX_ADAPTIVE_ORIGINALMAP_TSP);
        fi.open(str_sub.c_str() );
        if (!fi) {
            std::cout << "erreur read " << str_sub << endl; }
        fi >> adaptiveMapOri;
        fi.close();

    }

    //! WB.Q 101216 add to write original map for TSP
    void writeOriginalMap(string str){

        int pos = this->getPos(str);
        string str_sub;

        ofstream fo;
        //! write adaptiveMapOri;
        if(adaptiveMapOri.width != 0 && adaptiveMapOri.height != 0){
            str_sub = str.substr(0, pos);
            str_sub.append(NN_SFX_ADAPTIVE_ORIGINALMAP_TSP);
            fo.open(str_sub.c_str());
            if (!fo) {
                std::cout << "erreur  write " << str_sub << endl; }
            fo << adaptiveMapOri;
            fo.close();
        }
    }

    //!wb.Q: this function computes the total distance of a netLink
    template <typename Distance>
    double evaluateWeightOfSingleTree_Recursive(Distance dist)
    {
        //        this->grayValueMap.resetValue(0); // to count the number of nodes

        double totalWeight = 0;
        cout << ">>>> evaluateWeightOfSingleTree_Recursive: " << endl;

        vector<PointCoord> nodeAlreadyTraversed;
        nodeAlreadyTraversed.clear();

        PointCoord ps(0);

        evaluateWeightOfSingleTreeRecursive(ps, nodeAlreadyTraversed, totalWeight, dist);

        return totalWeight;
    }

    //! QWB setting all nodes between node2 and node3 can not execute 2-opt, do not use recursive function
    void setFixedlinksBetweenNode2Node3(PointCoord node1, PointCoord node2, PointCoord node3){

        // get the first node between node2 and node3
        Point2D pLinkOfNode2;
        PointCoord pco2(-1);
        this->networkLinks[node2[1]][node2[0]].get(0, pLinkOfNode2);
        pco2[0] = (int)pLinkOfNode2[0];
        pco2[1] = (int)pLinkOfNode2[1];
        if(pco2 == node1){
            this->networkLinks[node2[1]][node2[0]].get(1, pLinkOfNode2);
            pco2[0] = (int)pLinkOfNode2[0];
            pco2[1] = (int)pLinkOfNode2[1];
        }
        this->fixedMap[node2[1]][node2[0]] = 1;
        this->fixedMap[pco2[1]][pco2[0]] = 1;

        PointCoord nodeAvant;
        nodeAvant = node2;

        while(pco2 != node3){

            PointCoord pcoT(-1);
            this->networkLinks[pco2[1]][pco2[0]].get(0, pLinkOfNode2);
            pcoT[0] = (int)pLinkOfNode2[0];
            pcoT[1] = (int)pLinkOfNode2[1];
            if(pcoT == nodeAvant){
                this->networkLinks[pco2[1]][pco2[0]].get(1, pLinkOfNode2);
                pcoT[0] = (int)pLinkOfNode2[0];
                pcoT[1] = (int)pLinkOfNode2[1];
            }

            this->fixedMap[pcoT[1]][pcoT[0]] = 1;
            nodeAvant = pco2;
            pco2 = pcoT;
        }
    }


    //!QWB: generate initial 2 connected links according to memory, begin with (0, 0), for netLinks (1D or 2D) just by their subscripts
    //!QWB: [0][0] connects [0][1] and [_wr - 1][_hr - 1]
    //!QWB: note, do not use this function two times for the same netWorkLinks
    //    template <class Grid1>
    //    void generate2ConectedLinks(Grid1& networkLinks){
    void generate2ConectedLinks(){

        //        typedef typename Grid<BufferLink >::index_type PointCoord;
        //        typedef Index<2> PointCoord;

        int _hr = this->adaptiveMap.height;
        int _wr = this->adaptiveMap.width;

        for(int i = 0; i < _hr; i++)
            for(int j = 0; j < (_wr - 1); j++){
                PointCoord pCoord(j + 1, i);
                PointCoord ps(j , i);
                networkLinks(ps).insert_cpu(pCoord);
                networkLinks(pCoord).insert_cpu(ps);

                if (i == 0 && j == 0 && (_hr - 1) % 2){
                    PointCoord pEnd(0 , _hr - 1);
                    networkLinks(ps).insert_cpu(pEnd);
                    networkLinks(pEnd).insert_cpu(ps);
                }

                else if (i == 0 && j == 0 && (_hr - 1) % 2 == 0){
                    PointCoord pEnd(_wr - 1 , _hr - 1);
                    PointCoord pStart(0, 0);
                    networkLinks(pStart).insert_cpu(pEnd);
                    networkLinks(pEnd).insert_cpu(ps);
                }

                else if ((i % 2 == 0) && (j == _wr - 2) && (i + 1) < _hr){
                    PointCoord pCorner(j+1 , i+1);
                    PointCoord pCornerUp(j+1 , i);
                    networkLinks(pCornerUp).insert_cpu(pCorner);
                    networkLinks(pCorner).insert_cpu(pCornerUp);
                }

                else if ((i % 2 == 1) && (j == 0) && (i + 1) < _hr){
                    PointCoord pCorner(j , i+1);
                    networkLinks(ps).insert_cpu(pCorner);
                    networkLinks(pCorner).insert_cpu(ps);
                }
            }
    }// end generate 2-connected tour



    //! WB.Q add to judge interact of two 2opt, need to compare first2opt when push
    bool judgeNonInteractRocki(int & pi_s_previous_city, int & pi_k_previous_city,
                               int &pi_s_city, int & pi_k_city,
                               int first2opt){

        int N = this->adaptiveMap.width;

        int pi_k_pre = this->grayValueMap[0][pi_k_previous_city];
        int pi_k = this->grayValueMap[0][pi_k_city];

        if(pi_k < pi_k_pre)
        {
            //if first 2opt lies in first half
            return true;
        }
        else
            return false;
    }// end judgeInteract


    //! WB.Q 2024 add to judge non-interacted 23456-opt along current tour, and judge confict with the first2-opt
    void judgeNonInteract23456OptQiao(int node1_pre, int node3_pre,
                                      int node5_pre, int node7_pre,
                                      int node9_pre, int node11_pre,
                                      int node1, int node3, int node5, int node7,int node9, int node11,
                                      int first2opt, bool& nonInteracted){


        int densityValue = this->densityMap[0][node1]; //qiao
        int kValue = densityValue % 10;
        if(kValue > 6)
            cout << "error JudgeNon kValue = " << kValue << endl;

        int pi_node1_pre=-1, pi_node3_pre=-1,pi_node5_pre=-1,pi_node7_pre=-1,pi_node9_pre=-1,pi_node11_pre=-1;
        if(node1_pre>=0) pi_node1_pre = this->grayValueMap[0][node1_pre];
        if(node3_pre>=0) pi_node3_pre = this->grayValueMap[0][node3_pre];
        if(node5_pre>=0) pi_node5_pre = this->grayValueMap[0][node5_pre];
        if(node7_pre>=0) pi_node7_pre = this->grayValueMap[0][node7_pre];
        if(node9_pre>=0) pi_node9_pre = this->grayValueMap[0][node9_pre];
        if(node11_pre>=0) pi_node11_pre = this->grayValueMap[0][node11_pre];

        int pi_node1=-1, pi_node3=-1, pi_node5=-1, pi_node7=-1, pi_node9=-1, pi_node11=-1;
        if(node1>=0) pi_node1 = this->grayValueMap[0][node1];
        if(node3>=0) pi_node3 = this->grayValueMap[0][node3];
        if(node5>=0) pi_node5 = this->grayValueMap[0][node5];
        if(node7>=0) pi_node7 = this->grayValueMap[0][node7];
        if(node9>=0) pi_node9 = this->grayValueMap[0][node9];
        if(node11>=0) pi_node11 = this->grayValueMap[0][node11];

        //        cout << "Judging order current: " << pi_node1 << ", " <<pi_node3 << ", " << pi_node5 << ", " << pi_node7 << ", " << pi_node9 << ", " << pi_node11 << endl;
        //        cout << "Judging order previous: " <<pi_node1_pre << ", " <<pi_node3_pre << ", " << pi_node5_pre << ", " <<pi_node7_pre<< ", " <<pi_node9_pre<< ", " <<pi_node11_pre << endl;

        if(pi_node1_pre < pi_node1 && pi_node1 < pi_node3_pre) //lies in the first sub_tour
        {
            if(kValue == 2 && pi_node3 < pi_node3_pre)
            {
                nonInteracted = 1;
                return ;
            }
            else if(kValue == 3 && pi_node5 < pi_node3_pre)
            {
                nonInteracted = 1;
                return ;
            }
            else if(kValue == 4 && pi_node7 < pi_node3_pre)
            {
                nonInteracted = 1;
                return ;
            }
            else if(kValue == 5 && pi_node9 < pi_node3_pre)
            {
                nonInteracted = 1;
                return ;
            }
            else if(kValue == 6 && pi_node11 < pi_node3_pre)
            {
                nonInteracted = 1;
                return ;
            }

        }
        else if(pi_node3_pre > 0 && pi_node3_pre < pi_node1 && pi_node1 < pi_node5_pre)//locates in the second sub-tour
        {
            if(kValue == 2 && pi_node3 < pi_node5_pre)
            {
                nonInteracted = 1;
                return ;
            }
            else if(kValue == 3 && pi_node5 < pi_node5_pre)
            {
                nonInteracted = 1;
                return ;
            }
            else if(kValue == 4 && pi_node7 < pi_node5_pre)
            {
                nonInteracted = 1;
                return ;
            }
            else if(kValue == 5 && pi_node9 < pi_node5_pre)
            {
                nonInteracted = 1;
                return ;
            }
            else if(kValue == 6 && pi_node11 < pi_node5_pre)
            {
                nonInteracted = 1;
                return ;
            }

        }
        else if(pi_node5_pre > 0 && pi_node5_pre < pi_node1 && pi_node1 < pi_node7_pre)//locates in the third sub-tour
        {
            if(kValue == 2 && pi_node3 < pi_node7_pre)
            {
                nonInteracted = 1;
                return ;
            }
            else if(kValue == 3 && pi_node5 < pi_node7_pre)
            {
                nonInteracted = 1;
                return ;
            }
            else if(kValue == 4 && pi_node7 < pi_node7_pre)
            {
                nonInteracted = 1;
                return ;
            }
            else if(kValue == 5 && pi_node9 < pi_node7_pre)
            {
                nonInteracted = 1;
                return ;
            }
            else if(kValue == 6 && pi_node11 < pi_node7_pre)
            {
                nonInteracted = 1;
                return ;
            }
        }
        else if(pi_node7_pre > 0 && pi_node7_pre < pi_node1  && pi_node1 < pi_node9_pre)//locates in the fouth sub-tour
        {
            if(kValue == 2 && pi_node3 < pi_node9_pre)
            {
                nonInteracted = 1;
                return ;
            }
            else if(kValue == 3 && pi_node5 < pi_node9_pre)
            {

                nonInteracted = 1;
                return ;
            }
            else if(kValue == 4 && pi_node7 < pi_node9_pre)
            {
                nonInteracted = 1;
                return ;
            }
            else if(kValue == 5 && pi_node9 < pi_node9_pre)
            {
                nonInteracted = 1;
                return ;
            }
            else if(kValue == 6 && pi_node11 < pi_node9_pre)
            {
                nonInteracted = 1;
                return ;
            }
        }
        else if(pi_node9_pre > 0 && pi_node9_pre < pi_node1 && pi_node1 < pi_node11_pre)//locates in the fifth sub-tour
        {
            if(kValue == 2 && pi_node3 < pi_node11_pre)
            {
                nonInteracted = 1;
                return ;
            }
            else if(kValue == 3 && pi_node5 < pi_node11_pre)
            {

                nonInteracted = 1;
                return ;
            }
            else if(kValue == 4 && pi_node7 < pi_node11_pre)
            {
                nonInteracted = 1;
                return ;
            }
            else if(kValue == 5 && pi_node9 < pi_node11_pre)
            {
                nonInteracted = 1;
                return ;
            }
            else if(kValue == 6 && pi_node11 < pi_node11_pre)
            {
                nonInteracted = 1;
                return ;
            }
        }


    }// end judgeInteract

    //! WB.Q add first element to vacant stackes and mark
    int addEleToVacantStacksRocki(PointCoord pi_s_city, int pi_k_city, vector<int>& stackA, vector<int>& stackB){
        // if stack B is empty, push ps into stack a and b
        if(this->densityMap[0][pi_s_city[0]] >= 0){
            stackA.push_back(pi_s_city[0]);
            stackB.push_back(pi_k_city);

            // cout << "first 2opt push to A " << this->grayValueMap[0][pi_s_city[0]] << endl;
            // cout << "first 2-opt push to B " << this->grayValueMap[0][pi_k_city] << endl;

            //! mark pi_k_city is right bracket
            this->fixedMap[0][pi_k_city] = 1;
        }
        return this->grayValueMap[0][pi_s_city[0]];
    }// end add first element to stacks


    //! WB.Q 2024 add first 23456opt element to vacant stackes and mark
    int addFirstOptToVacantStacks(PointCoord pi_s_city, int node3, int node5, int node7, int node9, int node11,
                                  vector<int>& stackA, vector<int>& stackB, vector<int>& stackC,
                                  vector<int>& stackD, vector<int>& stackE, vector<int>& stackF)
    {
        // if stack B is empty, push ps into stack a and b
        int densityValue = this->densityMap[0][pi_s_city[0]];
        if(this->optCandidateMap[0][pi_s_city[0]] > 0 && densityValue > 0)
        {
            int kValue = densityValue % 10;
            if(kValue > 6)
                cout << "error addFrist kValue qiao = " << kValue << endl;

            this->fixedMap[0][pi_s_city[0]] = kValue;

            stackA.push_back(pi_s_city[0]);
            stackB.push_back(node3);
            //            cout << "first opt push to A " << this->grayValueMap[0][pi_s_city[0]]  << ", node1=" << pi_s_city[0] << endl;
            //            cout << "first opt push to B " << this->grayValueMap[0][node3] << ", node3= " << node3 << endl;

            this->fixedMap[0][node3] = kValue;
            if(kValue >= 3)
            {
                stackC.push_back(node5);
                this->fixedMap[0][node5] = kValue;

                //qiao only for test

                //                cout << "first opt push to C " << this->grayValueMap[0][node5] << ", node5= " << node5 << endl;

                if(kValue >= 4)
                {
                    stackD.push_back(node7);
                    this->fixedMap[0][node7] = kValue;


                    //                    //qiao only for test
                    //                    cout << "first opt push to D " << this->grayValueMap[0][node7]  << ", node7=" << node7 << endl;

                    if(kValue >= 5)
                    {
                        stackE.push_back(node9);
                        this->fixedMap[0][node9] = kValue;


                        //                        //qiao only for test
                        //                        cout << "first opt push to E " << this->grayValueMap[0][node9]  << ", node9=" << node9 << endl;

                        if(kValue >= 6)
                        {
                            stackF.push_back(node11);
                            this->fixedMap[0][node11] = kValue;

                            //                            //qiao only for test
                            //                            cout << "first opt push to F " << this->grayValueMap[0][node11]  << ", node9=" << node11 << endl;
                        }
                    }
                }
            }

        }
        return this->grayValueMap[0][pi_s_city[0]];
    }// end add first element to stacks



    //! WB.Q when the algorithm meet a city along the tour, it does the follwoing codes
    //! pop condition is not pi_k == B.top
    void operationOnOneCityAlongTourRocki(PointCoord pi_s_city, vector<int>& stackA, vector<int>& stackB,
                                          int& first2opt){

        //! if stackB is not vacant, judge pi_s_city is interacted or not, and judge if we need to pop elements
        //! if pi_s_city is marked as right bracket, pop element in A, B
        if (this->fixedMap[0][pi_s_city[0]] && stackA.size() > 0){ // fixedMap to mark right bracket

            int finalMark2opt = stackA.back(); // note, stackA.size + 1 = stackB.size
            this->activeMap[0][finalMark2opt] = 1; // activeMap to mark final selected 2-opt

            //test
            int finalMarkB = stackB.back();
            if(finalMarkB != pi_s_city[0]){
                cout << "error: stackB.popBack error. " << endl;
            }
            else{
                stackA.pop_back();
                stackB.pop_back();
            }
            return;
        }
        else if(this->densityMap[0][pi_s_city[0]] >=0){

            int pi_k_city = (int) this->densityMap[0][pi_s_city[0]] / decodePiK;

            if(stackB.size() == 0 && stackA.size() == 0){
                first2opt = addEleToVacantStacksRocki(pi_s_city, pi_k_city, stackA, stackB);
            }
            else{
                // pi_i is the top element of stackB, and is the coordinate of the city in the array
                int pi_s_previous_city = stackA.back();
                int pi_k_previous_city = stackB.back();

                bool nonInteracted = judgeNonInteractRocki(pi_s_previous_city, pi_k_previous_city, pi_s_city[0], pi_k_city, first2opt);
                if(nonInteracted){

                    //                    cout << "stack A not vacant, push to A " << this->grayValueMap[0][pi_s_city[0]] << endl;
                    //                    cout << "stack B not vacant, push to B " << this->grayValueMap[0][pi_k_city] << endl;

                    stackA.push_back(pi_s_city[0]);
                    stackB.push_back(pi_k_city);

                    this->fixedMap[0][pi_k_city] = 1;// mark pi_k in stackB
                }
            }
        }
        else
            return;
    }// end operation on one city

    //! WB.Q 2024 when the algorithm meet a city along the tour, it does the follwoing codes
    //! pop condition is not pi_k == B.top
    void operationOnOneCityAlongTour23456opt(PointCoord currentNode, vector<int>& stackA, vector<int>& stackB,vector<int>& stackC,
                                             vector<int>& stackD, vector<int>& stackE,vector<int>& stackF,
                                             int& firstopt){

        //! if stackB is not vacant, judge pi_s_city is interacted or not, and judge if we need to pop elements
        //! if pi_s_city is marked as right bracket, pop element in A, B
        if(this->fixedMap[0][currentNode[0]] > 1)//current node posseses a candidate k-opt
        {
            int kValue = this->fixedMap[0][currentNode[0]]; // check whether current node is already pushed into stacks as node3579\11

            //if kValue == 2, pop two stacks AB, and mark stackA.back() posses a non-interacted 2-opt
            if (kValue == 2 && stackA.size() > 0)
            {
                int selectOneopt = stackA.back();
                this->activeMap[0][selectOneopt] = 2; // mark this selected 2-opt

                //                cout << "Pop 2-opt stack order " << this->grayValueMap[0][stackA.back()]  <<",node3= " << this->grayValueMap[0][stackB.back()] << endl;

                stackA.pop_back();
                stackB.pop_back();
            }
            //if kValue == 3, judge whether current node is in stackB or stackC, in stackC pop
            if(kValue == 3 && stackA.size() > 0 && stackB.size() >0 )
            {
                int node3 = stackB.back();
                int node5 = stackC.back();

                if(currentNode[0] == node5)
                {
                    //only for test, check whether stackB.top is also 3-opt
                    if(this->fixedMap[0][stackA.back()] != kValue || this->fixedMap[0][node3] != kValue || this->fixedMap[0][node5] != kValue)
                        cout << "Error: stackB and stackC not the same 3-opt !" << endl;

                    else
                    {

                        int selectOneopt = stackA.back();
                        this->activeMap[0][selectOneopt] = kValue; // mark this selected 3-opt

                        //                        cout << " Pop 3-opt stack order " << this->grayValueMap[0][stackA.back()]  <<", node3= " << this->grayValueMap[0][stackB.back()] <<", node5= " << this->grayValueMap[0][stackC.back()]<< endl;

                        //pop
                        stackA.pop_back();
                        stackB.pop_back();
                        stackC.pop_back();
                    }
                }
            }
            //if kValue == 4, judge whether current node is in stack BCD, if in stackD, pop all
            if(kValue == 4 && stackA.size() > 0 && stackB.size() >0 && stackC.size() >0  )
            {
                int node3 = stackB.back();
                int node5 = stackC.back();
                int node7 = stackD.back();

                if(currentNode[0] == node7)
                {
                    //only for test, check whether stackBC.top is also 4-opt
                    if(this->fixedMap[0][stackA.back()] != kValue ||this->fixedMap[0][node3] != kValue || this->fixedMap[0][node5] != kValue|| this->fixedMap[0][node7] != kValue)
                        cout << "Error: stackB and stackC not the same 4-opt !" << endl;
                    //work code
                    else{

                        int selectOneopt = stackA.back();
                        this->activeMap[0][selectOneopt] = kValue; // mark this selected 4-opt

                        //                        cout << " Pop 4-opt stack order " << this->grayValueMap[0][stackA.back()]  <<", node3= " << this->grayValueMap[0][stackB.back()]
                        //                                <<", node5= " << this->grayValueMap[0][stackC.back()] <<", node7= " << this->grayValueMap[0][stackD.back()] << endl;

                        //pop all
                        stackA.pop_back();
                        stackB.pop_back();
                        stackC.pop_back();
                        stackD.pop_back();
                    }
                }
            }
            //if kValue ==5, judge whether current node is in stackBCDE, if in stackE,pop all
            if(kValue == 5)
            {
                int node3 = stackB.back();
                int node5 = stackC.back();
                int node7 = stackD.back();
                int node9 = stackE.back();

                if(currentNode[0] == node9)
                {
                    //only for test, check whether stackBC.top is also 5-opt
                    if(this->fixedMap[0][stackA.back()] != kValue ||this->fixedMap[0][node3] != kValue || this->fixedMap[0][node5] != kValue
                            || this->fixedMap[0][node7] != kValue || this->fixedMap[0][node9] != kValue)
                        cout << "Error: stackB C D E not the same 5-opt !" << endl;
                    //work code
                    else{

                        int selectOneopt = stackA.back();
                        this->activeMap[0][selectOneopt] = kValue; // mark this selected 5-opt

                        //                        cout << " Pop 5-opt stack order " << this->grayValueMap[0][stackA.back()]  <<", node3= " << this->grayValueMap[0][stackB.back()]
                        //                                <<", node5= " << this->grayValueMap[0][stackC.back()] <<", node7= " << this->grayValueMap[0][stackD.back()]  <<", node9= " << this->grayValueMap[0][stackE.back()] << endl;


                        //pop all
                        stackA.pop_back();
                        stackB.pop_back();
                        stackC.pop_back();
                        stackD.pop_back();
                        stackE.pop_back();
                    }
                }
            }
            //if kValue ==6, judge whether current node is in stackBCDEF, if in stackF,pop all
            if(kValue == 6)
            {
                int node3 = stackB.back();
                int node5 = stackC.back();
                int node7 = stackD.back();
                int node9 = stackE.back();
                int node11 = stackF.back();

                if(currentNode[0] == node11)
                {
                    //only for test, check whether stackBC.top is also 6-opt
                    if(this->fixedMap[0][stackA.back()] != kValue ||this->fixedMap[0][node3] != kValue || this->fixedMap[0][node5] != kValue
                            || this->fixedMap[0][node7] != kValue || this->fixedMap[0][node9] != kValue|| this->fixedMap[0][node11] != kValue)
                        cout << "Error: stackB C D E F not the same 6-opt !" << endl;
                    //work code
                    else{

                        int selectOneopt = stackA.back();
                        this->activeMap[0][selectOneopt] = kValue; // mark this selected 6-opt

                        //                        cout << " Pop 6-opt stack order " << this->grayValueMap[0][stackA.back()]  <<", node3= " << this->grayValueMap[0][stackB.back()]
                        //                                <<", node5= " << this->grayValueMap[0][stackC.back()] <<", node7= " << this->grayValueMap[0][stackD.back()]
                        //                                <<", node9= " << this->grayValueMap[0][stackE.back()] <<", node11= " << this->grayValueMap[0][stackF.back()]<< endl;


                        //pop all
                        stackA.pop_back();
                        stackB.pop_back();
                        stackC.pop_back();
                        stackD.pop_back();
                        stackE.pop_back();
                        stackF.pop_back();
                    }
                }
            }//end 6opt

            return;
        }// end if current node is in stackBCDEF

        else if(this->optCandidateMap[0][currentNode[0]] >0
                && this->densityMap[0][currentNode[0]] > 0
                )
        {

            //old select non-interacted using densityMap
            int densityValue = this->densityMap[0][currentNode[0]];//k of k-opt obtained when check GPU parallel k-opt
            int kValue = densityValue % 10;
            if(kValue > 6)
                cout << "k-opt k here error 1 = " << kValue << ", densityValue= " << densityValue << ", optValue "
                     << this->optCandidateMap[0][currentNode[0]] << ", currentNode[0]= " << currentNode[0] << ", order= " << this->grayValueMap[0][currentNode[0]] << endl;


            //new decode the node3 5 7 9 11
            unsigned long long optCandidats = this->optCandidateMap[0][currentNode[0]];
            unsigned long long nodeResult;
            if(kValue < 6)
                nodeResult = codeBit;
            if(kValue == 6)
                nodeResult = codeBit12;

            int width = this->densityMap.width;
            int node3=-1, node5 = -1, node7 = -1, node9 = -1, node11= -1;
            if(kValue == 2)
            {
                node3 = optCandidats & nodeResult;
                if(node3 > width || node3 < 0)
                    cout << "Error decode opt node3 > widht " << endl;
            }
            else if(kValue == 3)
            {
                node5 = optCandidats & nodeResult;
                optCandidats = optCandidats >> 16;
                node3 = optCandidats & nodeResult;

                if(node5 > width|| node5 < 0)
                    cout << "Error decode opt node5 > widht " << endl;
            }
            else if(kValue ==4)
            {
                node7 = optCandidats & nodeResult;
                optCandidats = optCandidats >> 16;
                node5 = optCandidats & nodeResult;
                optCandidats = optCandidats >> 16;
                node3 = optCandidats & nodeResult;
                if(node7 > width|| node7 < 0)
                    cout << "Error decode opt node7 > widht " << endl;
            }
            else if(kValue ==5)
            {
                node9 = optCandidats & nodeResult;
                optCandidats = optCandidats >> 16;
                node7 = optCandidats & nodeResult;
                optCandidats = optCandidats >> 16;
                node5 = optCandidats & nodeResult;
                optCandidats = optCandidats >> 16;
                node3 = optCandidats & nodeResult;
                if(node9 > width || node9 < 0)
                    cout << "Error decode opt node9 > widht " << endl;
            }
            else if(kValue ==6)
            {
                node11 = optCandidats & nodeResult;
                optCandidats = optCandidats >> 12; //need to check only for 6-opt
                node9 = optCandidats & nodeResult;
                optCandidats = optCandidats >> 12;
                node7 = optCandidats & nodeResult;
                optCandidats = optCandidats >> 12;
                node5 = optCandidats & nodeResult;
                optCandidats = optCandidats >> 12;
                node3 = optCandidats & nodeResult;
                if(node11 > width|| node11 < 0)
                    cout << "Error decode opt node11 > widht " << endl;
            }
            else
                cout << "Error selecting decode kValue= " << kValue << ", optCandidate= " << this->optCandidateMap[0][currentNode[0]] << endl;

            if(stackA.size() == 0){ // if stackA.size == 0, then all stacks' size are 0
                firstopt = addFirstOptToVacantStacks(currentNode, node3, node5, node7, node9, node11, stackA, stackB, stackC, stackD, stackE, stackF);
            }
            else{
                //                cout << "Selecting treat current node: " << currentNode[0] << ", node3=" << node3 << ", node5=" << node5 << ", node7=" << node7 << ", node9=" << node9 << ", node11=" << node11
                //                     << endl;
                //                cout << "Selecting treat current node tour order: " << this->grayValueMap[0][currentNode[0]] << ", " <<this->grayValueMap[0][node3]
                //                                                                                                                << ", " <<this->grayValueMap[0][node5] << endl;
                //                                                                                                                   <<", " << this->grayValueMap[0][node7]
                //                                                                                                                   <<", " << this->grayValueMap[0][node9]  <<", " << this->grayValueMap[0][node11] <<  endl;


                // pi_i is the top element of stackB, and is the coordinate of the city in the array
                int node1_pre= -1, node3_pre= -1,node5_pre= -1,node7_pre= -1,node9_pre= -1,node11_pre= -1;
                node1_pre = stackA.back();
                node3_pre = stackB.back();
                if(stackC.size()>0) node5_pre = stackC.back();
                if(stackD.size()>0) node7_pre = stackD.back();
                if(stackE.size()>0) node9_pre = stackE.back();
                if(stackF.size()>0) node11_pre = stackF.back();

                //                cout << "Selecting previous node tour order: " << this->grayValueMap[0][node1_pre] << ", "
                //                     <<this->grayValueMap[0][node3_pre] << ", " <<this->grayValueMap[0][node5_pre] <<", " << this->grayValueMap[0][node7_pre]
                //                       <<", " << this->grayValueMap[0][node9_pre] <<", " << this->grayValueMap[0][node11_pre] <<  endl;


                bool nonInteracted = 0;
                judgeNonInteract23456OptQiao(node1_pre, node3_pre, node5_pre, node7_pre, node9_pre, node11_pre,
                                             currentNode[0], node3, node5, node7, node9, node11,firstopt, nonInteracted);

                //                cout << "Selecting judge NonInteracted = " << nonInteracted << endl;
                if(nonInteracted){

                    this->fixedMap[0][currentNode[0]] = kValue;

                    stackA.push_back(currentNode[0]);
                    stackB.push_back(node3);
                    this->fixedMap[0][node3] = kValue;
                    if(kValue >= 3)
                    {
                        stackC.push_back(node5);
                        this->fixedMap[0][node5] = kValue;

                        //                        //qiao only for test
                        //                        cout << "opt push to A order " << this->grayValueMap[0][currentNode[0]] << ", node1= " << currentNode[0] << endl;
                        //                        cout << "opt push to B order  " << this->grayValueMap[0][node3] << ", node3= " << node3 << endl;
                        //                        cout << "opt push to C order " << this->grayValueMap[0][node5] << ", node5= " << node5 << endl;

                        if(kValue >= 4)
                        {
                            stackD.push_back(node7);
                            this->fixedMap[0][node7] = kValue;

                            //                            //qiao only for test
                            //                            cout << "opt push to D " << this->grayValueMap[0][node7] << ", node7= " << node7 << endl;

                            if(kValue >= 5)
                            {
                                stackE.push_back(node9);
                                this->fixedMap[0][node9] = kValue;

                                //                                //qiao only for test
                                //                                cout << "opt push to E " << this->grayValueMap[0][node9] << ", node9= " << node9 << endl;

                                if(kValue >= 6)
                                {
                                    stackF.push_back(node11);
                                    this->fixedMap[0][node11] = kValue;

                                    //                                    //qiao only for test
                                    //                                    cout << "opt push to F " << this->grayValueMap[0][node11] << ", node11= " << node11 << endl;

                                }
                            }
                        }
                    }


                }// end non-interacted insert into stacks
            }
        }
        else
            return;
    }// end operation on one city





    //!QWB add to select non-interacted 2-exchanges by rocki
    void selectNonIteracted2ExchangeRocki(PointCoord node1){

        int N =  this->adaptiveMap.width;
        int first2opt = initialPrepareValue;

        // each stack stores the coordinate of city[0], not the tour id pi of this city along the tour
        vector<int> stackA;
        vector<int> stackB;
        stackA.clear();
        stackB.clear();

        //! for each city in the tour
        if(this->networkLinks[node1[1]][node1[0]].numLinks == 2)
        {
            //! if node1 has 2-exchange
            if(this->densityMap[node1[1]][node1[0]] >= 0){
                //                 this->check2optQuality(node1);
                int pi_k_city = (int)this->densityMap[node1[1]][node1[0]] / decodePiK;
                first2opt = addEleToVacantStacksRocki(node1, pi_k_city, stackA, stackB);
            }

            PointCoord node2_(0, 0); //previously it is point2d
            PointCoord node2(0, 0);

            // node2
            this->networkLinks[node1[1]][node1[0]].get(0, node2_);
            node2[0] = node2_[0];
            node2[1] = node2_[1];

            //! make sure node2 is in right direction of node1
            if(this->grayValueMap[node1[1]][node1[0]] == N-1 && this->grayValueMap[node2[1]][node2[0]] != 0 )
            {
                this->networkLinks[node1[1]][node1[0]].get(1, node2_);
                node2[0] = node2_[0];
                node2[1] = node2_[1];
            }
            else if((this->grayValueMap[node1[1]][node1[0]] != N-1) &&
                    (this->grayValueMap[node2[1]][node2[0]] < this->grayValueMap[node1[1]][node1[0]]
                     || ABS(this->grayValueMap[node1[1]][node1[0]] - this->grayValueMap[node2[1]][node2[0]]) > 2) )
            {
                this->networkLinks[node1[1]][node1[0]].get(1, node2_);
                node2[0] = node2_[0];
                node2[1] = node2_[1];
            }

            //! operation for one node met on the rout, if node2 has 2opt
            //            cout << "current nodes to be treated , id " << this->grayValueMap[0][node2[0]] << endl;
            //            int correspon = this->densityMap[0][node2[0]];
            //            if(correspon >= 0)
            //            {
            //                // test print the corepondences
            //                if(this->grayValueMap[0][node2[0]] > this->grayValueMap[0][correspon])
            //                cout << "print correspondeces :  node1 id " << this->grayValueMap[0][node2[0]]
            //                        << ";  node3 id " << this->grayValueMap[0][correspon] << endl;
            //            }
            // to recovery
            operationOnOneCityAlongTourRocki(node2, stackA, stackB, first2opt);
            //             this->check2optQuality(node1);

            //! WB.Q node3 only indicate the third city along the tour
            PointCoord pLinkOfNode2;
            PointCoord node3(-1, -1);
            this->networkLinks[node2[1]][node2[0]].get(0, pLinkOfNode2);
            node3[0] = (int)pLinkOfNode2[0];
            node3[1] = (int)pLinkOfNode2[1];
            if(node3 == node1){
                this->networkLinks[node2[1]][node2[0]].get(1, pLinkOfNode2);
                node3[0] = (int)pLinkOfNode2[0];
                node3[1] = (int)pLinkOfNode2[1];
            }

            PointCoord pco2Avant;
            pco2Avant = node2;

            while(node3 != node1){// the selection step only loop from 0 to N

                PointCoord pLinkOfNode3;
                PointCoord pco3(-1, -1);
                if(this->networkLinks[node3[1]][node3[0]].numLinks == 2){

                    this->networkLinks[node3[1]][node3[0]].get(0, pLinkOfNode3);
                    pco3[0] = (int)pLinkOfNode3[0];
                    pco3[1] = (int)pLinkOfNode3[1];
                    if(pco3 == pco2Avant){
                        this->networkLinks[node3[1]][node3[0]].get(1, pLinkOfNode3);
                        pco3[0] = (int)pLinkOfNode3[0];
                        pco3[1] = (int)pLinkOfNode3[1];
                    }

                    if(pco3 == pco2Avant){
                        cout << "open circle select non-interacted 2 opt . " << endl;
                        break;
                    }

                    //! operation for one node met on the rout
                    //                    cout << "current nodes to be treated  id " << this->grayValueMap[0][node3[0]] << endl;
                    //                    correspon = this->densityMap[0][node3[0]];
                    //                    if(correspon >= 0)
                    //                    {
                    //                        // test print the corepondences
                    //                        if(this->grayValueMap[0][node3[0]] > this->grayValueMap[0][correspon])
                    //                        cout << "print correspondeces :  node1 id " << this->grayValueMap[0][node3[0]]
                    //                                << ";  node3 id " << this->grayValueMap[0][correspon] << endl;
                    //                    }
                    // to recovery the density value so comment this 300717
                    operationOnOneCityAlongTourRocki(node3, stackA, stackB, first2opt);


                    pco2Avant = node3;
                    node3 = pco3;
                }
                else{
                    cout << "error num link > 2, select non-iteratected 2-opt " << endl;
                    break;
                }
            } // end while loop the tour

            //! if stackA are not poped clearly
            while(stackA.size() > 0){
                int nodeActived = stackA.back();
                this->activeMap[0][nodeActived] = 1;
                stackA.pop_back();

            }

        }// end if not correct links
        else
            cout << "error select non-interacted 2exchange " << endl;

    }// select non-interacted 2-opt sequentially




    //!QWB 2024 add to select non-interacted 23456-exchanges at the same time
    void selectNonIteracted23456ExchangeQiao(PointCoord node1){

        //        cout <<  "Enter select non-interacted:::::: " << endl;

        int N = this->adaptiveMap.width;
        int firstopt = initialPrepareValue; //qiao firstopt is useless because tour order of node11 will not overpass N-1

        // each stack stores the coordinate of city[0], not the tour id pi of this city along the tour
        vector<int> stackA;
        vector<int> stackB;
        vector<int> stackC;
        vector<int> stackD;
        vector<int> stackE;
        vector<int> stackF;
        stackA.clear();
        stackB.clear();
        stackC.clear();
        stackD.clear();
        stackE.clear();
        stackF.clear();

        //! for each city in the tour
        if(this->networkLinks[node1[1]][node1[0]].numLinks == 2)
        {
            //! if node1 has 2-exchange
            if(this->optCandidateMap[node1[1]][node1[0]] > 0
                    &&  this->densityMap[node1[1]][node1[0]] > 0
                    )
            {
                int densityValue = (int) this->densityMap[node1[1]][node1[0]];// k of kopt;
                int kValue = densityValue % 10;
                if(kValue > 6)
                    cout << "k-opt k here error = " << kValue << ", densityValue= " << densityValue << ", optValue" << this->optCandidateMap[node1[1]][node1[0]] << endl;

                //new select non-interacted using optCandidateMap
                unsigned long long optCandidats = this->optCandidateMap[node1[1]][node1[0]];

                unsigned long long nodeResult;
                if(kValue < 6)
                    nodeResult = codeBit;
                if(kValue == 6)
                    nodeResult = codeBit12;

                int node3=initialPrepareValue,node5 = initialPrepareValue, node7 =initialPrepareValue, node9 = initialPrepareValue, node11=initialPrepareValue;
                if(kValue == 2)
                {
                    node3 = optCandidats & nodeResult;
                    if(node3 > N || node3 < 0)
                        cout << "Error decode opt node3 > N." << endl;
                }
                if(kValue == 3)
                {
                    node5 = optCandidats & nodeResult;
                    optCandidats = optCandidats >> 16;
                    node3 = optCandidats & nodeResult;
                    if(node5 > N || node5 < 0)
                        cout << "Error decode opt node5 > N." << endl;

                    //                    //qiao only for test
                    //                    cout << "tour order node3 " << this->grayValueMap[0][node3] << ", node5 " << this->grayValueMap[0][node5] << endl;

                }
                if(kValue ==4)
                {
                    node7 = optCandidats & nodeResult;
                    optCandidats = optCandidats >> 16;
                    node5 = optCandidats & nodeResult;
                    optCandidats = optCandidats >> 16;
                    node3 = optCandidats & nodeResult;
                    if(node7 > N || node7 < 0)
                        cout << "Error decode opt node7 > N." << endl;

                    //                    //qiao only for test
                    //                    cout <<"Selecting treat node1, node3, node5, node7 " << node1[0] << ", " << node3 << ", " << node5 << ", " << node7<< endl;
                    //                    cout << "Selecting treat tour order " << this->grayValueMap[0][node1[0]]  << "  " << this->grayValueMap[0][node3] << " " << this->grayValueMap[0][node5]  << " " << this->grayValueMap[0][node7]<< endl;


                }
                if(kValue ==5)
                {
                    node9 = optCandidats & nodeResult;
                    optCandidats = optCandidats >> 16;
                    node7 = optCandidats & nodeResult;
                    optCandidats = optCandidats >> 16;
                    node5 = optCandidats & nodeResult;
                    optCandidats = optCandidats >> 16;
                    node3 = optCandidats & nodeResult;
                    if(node9 > N || node9 < 0)
                        cout << "Error decode opt node9 > N." << endl;
                }
                if(kValue ==6)
                {
                    node11 = optCandidats & nodeResult;
                    optCandidats = optCandidats >> 12; //need to check only for 6-opt
                    node9 = optCandidats & nodeResult;
                    optCandidats = optCandidats >> 12;
                    node7 = optCandidats & nodeResult;
                    optCandidats = optCandidats >> 12;
                    node5 = optCandidats & nodeResult;
                    optCandidats = optCandidats >> 12;
                    node3 = optCandidats & nodeResult;
                    if(node11 > N || node11 < 0)
                        cout << "Error decode opt node11 > N." << endl;
                }

                //                optCandidats = optCandidats >> 16; // for 6-opt
                //                int node9 = optCandidats & nodeResult;
                //                cout << "node3 = " << node3 << ", node5=" << node5 << endl;



                firstopt = addFirstOptToVacantStacks(node1, node3, node5, node7, node9, node11, stackA, stackB, stackC, stackD, stackE, stackF);

            }

            PointCoord node2_(0, 0); //previously it is point2d
            PointCoord node2(0, 0);

            // node2
            this->networkLinks[node1[1]][node1[0]].get(0, node2_);
            node2[0] = node2_[0];
            node2[1] = node2_[1];

            //! make sure node2 is in right direction of node1
            if(this->grayValueMap[node1[1]][node1[0]] == N-1 && this->grayValueMap[node2[1]][node2[0]] != 0 )
            {
                this->networkLinks[node1[1]][node1[0]].get(1, node2_);
                node2[0] = node2_[0];
                node2[1] = node2_[1];
            }
            else if((this->grayValueMap[node1[1]][node1[0]] != N-1) &&
                    (this->grayValueMap[node2[1]][node2[0]] < this->grayValueMap[node1[1]][node1[0]]
                     || ABS(this->grayValueMap[node1[1]][node1[0]] - this->grayValueMap[node2[1]][node2[0]]) > 2) )
            {
                this->networkLinks[node1[1]][node1[0]].get(1, node2_);
                node2[0] = node2_[0];
                node2[1] = node2_[1];
            }

            //! operation for one node met on the rout, if node2 has 2opt
            //            cout << "Selecting treating current order: " << this->grayValueMap[0][node2[0]] << endl;

            operationOnOneCityAlongTour23456opt(node2, stackA, stackB, stackC,stackD, stackE, stackF, firstopt);

            //! WB.Q node3 only indicate the third city along the tour
            PointCoord pLinkOfNode2;
            PointCoord node3(-1, -1);
            this->networkLinks[node2[1]][node2[0]].get(0, pLinkOfNode2);
            node3[0] = (int)pLinkOfNode2[0];
            node3[1] = (int)pLinkOfNode2[1];
            if(node3 == node1){
                this->networkLinks[node2[1]][node2[0]].get(1, pLinkOfNode2);
                node3[0] = (int)pLinkOfNode2[0];
                node3[1] = (int)pLinkOfNode2[1];
            }

            PointCoord pco2Avant;
            pco2Avant = node2;

            //qiao add stop conditon for test
            int numWhile = 0;

            //            while(node3 != node1 )
            while(node3 != node1 && numWhile < N+10)
            {// the selection step only loop from 0 to N

                numWhile += 1;// qiao only for test
                if (numWhile > N)
                    cout << "Error selecting non-interacted numWhile > N ............ " << endl;

                PointCoord pLinkOfNode3;
                PointCoord pco3(-1, -1);
                if(this->networkLinks[node3[1]][node3[0]].numLinks == 2){

                    this->networkLinks[node3[1]][node3[0]].get(0, pLinkOfNode3);
                    pco3[0] = (int)pLinkOfNode3[0];
                    pco3[1] = (int)pLinkOfNode3[1];
                    if(pco3 == pco2Avant){
                        this->networkLinks[node3[1]][node3[0]].get(1, pLinkOfNode3);
                        pco3[0] = (int)pLinkOfNode3[0];
                        pco3[1] = (int)pLinkOfNode3[1];
                    }

                    if(pco3 == pco2Avant){
                        cout << "open circle select non-interacted 2 opt . " << endl;
                        break;
                    }

                    //! operation for one node met on the rout
                    //                    cout << "Selecting treating current node order: " << this->grayValueMap[0][node3[0]] << endl;


                    operationOnOneCityAlongTour23456opt(node3, stackA, stackB, stackC,stackD, stackE, stackF, firstopt);


                    pco2Avant = node3;
                    node3 = pco3;
                }
                else{
                    cout << "error num link > 2, select non-iteratected 2-opt " << endl;
                    break;
                }
            } // end while loop the tour

            //! if stackA are not poped clearly,qiao 2024 there should not exist this case
            if(stackA.size() > 0){
                cout << "Error ********* stackA are not poped clearly >>>>>>>>>>" << endl;
                //                int node1_left = stackA.back();
                //                int node3_left = stackB.back();
                //                int node5_left = stackC.back();

                //                cout << "stack Left A.size= " << stackA.size() << ", B.size= "<<stackB.size() << ", stackC.size= " << stackC.size() << endl;
                //                cout << "node1_left A " << this->grayValueMap[0][node1_left] << endl;
                //                cout << "node3_left B " << this->grayValueMap[0][node3_left] << endl;
                //                cout << "node5_left C " << this->grayValueMap[0][node5_left] << endl;

                //                this->activeMap[0][node1_left] = 1;
                //                stackA.pop_back();
                //                stackB.pop_back();
                //                stackC.pop_back();
                //                stackD.pop_back();
                //                stackE.pop_back();
                //                stackF.pop_back();

            }

        }// end if not correct links
        else
            cout << "error select non-interacted 23456exchange " << endl;



        //        cout << "Selected non interacted DONE " << endl ;

    }// select non-interacted 23456-opt sequentially




    //! WB.Q return changeLink of one node
    void returnChangeLinks(PointCoord node1, PointCoord& node2, int& changeLink1, int& changeLink2)
    {

        int N =  this->adaptiveMap.width;
        PointCoord node2_(0, 0);
        this->networkLinks[0][node1[0]].get(0, node2_);
        node2[0] = node2_[0];
        node2[1] = node2_[1];

        //        cout << "1 return node1 " << node1[0]  << endl;
        //        cout << "1 return change links id node1 " << this->grayValueMap[0][node1[0]] << ", id node2 " << this->grayValueMap[0][node2[0]] << endl;

        // make sure node2 is in right direction of node1
        if(this->grayValueMap[node1[1]][node1[0]] == N-1 && this->grayValueMap[node2[1]][node2[0]] != 0 )
        {
            this->networkLinks[node1[1]][node1[0]].get(1, node2_);
            node2[0] = node2_[0];
            node2[1] = node2_[1];
            changeLink1 = 1;
        }
        //                else if((this->grayValueMap[node1[1]][node1[0]] != N-1) &&
        //                        (this->grayValueMap[node2[1]][node2[0]] < this->grayValueMap[node1[1]][node1[0]]
        //                         || ABS(this->grayValueMap[node1[1]][node1[0]] - this->grayValueMap[node2[1]][node2[0]]) > 2) )
        else if((this->grayValueMap[node1[1]][node1[0]] != N-1) && this->grayValueMap[node2[1]][node2[0]] - 1 != this->grayValueMap[0][node1[0]] )
        {
            this->networkLinks[node1[1]][node1[0]].get(1, node2_);
            node2[0] = node2_[0];
            node2[1] = node2_[1];
            changeLink1 = 1;
        }
        else
            changeLink1 = 0;

        //        cout << "2 return change links id node1 " << this->grayValueMap[0][node1[0]] << ", id node2 " << this->grayValueMap[0][node2[0]] << endl;

        PointCoord node1_(0, 0);
        this->networkLinks[0][node2[0]].get(0, node1_);
        if((int)node1_[0] != node1[0] || (int)node1_[1] != node1[1])
        {
            changeLink2 = 1;
        }
        else
            changeLink2 = 0;

    }// end return changelinks for 2-opt


    //! WB.Q return changeLink of one node
    void returnChangeLinks(PointCoord node1, PointCoord& node2, int& changeLink1, int& changeLink2, Grid<BufferLinkPointCoord > networkLinksCP)
    {

        cout << "enter changelinks " << endl;

        int N =  this->adaptiveMap.width;
        PointCoord node2_(0, 0);
        networkLinksCP[0][node1[0]].get(0, node2_);
        node2[0] = node2_[0];
        node2[1] = node2_[1];

        cout << "1 return node1 " << node1[0] << ", node2 " << node2[0] << endl;
        cout << "1 return change links id node1 " << this->grayValueMap[0][node1[0]] << ", id node2 " << this->grayValueMap[0][node2[0]] << endl;

        // make sure node2 is in right direction of node1
        if(this->grayValueMap[node1[1]][node1[0]] == N-1 && this->grayValueMap[node2[1]][node2[0]] != 0 )
        {
            networkLinksCP[node1[1]][node1[0]].get(1, node2_);
            node2[0] = node2_[0];
            node2[1] = node2_[1];
            changeLink1 = 1;
        }
        //                else if((this->grayValueMap[node1[1]][node1[0]] != N-1) &&
        //                        (this->grayValueMap[node2[1]][node2[0]] < this->grayValueMap[node1[1]][node1[0]]
        //                         || ABS(this->grayValueMap[node1[1]][node1[0]] - this->grayValueMap[node2[1]][node2[0]]) > 2) )
        else if((this->grayValueMap[node1[1]][node1[0]] != N-1) && this->grayValueMap[node2[1]][node2[0]] - 1 != this->grayValueMap[0][node1[0]] )
        {
            networkLinksCP[node1[1]][node1[0]].get(1, node2_);
            node2[0] = node2_[0];
            node2[1] = node2_[1];
            changeLink1 = 1;
        }
        else
            changeLink1 = 0;

        cout << "2 return change links id node1 " << this->grayValueMap[0][node1[0]] << ", id node2 " << this->grayValueMap[0][node2[0]] << endl;

        PointCoord node1_(0, 0);
        networkLinksCP[0][node2[0]].get(0, node1_);
        if((int)node1_[0] != node1[0] || (int)node1_[1] != node1[1])
        {
            changeLink2 = 1;
        }
        else
            changeLink2 = 0;


        cout << " out changelinks " << endl;

    }// end return changelinks for 2-opt


    //!WB.Q execute non-interacted 2-opt with only node3 as code
    bool executeNonInteract2optOnlyNode3(int& numOptimized){

        for(int _x = 0; _x < this->adaptiveMap.width; _x++)
        {
            if(this->activeMap[0][_x])
            {
                //test
                //                float evaCurrentBefore = this->evaluateWeightOfTSP<CM_DistanceSquaredEuclidean>();

                PointCoord node1(_x,0);

                int node3_int = this->densityMap[0][_x];
                PointCoord node3(node3_int, 0);

                // node2
                PointCoord node2(0, 0);
                PointCoord node4(0, 0);

                int changeLink1 = 0;
                int changeLink2 = 0;
                int changeLink3 = 0;
                int changeLink4 = 0;

                returnChangeLinks(node1, node2, changeLink1, changeLink2);
                returnChangeLinks(node3, node4, changeLink3, changeLink4);

                this->networkLinks[0][node1[0]].bCell[changeLink1] = node3;
                this->networkLinks[0][node3_int].bCell[changeLink3] = node1;
                this->networkLinks[0][node2[0]].bCell[changeLink2] = node4;
                this->networkLinks[0][node4[0]].bCell[changeLink4] = node2;
                numOptimized ++;

            }
        }

        return true;
    }

    //we.Q 2024 execute

    //qiao only for test
    inline GLfloat distance(PointCoord& p1, PointCoord& p2, NN& nn1, NN& nn2)
    {
        PointEuclid pp1 = nn1.adaptiveMap[p1[1]][p1[0]];
        PointEuclid pp2 = nn2.adaptiveMap[p2[1]][p2[0]];
        return components::DistanceEuclidean<PointEuclid>()(pp1, pp2);
    }

    //!WB.Q 2024 execute non-interacted 23456-opt with only node3 as code
    bool executeNonInteract23456optOnlyNode3(int& numOptimized,  Grid<GLint> optPossibilitiesMap, NeuralNetEMST<BufferLinkPointCoord, PointE<2>>  nn){

        for(int _x = 0; _x < this->adaptiveMap.width; _x++)
        {
            if(this->activeMap[0][_x] > 0)
            {
                //test
                // float evaCurrentBefore = this->evaluateWeightOfTSP<CM_DistanceSquaredEuclidean>();
                PointCoord node1(_x,0);

                int densityValue = this->densityMap[node1[1]][node1[0]];// k of kopt;
                int kValue = densityValue % 10;
                if(kValue > 6)
                    cout << "k-opt k execute= " << kValue << ", optmode=" << densityValue << endl;
                int optmode = densityValue /100;


                //new select non-interacted using optCandidateMap
                unsigned long long optCandidats = this->optCandidateMap[node1[1]][node1[0]];
                unsigned long long nodeResult;
                if(kValue < 6)
                    nodeResult = codeBit;
                if(kValue == 6)
                    nodeResult = codeBit12;

                int node3=initialPrepareValue,node5 = initialPrepareValue, node7 =initialPrepareValue, node9 = initialPrepareValue, node11=initialPrepareValue;

                if(kValue == 2)
                {
                    node3 = optCandidats & nodeResult;

                    //execute 2-opt
                    PointCoord node3_(node3, 0);

                    // node2
                    PointCoord node2(0, 0);
                    PointCoord node4(0, 0);

                    int changeLink1 = 0;
                    int changeLink2 = 0;
                    int changeLink3 = 0;
                    int changeLink4 = 0;

                    returnChangeLinks(node1, node2, changeLink1, changeLink2);
                    returnChangeLinks(node3_, node4, changeLink3, changeLink4);

                    this->networkLinks[0][node1[0]].bCell[changeLink1] = node3_;
                    this->networkLinks[0][node3].bCell[changeLink3] = node1;
                    this->networkLinks[0][node2[0]].bCell[changeLink2] = node4;
                    this->networkLinks[0][node4[0]].bCell[changeLink4] = node2;

                }
                if(kValue == 3)
                {
                    node5 = optCandidats & nodeResult;
                    optCandidats = optCandidats >> 16;
                    node3 = optCandidats & nodeResult;

                    //execute 3-opt
                    PointCoord node3_(node3, 0);
                    PointCoord node5_(node5, 0);

                    // node2
                    PointCoord node2(0, 0);
                    PointCoord node4(0, 0);
                    PointCoord node6(0, 0);

                    int changeLink1 = 0;
                    int changeLink2 = 0;
                    int changeLink3 = 0;
                    int changeLink4 = 0;
                    int changeLink5 = 0;
                    int changeLink6 = 0;

                    returnChangeLinks(node1, node2, changeLink1, changeLink2);
                    returnChangeLinks(node3_, node4, changeLink3, changeLink4);
                    returnChangeLinks(node5_, node6, changeLink5, changeLink6);

                    cout << "Execute 3-opt mode= "<< optmode<< " tour order: " << this->grayValueMap[0][node1[0]] << ", " <<  this->grayValueMap[0][node3]
                                                                                                                     << ", " <<  this->grayValueMap[0][node5] << endl;
                    //                    cout << "Execute 3opt node1-3= " << node1[0] << ", " << node2[0] << ", " << node3 << ", " << node4[0] << ", " << node5 << ", " << node6[0] << endl;

                    if(optmode == 0)
                    {
                        this->networkLinks[0][node1[0]].bCell[changeLink1] = node4;
                        this->networkLinks[0][node2[0]].bCell[changeLink2] = node6;
                        this->networkLinks[0][node3].bCell[changeLink3] = node5_;
                        this->networkLinks[0][node4[0]].bCell[changeLink4] = node1;
                        this->networkLinks[0][node5].bCell[changeLink5] = node3_;
                        this->networkLinks[0][node6[0]].bCell[changeLink6] = node2;
                    }
                    if(optmode == 1)
                    {
                        this->networkLinks[0][node1[0]].bCell[changeLink1] = node5_;
                        this->networkLinks[0][node2[0]].bCell[changeLink2] = node4;
                        this->networkLinks[0][node3].bCell[changeLink3] = node6;
                        this->networkLinks[0][node4[0]].bCell[changeLink4] = node2;
                        this->networkLinks[0][node5].bCell[changeLink5] = node1;
                        this->networkLinks[0][node6[0]].bCell[changeLink6] = node3_;
                    }
                    if(optmode == 2)
                    {
                        this->networkLinks[0][node1[0]].bCell[changeLink1] = node3_;
                        this->networkLinks[0][node2[0]].bCell[changeLink2] = node5_;
                        this->networkLinks[0][node3].bCell[changeLink3] = node1;
                        this->networkLinks[0][node4[0]].bCell[changeLink4] = node6;
                        this->networkLinks[0][node5].bCell[changeLink5] = node2;
                        this->networkLinks[0][node6[0]].bCell[changeLink6] = node4;

                    }
                    if(optmode == 3)
                    {
                        this->networkLinks[0][node1[0]].bCell[changeLink1] = node4;
                        this->networkLinks[0][node2[0]].bCell[changeLink2] = node5_;
                        this->networkLinks[0][node3].bCell[changeLink3] = node6;
                        this->networkLinks[0][node4[0]].bCell[changeLink4] = node1;
                        this->networkLinks[0][node5].bCell[changeLink5] = node2;
                        this->networkLinks[0][node6[0]].bCell[changeLink6] = node3_;

                        //qiao only for test
                        //                        float oldLength = distance(node2, node1, nn, nn) + distance(node3_, node4, nn, nn) + distance(node5_, node6, nn, nn);
                        //                        float newLength = distance(node1, node4, nn, nn) + distance(node2, node5_, nn, nn) + distance(node3_, node6, nn, nn);
                        //                        cout << "Execute Mode3 newlength: " << newLength << ", oldlength: " << oldLength << endl;

                    }

                }
                if(kValue ==4)
                {

                    node7 = optCandidats & nodeResult;
                    optCandidats = optCandidats >> 16;
                    node5 = optCandidats & nodeResult;
                    optCandidats = optCandidats >> 16;
                    node3 = optCandidats & nodeResult;

                    //                    //execute 4-opt
                    //                    cout << "Execute 4-opt mark node1-7: " << node1[0] << ", " << node3 << ", " << node5 << ", " << node7 << ", " << endl;
                    cout << "Execute 4-opt mode= "<< optmode<< " tour order: " << this->grayValueMap[0][node1[0]] << ", " <<  this->grayValueMap[0][node3]
                                                                                                                     << ", " <<  this->grayValueMap[0][node5] << ", "  << this->grayValueMap[0][node7] << endl;


                    //execute 4-opt
                    PointCoord node3_(node3, 0);
                    PointCoord node5_(node5, 0);
                    PointCoord node7_(node7, 0);

                    // node2
                    PointCoord node2(0, 0);
                    PointCoord node4(0, 0);
                    PointCoord node6(0, 0);
                    PointCoord node8(0, 0);


                    int changeLink1 = 0;
                    int changeLink2 = 0;
                    int changeLink3 = 0;
                    int changeLink4 = 0;
                    int changeLink5 = 0;
                    int changeLink6 = 0;
                    int changeLink7 = 0;
                    int changeLink8 = 0;

                    returnChangeLinks(node1, node2, changeLink1, changeLink2);
                    returnChangeLinks(node3_, node4, changeLink3, changeLink4);
                    returnChangeLinks(node5_, node6, changeLink5, changeLink6);
                    returnChangeLinks(node7_, node8, changeLink7, changeLink8);

                    PointCoord array[8];
                    array[0] = node1;
                    array[1] = node2;
                    array[2] = node3_;
                    array[3] = node4;
                    array[4] = node5_;
                    array[5] = node6;
                    array[6] = node7_;
                    array[7] = node8;
                    int arrayChangLinks[8]={-1,-1,-1,-1,-1,-1,-1,-1};
                    arrayChangLinks[0] = changeLink1;
                    arrayChangLinks[1] = changeLink2;
                    arrayChangLinks[2] = changeLink3;
                    arrayChangLinks[3] = changeLink4;
                    arrayChangLinks[4] = changeLink5;
                    arrayChangLinks[5] = changeLink6;
                    arrayChangLinks[6] = changeLink7;
                    arrayChangLinks[7] = changeLink8;


                    int nd1, nd2, nd3, nd4, nd5,nd6,nd7,nd8;
                    int optSelected = optmode*8;

                    //1 7 2 8 3 5 4 6
                    nd1 = optPossibilitiesMap[0][optSelected] -1; // 0
                    nd2 = optPossibilitiesMap[0][optSelected+1] -1; // 6
                    nd3 = optPossibilitiesMap[0][optSelected+2] -1;
                    nd4 = optPossibilitiesMap[0][optSelected+3] -1;
                    nd5 = optPossibilitiesMap[0][optSelected+4] -1;
                    nd6 = optPossibilitiesMap[0][optSelected+5] -1;
                    nd7 = optPossibilitiesMap[0][optSelected+6] -1;
                    nd8 = optPossibilitiesMap[0][optSelected+7] -1;

                    //                    cout << "Execute 4-opt nd1-8 = " << nd1 << " " << nd2 << " "<< nd3<< " "
                    //                         << nd4 <<" " << nd5 << " " << nd6<< " " << nd7 << " " << nd8 << " " << endl;

                    //                    cout << "Execute 4-opt array[nd1][0]= "  << array[nd1][0] << ", array[nd2][0]= " << array[nd2][0]  << ", array[nd3][0]= " << array[nd3][0]  << ", array[nd4][0]= " << array[nd4][0]<< endl;

                    //error here, changelinks is also changing accordng to ndx
                    this->networkLinks[0][array[nd1][0]].bCell[arrayChangLinks[nd1]] = array[nd2];
                    this->networkLinks[0][array[nd2][0]].bCell[arrayChangLinks[nd2]] = array[nd1];
                    this->networkLinks[0][array[nd3][0]].bCell[arrayChangLinks[nd3]] = array[nd4];
                    this->networkLinks[0][array[nd4][0]].bCell[arrayChangLinks[nd4]] = array[nd3];
                    this->networkLinks[0][array[nd5][0]].bCell[arrayChangLinks[nd5]] = array[nd6];
                    this->networkLinks[0][array[nd6][0]].bCell[arrayChangLinks[nd6]] = array[nd5];
                    this->networkLinks[0][array[nd7][0]].bCell[arrayChangLinks[nd7]] = array[nd8];
                    this->networkLinks[0][array[nd8][0]].bCell[arrayChangLinks[nd8]] = array[nd7];

                }
                if(kValue ==5)
                {
                    node9 = optCandidats & nodeResult;
                    optCandidats = optCandidats >> 16;
                    node7 = optCandidats & nodeResult;
                    optCandidats = optCandidats >> 16;
                    node5 = optCandidats & nodeResult;
                    optCandidats = optCandidats >> 16;
                    node3 = optCandidats & nodeResult;

                    //execute 5-opt
                    PointCoord node3_(node3, 0);
                    PointCoord node5_(node5, 0);
                    PointCoord node7_(node7, 0);
                    PointCoord node9_(node9, 0);


                    cout << "Execute 5-opt mode= "<< optmode<< " tour order: " << this->grayValueMap[0][node1[0]] << ", " <<  this->grayValueMap[0][node3]
                                                                                                                     << ", " <<  this->grayValueMap[0][node5] << ", "  << this->grayValueMap[0][node7]  << ", "  << this->grayValueMap[0][node9]<< endl;

                    // node2
                    PointCoord node2(0, 0);
                    PointCoord node4(0, 0);
                    PointCoord node6(0, 0);
                    PointCoord node8(0, 0);
                    PointCoord node10(0, 0);

                    int changeLink1 = 0;
                    int changeLink2 = 0;
                    int changeLink3 = 0;
                    int changeLink4 = 0;
                    int changeLink5 = 0;
                    int changeLink6 = 0;
                    int changeLink7 = 0;
                    int changeLink8 = 0;
                    int changeLink9 = 0;
                    int changeLink10 = 0;

                    returnChangeLinks(node1, node2, changeLink1, changeLink2);
                    returnChangeLinks(node3_, node4, changeLink3, changeLink4);
                    returnChangeLinks(node5_, node6, changeLink5, changeLink6);
                    returnChangeLinks(node7_, node8, changeLink7, changeLink8);
                    returnChangeLinks(node9_, node10, changeLink9, changeLink10);


                    PointCoord array[10];
                    array[0] = node1;
                    array[1] = node2;
                    array[2] = node3_;
                    array[3] = node4;
                    array[4] = node5_;
                    array[5] = node6;
                    array[6] = node7_;
                    array[7] = node8;
                    array[8] = node9_;
                    array[9] = node10;
                    int arrayChangLinks[10]={-1,-1,-1,-1,-1,-1,-1,-1,-1,-1};
                    arrayChangLinks[0] = changeLink1;
                    arrayChangLinks[1] = changeLink2;
                    arrayChangLinks[2] = changeLink3;
                    arrayChangLinks[3] = changeLink4;
                    arrayChangLinks[4] = changeLink5;
                    arrayChangLinks[5] = changeLink6;
                    arrayChangLinks[6] = changeLink7;
                    arrayChangLinks[7] = changeLink8;
                    arrayChangLinks[8] = changeLink9;
                    arrayChangLinks[9] = changeLink10;

                    int nd1, nd2, nd3, nd4, nd5,nd6,nd7,nd8,nd9,nd10;
                    int optSelected = optmode*10;

                    //1 7 2 8 3 5 4 6
                    nd1 = optPossibilitiesMap[0][optSelected] -1; // 0
                    nd2 = optPossibilitiesMap[0][optSelected+1] -1; // 6
                    nd3 = optPossibilitiesMap[0][optSelected+2] -1;
                    nd4 = optPossibilitiesMap[0][optSelected+3] -1;
                    nd5 = optPossibilitiesMap[0][optSelected+4] -1;
                    nd6 = optPossibilitiesMap[0][optSelected+5] -1;
                    nd7 = optPossibilitiesMap[0][optSelected+6] -1;
                    nd8 = optPossibilitiesMap[0][optSelected+7] -1;
                    nd9 = optPossibilitiesMap[0][optSelected+8] -1;
                    nd10 = optPossibilitiesMap[0][optSelected+9] -1;


                    //error here, changelinks is also changing accordng to ndx
                    this->networkLinks[0][array[nd1][0]].bCell[arrayChangLinks[nd1]] = array[nd2];
                    this->networkLinks[0][array[nd2][0]].bCell[arrayChangLinks[nd2]] = array[nd1];
                    this->networkLinks[0][array[nd3][0]].bCell[arrayChangLinks[nd3]] = array[nd4];
                    this->networkLinks[0][array[nd4][0]].bCell[arrayChangLinks[nd4]] = array[nd3];
                    this->networkLinks[0][array[nd5][0]].bCell[arrayChangLinks[nd5]] = array[nd6];
                    this->networkLinks[0][array[nd6][0]].bCell[arrayChangLinks[nd6]] = array[nd5];
                    this->networkLinks[0][array[nd7][0]].bCell[arrayChangLinks[nd7]] = array[nd8];
                    this->networkLinks[0][array[nd8][0]].bCell[arrayChangLinks[nd8]] = array[nd7];
                    this->networkLinks[0][array[nd9][0]].bCell[arrayChangLinks[nd9]] = array[nd10];
                    this->networkLinks[0][array[nd10][0]].bCell[arrayChangLinks[nd10]] = array[nd9];

                }
                if(kValue ==6)
                {

                    node11 = optCandidats & nodeResult;
                    optCandidats = optCandidats >> 12; //need to check only for 6-opt
                    node9 = optCandidats & nodeResult;
                    optCandidats = optCandidats >> 12;
                    node7 = optCandidats & nodeResult;
                    optCandidats = optCandidats >> 12;
                    node5 = optCandidats & nodeResult;
                    optCandidats = optCandidats >> 12;
                    node3 = optCandidats & nodeResult;

                    //                    //execute 6-opt
                    //                    cout << "Execute 6-opt mark node1-11: " << node1[0] << ", " << node3 << ", " << node5 << ", " << node7 << ", " << node9 << ", " <<  node11 << ", " <<endl;
                    cout << "Execute 6-opt mode= "<< optmode<< " tour order: " << this->grayValueMap[0][node1[0]] << " " <<  this->grayValueMap[0][node3]
                                                                                                                     << " " << this->grayValueMap[0][node5] << " " << this->grayValueMap[0][node7]
                                                                                                                        << " " << this->grayValueMap[0][node9] << " " << this->grayValueMap[0][node11] << endl;

                    //execute 5-opt
                    PointCoord node3_(node3, 0);
                    PointCoord node5_(node5, 0);
                    PointCoord node7_(node7, 0);
                    PointCoord node9_(node9, 0);
                    PointCoord node11_(node11, 0);

                    // node2
                    PointCoord node2(0, 0);
                    PointCoord node4(0, 0);
                    PointCoord node6(0, 0);
                    PointCoord node8(0, 0);
                    PointCoord node10(0, 0);
                    PointCoord node12(0, 0);

                    int changeLink1 = 0;
                    int changeLink2 = 0;
                    int changeLink3 = 0;
                    int changeLink4 = 0;
                    int changeLink5 = 0;
                    int changeLink6 = 0;
                    int changeLink7 = 0;
                    int changeLink8 = 0;
                    int changeLink9 = 0;
                    int changeLink10 = 0;
                    int changeLink11 = 0;
                    int changeLink12 = 0;

                    returnChangeLinks(node1, node2, changeLink1, changeLink2);
                    returnChangeLinks(node3_, node4, changeLink3, changeLink4);
                    returnChangeLinks(node5_, node6, changeLink5, changeLink6);
                    returnChangeLinks(node7_, node8, changeLink7, changeLink8);
                    returnChangeLinks(node9_, node10, changeLink9, changeLink10);
                    returnChangeLinks(node11_, node12, changeLink11, changeLink12);


                    PointCoord array[12];
                    array[0] = node1;
                    array[1] = node2;
                    array[2] = node3_;
                    array[3] = node4;
                    array[4] = node5_;
                    array[5] = node6;
                    array[6] = node7_;
                    array[7] = node8;
                    array[8] = node9_;
                    array[9] = node10;
                    array[10] = node11_;
                    array[11] = node12;

                    int arrayChangLinks[12]={-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1};
                    arrayChangLinks[0] = changeLink1;
                    arrayChangLinks[1] = changeLink2;
                    arrayChangLinks[2] = changeLink3;
                    arrayChangLinks[3] = changeLink4;
                    arrayChangLinks[4] = changeLink5;
                    arrayChangLinks[5] = changeLink6;
                    arrayChangLinks[6] = changeLink7;
                    arrayChangLinks[7] = changeLink8;
                    arrayChangLinks[8] = changeLink9;
                    arrayChangLinks[9] = changeLink10;
                    arrayChangLinks[10] = changeLink11;
                    arrayChangLinks[11] = changeLink12;


                    int nd1, nd2, nd3, nd4, nd5,nd6,nd7,nd8,nd9,nd10,nd11,nd12;
                    int optSelected = optmode*12;

                    //1 7 2 8 3 5 4 6
                    nd1 = optPossibilitiesMap[0][optSelected] -1; // 0
                    nd2 = optPossibilitiesMap[0][optSelected+1] -1; // 6
                    nd3 = optPossibilitiesMap[0][optSelected+2] -1;
                    nd4 = optPossibilitiesMap[0][optSelected+3] -1;
                    nd5 = optPossibilitiesMap[0][optSelected+4] -1;
                    nd6 = optPossibilitiesMap[0][optSelected+5] -1;
                    nd7 = optPossibilitiesMap[0][optSelected+6] -1;
                    nd8 = optPossibilitiesMap[0][optSelected+7] -1;
                    nd9 = optPossibilitiesMap[0][optSelected+8] -1;
                    nd10 = optPossibilitiesMap[0][optSelected+9] -1;
                    nd11 = optPossibilitiesMap[0][optSelected+10] -1;
                    nd12 = optPossibilitiesMap[0][optSelected+11] -1;

                    //error here, changelinks is also changing accordng to ndx
                    this->networkLinks[0][array[nd1][0]].bCell[arrayChangLinks[nd1]] = array[nd2];
                    this->networkLinks[0][array[nd2][0]].bCell[arrayChangLinks[nd2]] = array[nd1];
                    this->networkLinks[0][array[nd3][0]].bCell[arrayChangLinks[nd3]] = array[nd4];
                    this->networkLinks[0][array[nd4][0]].bCell[arrayChangLinks[nd4]] = array[nd3];
                    this->networkLinks[0][array[nd5][0]].bCell[arrayChangLinks[nd5]] = array[nd6];
                    this->networkLinks[0][array[nd6][0]].bCell[arrayChangLinks[nd6]] = array[nd5];
                    this->networkLinks[0][array[nd7][0]].bCell[arrayChangLinks[nd7]] = array[nd8];
                    this->networkLinks[0][array[nd8][0]].bCell[arrayChangLinks[nd8]] = array[nd7];
                    this->networkLinks[0][array[nd9][0]].bCell[arrayChangLinks[nd9]] = array[nd10];
                    this->networkLinks[0][array[nd10][0]].bCell[arrayChangLinks[nd10]] = array[nd9];
                    this->networkLinks[0][array[nd11][0]].bCell[arrayChangLinks[nd11]] = array[nd12];
                    this->networkLinks[0][array[nd12][0]].bCell[arrayChangLinks[nd12]] = array[nd11];



                }

                numOptimized ++;


                //                // test
                //                float evaCurrentAfter = this->evaluateWeightOfTSP<CM_DistanceSquaredEuclidean>();
                //                if(evaCurrentAfter < evaCurrentBefore)
                //                {
                //                    // test
                //                    int idNode1 = this->grayValueMap[0][_x];
                //                    int idNode3 = this->grayValueMap[0][node3_int];
                //                    cout << " correct optimizing " << idNode1  <<  ", id son1 " << this->grayValueMap[0][node2[0]]
                //                            << ". idCorrs " << idNode3 << ", id son3 "  << this->grayValueMap[0][node4[0]]  << endl;

                //                }
                //                else if (evaCurrentAfter == evaCurrentBefore)
                //                {
                //                    int idNode1 = this->grayValueMap[0][_x];
                //                    int idNode3 = this->grayValueMap[0][node3_int];
                //                    cout << " equal optimizing " << idNode1  <<  ", id son1 " << this->grayValueMap[0][node2[0]]
                //                            << ". idCorrs " << idNode3 << ", id son3 "  << this->grayValueMap[0][node4[0]]  << endl;
                //                }
                //                else
                //                {
                //                    int idNode1 = this->grayValueMap[0][_x];
                //                    int idNode3 = this->grayValueMap[0][node3_int];
                //                    cout << " warning optimizing " << idNode1  <<  ", id son1 " << this->grayValueMap[0][node2[0]]
                //                            << ". idCorrs " << idNode3 << ", id son3 "  << this->grayValueMap[0][node4[0]]  << endl;
                //                }

            }
        }

        cout  << "numOptimized = " << numOptimized << endl;

        return true;
    }

    //!WB.Q 2024 execute non-interacted 23456-opt with only node3 as code
    bool executeNonInteract23456optOnlyNode3(int& numOptimized,  Grid<GLint> optPossibilitiesMap){

        int numOptEexcuted = 0;

        for(int _x = 0; _x < this->adaptiveMap.width; _x++)
        {
            if(this->activeMap[0][_x] > 0)
            {
                numOptEexcuted += 1;
                //                cout << "Execute numOptEexcuted = " << numOptEexcuted << endl;

                //test
                // float evaCurrentBefore = this->evaluateWeightOfTSP<CM_DistanceSquaredEuclidean>();

                PointCoord node1(_x,0);

                int densityValue = this->densityMap[node1[1]][node1[0]];// k of kopt;
                int kValue = densityValue % 10;
                if(kValue > 6)
                    cout << "k-opt k execute= " << kValue << ", optmode=" << densityValue << endl;
                int optmode = densityValue /100;


                //new select non-interacted using optCandidateMap
                unsigned long long optCandidats = this->optCandidateMap[node1[1]][node1[0]];
                unsigned long long nodeResult;
                if(kValue < 6)
                    nodeResult = codeBit;
                if(kValue == 6)
                    nodeResult = codeBit12;

                int node3=initialPrepareValue,node5 = initialPrepareValue, node7 =initialPrepareValue, node9 = initialPrepareValue, node11=initialPrepareValue;

                if(kValue == 2)
                {
                    node3 = optCandidats & nodeResult;

                    //execute 2-opt
                    PointCoord node3_(node3, 0);

                    //                    cout << "Execute 2-opt mode= "<< optmode<< " tour order: " << this->grayValueMap[0][node1[0]] << ", " <<  this->grayValueMap[0][node3] << endl;

                    // node2
                    PointCoord node2(0, 0);
                    PointCoord node4(0, 0);

                    int changeLink1 = 0;
                    int changeLink2 = 0;
                    int changeLink3 = 0;
                    int changeLink4 = 0;

                    returnChangeLinks(node1, node2, changeLink1, changeLink2);
                    returnChangeLinks(node3_, node4, changeLink3, changeLink4);

                    this->networkLinks[0][node1[0]].bCell[changeLink1] = node3_;
                    this->networkLinks[0][node3].bCell[changeLink3] = node1;
                    this->networkLinks[0][node2[0]].bCell[changeLink2] = node4;
                    this->networkLinks[0][node4[0]].bCell[changeLink4] = node2;

                }
                if(kValue == 3)
                {
                    node5 = optCandidats & nodeResult;
                    optCandidats = optCandidats >> 16;
                    node3 = optCandidats & nodeResult;

                    //execute 3-opt
                    PointCoord node3_(node3, 0);
                    PointCoord node5_(node5, 0);

                    // node2
                    PointCoord node2(0, 0);
                    PointCoord node4(0, 0);
                    PointCoord node6(0, 0);

                    int changeLink1 = 0;
                    int changeLink2 = 0;
                    int changeLink3 = 0;
                    int changeLink4 = 0;
                    int changeLink5 = 0;
                    int changeLink6 = 0;

                    returnChangeLinks(node1, node2, changeLink1, changeLink2);
                    returnChangeLinks(node3_, node4, changeLink3, changeLink4);
                    returnChangeLinks(node5_, node6, changeLink5, changeLink6);

                    //                    cout << "Execute 3-opt mode= "<< optmode<< " tour order: " << this->grayValueMap[0][node1[0]] << ", " <<  this->grayValueMap[0][node3]
                    //                                                                                                                     << ", " <<  this->grayValueMap[0][node5] << endl;

                    if(optmode == 0)
                    {
                        this->networkLinks[0][node1[0]].bCell[changeLink1] = node4;
                        this->networkLinks[0][node2[0]].bCell[changeLink2] = node6;
                        this->networkLinks[0][node3].bCell[changeLink3] = node5_;
                        this->networkLinks[0][node4[0]].bCell[changeLink4] = node1;
                        this->networkLinks[0][node5].bCell[changeLink5] = node3_;
                        this->networkLinks[0][node6[0]].bCell[changeLink6] = node2;
                    }
                    if(optmode == 1)
                    {
                        this->networkLinks[0][node1[0]].bCell[changeLink1] = node5_;
                        this->networkLinks[0][node2[0]].bCell[changeLink2] = node4;
                        this->networkLinks[0][node3].bCell[changeLink3] = node6;
                        this->networkLinks[0][node4[0]].bCell[changeLink4] = node2;
                        this->networkLinks[0][node5].bCell[changeLink5] = node1;
                        this->networkLinks[0][node6[0]].bCell[changeLink6] = node3_;
                    }
                    if(optmode == 2)
                    {
                        this->networkLinks[0][node1[0]].bCell[changeLink1] = node3_;
                        this->networkLinks[0][node2[0]].bCell[changeLink2] = node5_;
                        this->networkLinks[0][node3].bCell[changeLink3] = node1;
                        this->networkLinks[0][node4[0]].bCell[changeLink4] = node6;
                        this->networkLinks[0][node5].bCell[changeLink5] = node2;
                        this->networkLinks[0][node6[0]].bCell[changeLink6] = node4;

                    }
                    if(optmode == 3)
                    {
                        this->networkLinks[0][node1[0]].bCell[changeLink1] = node4;
                        this->networkLinks[0][node2[0]].bCell[changeLink2] = node5_;
                        this->networkLinks[0][node3].bCell[changeLink3] = node6;
                        this->networkLinks[0][node4[0]].bCell[changeLink4] = node1;
                        this->networkLinks[0][node5].bCell[changeLink5] = node2;
                        this->networkLinks[0][node6[0]].bCell[changeLink6] = node3_;

                    }

                }
                if(kValue == 4)
                {

                    node7 = optCandidats & nodeResult;
                    optCandidats = optCandidats >> 16;
                    node5 = optCandidats & nodeResult;
                    optCandidats = optCandidats >> 16;
                    node3 = optCandidats & nodeResult;

                    //                    //execute 4-opt
                    //                    cout << "Execute 4-opt mark node1-7: " << node1[0] << ", " << node3 << ", " << node5 << ", " << node7 << ", " << endl;
                    //                    cout << "Execute 4-opt mode= "<< optmode<< " tour order: " << this->grayValueMap[0][node1[0]] << ", " <<  this->grayValueMap[0][node3]
                    //                                                                                                                     << ", " <<  this->grayValueMap[0][node5] << ", "  << this->grayValueMap[0][node7] << endl;
                    //execute 4-opt
                    PointCoord node3_(node3, 0);
                    PointCoord node5_(node5, 0);
                    PointCoord node7_(node7, 0);

                    // node2
                    PointCoord node2(0, 0);
                    PointCoord node4(0, 0);
                    PointCoord node6(0, 0);
                    PointCoord node8(0, 0);


                    int changeLink1 = 0;
                    int changeLink2 = 0;
                    int changeLink3 = 0;
                    int changeLink4 = 0;
                    int changeLink5 = 0;
                    int changeLink6 = 0;
                    int changeLink7 = 0;
                    int changeLink8 = 0;

                    returnChangeLinks(node1, node2, changeLink1, changeLink2);
                    returnChangeLinks(node3_, node4, changeLink3, changeLink4);
                    returnChangeLinks(node5_, node6, changeLink5, changeLink6);
                    returnChangeLinks(node7_, node8, changeLink7, changeLink8);

                    PointCoord array[8];
                    array[0] = node1;
                    array[1] = node2;
                    array[2] = node3_;
                    array[3] = node4;
                    array[4] = node5_;
                    array[5] = node6;
                    array[6] = node7_;
                    array[7] = node8;
                    int arrayChangLinks[8]={-1,-1,-1,-1,-1,-1,-1,-1};
                    arrayChangLinks[0] = changeLink1;
                    arrayChangLinks[1] = changeLink2;
                    arrayChangLinks[2] = changeLink3;
                    arrayChangLinks[3] = changeLink4;
                    arrayChangLinks[4] = changeLink5;
                    arrayChangLinks[5] = changeLink6;
                    arrayChangLinks[6] = changeLink7;
                    arrayChangLinks[7] = changeLink8;


                    int nd1, nd2, nd3, nd4, nd5,nd6,nd7,nd8;
                    int optSelected = optmode*8;

                    //1 7 2 8 3 5 4 6
                    nd1 = optPossibilitiesMap[0][optSelected] -1; // 0
                    nd2 = optPossibilitiesMap[0][optSelected+1] -1; // 6
                    nd3 = optPossibilitiesMap[0][optSelected+2] -1;
                    nd4 = optPossibilitiesMap[0][optSelected+3] -1;
                    nd5 = optPossibilitiesMap[0][optSelected+4] -1;
                    nd6 = optPossibilitiesMap[0][optSelected+5] -1;
                    nd7 = optPossibilitiesMap[0][optSelected+6] -1;
                    nd8 = optPossibilitiesMap[0][optSelected+7] -1;

                    //                    cout << "Execute 4-opt nd1-8 = " << nd1 << " " << nd2 << " "<< nd3<< " "
                    //                         << nd4 <<" " << nd5 << " " << nd6<< " " << nd7 << " " << nd8 << " " << endl;

                    //                    cout << "Execute 4-opt array[nd1][0]= "  << array[nd1][0] << ", array[nd2][0]= " << array[nd2][0]  << ", array[nd3][0]= " << array[nd3][0]  << ", array[nd4][0]= " << array[nd4][0]<< endl;

                    //error here, changelinks is also changing accordng to ndx
                    this->networkLinks[0][array[nd1][0]].bCell[arrayChangLinks[nd1]] = array[nd2];
                    this->networkLinks[0][array[nd2][0]].bCell[arrayChangLinks[nd2]] = array[nd1];
                    this->networkLinks[0][array[nd3][0]].bCell[arrayChangLinks[nd3]] = array[nd4];
                    this->networkLinks[0][array[nd4][0]].bCell[arrayChangLinks[nd4]] = array[nd3];
                    this->networkLinks[0][array[nd5][0]].bCell[arrayChangLinks[nd5]] = array[nd6];
                    this->networkLinks[0][array[nd6][0]].bCell[arrayChangLinks[nd6]] = array[nd5];
                    this->networkLinks[0][array[nd7][0]].bCell[arrayChangLinks[nd7]] = array[nd8];
                    this->networkLinks[0][array[nd8][0]].bCell[arrayChangLinks[nd8]] = array[nd7];

                }
                if(kValue ==5)
                {
                    node9 = optCandidats & nodeResult;
                    optCandidats = optCandidats >> 16;
                    node7 = optCandidats & nodeResult;
                    optCandidats = optCandidats >> 16;
                    node5 = optCandidats & nodeResult;
                    optCandidats = optCandidats >> 16;
                    node3 = optCandidats & nodeResult;

                    //execute 5-opt
                    PointCoord node3_(node3, 0);
                    PointCoord node5_(node5, 0);
                    PointCoord node7_(node7, 0);
                    PointCoord node9_(node9, 0);


                    //                    cout << "Execute 5-opt mode= "<< optmode<< " tour order: " << this->grayValueMap[0][node1[0]] << ", " <<  this->grayValueMap[0][node3]
                    //                                                                                                                     << ", " <<  this->grayValueMap[0][node5] << ", "  << this->grayValueMap[0][node7]  << ", "  << this->grayValueMap[0][node9]<< endl;

                    // node2
                    PointCoord node2(0, 0);
                    PointCoord node4(0, 0);
                    PointCoord node6(0, 0);
                    PointCoord node8(0, 0);
                    PointCoord node10(0, 0);

                    int changeLink1 = 0;
                    int changeLink2 = 0;
                    int changeLink3 = 0;
                    int changeLink4 = 0;
                    int changeLink5 = 0;
                    int changeLink6 = 0;
                    int changeLink7 = 0;
                    int changeLink8 = 0;
                    int changeLink9 = 0;
                    int changeLink10 = 0;

                    returnChangeLinks(node1, node2, changeLink1, changeLink2);
                    returnChangeLinks(node3_, node4, changeLink3, changeLink4);
                    returnChangeLinks(node5_, node6, changeLink5, changeLink6);
                    returnChangeLinks(node7_, node8, changeLink7, changeLink8);
                    returnChangeLinks(node9_, node10, changeLink9, changeLink10);


                    PointCoord array[10];
                    array[0] = node1;
                    array[1] = node2;
                    array[2] = node3_;
                    array[3] = node4;
                    array[4] = node5_;
                    array[5] = node6;
                    array[6] = node7_;
                    array[7] = node8;
                    array[8] = node9_;
                    array[9] = node10;
                    int arrayChangLinks[10]={-1,-1,-1,-1,-1,-1,-1,-1,-1,-1};
                    arrayChangLinks[0] = changeLink1;
                    arrayChangLinks[1] = changeLink2;
                    arrayChangLinks[2] = changeLink3;
                    arrayChangLinks[3] = changeLink4;
                    arrayChangLinks[4] = changeLink5;
                    arrayChangLinks[5] = changeLink6;
                    arrayChangLinks[6] = changeLink7;
                    arrayChangLinks[7] = changeLink8;
                    arrayChangLinks[8] = changeLink9;
                    arrayChangLinks[9] = changeLink10;

                    int nd1, nd2, nd3, nd4, nd5,nd6,nd7,nd8,nd9,nd10;
                    int optSelected = optmode*10;

                    //1 7 2 8 3 5 4 6
                    nd1 = optPossibilitiesMap[0][optSelected] -1; // 0
                    nd2 = optPossibilitiesMap[0][optSelected+1] -1; // 6
                    nd3 = optPossibilitiesMap[0][optSelected+2] -1;
                    nd4 = optPossibilitiesMap[0][optSelected+3] -1;
                    nd5 = optPossibilitiesMap[0][optSelected+4] -1;
                    nd6 = optPossibilitiesMap[0][optSelected+5] -1;
                    nd7 = optPossibilitiesMap[0][optSelected+6] -1;
                    nd8 = optPossibilitiesMap[0][optSelected+7] -1;
                    nd9 = optPossibilitiesMap[0][optSelected+8] -1;
                    nd10 = optPossibilitiesMap[0][optSelected+9] -1;


                    //error here, changelinks is also changing accordng to ndx
                    this->networkLinks[0][array[nd1][0]].bCell[arrayChangLinks[nd1]] = array[nd2];
                    this->networkLinks[0][array[nd2][0]].bCell[arrayChangLinks[nd2]] = array[nd1];
                    this->networkLinks[0][array[nd3][0]].bCell[arrayChangLinks[nd3]] = array[nd4];
                    this->networkLinks[0][array[nd4][0]].bCell[arrayChangLinks[nd4]] = array[nd3];
                    this->networkLinks[0][array[nd5][0]].bCell[arrayChangLinks[nd5]] = array[nd6];
                    this->networkLinks[0][array[nd6][0]].bCell[arrayChangLinks[nd6]] = array[nd5];
                    this->networkLinks[0][array[nd7][0]].bCell[arrayChangLinks[nd7]] = array[nd8];
                    this->networkLinks[0][array[nd8][0]].bCell[arrayChangLinks[nd8]] = array[nd7];
                    this->networkLinks[0][array[nd9][0]].bCell[arrayChangLinks[nd9]] = array[nd10];
                    this->networkLinks[0][array[nd10][0]].bCell[arrayChangLinks[nd10]] = array[nd9];

                }
                if(kValue ==6)
                {

                    node11 = optCandidats & nodeResult;
                    optCandidats = optCandidats >> 12; //need to check only for 6-opt
                    node9 = optCandidats & nodeResult;
                    optCandidats = optCandidats >> 12;
                    node7 = optCandidats & nodeResult;
                    optCandidats = optCandidats >> 12;
                    node5 = optCandidats & nodeResult;
                    optCandidats = optCandidats >> 12;
                    node3 = optCandidats & nodeResult;

                    //                    //execute 6-opt
                    //                    cout << "Execute 6-opt mark node1-11: " << node1[0] << ", " << node3 << ", " << node5 << ", " << node7 << ", " << node9 << ", " <<  node11 << ", " <<endl;
                    //                    cout << "Execute 6-opt mode= "<< optmode<< " tour order: " << this->grayValueMap[0][node1[0]] << " " <<  this->grayValueMap[0][node3]
                    //                                                                                                                     << " " << this->grayValueMap[0][node5] << " " << this->grayValueMap[0][node7]
                    //                                                                                                                        << " " << this->grayValueMap[0][node9] << " " << this->grayValueMap[0][node11] << endl;

                    //execute 5-opt
                    PointCoord node3_(node3, 0);
                    PointCoord node5_(node5, 0);
                    PointCoord node7_(node7, 0);
                    PointCoord node9_(node9, 0);
                    PointCoord node11_(node11, 0);

                    // node2
                    PointCoord node2(0, 0);
                    PointCoord node4(0, 0);
                    PointCoord node6(0, 0);
                    PointCoord node8(0, 0);
                    PointCoord node10(0, 0);
                    PointCoord node12(0, 0);

                    int changeLink1 = 0;
                    int changeLink2 = 0;
                    int changeLink3 = 0;
                    int changeLink4 = 0;
                    int changeLink5 = 0;
                    int changeLink6 = 0;
                    int changeLink7 = 0;
                    int changeLink8 = 0;
                    int changeLink9 = 0;
                    int changeLink10 = 0;
                    int changeLink11 = 0;
                    int changeLink12 = 0;

                    returnChangeLinks(node1, node2, changeLink1, changeLink2);
                    returnChangeLinks(node3_, node4, changeLink3, changeLink4);
                    returnChangeLinks(node5_, node6, changeLink5, changeLink6);
                    returnChangeLinks(node7_, node8, changeLink7, changeLink8);
                    returnChangeLinks(node9_, node10, changeLink9, changeLink10);
                    returnChangeLinks(node11_, node12, changeLink11, changeLink12);


                    PointCoord array[12];
                    array[0] = node1;
                    array[1] = node2;
                    array[2] = node3_;
                    array[3] = node4;
                    array[4] = node5_;
                    array[5] = node6;
                    array[6] = node7_;
                    array[7] = node8;
                    array[8] = node9_;
                    array[9] = node10;
                    array[10] = node11_;
                    array[11] = node12;

                    int arrayChangLinks[12]={-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1};
                    arrayChangLinks[0] = changeLink1;
                    arrayChangLinks[1] = changeLink2;
                    arrayChangLinks[2] = changeLink3;
                    arrayChangLinks[3] = changeLink4;
                    arrayChangLinks[4] = changeLink5;
                    arrayChangLinks[5] = changeLink6;
                    arrayChangLinks[6] = changeLink7;
                    arrayChangLinks[7] = changeLink8;
                    arrayChangLinks[8] = changeLink9;
                    arrayChangLinks[9] = changeLink10;
                    arrayChangLinks[10] = changeLink11;
                    arrayChangLinks[11] = changeLink12;


                    int nd1, nd2, nd3, nd4, nd5,nd6,nd7,nd8,nd9,nd10,nd11,nd12;
                    int optSelected = optmode*12;

                    //1 7 2 8 3 5 4 6
                    nd1 = optPossibilitiesMap[0][optSelected] -1; // 0
                    nd2 = optPossibilitiesMap[0][optSelected+1] -1; // 6
                    nd3 = optPossibilitiesMap[0][optSelected+2] -1;
                    nd4 = optPossibilitiesMap[0][optSelected+3] -1;
                    nd5 = optPossibilitiesMap[0][optSelected+4] -1;
                    nd6 = optPossibilitiesMap[0][optSelected+5] -1;
                    nd7 = optPossibilitiesMap[0][optSelected+6] -1;
                    nd8 = optPossibilitiesMap[0][optSelected+7] -1;
                    nd9 = optPossibilitiesMap[0][optSelected+8] -1;
                    nd10 = optPossibilitiesMap[0][optSelected+9] -1;
                    nd11 = optPossibilitiesMap[0][optSelected+10] -1;
                    nd12 = optPossibilitiesMap[0][optSelected+11] -1;

                    //error here, changelinks is also changing accordng to ndx
                    this->networkLinks[0][array[nd1][0]].bCell[arrayChangLinks[nd1]] = array[nd2];
                    this->networkLinks[0][array[nd2][0]].bCell[arrayChangLinks[nd2]] = array[nd1];
                    this->networkLinks[0][array[nd3][0]].bCell[arrayChangLinks[nd3]] = array[nd4];
                    this->networkLinks[0][array[nd4][0]].bCell[arrayChangLinks[nd4]] = array[nd3];
                    this->networkLinks[0][array[nd5][0]].bCell[arrayChangLinks[nd5]] = array[nd6];
                    this->networkLinks[0][array[nd6][0]].bCell[arrayChangLinks[nd6]] = array[nd5];
                    this->networkLinks[0][array[nd7][0]].bCell[arrayChangLinks[nd7]] = array[nd8];
                    this->networkLinks[0][array[nd8][0]].bCell[arrayChangLinks[nd8]] = array[nd7];
                    this->networkLinks[0][array[nd9][0]].bCell[arrayChangLinks[nd9]] = array[nd10];
                    this->networkLinks[0][array[nd10][0]].bCell[arrayChangLinks[nd10]] = array[nd9];
                    this->networkLinks[0][array[nd11][0]].bCell[arrayChangLinks[nd11]] = array[nd12];
                    this->networkLinks[0][array[nd12][0]].bCell[arrayChangLinks[nd12]] = array[nd11];



                }

                numOptimized ++;


                //                // test
                //                float evaCurrentAfter = this->evaluateWeightOfTSP<CM_DistanceSquaredEuclidean>();
                //                if(evaCurrentAfter < evaCurrentBefore)
                //                {
                //                    // test
                //                    int idNode1 = this->grayValueMap[0][_x];
                //                    int idNode3 = this->grayValueMap[0][node3_int];
                //                    cout << " correct optimizing " << idNode1  <<  ", id son1 " << this->grayValueMap[0][node2[0]]
                //                            << ". idCorrs " << idNode3 << ", id son3 "  << this->grayValueMap[0][node4[0]]  << endl;

                //                }
                //                else if (evaCurrentAfter == evaCurrentBefore)
                //                {
                //                    int idNode1 = this->grayValueMap[0][_x];
                //                    int idNode3 = this->grayValueMap[0][node3_int];
                //                    cout << " equal optimizing " << idNode1  <<  ", id son1 " << this->grayValueMap[0][node2[0]]
                //                            << ". idCorrs " << idNode3 << ", id son3 "  << this->grayValueMap[0][node4[0]]  << endl;
                //                }
                //                else
                //                {
                //                    int idNode1 = this->grayValueMap[0][_x];
                //                    int idNode3 = this->grayValueMap[0][node3_int];
                //                    cout << " warning optimizing " << idNode1  <<  ", id son1 " << this->grayValueMap[0][node2[0]]
                //                            << ". idCorrs " << idNode3 << ", id son3 "  << this->grayValueMap[0][node4[0]]  << endl;
                //                }

            }
        }

        //        cout  << "numOptimized = " << numOptimized << endl;

        return true;
    }

    //!WB.Q 2024 execute non-interacted 23456-opt with only node3 as code
    bool executeNonInteract23456optOnlyNode3(int& numOptimized,  Grid<GLint> optPossibilitiesMap, Grid<BufferLinkPointCoord > networkLinksCP){

        int numOptEexcuted = 0;

        for(int _x = 0; _x < this->adaptiveMap.width; _x++)
        {
            if(this->activeMap[0][_x] > 0)
            {
                numOptEexcuted += 1;
                cout << "Execute numOptEexcuted = " << numOptEexcuted << endl;

                //test
                // float evaCurrentBefore = this->evaluateWeightOfTSP<CM_DistanceSquaredEuclidean>();

                PointCoord node1(_x,0);

                int densityValue = this->densityMap[node1[1]][node1[0]];// k of kopt;
                int kValue = densityValue % 10;
                if(kValue > 6)
                    cout << "k-opt k execute= " << kValue << ", optmode=" << densityValue << endl;
                int optmode = densityValue /100;


                //new select non-interacted using optCandidateMap
                unsigned long long optCandidats = this->optCandidateMap[node1[1]][node1[0]];
                unsigned long long nodeResult;
                if(kValue < 6)
                    nodeResult = codeBit;
                if(kValue == 6)
                    nodeResult = codeBit12;

                int node3=initialPrepareValue,node5 = initialPrepareValue, node7 =initialPrepareValue, node9 = initialPrepareValue, node11=initialPrepareValue;

                if(kValue == 2)
                {
                    node3 = optCandidats & nodeResult;

                    //execute 2-opt
                    PointCoord node3_(node3, 0);

                    cout << "Execute 2-opt mode= "<< optmode<< " tour order: " << this->grayValueMap[0][node1[0]] << ", " <<  this->grayValueMap[0][node3] << endl;

                    // node2
                    PointCoord node2(0, 0);
                    PointCoord node4(0, 0);

                    int changeLink1 = 0;
                    int changeLink2 = 0;
                    int changeLink3 = 0;
                    int changeLink4 = 0;

                    returnChangeLinks(node1, node2, changeLink1, changeLink2, networkLinksCP);
                    returnChangeLinks(node3_, node4, changeLink3, changeLink4, networkLinksCP);

                    this->networkLinks[0][node1[0]].bCell[changeLink1] = node3_;
                    this->networkLinks[0][node3].bCell[changeLink3] = node1;
                    this->networkLinks[0][node2[0]].bCell[changeLink2] = node4;
                    this->networkLinks[0][node4[0]].bCell[changeLink4] = node2;

                }
                if(kValue == 3)
                {
                    node5 = optCandidats & nodeResult;
                    optCandidats = optCandidats >> 16;
                    node3 = optCandidats & nodeResult;

                    //execute 3-opt
                    PointCoord node3_(node3, 0);
                    PointCoord node5_(node5, 0);

                    // node2
                    PointCoord node2(0, 0);
                    PointCoord node4(0, 0);
                    PointCoord node6(0, 0);

                    int changeLink1 = 0;
                    int changeLink2 = 0;
                    int changeLink3 = 0;
                    int changeLink4 = 0;
                    int changeLink5 = 0;
                    int changeLink6 = 0;

                    returnChangeLinks(node1, node2, changeLink1, changeLink2, networkLinksCP);
                    returnChangeLinks(node3_, node4, changeLink3, changeLink4, networkLinksCP);
                    returnChangeLinks(node5_, node6, changeLink5, changeLink6, networkLinksCP);

                    cout << "Execute 3-opt mode= "<< optmode<< " tour order: " << this->grayValueMap[0][node1[0]] << ", " <<  this->grayValueMap[0][node3]
                                                                                                                     << ", " <<  this->grayValueMap[0][node5] << endl;

                    if(optmode == 0)
                    {
                        this->networkLinks[0][node1[0]].bCell[changeLink1] = node4;
                        this->networkLinks[0][node2[0]].bCell[changeLink2] = node6;
                        this->networkLinks[0][node3].bCell[changeLink3] = node5_;
                        this->networkLinks[0][node4[0]].bCell[changeLink4] = node1;
                        this->networkLinks[0][node5].bCell[changeLink5] = node3_;
                        this->networkLinks[0][node6[0]].bCell[changeLink6] = node2;
                    }
                    if(optmode == 1)
                    {
                        this->networkLinks[0][node1[0]].bCell[changeLink1] = node5_;
                        this->networkLinks[0][node2[0]].bCell[changeLink2] = node4;
                        this->networkLinks[0][node3].bCell[changeLink3] = node6;
                        this->networkLinks[0][node4[0]].bCell[changeLink4] = node2;
                        this->networkLinks[0][node5].bCell[changeLink5] = node1;
                        this->networkLinks[0][node6[0]].bCell[changeLink6] = node3_;
                    }
                    if(optmode == 2)
                    {
                        this->networkLinks[0][node1[0]].bCell[changeLink1] = node3_;
                        this->networkLinks[0][node2[0]].bCell[changeLink2] = node5_;
                        this->networkLinks[0][node3].bCell[changeLink3] = node1;
                        this->networkLinks[0][node4[0]].bCell[changeLink4] = node6;
                        this->networkLinks[0][node5].bCell[changeLink5] = node2;
                        this->networkLinks[0][node6[0]].bCell[changeLink6] = node4;

                    }
                    if(optmode == 3)
                    {
                        this->networkLinks[0][node1[0]].bCell[changeLink1] = node4;
                        this->networkLinks[0][node2[0]].bCell[changeLink2] = node5_;
                        this->networkLinks[0][node3].bCell[changeLink3] = node6;
                        this->networkLinks[0][node4[0]].bCell[changeLink4] = node1;
                        this->networkLinks[0][node5].bCell[changeLink5] = node2;
                        this->networkLinks[0][node6[0]].bCell[changeLink6] = node3_;

                    }




                }
                if(kValue == 4)
                {

                    node7 = optCandidats & nodeResult;
                    optCandidats = optCandidats >> 16;
                    node5 = optCandidats & nodeResult;
                    optCandidats = optCandidats >> 16;
                    node3 = optCandidats & nodeResult;

                    //                    //execute 4-opt
                    //                    cout << "Execute 4-opt mark node1-7: " << node1[0] << ", " << node3 << ", " << node5 << ", " << node7 << ", " << endl;
                    cout << "Execute 4-opt mode= "<< optmode<< " tour order: " << this->grayValueMap[0][node1[0]] << ", " <<  this->grayValueMap[0][node3]
                                                                                                                     << ", " <<  this->grayValueMap[0][node5] << ", "  << this->grayValueMap[0][node7] << endl;
                    //execute 4-opt
                    PointCoord node3_(node3, 0);
                    PointCoord node5_(node5, 0);
                    PointCoord node7_(node7, 0);

                    // node2
                    PointCoord node2(0, 0);
                    PointCoord node4(0, 0);
                    PointCoord node6(0, 0);
                    PointCoord node8(0, 0);


                    int changeLink1 = 0;
                    int changeLink2 = 0;
                    int changeLink3 = 0;
                    int changeLink4 = 0;
                    int changeLink5 = 0;
                    int changeLink6 = 0;
                    int changeLink7 = 0;
                    int changeLink8 = 0;

                    returnChangeLinks(node1, node2, changeLink1, changeLink2, networkLinksCP);
                    returnChangeLinks(node3_, node4, changeLink3, changeLink4, networkLinksCP);
                    returnChangeLinks(node5_, node6, changeLink5, changeLink6, networkLinksCP);
                    returnChangeLinks(node7_, node8, changeLink7, changeLink8, networkLinksCP);

                    PointCoord array[8];
                    array[0] = node1;
                    array[1] = node2;
                    array[2] = node3_;
                    array[3] = node4;
                    array[4] = node5_;
                    array[5] = node6;
                    array[6] = node7_;
                    array[7] = node8;
                    int arrayChangLinks[8]={-1,-1,-1,-1,-1,-1,-1,-1};
                    arrayChangLinks[0] = changeLink1;
                    arrayChangLinks[1] = changeLink2;
                    arrayChangLinks[2] = changeLink3;
                    arrayChangLinks[3] = changeLink4;
                    arrayChangLinks[4] = changeLink5;
                    arrayChangLinks[5] = changeLink6;
                    arrayChangLinks[6] = changeLink7;
                    arrayChangLinks[7] = changeLink8;


                    int nd1, nd2, nd3, nd4, nd5,nd6,nd7,nd8;
                    int optSelected = optmode*8;

                    //1 7 2 8 3 5 4 6
                    nd1 = optPossibilitiesMap[0][optSelected] -1; // 0
                    nd2 = optPossibilitiesMap[0][optSelected+1] -1; // 6
                    nd3 = optPossibilitiesMap[0][optSelected+2] -1;
                    nd4 = optPossibilitiesMap[0][optSelected+3] -1;
                    nd5 = optPossibilitiesMap[0][optSelected+4] -1;
                    nd6 = optPossibilitiesMap[0][optSelected+5] -1;
                    nd7 = optPossibilitiesMap[0][optSelected+6] -1;
                    nd8 = optPossibilitiesMap[0][optSelected+7] -1;

                    //                    cout << "Execute 4-opt nd1-8 = " << nd1 << " " << nd2 << " "<< nd3<< " "
                    //                         << nd4 <<" " << nd5 << " " << nd6<< " " << nd7 << " " << nd8 << " " << endl;

                    //                    cout << "Execute 4-opt array[nd1][0]= "  << array[nd1][0] << ", array[nd2][0]= " << array[nd2][0]  << ", array[nd3][0]= " << array[nd3][0]  << ", array[nd4][0]= " << array[nd4][0]<< endl;

                    //error here, changelinks is also changing accordng to ndx
                    this->networkLinks[0][array[nd1][0]].bCell[arrayChangLinks[nd1]] = array[nd2];
                    this->networkLinks[0][array[nd2][0]].bCell[arrayChangLinks[nd2]] = array[nd1];
                    this->networkLinks[0][array[nd3][0]].bCell[arrayChangLinks[nd3]] = array[nd4];
                    this->networkLinks[0][array[nd4][0]].bCell[arrayChangLinks[nd4]] = array[nd3];
                    this->networkLinks[0][array[nd5][0]].bCell[arrayChangLinks[nd5]] = array[nd6];
                    this->networkLinks[0][array[nd6][0]].bCell[arrayChangLinks[nd6]] = array[nd5];
                    this->networkLinks[0][array[nd7][0]].bCell[arrayChangLinks[nd7]] = array[nd8];
                    this->networkLinks[0][array[nd8][0]].bCell[arrayChangLinks[nd8]] = array[nd7];

                }
                if(kValue ==5)
                {
                    node9 = optCandidats & nodeResult;
                    optCandidats = optCandidats >> 16;
                    node7 = optCandidats & nodeResult;
                    optCandidats = optCandidats >> 16;
                    node5 = optCandidats & nodeResult;
                    optCandidats = optCandidats >> 16;
                    node3 = optCandidats & nodeResult;

                    //execute 5-opt
                    PointCoord node3_(node3, 0);
                    PointCoord node5_(node5, 0);
                    PointCoord node7_(node7, 0);
                    PointCoord node9_(node9, 0);


                    cout << "Execute 5-opt mode= "<< optmode<< " tour order: " << this->grayValueMap[0][node1[0]] << ", " <<  this->grayValueMap[0][node3]
                                                                                                                     << ", " <<  this->grayValueMap[0][node5] << ", "  << this->grayValueMap[0][node7]  << ", "  << this->grayValueMap[0][node9]<< endl;

                    // node2
                    PointCoord node2(0, 0);
                    PointCoord node4(0, 0);
                    PointCoord node6(0, 0);
                    PointCoord node8(0, 0);
                    PointCoord node10(0, 0);

                    int changeLink1 = 0;
                    int changeLink2 = 0;
                    int changeLink3 = 0;
                    int changeLink4 = 0;
                    int changeLink5 = 0;
                    int changeLink6 = 0;
                    int changeLink7 = 0;
                    int changeLink8 = 0;
                    int changeLink9 = 0;
                    int changeLink10 = 0;

                    returnChangeLinks(node1, node2, changeLink1, changeLink2, networkLinksCP);
                    returnChangeLinks(node3_, node4, changeLink3, changeLink4, networkLinksCP);
                    returnChangeLinks(node5_, node6, changeLink5, changeLink6, networkLinksCP);
                    returnChangeLinks(node7_, node8, changeLink7, changeLink8, networkLinksCP);
                    returnChangeLinks(node9_, node10, changeLink9, changeLink10, networkLinksCP);


                    PointCoord array[10];
                    array[0] = node1;
                    array[1] = node2;
                    array[2] = node3_;
                    array[3] = node4;
                    array[4] = node5_;
                    array[5] = node6;
                    array[6] = node7_;
                    array[7] = node8;
                    array[8] = node9_;
                    array[9] = node10;
                    int arrayChangLinks[10]={-1,-1,-1,-1,-1,-1,-1,-1,-1,-1};
                    arrayChangLinks[0] = changeLink1;
                    arrayChangLinks[1] = changeLink2;
                    arrayChangLinks[2] = changeLink3;
                    arrayChangLinks[3] = changeLink4;
                    arrayChangLinks[4] = changeLink5;
                    arrayChangLinks[5] = changeLink6;
                    arrayChangLinks[6] = changeLink7;
                    arrayChangLinks[7] = changeLink8;
                    arrayChangLinks[8] = changeLink9;
                    arrayChangLinks[9] = changeLink10;

                    int nd1, nd2, nd3, nd4, nd5,nd6,nd7,nd8,nd9,nd10;
                    int optSelected = optmode*10;

                    //1 7 2 8 3 5 4 6
                    nd1 = optPossibilitiesMap[0][optSelected] -1; // 0
                    nd2 = optPossibilitiesMap[0][optSelected+1] -1; // 6
                    nd3 = optPossibilitiesMap[0][optSelected+2] -1;
                    nd4 = optPossibilitiesMap[0][optSelected+3] -1;
                    nd5 = optPossibilitiesMap[0][optSelected+4] -1;
                    nd6 = optPossibilitiesMap[0][optSelected+5] -1;
                    nd7 = optPossibilitiesMap[0][optSelected+6] -1;
                    nd8 = optPossibilitiesMap[0][optSelected+7] -1;
                    nd9 = optPossibilitiesMap[0][optSelected+8] -1;
                    nd10 = optPossibilitiesMap[0][optSelected+9] -1;


                    //error here, changelinks is also changing accordng to ndx
                    this->networkLinks[0][array[nd1][0]].bCell[arrayChangLinks[nd1]] = array[nd2];
                    this->networkLinks[0][array[nd2][0]].bCell[arrayChangLinks[nd2]] = array[nd1];
                    this->networkLinks[0][array[nd3][0]].bCell[arrayChangLinks[nd3]] = array[nd4];
                    this->networkLinks[0][array[nd4][0]].bCell[arrayChangLinks[nd4]] = array[nd3];
                    this->networkLinks[0][array[nd5][0]].bCell[arrayChangLinks[nd5]] = array[nd6];
                    this->networkLinks[0][array[nd6][0]].bCell[arrayChangLinks[nd6]] = array[nd5];
                    this->networkLinks[0][array[nd7][0]].bCell[arrayChangLinks[nd7]] = array[nd8];
                    this->networkLinks[0][array[nd8][0]].bCell[arrayChangLinks[nd8]] = array[nd7];
                    this->networkLinks[0][array[nd9][0]].bCell[arrayChangLinks[nd9]] = array[nd10];
                    this->networkLinks[0][array[nd10][0]].bCell[arrayChangLinks[nd10]] = array[nd9];

                }
                if(kValue ==6)
                {

                    node11 = optCandidats & nodeResult;
                    optCandidats = optCandidats >> 12; //need to check only for 6-opt
                    node9 = optCandidats & nodeResult;
                    optCandidats = optCandidats >> 12;
                    node7 = optCandidats & nodeResult;
                    optCandidats = optCandidats >> 12;
                    node5 = optCandidats & nodeResult;
                    optCandidats = optCandidats >> 12;
                    node3 = optCandidats & nodeResult;

                    //                    //execute 6-opt
                    //                    cout << "Execute 6-opt mark node1-11: " << node1[0] << ", " << node3 << ", " << node5 << ", " << node7 << ", " << node9 << ", " <<  node11 << ", " <<endl;
                    cout << "Execute 6-opt mode= "<< optmode<< " tour order: " << this->grayValueMap[0][node1[0]] << " " <<  this->grayValueMap[0][node3]
                                                                                                                     << " " << this->grayValueMap[0][node5] << " " << this->grayValueMap[0][node7]
                                                                                                                        << " " << this->grayValueMap[0][node9] << " " << this->grayValueMap[0][node11] << endl;

                    //execute 5-opt
                    PointCoord node3_(node3, 0);
                    PointCoord node5_(node5, 0);
                    PointCoord node7_(node7, 0);
                    PointCoord node9_(node9, 0);
                    PointCoord node11_(node11, 0);

                    // node2
                    PointCoord node2(0, 0);
                    PointCoord node4(0, 0);
                    PointCoord node6(0, 0);
                    PointCoord node8(0, 0);
                    PointCoord node10(0, 0);
                    PointCoord node12(0, 0);

                    int changeLink1 = 0;
                    int changeLink2 = 0;
                    int changeLink3 = 0;
                    int changeLink4 = 0;
                    int changeLink5 = 0;
                    int changeLink6 = 0;
                    int changeLink7 = 0;
                    int changeLink8 = 0;
                    int changeLink9 = 0;
                    int changeLink10 = 0;
                    int changeLink11 = 0;
                    int changeLink12 = 0;

                    returnChangeLinks(node1, node2, changeLink1, changeLink2, networkLinksCP);
                    returnChangeLinks(node3_, node4, changeLink3, changeLink4, networkLinksCP);
                    returnChangeLinks(node5_, node6, changeLink5, changeLink6, networkLinksCP);
                    returnChangeLinks(node7_, node8, changeLink7, changeLink8, networkLinksCP);
                    returnChangeLinks(node9_, node10, changeLink9, changeLink10, networkLinksCP);
                    returnChangeLinks(node11_, node12, changeLink11, changeLink12, networkLinksCP);


                    PointCoord array[12];
                    array[0] = node1;
                    array[1] = node2;
                    array[2] = node3_;
                    array[3] = node4;
                    array[4] = node5_;
                    array[5] = node6;
                    array[6] = node7_;
                    array[7] = node8;
                    array[8] = node9_;
                    array[9] = node10;
                    array[10] = node11_;
                    array[11] = node12;

                    int arrayChangLinks[12]={-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1};
                    arrayChangLinks[0] = changeLink1;
                    arrayChangLinks[1] = changeLink2;
                    arrayChangLinks[2] = changeLink3;
                    arrayChangLinks[3] = changeLink4;
                    arrayChangLinks[4] = changeLink5;
                    arrayChangLinks[5] = changeLink6;
                    arrayChangLinks[6] = changeLink7;
                    arrayChangLinks[7] = changeLink8;
                    arrayChangLinks[8] = changeLink9;
                    arrayChangLinks[9] = changeLink10;
                    arrayChangLinks[10] = changeLink11;
                    arrayChangLinks[11] = changeLink12;


                    int nd1, nd2, nd3, nd4, nd5,nd6,nd7,nd8,nd9,nd10,nd11,nd12;
                    int optSelected = optmode*12;

                    //1 7 2 8 3 5 4 6
                    nd1 = optPossibilitiesMap[0][optSelected] -1; // 0
                    nd2 = optPossibilitiesMap[0][optSelected+1] -1; // 6
                    nd3 = optPossibilitiesMap[0][optSelected+2] -1;
                    nd4 = optPossibilitiesMap[0][optSelected+3] -1;
                    nd5 = optPossibilitiesMap[0][optSelected+4] -1;
                    nd6 = optPossibilitiesMap[0][optSelected+5] -1;
                    nd7 = optPossibilitiesMap[0][optSelected+6] -1;
                    nd8 = optPossibilitiesMap[0][optSelected+7] -1;
                    nd9 = optPossibilitiesMap[0][optSelected+8] -1;
                    nd10 = optPossibilitiesMap[0][optSelected+9] -1;
                    nd11 = optPossibilitiesMap[0][optSelected+10] -1;
                    nd12 = optPossibilitiesMap[0][optSelected+11] -1;

                    //error here, changelinks is also changing accordng to ndx
                    this->networkLinks[0][array[nd1][0]].bCell[arrayChangLinks[nd1]] = array[nd2];
                    this->networkLinks[0][array[nd2][0]].bCell[arrayChangLinks[nd2]] = array[nd1];
                    this->networkLinks[0][array[nd3][0]].bCell[arrayChangLinks[nd3]] = array[nd4];
                    this->networkLinks[0][array[nd4][0]].bCell[arrayChangLinks[nd4]] = array[nd3];
                    this->networkLinks[0][array[nd5][0]].bCell[arrayChangLinks[nd5]] = array[nd6];
                    this->networkLinks[0][array[nd6][0]].bCell[arrayChangLinks[nd6]] = array[nd5];
                    this->networkLinks[0][array[nd7][0]].bCell[arrayChangLinks[nd7]] = array[nd8];
                    this->networkLinks[0][array[nd8][0]].bCell[arrayChangLinks[nd8]] = array[nd7];
                    this->networkLinks[0][array[nd9][0]].bCell[arrayChangLinks[nd9]] = array[nd10];
                    this->networkLinks[0][array[nd10][0]].bCell[arrayChangLinks[nd10]] = array[nd9];
                    this->networkLinks[0][array[nd11][0]].bCell[arrayChangLinks[nd11]] = array[nd12];
                    this->networkLinks[0][array[nd12][0]].bCell[arrayChangLinks[nd12]] = array[nd11];



                }

                numOptimized ++;


                //                // test
                //                float evaCurrentAfter = this->evaluateWeightOfTSP<CM_DistanceSquaredEuclidean>();
                //                if(evaCurrentAfter < evaCurrentBefore)
                //                {
                //                    // test
                //                    int idNode1 = this->grayValueMap[0][_x];
                //                    int idNode3 = this->grayValueMap[0][node3_int];
                //                    cout << " correct optimizing " << idNode1  <<  ", id son1 " << this->grayValueMap[0][node2[0]]
                //                            << ". idCorrs " << idNode3 << ", id son3 "  << this->grayValueMap[0][node4[0]]  << endl;

                //                }
                //                else if (evaCurrentAfter == evaCurrentBefore)
                //                {
                //                    int idNode1 = this->grayValueMap[0][_x];
                //                    int idNode3 = this->grayValueMap[0][node3_int];
                //                    cout << " equal optimizing " << idNode1  <<  ", id son1 " << this->grayValueMap[0][node2[0]]
                //                            << ". idCorrs " << idNode3 << ", id son3 "  << this->grayValueMap[0][node4[0]]  << endl;
                //                }
                //                else
                //                {
                //                    int idNode1 = this->grayValueMap[0][_x];
                //                    int idNode3 = this->grayValueMap[0][node3_int];
                //                    cout << " warning optimizing " << idNode1  <<  ", id son1 " << this->grayValueMap[0][node2[0]]
                //                            << ". idCorrs " << idNode3 << ", id son3 "  << this->grayValueMap[0][node4[0]]  << endl;
                //                }

            }
        }

        cout  << "numOptimized = " << numOptimized << endl;

        return true;
    }//end execute



    //!WB.Q 2024 execute non-interacted 23456-opt with only node3 as code
    bool executeNonInteract23456optOnlyNode3(int& numOptimized,
                                             Grid<int> fourOptPossibilitiesMap,
                                             Grid<int> fiveOptPossibilitiesMap,
                                             Grid<int> sixOptPossibilitiesMap){

        int numOptEexcuted = 0;

        for(int _x = 0; _x < this->adaptiveMap.width; _x++)
        {
            if(this->activeMap[0][_x] > 0)
            {
                numOptEexcuted += 1;
                //                cout << "Execute numOptEexcuted = " << numOptEexcuted << endl;

                //test
                // float evaCurrentBefore = this->evaluateWeightOfTSP<CM_DistanceSquaredEuclidean>();

                PointCoord node1(_x,0);

                int densityValue = this->densityMap[node1[1]][node1[0]];// k of kopt;
                int kValue = densityValue % 10;
                if(kValue > 6)
                    cout << "k-opt k execute= " << kValue << ", optmode=" << densityValue << endl;
                int optmode = densityValue /100;


                //new select non-interacted using optCandidateMap
                unsigned long long optCandidats = this->optCandidateMap[node1[1]][node1[0]];
                unsigned long long nodeResult;
                if(kValue < 6)
                    nodeResult = codeBit;
                if(kValue == 6)
                    nodeResult = codeBit12;

                int node3=initialPrepareValue,node5 = initialPrepareValue, node7 =initialPrepareValue, node9 = initialPrepareValue, node11=initialPrepareValue;

                if(kValue == 2)
                {
                    node3 = optCandidats & nodeResult;

                    //execute 2-opt
                    PointCoord node3_(node3, 0);

                    // node2
                    PointCoord node2(0, 0);
                    PointCoord node4(0, 0);

                    int changeLink1 = 0;
                    int changeLink2 = 0;
                    int changeLink3 = 0;
                    int changeLink4 = 0;

                    returnChangeLinks(node1, node2, changeLink1, changeLink2);
                    returnChangeLinks(node3_, node4, changeLink3, changeLink4);

                    this->networkLinks[0][node1[0]].bCell[changeLink1] = node3_;
                    this->networkLinks[0][node3].bCell[changeLink3] = node1;
                    this->networkLinks[0][node2[0]].bCell[changeLink2] = node4;
                    this->networkLinks[0][node4[0]].bCell[changeLink4] = node2;

                }
                if(kValue == 3)
                {
                    node5 = optCandidats & nodeResult;
                    optCandidats = optCandidats >> 16;
                    node3 = optCandidats & nodeResult;

                    //execute 3-opt
                    PointCoord node3_(node3, 0);
                    PointCoord node5_(node5, 0);

                    // node2
                    PointCoord node2(0, 0);
                    PointCoord node4(0, 0);
                    PointCoord node6(0, 0);

                    int changeLink1 = 0;
                    int changeLink2 = 0;
                    int changeLink3 = 0;
                    int changeLink4 = 0;
                    int changeLink5 = 0;
                    int changeLink6 = 0;

                    returnChangeLinks(node1, node2, changeLink1, changeLink2);
                    returnChangeLinks(node3_, node4, changeLink3, changeLink4);
                    returnChangeLinks(node5_, node6, changeLink5, changeLink6);


                    //                    //qiao only for test why optimization > old, old length, compute edge's length according to links and
                    //                    double oldLength = dist(node1, node2, *this, *this) + dist(node3_, node4, *this, *this) + dist(node5_, node6, *this, *this);


                    //                                        cout << "Execute 3-opt mode= "<< optmode<< " tour order: " << this->grayValueMap[0][node1[0]] << ", " <<  this->grayValueMap[0][node3]
                    //                                                                                                                                         << ", " <<  this->grayValueMap[0][node5]
                    //                                                                                                                                            << ", " <<  this->grayValueMap[0][node6[0]]<< endl;


                    //                    //qiao only for test why optimization > old
                    //                    double newLength = 0;



                    //work code
                    if(optmode == 0)
                    {
                        this->networkLinks[0][node1[0]].bCell[changeLink1] = node4;
                        this->networkLinks[0][node2[0]].bCell[changeLink2] = node6;
                        this->networkLinks[0][node3].bCell[changeLink3] = node5_;
                        this->networkLinks[0][node4[0]].bCell[changeLink4] = node1;
                        this->networkLinks[0][node5].bCell[changeLink5] = node3_;
                        this->networkLinks[0][node6[0]].bCell[changeLink6] = node2;

                        //                        //qiao test code
                        //                        newLength =  dist(node1, node4, *this, *this) + dist(node2, node6, *this, *this) + dist(node5_, node3_, *this, *this);

                    }
                    if(optmode == 1)
                    {
                        this->networkLinks[0][node1[0]].bCell[changeLink1] = node5_;
                        this->networkLinks[0][node2[0]].bCell[changeLink2] = node4;
                        this->networkLinks[0][node3].bCell[changeLink3] = node6;
                        this->networkLinks[0][node4[0]].bCell[changeLink4] = node2;
                        this->networkLinks[0][node5].bCell[changeLink5] = node1;
                        this->networkLinks[0][node6[0]].bCell[changeLink6] = node3_;

                        //                        //qiao test code
                        //                        newLength =  dist(node1, node5_, *this, *this) + dist(node2, node4, *this, *this) + dist(node6, node3_, *this, *this);

                    }
                    if(optmode == 2)
                    {
                        this->networkLinks[0][node1[0]].bCell[changeLink1] = node3_;
                        this->networkLinks[0][node2[0]].bCell[changeLink2] = node5_;
                        this->networkLinks[0][node3].bCell[changeLink3] = node1;
                        this->networkLinks[0][node4[0]].bCell[changeLink4] = node6;
                        this->networkLinks[0][node5].bCell[changeLink5] = node2;
                        this->networkLinks[0][node6[0]].bCell[changeLink6] = node4;

                        //                        //qiao test code
                        //                        newLength =  dist(node1, node3_, *this, *this) + dist(node2, node5_, *this, *this) + dist(node4, node6, *this, *this);


                    }
                    if(optmode == 3)
                    {
                        this->networkLinks[0][node1[0]].bCell[changeLink1] = node4;
                        this->networkLinks[0][node2[0]].bCell[changeLink2] = node5_;
                        this->networkLinks[0][node3].bCell[changeLink3] = node6;
                        this->networkLinks[0][node4[0]].bCell[changeLink4] = node1;
                        this->networkLinks[0][node5].bCell[changeLink5] = node2;
                        this->networkLinks[0][node6[0]].bCell[changeLink6] = node3_;

                        //                        //qiao test code
                        //                        newLength =  dist(node1, node4, *this, *this) + dist(node2, node5_, *this, *this) + dist(node6, node3_, *this, *this);

                    }

                    //                    //new length
                    //                    if(newLength > oldLength)
                    //                        cout << "Error execute 3-opt: mode= " <<  optmode<< ", tour order: " << this->grayValueMap[0][node1[0]] << ", " <<  this->grayValueMap[0][node3]
                    //                                                                                            << ", " <<  this->grayValueMap[0][node5]
                    //                                                                                               << ", " <<  this->grayValueMap[0][node6[0]]<< endl;



                }
                if(kValue ==4)
                {

                    node7 = optCandidats & nodeResult;
                    optCandidats = optCandidats >> 16;
                    node5 = optCandidats & nodeResult;
                    optCandidats = optCandidats >> 16;
                    node3 = optCandidats & nodeResult;

                    //                    //execute 4-opt
                    //                    cout << "Execute 4-opt mark node1-7: " << node1[0] << ", " << node3 << ", " << node5 << ", " << node7 << ", " << endl;
                    //                    cout << "Execute 4-opt mode= "<< optmode<< " tour order: " << this->grayValueMap[0][node1[0]] << ", " <<  this->grayValueMap[0][node3]
                    //                                                                                                                     << ", " <<  this->grayValueMap[0][node5] << ", "  << this->grayValueMap[0][node7] << endl;

                    //execute 4-opt
                    PointCoord node3_(node3, 0);
                    PointCoord node5_(node5, 0);
                    PointCoord node7_(node7, 0);

                    // node2
                    PointCoord node2(0, 0);
                    PointCoord node4(0, 0);
                    PointCoord node6(0, 0);
                    PointCoord node8(0, 0);


                    int changeLink1 = 0;
                    int changeLink2 = 0;
                    int changeLink3 = 0;
                    int changeLink4 = 0;
                    int changeLink5 = 0;
                    int changeLink6 = 0;
                    int changeLink7 = 0;
                    int changeLink8 = 0;

                    returnChangeLinks(node1, node2, changeLink1, changeLink2);
                    returnChangeLinks(node3_, node4, changeLink3, changeLink4);
                    returnChangeLinks(node5_, node6, changeLink5, changeLink6);
                    returnChangeLinks(node7_, node8, changeLink7, changeLink8);

                    PointCoord array[8];
                    array[0] = node1;
                    array[1] = node2;
                    array[2] = node3_;
                    array[3] = node4;
                    array[4] = node5_;
                    array[5] = node6;
                    array[6] = node7_;
                    array[7] = node8;
                    int arrayChangLinks[8]={-1,-1,-1,-1,-1,-1,-1,-1};
                    arrayChangLinks[0] = changeLink1;
                    arrayChangLinks[1] = changeLink2;
                    arrayChangLinks[2] = changeLink3;
                    arrayChangLinks[3] = changeLink4;
                    arrayChangLinks[4] = changeLink5;
                    arrayChangLinks[5] = changeLink6;
                    arrayChangLinks[6] = changeLink7;
                    arrayChangLinks[7] = changeLink8;


                    int nd1, nd2, nd3, nd4, nd5,nd6,nd7,nd8;
                    int optSelected = optmode*8;

                    //1 7 2 8 3 5 4 6
                    nd1 = fourOptPossibilitiesMap[0][optSelected] -1; // 0
                    nd2 = fourOptPossibilitiesMap[0][optSelected+1] -1; // 6
                    nd3 = fourOptPossibilitiesMap[0][optSelected+2] -1;
                    nd4 = fourOptPossibilitiesMap[0][optSelected+3] -1;
                    nd5 = fourOptPossibilitiesMap[0][optSelected+4] -1;
                    nd6 = fourOptPossibilitiesMap[0][optSelected+5] -1;
                    nd7 = fourOptPossibilitiesMap[0][optSelected+6] -1;
                    nd8 = fourOptPossibilitiesMap[0][optSelected+7] -1;

                    //                    cout << "Execute 4-opt nd1-8 = " << nd1 << " " << nd2 << " "<< nd3<< " "
                    //                         << nd4 <<" " << nd5 << " " << nd6<< " " << nd7 << " " << nd8 << " " << endl;

                    //                    cout << "Execute 4-opt array[nd1][0]= "  << array[nd1][0] << ", array[nd2][0]= " << array[nd2][0]  << ", array[nd3][0]= " << array[nd3][0]  << ", array[nd4][0]= " << array[nd4][0]<< endl;

                    //error here, changelinks is also changing accordng to ndx
                    this->networkLinks[0][array[nd1][0]].bCell[arrayChangLinks[nd1]] = array[nd2];
                    this->networkLinks[0][array[nd2][0]].bCell[arrayChangLinks[nd2]] = array[nd1];
                    this->networkLinks[0][array[nd3][0]].bCell[arrayChangLinks[nd3]] = array[nd4];
                    this->networkLinks[0][array[nd4][0]].bCell[arrayChangLinks[nd4]] = array[nd3];
                    this->networkLinks[0][array[nd5][0]].bCell[arrayChangLinks[nd5]] = array[nd6];
                    this->networkLinks[0][array[nd6][0]].bCell[arrayChangLinks[nd6]] = array[nd5];
                    this->networkLinks[0][array[nd7][0]].bCell[arrayChangLinks[nd7]] = array[nd8];
                    this->networkLinks[0][array[nd8][0]].bCell[arrayChangLinks[nd8]] = array[nd7];

                }
                if(kValue ==5)
                {
                    node9 = optCandidats & nodeResult;
                    optCandidats = optCandidats >> 16;
                    node7 = optCandidats & nodeResult;
                    optCandidats = optCandidats >> 16;
                    node5 = optCandidats & nodeResult;
                    optCandidats = optCandidats >> 16;
                    node3 = optCandidats & nodeResult;

                    //execute 5-opt
                    PointCoord node3_(node3, 0);
                    PointCoord node5_(node5, 0);
                    PointCoord node7_(node7, 0);
                    PointCoord node9_(node9, 0);


                    //                    cout << "Execute 5-opt mode= "<< optmode<< " tour order: " << this->grayValueMap[0][node1[0]] << ", " <<  this->grayValueMap[0][node3]
                    //                                                                                                                     << ", " <<  this->grayValueMap[0][node5] << ", "  << this->grayValueMap[0][node7]  << ", "  << this->grayValueMap[0][node9]<< endl;

                    // node2
                    PointCoord node2(0, 0);
                    PointCoord node4(0, 0);
                    PointCoord node6(0, 0);
                    PointCoord node8(0, 0);
                    PointCoord node10(0, 0);

                    int changeLink1 = 0;
                    int changeLink2 = 0;
                    int changeLink3 = 0;
                    int changeLink4 = 0;
                    int changeLink5 = 0;
                    int changeLink6 = 0;
                    int changeLink7 = 0;
                    int changeLink8 = 0;
                    int changeLink9 = 0;
                    int changeLink10 = 0;

                    returnChangeLinks(node1, node2, changeLink1, changeLink2);
                    returnChangeLinks(node3_, node4, changeLink3, changeLink4);
                    returnChangeLinks(node5_, node6, changeLink5, changeLink6);
                    returnChangeLinks(node7_, node8, changeLink7, changeLink8);
                    returnChangeLinks(node9_, node10, changeLink9, changeLink10);


                    PointCoord array[10];
                    array[0] = node1;
                    array[1] = node2;
                    array[2] = node3_;
                    array[3] = node4;
                    array[4] = node5_;
                    array[5] = node6;
                    array[6] = node7_;
                    array[7] = node8;
                    array[8] = node9_;
                    array[9] = node10;
                    int arrayChangLinks[10]={-1,-1,-1,-1,-1,-1,-1,-1,-1,-1};
                    arrayChangLinks[0] = changeLink1;
                    arrayChangLinks[1] = changeLink2;
                    arrayChangLinks[2] = changeLink3;
                    arrayChangLinks[3] = changeLink4;
                    arrayChangLinks[4] = changeLink5;
                    arrayChangLinks[5] = changeLink6;
                    arrayChangLinks[6] = changeLink7;
                    arrayChangLinks[7] = changeLink8;
                    arrayChangLinks[8] = changeLink9;
                    arrayChangLinks[9] = changeLink10;

                    int nd1, nd2, nd3, nd4, nd5,nd6,nd7,nd8,nd9,nd10;
                    int optSelected = optmode*10;

                    //1 7 2 8 3 5 4 6
                    nd1 = fiveOptPossibilitiesMap[0][optSelected] -1; // 0
                    nd2 = fiveOptPossibilitiesMap[0][optSelected+1] -1; // 6
                    nd3 = fiveOptPossibilitiesMap[0][optSelected+2] -1;
                    nd4 = fiveOptPossibilitiesMap[0][optSelected+3] -1;
                    nd5 = fiveOptPossibilitiesMap[0][optSelected+4] -1;
                    nd6 = fiveOptPossibilitiesMap[0][optSelected+5] -1;
                    nd7 = fiveOptPossibilitiesMap[0][optSelected+6] -1;
                    nd8 = fiveOptPossibilitiesMap[0][optSelected+7] -1;
                    nd9 = fiveOptPossibilitiesMap[0][optSelected+8] -1;
                    nd10 = fiveOptPossibilitiesMap[0][optSelected+9] -1;


                    //error here, changelinks is also changing accordng to ndx
                    this->networkLinks[0][array[nd1][0]].bCell[arrayChangLinks[nd1]] = array[nd2];
                    this->networkLinks[0][array[nd2][0]].bCell[arrayChangLinks[nd2]] = array[nd1];
                    this->networkLinks[0][array[nd3][0]].bCell[arrayChangLinks[nd3]] = array[nd4];
                    this->networkLinks[0][array[nd4][0]].bCell[arrayChangLinks[nd4]] = array[nd3];
                    this->networkLinks[0][array[nd5][0]].bCell[arrayChangLinks[nd5]] = array[nd6];
                    this->networkLinks[0][array[nd6][0]].bCell[arrayChangLinks[nd6]] = array[nd5];
                    this->networkLinks[0][array[nd7][0]].bCell[arrayChangLinks[nd7]] = array[nd8];
                    this->networkLinks[0][array[nd8][0]].bCell[arrayChangLinks[nd8]] = array[nd7];
                    this->networkLinks[0][array[nd9][0]].bCell[arrayChangLinks[nd9]] = array[nd10];
                    this->networkLinks[0][array[nd10][0]].bCell[arrayChangLinks[nd10]] = array[nd9];

                }
                if(kValue ==6)
                {

                    node11 = optCandidats & nodeResult;
                    optCandidats = optCandidats >> 12; //need to check only for 6-opt
                    node9 = optCandidats & nodeResult;
                    optCandidats = optCandidats >> 12;
                    node7 = optCandidats & nodeResult;
                    optCandidats = optCandidats >> 12;
                    node5 = optCandidats & nodeResult;
                    optCandidats = optCandidats >> 12;
                    node3 = optCandidats & nodeResult;

                    //                    //execute 6-opt
                    //                    cout << "Execute 6-opt mark node1-11: " << node1[0] << ", " << node3 << ", " << node5 << ", " << node7 << ", " << node9 << ", " <<  node11 << ", " <<endl;
                    //                    cout << "Execute 6-opt mode= "<< optmode<< " tour order: "
                    //                         << this->grayValueMap[0][node1[0]]
                    //                            << " " <<  this->grayValueMap[0][node3]
                    //                               << " " << this->grayValueMap[0][node5]
                    //                                  << " " << this->grayValueMap[0][node7]
                    //                                  << " " << this->grayValueMap[0][node9]
                    //                                     << " " << this->grayValueMap[0][node11] << endl;

                    //execute 6-opt
                    PointCoord node3_(node3, 0);
                    PointCoord node5_(node5, 0);
                    PointCoord node7_(node7, 0);
                    PointCoord node9_(node9, 0);
                    PointCoord node11_(node11, 0);

                    // node2
                    PointCoord node2(0, 0);
                    PointCoord node4(0, 0);
                    PointCoord node6(0, 0);
                    PointCoord node8(0, 0);
                    PointCoord node10(0, 0);
                    PointCoord node12(0, 0);

                    int changeLink1 = 0;
                    int changeLink2 = 0;
                    int changeLink3 = 0;
                    int changeLink4 = 0;
                    int changeLink5 = 0;
                    int changeLink6 = 0;
                    int changeLink7 = 0;
                    int changeLink8 = 0;
                    int changeLink9 = 0;
                    int changeLink10 = 0;
                    int changeLink11 = 0;
                    int changeLink12 = 0;

                    returnChangeLinks(node1, node2, changeLink1, changeLink2);
                    returnChangeLinks(node3_, node4, changeLink3, changeLink4);
                    returnChangeLinks(node5_, node6, changeLink5, changeLink6);
                    returnChangeLinks(node7_, node8, changeLink7, changeLink8);
                    returnChangeLinks(node9_, node10, changeLink9, changeLink10);
                    returnChangeLinks(node11_, node12, changeLink11, changeLink12);

                    //                    //qiao only for test why optimization > old, old length, compute edge's length according to links and
                    //                    double oldLength = dist(node1, node2, *this, *this) + dist(node3_, node4, *this, *this) + dist(node5_, node6, *this, *this)
                    //                            + dist(node7_, node8, *this, *this) + dist(node9_, node10, *this, *this) + dist(node11_, node12, *this, *this);


                    //                    cout << "Execute 6-opt mode= "<< optmode<< " tour order: " << this->grayValueMap[0][node1[0]] << ", " <<  this->grayValueMap[0][node3]
                    //                                                                                                                     << ", " <<  this->grayValueMap[0][node5] << ", " <<  this->grayValueMap[0][node6[0]] << ", " <<  this->grayValueMap[0][node9_[0]] << ", " <<  this->grayValueMap[0][node11]<< endl;

                    //qiao only for test why optimization > old
                    double newLength = 0;

                    PointCoord array[12];
                    array[0] = node1;
                    array[1] = node2;
                    array[2] = node3_;
                    array[3] = node4;
                    array[4] = node5_;
                    array[5] = node6;
                    array[6] = node7_;
                    array[7] = node8;
                    array[8] = node9_;
                    array[9] = node10;
                    array[10] = node11_;
                    array[11] = node12;

                    int arrayChangLinks[12]={-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1};
                    arrayChangLinks[0] = changeLink1;
                    arrayChangLinks[1] = changeLink2;
                    arrayChangLinks[2] = changeLink3;
                    arrayChangLinks[3] = changeLink4;
                    arrayChangLinks[4] = changeLink5;
                    arrayChangLinks[5] = changeLink6;
                    arrayChangLinks[6] = changeLink7;
                    arrayChangLinks[7] = changeLink8;
                    arrayChangLinks[8] = changeLink9;
                    arrayChangLinks[9] = changeLink10;
                    arrayChangLinks[10] = changeLink11;
                    arrayChangLinks[11] = changeLink12;


                    int nd1, nd2, nd3, nd4, nd5,nd6,nd7,nd8,nd9,nd10,nd11,nd12;
                    int optSelected = optmode*12;

                    //1 7 2 8 3 5 4 6
                    nd1 = sixOptPossibilitiesMap[0][optSelected] -1; // 0
                    nd2 = sixOptPossibilitiesMap[0][optSelected+1] -1; // 6
                    nd3 = sixOptPossibilitiesMap[0][optSelected+2] -1;
                    nd4 = sixOptPossibilitiesMap[0][optSelected+3] -1;
                    nd5 = sixOptPossibilitiesMap[0][optSelected+4] -1;
                    nd6 = sixOptPossibilitiesMap[0][optSelected+5] -1;
                    nd7 = sixOptPossibilitiesMap[0][optSelected+6] -1;
                    nd8 = sixOptPossibilitiesMap[0][optSelected+7] -1;
                    nd9 = sixOptPossibilitiesMap[0][optSelected+8] -1;
                    nd10 = sixOptPossibilitiesMap[0][optSelected+9] -1;
                    nd11 = sixOptPossibilitiesMap[0][optSelected+10] -1;
                    nd12 = sixOptPossibilitiesMap[0][optSelected+11] -1;

                    this->networkLinks[0][array[nd1][0]].bCell[arrayChangLinks[nd1]] = array[nd2];
                    this->networkLinks[0][array[nd2][0]].bCell[arrayChangLinks[nd2]] = array[nd1];
                    this->networkLinks[0][array[nd3][0]].bCell[arrayChangLinks[nd3]] = array[nd4];
                    this->networkLinks[0][array[nd4][0]].bCell[arrayChangLinks[nd4]] = array[nd3];
                    this->networkLinks[0][array[nd5][0]].bCell[arrayChangLinks[nd5]] = array[nd6];
                    this->networkLinks[0][array[nd6][0]].bCell[arrayChangLinks[nd6]] = array[nd5];
                    this->networkLinks[0][array[nd7][0]].bCell[arrayChangLinks[nd7]] = array[nd8];
                    this->networkLinks[0][array[nd8][0]].bCell[arrayChangLinks[nd8]] = array[nd7];
                    this->networkLinks[0][array[nd9][0]].bCell[arrayChangLinks[nd9]] = array[nd10];
                    this->networkLinks[0][array[nd10][0]].bCell[arrayChangLinks[nd10]] = array[nd9];
                    this->networkLinks[0][array[nd11][0]].bCell[arrayChangLinks[nd11]] = array[nd12];
                    this->networkLinks[0][array[nd12][0]].bCell[arrayChangLinks[nd12]] = array[nd11];

                    //                    //qiao test code
                    //                    newLength =  dist(array[nd1], array[nd2], *this, *this) + dist(array[nd3], array[nd4], *this, *this) + dist(array[nd5], array[nd6], *this, *this)
                    //                            + dist(array[nd7], array[nd8], *this, *this) + dist(array[nd9], array[nd10], *this, *this) + dist(array[nd11], array[nd12], *this, *this);


                    //                    //new length
                    //                    if(newLength > oldLength)
                    //                    cout << "ERROR Execute 6-opt mode= "<< optmode<< " tour order: "
                    //                         << this->grayValueMap[0][node1[0]]
                    //                            << " " <<  this->grayValueMap[0][node3]
                    //                               << " " << this->grayValueMap[0][node5]
                    //                                  << " " << this->grayValueMap[0][node7]
                    //                                    << " " << this->grayValueMap[0][node9]
                    //                                     << " " << this->grayValueMap[0][node11] << endl;
                    //                    else
                    //                        cout << "Correct execute 6-opt" << endl;

                }

                numOptimized ++;


                //                // test
                //                float evaCurrentAfter = this->evaluateWeightOfTSP<CM_DistanceSquaredEuclidean>();
                //                if(evaCurrentAfter < evaCurrentBefore)
                //                {
                //                    // test
                //                    int idNode1 = this->grayValueMap[0][_x];
                //                    int idNode3 = this->grayValueMap[0][node3_int];
                //                    cout << " correct optimizing " << idNode1  <<  ", id son1 " << this->grayValueMap[0][node2[0]]
                //                            << ". idCorrs " << idNode3 << ", id son3 "  << this->grayValueMap[0][node4[0]]  << endl;

                //                }
                //                else if (evaCurrentAfter == evaCurrentBefore)
                //                {
                //                    int idNode1 = this->grayValueMap[0][_x];
                //                    int idNode3 = this->grayValueMap[0][node3_int];
                //                    cout << " equal optimizing " << idNode1  <<  ", id son1 " << this->grayValueMap[0][node2[0]]
                //                            << ". idCorrs " << idNode3 << ", id son3 "  << this->grayValueMap[0][node4[0]]  << endl;
                //                }
                //                else
                //                {
                //                    int idNode1 = this->grayValueMap[0][_x];
                //                    int idNode3 = this->grayValueMap[0][node3_int];
                //                    cout << " warning optimizing " << idNode1  <<  ", id son1 " << this->grayValueMap[0][node2[0]]
                //                            << ". idCorrs " << idNode3 << ", id son3 "  << this->grayValueMap[0][node4[0]]  << endl;
                //                }

            }
        }

        cout  << "numOptimized = " << numOptimized << endl;

        return true;
    }//end execute



    //!WB.Q 2024 execute non-interacted 23456-opt with only node3 as code
    bool executeNonInteract23456optOnlyNode3(int& numOptimized,  Grid<GLint> fourOptPossibilitiesMap,  Grid<GLint> fiveOptPossibilitiesMap,
                                             Grid<GLint> sixOptPossibilitiesMap, Grid<BufferLinkPointCoord > networkLinksCP){

        int numOptEexcuted = 0;

        for(int _x = 0; _x < this->adaptiveMap.width; _x++)
        {
            if(this->activeMap[0][_x] > 0)
            {
                numOptEexcuted += 1;
                cout << "Execute numOptEexcuted PB = " << numOptEexcuted << endl;

                //test
                // float evaCurrentBefore = this->evaluateWeightOfTSP<CM_DistanceSquaredEuclidean>();

                PointCoord node1(_x,0);

                int densityValue = this->densityMap[node1[1]][node1[0]];// k of kopt;
                int kValue = densityValue % 10;
                if(kValue > 6)
                    cout << "k-opt k execute= " << kValue << ", optmode=" << densityValue << endl;
                int optmode = densityValue /100;

                cout << " kValue " << kValue << endl;


                //new select non-interacted using optCandidateMap
                unsigned long long optCandidats = this->optCandidateMap[node1[1]][node1[0]];
                unsigned long long nodeResult;
                if(kValue < 6)
                    nodeResult = codeBit;
                if(kValue == 6)
                    nodeResult = codeBit12;

                cout << " optCandidats " << optCandidats << ", nodeResult: " << nodeResult << endl;

                int node3=initialPrepareValue,node5 = initialPrepareValue, node7 =initialPrepareValue, node9 = initialPrepareValue, node11=initialPrepareValue;

                if(kValue == 2)
                {
                    node3 = optCandidats & nodeResult;

                    //execute 2-opt
                    PointCoord node3_(node3, 0);

                    cout << "Execute 2-opt mode= "<< optmode<< " tour order: " << this->grayValueMap[0][node1[0]] << ", " <<  this->grayValueMap[0][node3] << endl;

                    // node2
                    PointCoord node2(0, 0);
                    PointCoord node4(0, 0);

                    int changeLink1 = 0;
                    int changeLink2 = 0;
                    int changeLink3 = 0;
                    int changeLink4 = 0;

                    cout << "Execute 2-opt changelinks " << changeLink1 << ", " << changeLink2 << ", " << changeLink3 << ", " << changeLink4 << endl;
                    returnChangeLinks(node1, node2, changeLink1, changeLink2, networkLinksCP);
                    returnChangeLinks(node3_, node4, changeLink3, changeLink4, networkLinksCP);

                    this->networkLinks[0][node1[0]].bCell[changeLink1] = node3_;
                    this->networkLinks[0][node3].bCell[changeLink3] = node1;
                    this->networkLinks[0][node2[0]].bCell[changeLink2] = node4;
                    this->networkLinks[0][node4[0]].bCell[changeLink4] = node2;

                }
                if(kValue == 3)
                {
                    node5 = optCandidats & nodeResult;
                    optCandidats = optCandidats >> 16;
                    node3 = optCandidats & nodeResult;


                    //execute 3-opt
                    PointCoord node3_(node3, 0);
                    PointCoord node5_(node5, 0);

                    // node2
                    PointCoord node2(0, 0);
                    PointCoord node4(0, 0);
                    PointCoord node6(0, 0);

                    int changeLink1 = 0;
                    int changeLink2 = 0;
                    int changeLink3 = 0;
                    int changeLink4 = 0;
                    int changeLink5 = 0;
                    int changeLink6 = 0;

                    cout << "node3 " << node3 << ", node5 " << node5 << endl;

                    returnChangeLinks(node1, node2, changeLink1, changeLink2, networkLinksCP);
                    returnChangeLinks(node3_, node4, changeLink3, changeLink4, networkLinksCP);
                    returnChangeLinks(node5_, node6, changeLink5, changeLink6, networkLinksCP);

                    cout << "Execute 3-opt mode= "<< optmode<< " tour order: " << this->grayValueMap[0][node1[0]] << ", " <<  this->grayValueMap[0][node3]
                                                                                                                     << ", " <<  this->grayValueMap[0][node5] << endl;


                    cout << "Execute 2 3-opt changelinks " << changeLink1 << ", " << changeLink2 << ", " << changeLink3 << ", " << changeLink4 << ", " <<  changeLink5 << ", " << changeLink6 << endl;


                    if(optmode == 0)
                    {
                        this->networkLinks[0][node1[0]].bCell[changeLink1] = node4;
                        this->networkLinks[0][node2[0]].bCell[changeLink2] = node6;
                        this->networkLinks[0][node3].bCell[changeLink3] = node5_;
                        this->networkLinks[0][node4[0]].bCell[changeLink4] = node1;
                        this->networkLinks[0][node5].bCell[changeLink5] = node3_;
                        this->networkLinks[0][node6[0]].bCell[changeLink6] = node2;
                    }
                    if(optmode == 1)
                    {
                        this->networkLinks[0][node1[0]].bCell[changeLink1] = node5_;
                        this->networkLinks[0][node2[0]].bCell[changeLink2] = node4;
                        this->networkLinks[0][node3].bCell[changeLink3] = node6;
                        this->networkLinks[0][node4[0]].bCell[changeLink4] = node2;
                        this->networkLinks[0][node5].bCell[changeLink5] = node1;
                        this->networkLinks[0][node6[0]].bCell[changeLink6] = node3_;
                    }
                    if(optmode == 2)
                    {
                        this->networkLinks[0][node1[0]].bCell[changeLink1] = node3_;
                        this->networkLinks[0][node2[0]].bCell[changeLink2] = node5_;
                        this->networkLinks[0][node3].bCell[changeLink3] = node1;
                        this->networkLinks[0][node4[0]].bCell[changeLink4] = node6;
                        this->networkLinks[0][node5].bCell[changeLink5] = node2;
                        this->networkLinks[0][node6[0]].bCell[changeLink6] = node4;

                    }
                    if(optmode == 3)
                    {
                        this->networkLinks[0][node1[0]].bCell[changeLink1] = node4;
                        this->networkLinks[0][node2[0]].bCell[changeLink2] = node5_;
                        this->networkLinks[0][node3].bCell[changeLink3] = node6;
                        this->networkLinks[0][node4[0]].bCell[changeLink4] = node1;
                        this->networkLinks[0][node5].bCell[changeLink5] = node2;
                        this->networkLinks[0][node6[0]].bCell[changeLink6] = node3_;

                    }


                    cout << "Execute 2 3-opt changelinks " << changeLink1 << ", " << changeLink2 << ", " << changeLink3 << ", " << changeLink4 << ", " <<  changeLink5 << ", " << changeLink6 << endl;




                }
                if(kValue == 4)
                {

                    node7 = optCandidats & nodeResult;
                    optCandidats = optCandidats >> 16;
                    node5 = optCandidats & nodeResult;
                    optCandidats = optCandidats >> 16;
                    node3 = optCandidats & nodeResult;

                    //                    //execute 4-opt
                    //                    cout << "Execute 4-opt mark node1-7: " << node1[0] << ", " << node3 << ", " << node5 << ", " << node7 << ", " << endl;
                    cout << "Execute 4-opt mode= "<< optmode<< " tour order: " << this->grayValueMap[0][node1[0]] << ", " <<  this->grayValueMap[0][node3]
                                                                                                                     << ", " <<  this->grayValueMap[0][node5] << ", "  << this->grayValueMap[0][node7] << endl;
                    //execute 4-opt
                    PointCoord node3_(node3, 0);
                    PointCoord node5_(node5, 0);
                    PointCoord node7_(node7, 0);

                    // node2
                    PointCoord node2(0, 0);
                    PointCoord node4(0, 0);
                    PointCoord node6(0, 0);
                    PointCoord node8(0, 0);


                    int changeLink1 = 0;
                    int changeLink2 = 0;
                    int changeLink3 = 0;
                    int changeLink4 = 0;
                    int changeLink5 = 0;
                    int changeLink6 = 0;
                    int changeLink7 = 0;
                    int changeLink8 = 0;

                    returnChangeLinks(node1, node2, changeLink1, changeLink2, networkLinksCP);
                    returnChangeLinks(node3_, node4, changeLink3, changeLink4, networkLinksCP);
                    returnChangeLinks(node5_, node6, changeLink5, changeLink6, networkLinksCP);
                    returnChangeLinks(node7_, node8, changeLink7, changeLink8, networkLinksCP);

                    PointCoord array[8];
                    array[0] = node1;
                    array[1] = node2;
                    array[2] = node3_;
                    array[3] = node4;
                    array[4] = node5_;
                    array[5] = node6;
                    array[6] = node7_;
                    array[7] = node8;
                    int arrayChangLinks[8]={-1,-1,-1,-1,-1,-1,-1,-1};
                    arrayChangLinks[0] = changeLink1;
                    arrayChangLinks[1] = changeLink2;
                    arrayChangLinks[2] = changeLink3;
                    arrayChangLinks[3] = changeLink4;
                    arrayChangLinks[4] = changeLink5;
                    arrayChangLinks[5] = changeLink6;
                    arrayChangLinks[6] = changeLink7;
                    arrayChangLinks[7] = changeLink8;


                    int nd1, nd2, nd3, nd4, nd5,nd6,nd7,nd8;
                    int optSelected = optmode*8;

                    //1 7 2 8 3 5 4 6
                    nd1 = fourOptPossibilitiesMap[0][optSelected] -1; // 0
                    nd2 = fourOptPossibilitiesMap[0][optSelected+1] -1; // 6
                    nd3 = fourOptPossibilitiesMap[0][optSelected+2] -1;
                    nd4 = fourOptPossibilitiesMap[0][optSelected+3] -1;
                    nd5 = fourOptPossibilitiesMap[0][optSelected+4] -1;
                    nd6 = fourOptPossibilitiesMap[0][optSelected+5] -1;
                    nd7 = fourOptPossibilitiesMap[0][optSelected+6] -1;
                    nd8 = fourOptPossibilitiesMap[0][optSelected+7] -1;

                    cout << "Execute 4-opt nd1-8 = " << nd1 << " " << nd2 << " "<< nd3<< " "
                         << nd4 <<" " << nd5 << " " << nd6<< " " << nd7 << " " << nd8 << " " << endl;

                    cout << "Execute 4-opt array[nd1][0]= "  << array[nd1][0] << ", array[nd2][0]= " << array[nd2][0]  << ", array[nd3][0]= " << array[nd3][0]  << ", array[nd4][0]= " << array[nd4][0]<< endl;

                    //error here, changelinks is also changing accordng to ndx
                    this->networkLinks[0][array[nd1][0]].bCell[arrayChangLinks[nd1]] = array[nd2];
                    this->networkLinks[0][array[nd2][0]].bCell[arrayChangLinks[nd2]] = array[nd1];
                    this->networkLinks[0][array[nd3][0]].bCell[arrayChangLinks[nd3]] = array[nd4];
                    this->networkLinks[0][array[nd4][0]].bCell[arrayChangLinks[nd4]] = array[nd3];
                    this->networkLinks[0][array[nd5][0]].bCell[arrayChangLinks[nd5]] = array[nd6];
                    this->networkLinks[0][array[nd6][0]].bCell[arrayChangLinks[nd6]] = array[nd5];
                    this->networkLinks[0][array[nd7][0]].bCell[arrayChangLinks[nd7]] = array[nd8];
                    this->networkLinks[0][array[nd8][0]].bCell[arrayChangLinks[nd8]] = array[nd7];

                }
                if(kValue ==5)
                {
                    node9 = optCandidats & nodeResult;
                    optCandidats = optCandidats >> 16;
                    node7 = optCandidats & nodeResult;
                    optCandidats = optCandidats >> 16;
                    node5 = optCandidats & nodeResult;
                    optCandidats = optCandidats >> 16;
                    node3 = optCandidats & nodeResult;

                    //execute 5-opt
                    PointCoord node3_(node3, 0);
                    PointCoord node5_(node5, 0);
                    PointCoord node7_(node7, 0);
                    PointCoord node9_(node9, 0);


                    cout << "Execute 5-opt mode= "<< optmode<< " tour order: " << this->grayValueMap[0][node1[0]] << ", " <<  this->grayValueMap[0][node3]
                                                                                                                     << ", " <<  this->grayValueMap[0][node5] << ", "  << this->grayValueMap[0][node7]  << ", "  << this->grayValueMap[0][node9]<< endl;

                    // node2
                    PointCoord node2(0, 0);
                    PointCoord node4(0, 0);
                    PointCoord node6(0, 0);
                    PointCoord node8(0, 0);
                    PointCoord node10(0, 0);

                    int changeLink1 = 0;
                    int changeLink2 = 0;
                    int changeLink3 = 0;
                    int changeLink4 = 0;
                    int changeLink5 = 0;
                    int changeLink6 = 0;
                    int changeLink7 = 0;
                    int changeLink8 = 0;
                    int changeLink9 = 0;
                    int changeLink10 = 0;

                    returnChangeLinks(node1, node2, changeLink1, changeLink2, networkLinksCP);
                    returnChangeLinks(node3_, node4, changeLink3, changeLink4, networkLinksCP);
                    returnChangeLinks(node5_, node6, changeLink5, changeLink6, networkLinksCP);
                    returnChangeLinks(node7_, node8, changeLink7, changeLink8, networkLinksCP);
                    returnChangeLinks(node9_, node10, changeLink9, changeLink10, networkLinksCP);


                    PointCoord array[10];
                    array[0] = node1;
                    array[1] = node2;
                    array[2] = node3_;
                    array[3] = node4;
                    array[4] = node5_;
                    array[5] = node6;
                    array[6] = node7_;
                    array[7] = node8;
                    array[8] = node9_;
                    array[9] = node10;
                    int arrayChangLinks[10]={-1,-1,-1,-1,-1,-1,-1,-1,-1,-1};
                    arrayChangLinks[0] = changeLink1;
                    arrayChangLinks[1] = changeLink2;
                    arrayChangLinks[2] = changeLink3;
                    arrayChangLinks[3] = changeLink4;
                    arrayChangLinks[4] = changeLink5;
                    arrayChangLinks[5] = changeLink6;
                    arrayChangLinks[6] = changeLink7;
                    arrayChangLinks[7] = changeLink8;
                    arrayChangLinks[8] = changeLink9;
                    arrayChangLinks[9] = changeLink10;

                    int nd1, nd2, nd3, nd4, nd5,nd6,nd7,nd8,nd9,nd10;
                    int optSelected = optmode*10;

                    //1 7 2 8 3 5 4 6
                    nd1 = fiveOptPossibilitiesMap[0][optSelected] -1; // 0
                    nd2 = fiveOptPossibilitiesMap[0][optSelected+1] -1; // 6
                    nd3 = fiveOptPossibilitiesMap[0][optSelected+2] -1;
                    nd4 = fiveOptPossibilitiesMap[0][optSelected+3] -1;
                    nd5 = fiveOptPossibilitiesMap[0][optSelected+4] -1;
                    nd6 = fiveOptPossibilitiesMap[0][optSelected+5] -1;
                    nd7 = fiveOptPossibilitiesMap[0][optSelected+6] -1;
                    nd8 = fiveOptPossibilitiesMap[0][optSelected+7] -1;
                    nd9 = fiveOptPossibilitiesMap[0][optSelected+8] -1;
                    nd10 = fiveOptPossibilitiesMap[0][optSelected+9] -1;


                    //error here, changelinks is also changing accordng to ndx
                    this->networkLinks[0][array[nd1][0]].bCell[arrayChangLinks[nd1]] = array[nd2];
                    this->networkLinks[0][array[nd2][0]].bCell[arrayChangLinks[nd2]] = array[nd1];
                    this->networkLinks[0][array[nd3][0]].bCell[arrayChangLinks[nd3]] = array[nd4];
                    this->networkLinks[0][array[nd4][0]].bCell[arrayChangLinks[nd4]] = array[nd3];
                    this->networkLinks[0][array[nd5][0]].bCell[arrayChangLinks[nd5]] = array[nd6];
                    this->networkLinks[0][array[nd6][0]].bCell[arrayChangLinks[nd6]] = array[nd5];
                    this->networkLinks[0][array[nd7][0]].bCell[arrayChangLinks[nd7]] = array[nd8];
                    this->networkLinks[0][array[nd8][0]].bCell[arrayChangLinks[nd8]] = array[nd7];
                    this->networkLinks[0][array[nd9][0]].bCell[arrayChangLinks[nd9]] = array[nd10];
                    this->networkLinks[0][array[nd10][0]].bCell[arrayChangLinks[nd10]] = array[nd9];

                }
                if(kValue ==6)
                {

                    node11 = optCandidats & nodeResult;
                    optCandidats = optCandidats >> 12; //need to check only for 6-opt
                    node9 = optCandidats & nodeResult;
                    optCandidats = optCandidats >> 12;
                    node7 = optCandidats & nodeResult;
                    optCandidats = optCandidats >> 12;
                    node5 = optCandidats & nodeResult;
                    optCandidats = optCandidats >> 12;
                    node3 = optCandidats & nodeResult;

                    //                    //execute 6-opt
                    //                    cout << "Execute 6-opt mark node1-11: " << node1[0] << ", " << node3 << ", " << node5 << ", " << node7 << ", " << node9 << ", " <<  node11 << ", " <<endl;
                    cout << "Execute 6-opt mode= "<< optmode<< " tour order: " << this->grayValueMap[0][node1[0]] << " " <<  this->grayValueMap[0][node3]
                                                                                                                     << " " << this->grayValueMap[0][node5] << " " << this->grayValueMap[0][node7]
                                                                                                                        << " " << this->grayValueMap[0][node9] << " " << this->grayValueMap[0][node11] << endl;

                    //execute 5-opt
                    PointCoord node3_(node3, 0);
                    PointCoord node5_(node5, 0);
                    PointCoord node7_(node7, 0);
                    PointCoord node9_(node9, 0);
                    PointCoord node11_(node11, 0);

                    // node2
                    PointCoord node2(0, 0);
                    PointCoord node4(0, 0);
                    PointCoord node6(0, 0);
                    PointCoord node8(0, 0);
                    PointCoord node10(0, 0);
                    PointCoord node12(0, 0);

                    int changeLink1 = 0;
                    int changeLink2 = 0;
                    int changeLink3 = 0;
                    int changeLink4 = 0;
                    int changeLink5 = 0;
                    int changeLink6 = 0;
                    int changeLink7 = 0;
                    int changeLink8 = 0;
                    int changeLink9 = 0;
                    int changeLink10 = 0;
                    int changeLink11 = 0;
                    int changeLink12 = 0;

                    returnChangeLinks(node1, node2, changeLink1, changeLink2, networkLinksCP);
                    returnChangeLinks(node3_, node4, changeLink3, changeLink4, networkLinksCP);
                    returnChangeLinks(node5_, node6, changeLink5, changeLink6, networkLinksCP);
                    returnChangeLinks(node7_, node8, changeLink7, changeLink8, networkLinksCP);
                    returnChangeLinks(node9_, node10, changeLink9, changeLink10, networkLinksCP);
                    returnChangeLinks(node11_, node12, changeLink11, changeLink12, networkLinksCP);


                    PointCoord array[12];
                    array[0] = node1;
                    array[1] = node2;
                    array[2] = node3_;
                    array[3] = node4;
                    array[4] = node5_;
                    array[5] = node6;
                    array[6] = node7_;
                    array[7] = node8;
                    array[8] = node9_;
                    array[9] = node10;
                    array[10] = node11_;
                    array[11] = node12;

                    int arrayChangLinks[12]={-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1};
                    arrayChangLinks[0] = changeLink1;
                    arrayChangLinks[1] = changeLink2;
                    arrayChangLinks[2] = changeLink3;
                    arrayChangLinks[3] = changeLink4;
                    arrayChangLinks[4] = changeLink5;
                    arrayChangLinks[5] = changeLink6;
                    arrayChangLinks[6] = changeLink7;
                    arrayChangLinks[7] = changeLink8;
                    arrayChangLinks[8] = changeLink9;
                    arrayChangLinks[9] = changeLink10;
                    arrayChangLinks[10] = changeLink11;
                    arrayChangLinks[11] = changeLink12;


                    int nd1, nd2, nd3, nd4, nd5,nd6,nd7,nd8,nd9,nd10,nd11,nd12;
                    int optSelected = optmode*12;

                    //1 7 2 8 3 5 4 6
                    nd1 = sixOptPossibilitiesMap[0][optSelected] -1; // 0
                    nd2 = sixOptPossibilitiesMap[0][optSelected+1] -1; // 6
                    nd3 = sixOptPossibilitiesMap[0][optSelected+2] -1;
                    nd4 = sixOptPossibilitiesMap[0][optSelected+3] -1;
                    nd5 = sixOptPossibilitiesMap[0][optSelected+4] -1;
                    nd6 = sixOptPossibilitiesMap[0][optSelected+5] -1;
                    nd7 = sixOptPossibilitiesMap[0][optSelected+6] -1;
                    nd8 = sixOptPossibilitiesMap[0][optSelected+7] -1;
                    nd9 = sixOptPossibilitiesMap[0][optSelected+8] -1;
                    nd10 = sixOptPossibilitiesMap[0][optSelected+9] -1;
                    nd11 = sixOptPossibilitiesMap[0][optSelected+10] -1;
                    nd12 = sixOptPossibilitiesMap[0][optSelected+11] -1;

                    //error here, changelinks is also changing accordng to ndx
                    this->networkLinks[0][array[nd1][0]].bCell[arrayChangLinks[nd1]] = array[nd2];
                    this->networkLinks[0][array[nd2][0]].bCell[arrayChangLinks[nd2]] = array[nd1];
                    this->networkLinks[0][array[nd3][0]].bCell[arrayChangLinks[nd3]] = array[nd4];
                    this->networkLinks[0][array[nd4][0]].bCell[arrayChangLinks[nd4]] = array[nd3];
                    this->networkLinks[0][array[nd5][0]].bCell[arrayChangLinks[nd5]] = array[nd6];
                    this->networkLinks[0][array[nd6][0]].bCell[arrayChangLinks[nd6]] = array[nd5];
                    this->networkLinks[0][array[nd7][0]].bCell[arrayChangLinks[nd7]] = array[nd8];
                    this->networkLinks[0][array[nd8][0]].bCell[arrayChangLinks[nd8]] = array[nd7];
                    this->networkLinks[0][array[nd9][0]].bCell[arrayChangLinks[nd9]] = array[nd10];
                    this->networkLinks[0][array[nd10][0]].bCell[arrayChangLinks[nd10]] = array[nd9];
                    this->networkLinks[0][array[nd11][0]].bCell[arrayChangLinks[nd11]] = array[nd12];
                    this->networkLinks[0][array[nd12][0]].bCell[arrayChangLinks[nd12]] = array[nd11];



                }

                numOptimized ++;


                //                // test
                //                float evaCurrentAfter = this->evaluateWeightOfTSP<CM_DistanceSquaredEuclidean>();
                //                if(evaCurrentAfter < evaCurrentBefore)
                //                {
                //                    // test
                //                    int idNode1 = this->grayValueMap[0][_x];
                //                    int idNode3 = this->grayValueMap[0][node3_int];
                //                    cout << " correct optimizing " << idNode1  <<  ", id son1 " << this->grayValueMap[0][node2[0]]
                //                            << ". idCorrs " << idNode3 << ", id son3 "  << this->grayValueMap[0][node4[0]]  << endl;

                //                }
                //                else if (evaCurrentAfter == evaCurrentBefore)
                //                {
                //                    int idNode1 = this->grayValueMap[0][_x];
                //                    int idNode3 = this->grayValueMap[0][node3_int];
                //                    cout << " equal optimizing " << idNode1  <<  ", id son1 " << this->grayValueMap[0][node2[0]]
                //                            << ". idCorrs " << idNode3 << ", id son3 "  << this->grayValueMap[0][node4[0]]  << endl;
                //                }
                //                else
                //                {
                //                    int idNode1 = this->grayValueMap[0][_x];
                //                    int idNode3 = this->grayValueMap[0][node3_int];
                //                    cout << " warning optimizing " << idNode1  <<  ", id son1 " << this->grayValueMap[0][node2[0]]
                //                            << ". idCorrs " << idNode3 << ", id son3 "  << this->grayValueMap[0][node4[0]]  << endl;
                //                }

            }
        }

        cout  << "numOptimized = " << numOptimized << endl;

        return true;
    }//end execute


    float dist(int i, int j, doubleLinkedEdgeForTSP* coords){
        float dx, dy;
        dx = coords[i].currentCoord[0] - coords[j].currentCoord[0];
        dy = coords[i].currentCoord[1] - coords[j].currentCoord[1];
        return (dx*dx + dy*dy);
        //     double dist = (dx*dx + dy*dy);
        //     return sqrt(dist);
    }

    //qiao add sequential 2-opt, using 2 loops of linkCoordTourCpu
    void sequential2optBest(Grid<doubleLinkedEdgeForTSP>& linkCoordTourCpu)
    {

        cout << "Enter sequential 2-opt " << endl;
        int N =  this->adaptiveMap.width;
        for(int j = 1; j < N; j++)
        {

            float oldLength0 =  dist(j-1, j, linkCoordTourCpu[0]);

            //            cout << "oldLen0 = " << oldLength0 << endl;

            for(int i = j+1; i < N; i++)
            {
                float oldLength = oldLength0;
                oldLength += dist(i-1, i, linkCoordTourCpu[0]);

                float newLength = dist(j-1, i-1, linkCoordTourCpu[0]) + dist(j, i, linkCoordTourCpu[0]);

                //                cout << "newLength = " << newLength << endl;


                if(newLength < oldLength)
                {
                    float optimization = oldLength - newLength;

                    int node1 = (int)linkCoordTourCpu[0][j-1].current;
                    int node3 = (int)linkCoordTourCpu[0][i-1].current;

                    float localMinChange = this->minRadiusMap[0][node1];

                    if(optimization > localMinChange)
                    {
                        this->densityMap[0][node1]= node3; // WB.Q this way can work for multi-thread operation
                        this->minRadiusMap[0][node1]= optimization;
                    }

                }

            }

        }

    }//qiao end sequential 2-opt with best opt for one node


    //qiao add sequential 2-opt, using 2 loops of linkCoordTourCpu
    void sequential2optFirst(Grid<doubleLinkedEdgeForTSP>& linkCoordTourCpu)
    {

        cout << "Enter sequential 2-opt FIRST " << endl;
        int N =  this->adaptiveMap.width;
        for(int j = 1; j < N; j++)
        {

            float oldLength0 =  dist(j-1, j, linkCoordTourCpu[0]);

            //            cout << "oldLen0 = " << oldLength0 << endl;

            for(int i = j+1; i < N; i++)
            {
                float oldLength = oldLength0;
                oldLength += dist(i-1, i, linkCoordTourCpu[0]);

                float newLength = dist(j-1, i-1, linkCoordTourCpu[0]) + dist(j, i, linkCoordTourCpu[0]);

                //                cout << "newLength = " << newLength << endl;


                if(newLength < oldLength)
                {
                    float optimization = oldLength - newLength;

                    int node1 = (int)linkCoordTourCpu[0][j-1].current;
                    int node3 = (int)linkCoordTourCpu[0][i-1].current;

                    //                    float localMinChange = this->minRadiusMap[0][node1];

                    if(optimization > 0)
                    {
                        this->densityMap[0][node1]= node3; // WB.Q this way can work for multi-thread operation
                        //                        this->minRadiusMap[0][node1]= optimization;
                        break;
                    }

                }

            }

        }

    }//qiao end sequential 2-opt with best opt for one node




    //qiao add sequential 4-opt, using 4 loops of linkCoordTourCpu
    void sequential4optFirst(Grid<doubleLinkedEdgeForTSP>& linkCoordTourCpu,  Grid<GLint> optPossibilitiesMap)
    {

        //        cout << "Enter sequential 4-opt " << endl;
        int N =  this->adaptiveMap.width;
        for(int w = 1; w < N; w++)
        {

            float oldLength0 = dist(w-1, w, linkCoordTourCpu[0]);

            //           cout << "oldLen0 = " << oldLength0 << endl;
            //                       cout << " ith  = " << w << endl;

            bool foundOpt = 0;

            for(int j = w+1; j < N; j++)
            {
                float oldLength1 = oldLength0;
                oldLength1 += dist(j-1, j, linkCoordTourCpu[0]);

                //                cout << "    jth  = " << j << endl;

                for(int p = j+1; p <N; p++)
                {
                    float oldLength2 = oldLength1;
                    oldLength2 += dist(p-1, p, linkCoordTourCpu[0]);

                    //                    cout << "oldLength2 = " << oldLength2 << endl;

                    //                    cout << "          pth  = " << p << endl;

                    for(int k = p+1; k<N; k++)
                    {

                        //                        cout << k << endl;

                        float oldLength = oldLength2;
                        oldLength += dist(k-1, k, linkCoordTourCpu[0]);

                        //                                                cout << "oldLength = " << oldLength << endl;

                        float newLength;
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
                        //                        float optimiz = -INFINITY;

                        for(int opt = 0; opt < 200; opt +=8) //  4 edges 8 nodes
                        {
                            int nd1 = optPossibilitiesMap[0][opt] -1;
                            int nd2 = optPossibilitiesMap[0][opt+1] -1;
                            int nd3 = optPossibilitiesMap[0][opt+2] -1;
                            int nd4 = optPossibilitiesMap[0][opt+3] -1;
                            int nd5 = optPossibilitiesMap[0][opt+4] -1;
                            int nd6 = optPossibilitiesMap[0][opt+5] -1;
                            int nd7 = optPossibilitiesMap[0][opt+6] -1;
                            int nd8 = optPossibilitiesMap[0][opt+7] -1;

                            //                            if(nd8 > 11 || nd8 <0)
                            //                            cout << "nd8 = " << nd8<< endl;

                            int optCandi = opt / 8;
                            newLength= dist(array[nd1],array[nd2], linkCoordTourCpu[0]) + dist(array[nd3],array[nd4], linkCoordTourCpu[0]) + dist(array[nd5],array[nd6], linkCoordTourCpu[0])+ dist(array[nd7],array[nd8], linkCoordTourCpu[0]);

                            float opti = oldLength - newLength;
                            //                            if(opti > 0 && opti > optimiz)
                            if(opti > 0)
                            {
                                finalSelect = optCandi;
                                //                                optimiz = opti;
                                //                                cout << "selected " << opti << endl;
                                break;

                            }
                        }

                        if(finalSelect >= 0)
                        {

                            unsigned int node1 = (int)linkCoordTourCpu[0][w-1].current;
                            unsigned int node3 = (int)linkCoordTourCpu[0][j-1].current;
                            unsigned int node5 = (int)linkCoordTourCpu[0][p-1].current;
                            unsigned int node7 = (int)linkCoordTourCpu[0][k-1].current;

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

                                this->optCandidateMap[0][node1] = result; // WB.Q 2024 find a solution to judge non-interacted 23456-opt
                                this->densityMap[0][node1] = codekopt;
                            }

                            foundOpt = 1;

                            break;
                        }

                    }
                    if(foundOpt)
                        break;

                }

                if(foundOpt)
                    break;

            }

            if(foundOpt)
                continue;
        }

    }//qiao end sequential 4-opt with first opt for one node



    //qiao add sequential 4-opt, using 4 loops of linkCoordTourCpu
    void sequential4optBest(Grid<doubleLinkedEdgeForTSP>& linkCoordTourCpu,  Grid<GLint> optPossibilitiesMap)
    {

        cout << "Enter sequential 4-opt " << endl;
        int N =  this->adaptiveMap.width;
        for(int w = 1; w < N; w++)
        {

            float oldLength0 = dist(w-1, w, linkCoordTourCpu[0]);

            //           cout << "oldLen0 = " << oldLength0 << endl;
            //                       cout << " ith  = " << w << endl;

            for(int j = w+1; j < N; j++)
            {
                float oldLength1 = oldLength0;
                oldLength1 += dist(j-1, j, linkCoordTourCpu[0]);

                //                cout << "    jth  = " << j << endl;

                for(int p = j+1; p <N; p++)
                {
                    float oldLength2 = oldLength1;
                    oldLength2 += dist(p-1, p, linkCoordTourCpu[0]);

                    for(int k = p+1; k<N; k++)
                    {

                        float oldLength = oldLength2;
                        oldLength += dist(k-1, k, linkCoordTourCpu[0]);

                        float newLength;
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

                        unsigned int node1 = (int)linkCoordTourCpu[0][w-1].current;
                        float localMinChange = this->minRadiusMap[0][node1];

                        for(int opt = 0; opt < 200; opt +=8) //  4 edges 8 nodes
                        {
                            int nd1 = optPossibilitiesMap[0][opt] -1;
                            int nd2 = optPossibilitiesMap[0][opt+1] -1;
                            int nd3 = optPossibilitiesMap[0][opt+2] -1;
                            int nd4 = optPossibilitiesMap[0][opt+3] -1;
                            int nd5 = optPossibilitiesMap[0][opt+4] -1;
                            int nd6 = optPossibilitiesMap[0][opt+5] -1;
                            int nd7 = optPossibilitiesMap[0][opt+6] -1;
                            int nd8 = optPossibilitiesMap[0][opt+7] -1;

                            //                            if(nd8 > 11 || nd8 <0)
                            //                            cout << "nd8 = " << nd8<< endl;

                            int optCandi = opt / 8;
                            newLength= dist(array[nd1],array[nd2], linkCoordTourCpu[0]) + dist(array[nd3],array[nd4], linkCoordTourCpu[0]) + dist(array[nd5],array[nd6], linkCoordTourCpu[0])+ dist(array[nd7],array[nd8], linkCoordTourCpu[0]);

                            float opti = oldLength - newLength;
                            if(opti > 0 && opti > optimiz)
                            {
                                finalSelect = optCandi;
                                optimiz = opti;

                            }
                        }

                        if(optimiz > localMinChange)
                        {
                            unsigned int node3 = (int)linkCoordTourCpu[0][j-1].current;
                            unsigned int node5 = (int)linkCoordTourCpu[0][p-1].current;
                            unsigned int node7 = (int)linkCoordTourCpu[0][k-1].current;

                            unsigned long long result = 0;
                            result = result | node3;
                            result = result << 16;
                            result = result | node5;
                            result = result << 16;
                            result = result | node7;

                            float codekopt = finalSelect; //WB.Q 2024 densityMap only mark the k value of k-opt, and mark which mode of k-exchange
                            codekopt = finalSelect * 100 + 4;

                            this->optCandidateMap[0][node1] = result; // WB.Q 2024 find a solution to judge non-interacted 23456-opt
                            this->densityMap[0][node1] = codekopt;
                            this->minRadiusMap[0][node1]= optimiz;

                        }

                    }
                }
            }

        }

    }//qiao end sequential 4-opt with first opt for one node



    //qiao add sequential 5-opt, using 5loops of linkCoordTourCpu
    void sequential5optFirst(Grid<doubleLinkedEdgeForTSP>& linkCoordTourCpu,  Grid<GLint> optPossibilitiesMap)
    {

        //        cout << "Enter sequential 5-opt " << endl;
        int N =  this->adaptiveMap.width;
        for(int w = 1; w < N; w++)
        {

            float oldLength0 = dist(w-1, w, linkCoordTourCpu[0]);

            //             cout << "oldLen0 = " << oldLength0 << endl;

            bool foundOpt = 0;

            for(int j = w+1; j < N; j++)
            {
                float oldLength1 = oldLength0;
                oldLength1 += dist(j-1, j, linkCoordTourCpu[0]);

                for(int p = j+1; p <N; p++)
                {
                    float oldLength2 = oldLength1;
                    oldLength2 += dist(p-1, p, linkCoordTourCpu[0]);

                    for(int k = p+1; k<N; k++)
                    {

                        float oldLength3 = oldLength2;
                        oldLength3 += dist(k-1, k, linkCoordTourCpu[0]);


                        for(int idRow5th = k+1; idRow5th < N; idRow5th++)
                        {
                            float oldLength = oldLength3;
                            oldLength += dist(idRow5th-1, idRow5th, linkCoordTourCpu[0]);


                            //                                                    cout << "oldLength = " << oldLength << endl;

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
                            //                        float optimiz = -INFINITY;

                            for(int opt = 0; opt < 2080; opt +=10) //  4 edges 8 nodes
                            {
                                int nd1 = optPossibilitiesMap[0][opt] -1;
                                int nd2 = optPossibilitiesMap[0][opt+1] -1;
                                int nd3 = optPossibilitiesMap[0][opt+2] -1;
                                int nd4 = optPossibilitiesMap[0][opt+3] -1;
                                int nd5 = optPossibilitiesMap[0][opt+4] -1;
                                int nd6 = optPossibilitiesMap[0][opt+5] -1;
                                int nd7 = optPossibilitiesMap[0][opt+6] -1;
                                int nd8 = optPossibilitiesMap[0][opt+7] -1;
                                int nd9 = optPossibilitiesMap[0][opt+8] -1;
                                int nd10 = optPossibilitiesMap[0][opt+9] -1;


                                int optCandi = opt / 10;
                                // printf("GPU search nd1-8 %d, %d, %d, %d, %d, %d, %d, %d; optCandi=%d \n", nd1, nd2, nd3, nd4, nd5, nd6, nd7, nd8, optCandi);
                                newLength = dist(array[nd1],array[nd2], linkCoordTourCpu[0]) + dist(array[nd3],array[nd4], linkCoordTourCpu[0])
                                        + dist(array[nd5],array[nd6], linkCoordTourCpu[0])+ dist(array[nd7],array[nd8], linkCoordTourCpu[0]) + dist(array[nd9],array[nd10], linkCoordTourCpu[0]);

                                //                                cout << "newLength = " << newLength << endl; // newlength  == infinity
                                float opti = oldLength - newLength;
                                //                                if(opti > 0 && opti > optimiz)
                                if(opti > 0)
                                {
                                    finalSelect = optCandi;
                                    break;
                                }
                            }

                            if(finalSelect >= 0)
                            {

                                unsigned int node1 = (int)linkCoordTourCpu[0][w-1].current;
                                unsigned int node3 = (int)linkCoordTourCpu[0][j-1].current;
                                unsigned int node5 = (int)linkCoordTourCpu[0][p-1].current;
                                unsigned int node7 = (int)linkCoordTourCpu[0][k-1].current;
                                unsigned int node9 = (int)linkCoordTourCpu[0][idRow5th-1].current;

                                unsigned long long result = 0;
                                result = result | node3;
                                result = result << 16;
                                result = result | node5;
                                result = result << 16;
                                result = result | node7;
                                result = result << 16;
                                result = result | node9;

                                if(node9 > N || node9 < 0)
                                    cout << "Error searching decode opt node9 > widht " << endl;

                                float codekopt = finalSelect; //WB.Q 2024 densityMap only mark the k value of k-opt, and mark which mode of k-exchange
                                codekopt = finalSelect * 100 + 5;

                                this->optCandidateMap[0][node1] = result; // WB.Q 2024 find a solution to judge non-interacted 23456-opt
                                this->densityMap[0][node1] = codekopt;

                                foundOpt = 1;
                                break;
                            }
                        }

                        if(foundOpt)
                            break;
                    }
                    if(foundOpt)
                        break;

                }

                if(foundOpt)
                    break;

            }

            if(foundOpt)
                continue;
        }

    }//qiao end sequential 5-opt with first opt for one node



    //qiao add sequential 5-opt, using 5loops of linkCoordTourCpu
    void sequential5optBest(Grid<doubleLinkedEdgeForTSP>& linkCoordTourCpu,  Grid<GLint> optPossibilitiesMap)
    {

        cout << "Enter sequential 5-opt " << endl;
        int N =  this->adaptiveMap.width;
        for(int w = 1; w < N; w++)
        {

            float oldLength0 = dist(w-1, w, linkCoordTourCpu[0]);

            //             cout << "oldLen0 = " << oldLength0 << endl;

            for(int j = w+1; j < N; j++)
            {
                float oldLength1 = oldLength0;
                oldLength1 += dist(j-1, j, linkCoordTourCpu[0]);

                for(int p = j+1; p <N; p++)
                {
                    float oldLength2 = oldLength1;
                    oldLength2 += dist(p-1, p, linkCoordTourCpu[0]);

                    for(int k = p+1; k<N; k++)
                    {

                        float oldLength3 = oldLength2;
                        oldLength3 += dist(k-1, k, linkCoordTourCpu[0]);


                        for(int idRow5th = k+1; idRow5th < N; idRow5th++)
                        {
                            float oldLength = oldLength3;
                            oldLength += dist(idRow5th-1, idRow5th, linkCoordTourCpu[0]);

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


                            unsigned int node1 = (int)linkCoordTourCpu[0][w-1].current;
                            float localMinChange = this->minRadiusMap[0][node1];


                            int finalSelect = -1;
                            float optimiz = -INFINITY;

                            for(int opt = 0; opt < 2080; opt +=10) //  4 edges 8 nodes
                            {
                                int nd1 = optPossibilitiesMap[0][opt] -1;
                                int nd2 = optPossibilitiesMap[0][opt+1] -1;
                                int nd3 = optPossibilitiesMap[0][opt+2] -1;
                                int nd4 = optPossibilitiesMap[0][opt+3] -1;
                                int nd5 = optPossibilitiesMap[0][opt+4] -1;
                                int nd6 = optPossibilitiesMap[0][opt+5] -1;
                                int nd7 = optPossibilitiesMap[0][opt+6] -1;
                                int nd8 = optPossibilitiesMap[0][opt+7] -1;
                                int nd9 = optPossibilitiesMap[0][opt+8] -1;
                                int nd10 = optPossibilitiesMap[0][opt+9] -1;


                                int optCandi = opt / 10;
                                // printf("GPU search nd1-8 %d, %d, %d, %d, %d, %d, %d, %d; optCandi=%d \n", nd1, nd2, nd3, nd4, nd5, nd6, nd7, nd8, optCandi);
                                newLength = dist(array[nd1],array[nd2], linkCoordTourCpu[0]) + dist(array[nd3],array[nd4], linkCoordTourCpu[0])
                                        + dist(array[nd5],array[nd6], linkCoordTourCpu[0])+ dist(array[nd7],array[nd8], linkCoordTourCpu[0]) + dist(array[nd9],array[nd10], linkCoordTourCpu[0]);

                                //                                cout << "newLength = " << newLength << endl; // newlength  == infinity
                                float opti = oldLength - newLength;
                                //                                if(opti > 0 && opti > optimiz)
                                if(opti > 0 && opti > optimiz)
                                {
                                    finalSelect = optCandi;
                                    optimiz = opti;

                                }
                            }

                            if(optimiz > localMinChange)
                            {

                                unsigned int node3 = (int)linkCoordTourCpu[0][j-1].current;
                                unsigned int node5 = (int)linkCoordTourCpu[0][p-1].current;
                                unsigned int node7 = (int)linkCoordTourCpu[0][k-1].current;
                                unsigned int node9 = (int)linkCoordTourCpu[0][idRow5th-1].current;

                                unsigned long long result = 0;
                                result = result | node3;
                                result = result << 16;
                                result = result | node5;
                                result = result << 16;
                                result = result | node7;
                                result = result << 16;
                                result = result | node9;

                                if(node9 > N || node9 < 0)
                                    cout << "Error searching decode opt node9 > widht " << endl;

                                float codekopt = finalSelect; //WB.Q 2024 densityMap only mark the k value of k-opt, and mark which mode of k-exchange
                                codekopt = finalSelect * 100 + 5;

                                this->optCandidateMap[0][node1] = result; // WB.Q 2024 find a solution to judge non-interacted 23456-opt
                                this->densityMap[0][node1] = codekopt;
                                this->minRadiusMap[0][node1]= optimiz;

                            }
                        }

                    }
                }

            }

        }

    }//qiao end sequential 5-opt with first opt for one node




    //qiao add sequential 3-opt, using 5loops of linkCoordTourCpu
    void sequential3optFirst(Grid<doubleLinkedEdgeForTSP>& linkCoordTourCpu)
    {

        cout << "Enter sequential 3-opt " << endl;
        int N =  this->adaptiveMap.width;
        for(int j = 1; j < N; j++)
        {

            float oldLength0 =  dist(j-1, j, linkCoordTourCpu[0]);

            //            cout << "oldLen0 = " << oldLength0 << endl;
            bool foundOpt = 0;

            for(int i = j+1; i < N; i++)
            {
                float oldLength1 = oldLength0;
                oldLength1 += dist(i-1, i, linkCoordTourCpu[0]);

                for(int row = i +1; row <N; row ++)
                {

                    float oldLength = oldLength1;
                    oldLength += dist(row-1, row, linkCoordTourCpu[0]);

                    double newLength[4];
                    newLength[0] = dist(j-1, i, linkCoordTourCpu[0]) + dist(row-1, i-1, linkCoordTourCpu[0]) + dist(row, j, linkCoordTourCpu[0]);
                    newLength[1] = dist(j-1, row-1,linkCoordTourCpu[0]) + dist(row, i-1, linkCoordTourCpu[0]) + dist(i,j,linkCoordTourCpu[0]);
                    newLength[2] = dist(j-1, i-1, linkCoordTourCpu[0]) + dist(row-1, j, linkCoordTourCpu[0]) + dist(row, i,linkCoordTourCpu[0]);
                    newLength[3] = dist(j-1, i, linkCoordTourCpu[0]) + dist(row-1, j,linkCoordTourCpu[0]) + dist(row,i-1,linkCoordTourCpu[0]);

                    int finalSelect = -1;
                    for(int i = 0; i < 4; i++)
                    {
                        float opti = oldLength - newLength[i];
                        if(opti > 0)
                        {
                            finalSelect = i;
                            break;
                        }
                    }

                    if(finalSelect >= 0)
                    {

                        unsigned int node1 = (int)linkCoordTourCpu[0][j-1].current;
                        unsigned int node3 = (int)linkCoordTourCpu[0][i-1].current;
                        unsigned int node5 = (int)linkCoordTourCpu[0][row-1].current;

                        unsigned long long result = 0;
                        result = result | node3;
                        result = result << 16;
                        result = result | node5;

                        float codekopt = finalSelect; //WB.Q 2024 densityMap only mark the k value of k-opt, and mark which mode of k-exchange
                        codekopt = finalSelect * 100 + 3;

                        this->optCandidateMap[0][node1]= result; // WB.Q 2024 find a solution to judge non-interacted 23456-opt
                        this->densityMap[0][node1] = codekopt; // WB.Q 2024 densityMap only mark the k value of k-opt, and mark which mode of k-exchange

                        foundOpt = 1;
                        break;

                    }

                }
                if(foundOpt)
                    break;

            }
            if(foundOpt)
                continue;
        }

    }//qiao end sequential 3-opt with first opt for one node


    //qiao add sequential 3-opt, using 5loops of linkCoordTourCpu
    void sequential3optBest(Grid<doubleLinkedEdgeForTSP>& linkCoordTourCpu)
    {

        cout << "Enter sequential 3-opt Best " << endl;
        int N =  this->adaptiveMap.width;
        for(int j = 1; j < N; j++)
        {

            float oldLength0 =  dist(j-1, j, linkCoordTourCpu[0]);

            for(int i = j+1; i < N; i++)
            {
                float oldLength1 = oldLength0;
                oldLength1 += dist(i-1, i, linkCoordTourCpu[0]);

                for(int row = i +1; row <N; row ++)
                {

                    float oldLength = oldLength1;
                    oldLength += dist(row-1, row, linkCoordTourCpu[0]);

                    double newLength[4];
                    newLength[0] = dist(j-1, i, linkCoordTourCpu[0]) + dist(row-1, i-1, linkCoordTourCpu[0]) + dist(row, j, linkCoordTourCpu[0]);
                    newLength[1] = dist(j-1, row-1,linkCoordTourCpu[0]) + dist(row, i-1, linkCoordTourCpu[0]) + dist(i,j,linkCoordTourCpu[0]);
                    newLength[2] = dist(j-1, i-1, linkCoordTourCpu[0]) + dist(row-1, j, linkCoordTourCpu[0]) + dist(row, i,linkCoordTourCpu[0]);
                    newLength[3] = dist(j-1, i, linkCoordTourCpu[0]) + dist(row-1, j,linkCoordTourCpu[0]) + dist(row,i-1,linkCoordTourCpu[0]);

                    unsigned int node1 = (int)linkCoordTourCpu[0][j-1].current;
                    float localMinChange = this->minRadiusMap[0][node1];

                    int finalSelect = -1;
                    float optimiz = -INFINITY;
                    for(int i = 0; i < 4; i++)
                    {
                        float opti = oldLength - newLength[i];

                        if(opti > 0 && opti > optimiz )
                        {
                            optimiz = opti;
                            finalSelect = i;
                        }
                    }

                    if(optimiz > localMinChange)
                    {
                        unsigned int node3 = (int)linkCoordTourCpu[0][i-1].current;
                        unsigned int node5 = (int)linkCoordTourCpu[0][row-1].current;

                        unsigned long long result = 0;
                        result = result | node3;
                        result = result << 16;
                        result = result | node5;

                        float codekopt = finalSelect; //WB.Q 2024 densityMap only mark the k value of k-opt, and mark which mode of k-exchange
                        codekopt = finalSelect * 100 + 3;

                        this->optCandidateMap[0][node1]= result; // WB.Q 2024 find a solution to judge non-interacted 23456-opt
                        this->densityMap[0][node1] = codekopt; // WB.Q 2024 densityMap only mark the k value of k-opt, and mark which mode of k-exchange
                        this->minRadiusMap[0][node1]= optimiz;

                    }

                }

            }
        }

    }//qiao end sequential 3-opt with first opt for one node


    //qiao add sequential 6-opt, using 6 loops of linkCoordTourCpu
    void sequential6optFirst(Grid<doubleLinkedEdgeForTSP>& linkCoordTourCpu,  Grid<GLint> optPossibilitiesMap)
    {

        cout << "Enter sequential 6-opt first" << endl;
        int N =  this->adaptiveMap.width;
        for(int j_1 = 1; j_1 < N; j_1++)
        {

            float oldLength0 = dist(j_1-1, j_1, linkCoordTourCpu[0]);

            // cout << "oldLen0 = " << oldLength0 << endl;

            bool foundOpt = 0;

            for(int i_1 = j_1+1; i_1 < N; i_1++)
            {
                float oldLength1 = oldLength0;
                oldLength1 += dist(i_1-1, i_1, linkCoordTourCpu[0]);

                for(int row_1 = i_1+1; row_1 <N; row_1++)
                {
                    float oldLength2 = oldLength1;
                    oldLength2 += dist(row_1 -1, row_1, linkCoordTourCpu[0]);

                    for(int j = row_1 +1; j<N; j++)
                    {

                        float oldLength3 = oldLength2;
                        oldLength3 += dist(j-1, j, linkCoordTourCpu[0]);


                        for(int i = j+1; i < N; i++)
                        {
                            float oldLength4 = oldLength3;
                            oldLength4 += dist(i-1, i, linkCoordTourCpu[0]);

                            for(int row = i+1; row < N; row++)
                            {
                                float oldLength = oldLength4;
                                oldLength += dist(row-1, row, linkCoordTourCpu[0]);

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

                                for(int opt = 0; opt < 23220; opt +=12) //  6 edges 12 nodes 1935 sets 1935*12=23220 nodes
                                {
                                    int nd1 = optPossibilitiesMap[0][opt] -1;
                                    int nd2 = optPossibilitiesMap[0][opt+1] -1;
                                    int nd3 = optPossibilitiesMap[0][opt+2] -1;
                                    int nd4 = optPossibilitiesMap[0][opt+3] -1;
                                    int nd5 = optPossibilitiesMap[0][opt+4] -1;
                                    int nd6 = optPossibilitiesMap[0][opt+5] -1;
                                    int nd7 = optPossibilitiesMap[0][opt+6] -1;
                                    int nd8 = optPossibilitiesMap[0][opt+7] -1;
                                    int nd9 = optPossibilitiesMap[0][opt+8] -1;
                                    int nd10 = optPossibilitiesMap[0][opt+9] -1;
                                    int nd11 = optPossibilitiesMap[0][opt+10] -1;
                                    int nd12 = optPossibilitiesMap[0][opt+11] -1;

                                    int optCandi = opt / 12;

                                    newLength = dist(array[nd1],array[nd2], linkCoordTourCpu[0]) + dist(array[nd3],array[nd4], linkCoordTourCpu[0])
                                            + dist(array[nd5],array[nd6], linkCoordTourCpu[0])+ dist(array[nd7],array[nd8], linkCoordTourCpu[0])
                                            + dist(array[nd9],array[nd10], linkCoordTourCpu[0]) + dist(array[nd11],array[nd12], linkCoordTourCpu[0] );

                                    float opti = oldLength - newLength;
                                    if(opti > 0)
                                    {
                                        finalSelect = optCandi;

                                        break;
                                    }

                                }
                                if(finalSelect >= 0)
                                {

                                    int node1 = (int)linkCoordTourCpu[0][j_1-1].current;
                                    int node3 = (int)linkCoordTourCpu[0][i_1-1].current;
                                    int node5 = (int)linkCoordTourCpu[0][row_1-1].current;
                                    int node7 = (int)linkCoordTourCpu[0][j-1].current;
                                    int node9 = (int)linkCoordTourCpu[0][i-1].current;
                                    int node11 = (int)linkCoordTourCpu[0][row-1].current;

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

                                    this->optCandidateMap[0][node1] = result;
                                    this->densityMap[0][node1] = codekopt;

                                    foundOpt = 1;
                                    break;

                                }

                            }

                            if(foundOpt)
                                break;

                        }

                        if(foundOpt)
                            break;
                    }
                    if(foundOpt)
                        break;

                }

                if(foundOpt)
                    break;

            }

            if(foundOpt)
                continue;
        }

    }//qiao end sequential 5-opt with first opt for one node



    //qiao add sequential 6-opt, using 6 loops of linkCoordTourCpu
    void sequential6optBest(Grid<doubleLinkedEdgeForTSP>& linkCoordTourCpu,  Grid<GLint> optPossibilitiesMap)
    {

        cout << "Enter sequential 6-opt " << endl;
        int N =  this->adaptiveMap.width;
        for(int j_1 = 1; j_1 < N; j_1++)
        {

            float oldLength0 = dist(j_1-1, j_1, linkCoordTourCpu[0]);

            // cout << "oldLen0 = " << oldLength0 << endl;

            for(int i_1 = j_1+1; i_1 < N; i_1++)
            {
                float oldLength1 = oldLength0;
                oldLength1 += dist(i_1-1, i_1, linkCoordTourCpu[0]);

                for(int row_1 = i_1+1; row_1 <N; row_1++)
                {
                    float oldLength2 = oldLength1;
                    oldLength2 += dist(row_1 -1, row_1, linkCoordTourCpu[0]);

                    for(int j = row_1 +1; j<N; j++)
                    {

                        float oldLength3 = oldLength2;
                        oldLength3 += dist(j-1, j, linkCoordTourCpu[0]);


                        for(int i = j+1; i < N; i++)
                        {
                            float oldLength4 = oldLength3;
                            oldLength4 += dist(i-1, i, linkCoordTourCpu[0]);

                            for(int row = i+1; row < N; row++)
                            {
                                float oldLength = oldLength4;
                                oldLength += dist(row-1, row, linkCoordTourCpu[0]);

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


                                int node1 = (int)linkCoordTourCpu[0][j_1-1].current;
                                float localMinChange = this->minRadiusMap[0][node1];

                                int finalSelect = -1;
                                float optimiz = -INFINITY;


                                for(int opt = 0; opt < 23220; opt +=12) //  6 edges 12 nodes 1935 sets 1935*12=23220 nodes
                                {
                                    int nd1 = optPossibilitiesMap[0][opt] -1;
                                    int nd2 = optPossibilitiesMap[0][opt+1] -1;
                                    int nd3 = optPossibilitiesMap[0][opt+2] -1;
                                    int nd4 = optPossibilitiesMap[0][opt+3] -1;
                                    int nd5 = optPossibilitiesMap[0][opt+4] -1;
                                    int nd6 = optPossibilitiesMap[0][opt+5] -1;
                                    int nd7 = optPossibilitiesMap[0][opt+6] -1;
                                    int nd8 = optPossibilitiesMap[0][opt+7] -1;
                                    int nd9 = optPossibilitiesMap[0][opt+8] -1;
                                    int nd10 = optPossibilitiesMap[0][opt+9] -1;
                                    int nd11 = optPossibilitiesMap[0][opt+10] -1;
                                    int nd12 = optPossibilitiesMap[0][opt+11] -1;

                                    int optCandi = opt / 12;

                                    newLength = dist(array[nd1],array[nd2], linkCoordTourCpu[0]) + dist(array[nd3],array[nd4], linkCoordTourCpu[0])
                                            + dist(array[nd5],array[nd6], linkCoordTourCpu[0])+ dist(array[nd7],array[nd8], linkCoordTourCpu[0])
                                            + dist(array[nd9],array[nd10], linkCoordTourCpu[0]) + dist(array[nd11],array[nd12], linkCoordTourCpu[0] );

                                    float opti = oldLength - newLength;
                                    if(opti > 0 && opti > optimiz)
                                    {
                                        finalSelect = optCandi;
                                        optimiz = opti;
                                    }
                                }

                                if(optimiz > localMinChange)
                                {

                                    int node3 = (int)linkCoordTourCpu[0][i_1-1].current;
                                    int node5 = (int)linkCoordTourCpu[0][row_1-1].current;
                                    int node7 = (int)linkCoordTourCpu[0][j-1].current;
                                    int node9 = (int)linkCoordTourCpu[0][i-1].current;
                                    int node11 = (int)linkCoordTourCpu[0][row-1].current;

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

                                    this->optCandidateMap[0][node1] = result;
                                    this->densityMap[0][node1] = codekopt;
                                    this->minRadiusMap[0][node1]= optimiz;

                                }

                            }

                        }

                    }

                }

            }

        }

    }//qiao end sequential 5-opt with first opt for one node



    //qiao add sequential 6-opt, using 6 loops of linkCoordTourCpu which is already ordered TSP tour
    void sequentialVariable6optFirst(Grid<doubleLinkedEdgeForTSP>& linkCoordTourCpu, Grid<GLint> optPossibilitiesMap, Grid<GLint> fiveOptPossibilitiesMap, Grid<GLint> sixOptPossibilitiesMap)
    {

        cout << "Enter sequential variable k-opt " << ", order: " <<  this->grayValueMap[0][(int)linkCoordTourCpu[0][0].current] << endl;

        int N =  this->adaptiveMap.width;
        for(int j_1 = 1; j_1 < N; j_1++)
        {
            float oldLength0 = dist(j_1-1, j_1, linkCoordTourCpu[0]);
            //             cout << "oldLen0 = " << oldLength0 << endl;
            // unsigned int node3, node5, node7, node9, node11;

            //begin 2-opt
            for(int i_1 = j_1+2; i_1 < N; i_1++)
            {
                float oldLength1 = oldLength0;
                oldLength1 += dist(i_1-1, i_1, linkCoordTourCpu[0]);
                //                cout << "2-opt oldLen = " << oldLength1 << endl;

                //judge 2-opt if exist break and continue next j_1
                float newLength_2 = dist(j_1-1, i_1-1, linkCoordTourCpu[0]) + dist(j_1, i_1, linkCoordTourCpu[0]);

                if(newLength_2 < oldLength1)
                {

                    unsigned int node1 = (unsigned int)linkCoordTourCpu[0][j_1-1].current;
                    unsigned int node3 = (unsigned int)linkCoordTourCpu[0][i_1-1].current;


                    unsigned long long result = 0;
                    result = result | node3;
                    float codekopt = 2;

                    this->optCandidateMap[0][node1] = result;
                    this->densityMap[0][node1] = codekopt;

                    //                    cout <<"search select 2-opt: " << j_1   << ", tour order  " << this->grayValueMap[0][node1] << ", " << this->grayValueMap[0][node3] << endl;
                    break;
                }

                bool foundOpt = 0;

                //begin 3-opt
                for(int row_1 = i_1+2; row_1 <N; row_1++)
                {
                    float oldLength2 = oldLength1;
                    oldLength2 += dist(row_1 -1, row_1, linkCoordTourCpu[0]);

                    //                    cout << "3-opt oldLen = " << oldLength2 << endl;

                    //judge 3-opt if exist break and continue next j_1

                    double newLength_3[4];
                    newLength_3[0] = dist(j_1-1, i_1, linkCoordTourCpu[0]) + dist(row_1-1, i_1-1, linkCoordTourCpu[0]) + dist(row_1, j_1, linkCoordTourCpu[0]);
                    newLength_3[1] = dist(j_1-1, row_1-1,linkCoordTourCpu[0]) + dist(row_1, i_1-1, linkCoordTourCpu[0]) + dist(i_1,j_1,linkCoordTourCpu[0]);
                    newLength_3[2] = dist(j_1-1, i_1-1, linkCoordTourCpu[0]) + dist(row_1-1, j_1, linkCoordTourCpu[0]) + dist(row_1, i_1,linkCoordTourCpu[0]);
                    newLength_3[3] = dist(j_1-1, i_1, linkCoordTourCpu[0]) + dist(row_1-1, j_1,linkCoordTourCpu[0]) + dist(row_1,i_1-1,linkCoordTourCpu[0]);

                    int finalSelect = -1;
                    for(int i = 0; i < 4; i++)
                    {
                        float opti = oldLength2 - newLength_3[i];
                        if(opti > 0 )
                        {
                            finalSelect = i;
                            break;
                        }
                    }

                    if(finalSelect >= 0)
                    {

                        unsigned int node1 = (unsigned int)linkCoordTourCpu[0][j_1-1].current;
                        unsigned int node3 = (unsigned int)linkCoordTourCpu[0][i_1-1].current;
                        unsigned int node5 = (unsigned int)linkCoordTourCpu[0][row_1-1].current;

                        unsigned long long result = 0;
                        result = result | node3;
                        result = result << 16;
                        result = result | node5;

                        float codekopt = finalSelect; //WB.Q 2024 densityMap only mark the k value of k-opt, and mark which mode of k-exchange
                        codekopt = finalSelect * 100 + 3;

                        this->optCandidateMap[0][node1] = result;
                        this->densityMap[0][node1] = codekopt;

                        //                        cout << "search select 3-opt: " << j_1 << ", tour order " << this->grayValueMap[0][node1] << ", " << this->grayValueMap[0][node3]  << ", " << this->grayValueMap[0][node5]<< endl;


                        foundOpt = 1;
                        break;

                    }// end if foundOpt jump out of 3rd loop

                    //begin 4-opt
                    for(int j = row_1 +2; j<N; j++)
                    {

                        float oldLength3 = oldLength2;
                        oldLength3 += dist(j-1, j, linkCoordTourCpu[0]);
                        //  cout << "4-opt oldLen = " << oldLength3 << endl;


                        //judge 4-opt if exist break and continue next j_1

                        float newLength_4;//25 is fixed for 4-opt
                        int array[8];
                        array[0] = j_1-1;
                        array[1] = j_1;
                        array[2] = i_1-1;
                        array[3] = i_1;
                        array[4] = row_1-1;
                        array[5] = row_1;
                        array[6] = j-1;
                        array[7] = j;

                        int finalSelect = -1;

                        for(int opt = 0; opt < 200; opt +=8) //  4 edges 8 nodes
                        {
                            int nd1 = optPossibilitiesMap[0][opt] -1;
                            int nd2 = optPossibilitiesMap[0][opt+1] -1;
                            int nd3 = optPossibilitiesMap[0][opt+2] -1;
                            int nd4 = optPossibilitiesMap[0][opt+3] -1;
                            int nd5 = optPossibilitiesMap[0][opt+4] -1;
                            int nd6 = optPossibilitiesMap[0][opt+5] -1;
                            int nd7 = optPossibilitiesMap[0][opt+6] -1;
                            int nd8 = optPossibilitiesMap[0][opt+7] -1;

                            //  cout << ", nd1= " << nd1 << endl;

                            int optCandi = opt / 8;
                            //printf("GPU search nd1-8 %d, %d, %d, %d, %d, %d, %d, %d; optCandi=%d \n", nd1, nd2, nd3, nd4, nd5, nd6, nd7, nd8, optCandi);
                            newLength_4 = dist(array[nd1],array[nd2], linkCoordTourCpu[0]) + dist(array[nd3],array[nd4], linkCoordTourCpu[0]) + dist(array[nd5],array[nd6], linkCoordTourCpu[0])+ dist(array[nd7],array[nd8], linkCoordTourCpu[0]);

                            float opti = oldLength3 - newLength_4;
                            if(opti > 0)
                            {
                                finalSelect = optCandi;
                                break;

                            }
                        }

                        if(finalSelect >= 0)
                        {

                            unsigned int node1 = (unsigned int)linkCoordTourCpu[0][j_1-1].current;
                            unsigned int node3 = (unsigned int)linkCoordTourCpu[0][i_1-1].current;
                            unsigned int node5 = (unsigned int)linkCoordTourCpu[0][row_1-1].current;
                            unsigned int node7 = (unsigned int)linkCoordTourCpu[0][j-1].current;


                            unsigned long long result = 0;
                            result = result | node3;
                            result = result << 16;
                            result = result | node5;
                            result = result << 16;
                            result = result | node7;

                            float codekopt = finalSelect; //WB.Q 2024 densityMap only mark the k value of k-opt, and mark which mode of k-exchange
                            codekopt = finalSelect * 100 + 4;

                            this->optCandidateMap[0][node1] = result;
                            this->densityMap[0][node1] = codekopt;

                            //                            cout << "search select 4-opt: " << j_1 << ", tour order " << this->grayValueMap[0][node1] << ", " << this->grayValueMap[0][node3]
                            //                                    << ", " << this->grayValueMap[0][node5] << ", " << this->grayValueMap[0][node7] << endl;

                            foundOpt = 1;
                            break;
                        } // end 4-opt


                        //begin 5-opt
                        for(int i = j+2; i < N; i++)
                        {
                            float oldLength4 = oldLength3;
                            oldLength4 += dist(i-1, i, linkCoordTourCpu[0]);

                            // cout << "5-opt oldLen = " << oldLength4 << endl;

                            //judge 5-opt if exist break and continue next j_1
                            float newLength_5;
                            int array[10];
                            array[0] = j_1-1;
                            array[1] = j_1;
                            array[2] = i_1-1;
                            array[3] = i;
                            array[4] = row_1-1;
                            array[5] = row_1;
                            array[6] = j-1;
                            array[7] = j;
                            array[8] = i-1;
                            array[9] = i;

                            int finalSelect = -1;

                            for(int opt = 0; opt < 2080; opt +=10) //  4 edges 8 nodes
                            {
                                int nd1 = fiveOptPossibilitiesMap[0][opt] -1;
                                int nd2 = fiveOptPossibilitiesMap[0][opt+1] -1;
                                int nd3 = fiveOptPossibilitiesMap[0][opt+2] -1;
                                int nd4 = fiveOptPossibilitiesMap[0][opt+3] -1;
                                int nd5 = fiveOptPossibilitiesMap[0][opt+4] -1;
                                int nd6 = fiveOptPossibilitiesMap[0][opt+5] -1;
                                int nd7 = fiveOptPossibilitiesMap[0][opt+6] -1;
                                int nd8 = fiveOptPossibilitiesMap[0][opt+7] -1;
                                int nd9 = fiveOptPossibilitiesMap[0][opt+8] -1;
                                int nd10 = fiveOptPossibilitiesMap[0][opt+9] -1;

                                int optCandi = opt / 10;
                                // printf("GPU search nd1-8 %d, %d, %d, %d, %d, %d, %d, %d; optCandi=%d \n", nd1, nd2, nd3, nd4, nd5, nd6, nd7, nd8, optCandi);
                                newLength_5 = dist(array[nd1],array[nd2], linkCoordTourCpu[0]) + dist(array[nd3],array[nd4], linkCoordTourCpu[0])
                                        + dist(array[nd5],array[nd6], linkCoordTourCpu[0])+ dist(array[nd7],array[nd8], linkCoordTourCpu[0]) + dist(array[nd9],array[nd10], linkCoordTourCpu[0]);

                                float opti = oldLength4 - newLength_5;
                                if(opti > 0 )
                                {
                                    finalSelect = optCandi;
                                    break;
                                }
                            }

                            if(finalSelect >= 0)
                            {

                                unsigned int node1 = (unsigned int)linkCoordTourCpu[0][j_1-1].current;
                                unsigned int node3 = (unsigned int)linkCoordTourCpu[0][i_1-1].current;
                                unsigned int node5 = (unsigned int)linkCoordTourCpu[0][row_1-1].current;
                                unsigned int node7 = (unsigned int)linkCoordTourCpu[0][j-1].current;
                                unsigned int node9 = (unsigned int)linkCoordTourCpu[0][i-1].current;

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

                                this->optCandidateMap[0][node1] = result;
                                this->densityMap[0][node1] = codekopt;

                                //                                cout << "search select 5-opt: " << j_1  << ", tour order " << this->grayValueMap[0][node1] << ", " << this->grayValueMap[0][node3]
                                //                                        << ", " << this->grayValueMap[0][node5] << ", "
                                //                                        << this->grayValueMap[0][node7] << ", " << this->grayValueMap[0][node9] << endl;

                                foundOpt = 1;
                                break;
                            }

                            //begin 6-opt
                            for(int row = i+2; row < N; row++)
                            {

                                //judge 6-opt if exist break and continue next j_1
                                float oldLength = oldLength4;
                                oldLength += dist(row-1, row, linkCoordTourCpu[0]);

                                // cout << "6-opt oldLen = " << oldLength << endl;
                                //                                 cout << "6-opt row = " << row << ", i= " << i << ", j= " << j << ", row_1" << row_1 << ", i_1=" << i_1 << ", j_1= " << j_1 << endl;

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

                                for(int opt = 0; opt < 23220; opt +=12) //  6 edges 12 nodes 1935 sets 1935*12=23220 nodes
                                {
                                    int nd1 = sixOptPossibilitiesMap[0][opt] -1;
                                    int nd2 = sixOptPossibilitiesMap[0][opt+1] -1;
                                    int nd3 = sixOptPossibilitiesMap[0][opt+2] -1;
                                    int nd4 = sixOptPossibilitiesMap[0][opt+3] -1;
                                    int nd5 = sixOptPossibilitiesMap[0][opt+4] -1;
                                    int nd6 = sixOptPossibilitiesMap[0][opt+5] -1;
                                    int nd7 = sixOptPossibilitiesMap[0][opt+6] -1;
                                    int nd8 = sixOptPossibilitiesMap[0][opt+7] -1;
                                    int nd9 = sixOptPossibilitiesMap[0][opt+8] -1;
                                    int nd10 = sixOptPossibilitiesMap[0][opt+9] -1;
                                    int nd11 = sixOptPossibilitiesMap[0][opt+10] -1;
                                    int nd12 = sixOptPossibilitiesMap[0][opt+11] -1;

                                    int optCandi = opt / 12;

                                    newLength = dist(array[nd1],array[nd2], linkCoordTourCpu[0]) + dist(array[nd3],array[nd4], linkCoordTourCpu[0])
                                            + dist(array[nd5],array[nd6], linkCoordTourCpu[0])+ dist(array[nd7],array[nd8], linkCoordTourCpu[0])
                                            + dist(array[nd9],array[nd10], linkCoordTourCpu[0]) + dist(array[nd11],array[nd12], linkCoordTourCpu[0] );

                                    float opti = oldLength - newLength;
                                    if(opti > 0)
                                    {
                                        finalSelect = optCandi;

                                        break;
                                    }

                                }
                                if(finalSelect >= 0)
                                {

                                    unsigned int node1 = (unsigned int)linkCoordTourCpu[0][j_1-1].current;
                                    unsigned int node3 = (unsigned int)linkCoordTourCpu[0][i_1-1].current;
                                    unsigned int node5 = (unsigned int)linkCoordTourCpu[0][row_1-1].current;
                                    unsigned int node7 = (unsigned int)linkCoordTourCpu[0][j-1].current;
                                    unsigned int node9 = (unsigned int)linkCoordTourCpu[0][i-1].current;
                                    unsigned int node11 = (unsigned int)linkCoordTourCpu[0][row-1].current;

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

                                    this->optCandidateMap[0][node1] = result;
                                    this->densityMap[0][node1] = codekopt;

                                    //                                    cout <<"search select 6-opt: " << j_1  << ", tour order " << this->grayValueMap[0][node1] << ", " << this->grayValueMap[0][node3]
                                    //                                           << ", " << this->grayValueMap[0][node5] << ", " << this->grayValueMap[0][node7]
                                    //                                              << ", " << this->grayValueMap[0][node9] << ", " << this->grayValueMap[0][node11] << endl;

                                    foundOpt = 1;
                                    break;

                                }

                            }

                            if(foundOpt)
                                break;

                        }

                        if(foundOpt)
                            break;
                    }
                    if(foundOpt)
                        break;

                }

                if(foundOpt)
                    break;

            }

            //            if(foundOpt)
            //                continue;
        }

    }//qiao end sequential 5-opt with first opt for one node




    //qiao add sequential 6-opt, using 6 loops of linkCoordTourCpu which is already ordered TSP tour
    void sequentialVariable3optFirst(Grid<doubleLinkedEdgeForTSP>& linkCoordTourCpu, Grid<GLint> optPossibilitiesMap, Grid<GLint> fiveOptPossibilitiesMap, Grid<GLint> sixOptPossibilitiesMap)
    {

        //        cout << "Enter sequential variable k-opt " << ", order: " <<  this->grayValueMap[0][(int)linkCoordTourCpu[0][0].current] << endl;

        int N =  this->adaptiveMap.width;
        for(int j_1 = 1; j_1 < N; j_1++)
        {
            float oldLength0 = dist(j_1-1, j_1, linkCoordTourCpu[0]);
            //             cout << "oldLen0 = " << oldLength0 << endl;
            // unsigned int node3, node5, node7, node9, node11;

            //begin 2-opt
            for(int i_1 = j_1+2; i_1 < N; i_1++)
            {
                float oldLength1 = oldLength0;
                oldLength1 += dist(i_1-1, i_1, linkCoordTourCpu[0]);
                //                cout << "2-opt oldLen = " << oldLength1 << endl;

                //judge 2-opt if exist break and continue next j_1
                float newLength_2 = dist(j_1-1, i_1-1, linkCoordTourCpu[0]) + dist(j_1, i_1, linkCoordTourCpu[0]);

                if(newLength_2 < oldLength1)
                {

                    unsigned int node1 = (unsigned int)linkCoordTourCpu[0][j_1-1].current;
                    unsigned int node3 = (unsigned int)linkCoordTourCpu[0][i_1-1].current;

                    unsigned long long result = 0;
                    result = result | node3;
                    float codekopt = 2;

                    this->optCandidateMap[0][node1] = result;
                    this->densityMap[0][node1] = codekopt;

                    //                    cout <<"search select 2-opt: " << j_1   << ", tour order  " << this->grayValueMap[0][node1] << ", " << this->grayValueMap[0][node3] << endl;
                    break;
                }

                bool foundOpt = 0;

                //begin 3-opt
                for(int row_1 = i_1+2; row_1 <N; row_1++)
                {
                    float oldLength2 = oldLength1;
                    oldLength2 += dist(row_1 -1, row_1, linkCoordTourCpu[0]);

                    //                    cout << "3-opt oldLen = " << oldLength2 << endl;

                    //judge 3-opt if exist break and continue next j_1

                    double newLength_3[4];
                    newLength_3[0] = dist(j_1-1, i_1, linkCoordTourCpu[0]) + dist(row_1-1, i_1-1, linkCoordTourCpu[0]) + dist(row_1, j_1, linkCoordTourCpu[0]);
                    newLength_3[1] = dist(j_1-1, row_1-1,linkCoordTourCpu[0]) + dist(row_1, i_1-1, linkCoordTourCpu[0]) + dist(i_1,j_1,linkCoordTourCpu[0]);
                    newLength_3[2] = dist(j_1-1, i_1-1, linkCoordTourCpu[0]) + dist(row_1-1, j_1, linkCoordTourCpu[0]) + dist(row_1, i_1,linkCoordTourCpu[0]);
                    newLength_3[3] = dist(j_1-1, i_1, linkCoordTourCpu[0]) + dist(row_1-1, j_1,linkCoordTourCpu[0]) + dist(row_1,i_1-1,linkCoordTourCpu[0]);

                    int finalSelect = -1;
                    for(int i = 0; i < 4; i++)
                    {
                        float opti = oldLength2 - newLength_3[i];
                        if(opti > 0 )
                        {
                            finalSelect = i;
                            break;
                        }
                    }

                    if(finalSelect >= 0)
                    {

                        unsigned int node1 = (unsigned int)linkCoordTourCpu[0][j_1-1].current;
                        unsigned int node3 = (unsigned int)linkCoordTourCpu[0][i_1-1].current;
                        unsigned int node5 = (unsigned int)linkCoordTourCpu[0][row_1-1].current;

                        unsigned long long result = 0;
                        result = result | node3;
                        result = result << 16;
                        result = result | node5;

                        float codekopt = finalSelect; //WB.Q 2024 densityMap only mark the k value of k-opt, and mark which mode of k-exchange
                        codekopt = finalSelect * 100 + 3;

                        this->optCandidateMap[0][node1] = result;
                        this->densityMap[0][node1] = codekopt;

                        //                        cout << "search select 3-opt: " << j_1 << ", tour order " << this->grayValueMap[0][node1] << ", " << this->grayValueMap[0][node3]  << ", " << this->grayValueMap[0][node5]<< endl;


                        foundOpt = 1;
                        break;

                    }// end if foundOpt jump out of 3rd loop

                    if(foundOpt)
                        break;

                }

                if(foundOpt)
                    break;

            }
        }

    }//qiao end sequential 3-opt with first opt for one node



    //qiao add sequential 6-opt, using 6 loops of linkCoordTourCpu which is already ordered TSP tour
    void sequentialVariable4optFirst(Grid<doubleLinkedEdgeForTSP>& linkCoordTourCpu, Grid<GLint> optPossibilitiesMap, Grid<GLint> fiveOptPossibilitiesMap, Grid<GLint> sixOptPossibilitiesMap)
    {

        //        cout << "Enter sequential variable k-opt " << ", order: " <<  this->grayValueMap[0][(int)linkCoordTourCpu[0][0].current] << endl;

        int N =  this->adaptiveMap.width;
        for(int j_1 = 1; j_1 < N; j_1++)
        {
            float oldLength0 = dist(j_1-1, j_1, linkCoordTourCpu[0]);
            //  cout << "oldLen0 = " << oldLength0 << endl;
            // unsigned int node3, node5, node7, node9, node11;

            //begin 2-opt
            for(int i_1 = j_1 + 2; i_1 < N; i_1++)
            {
                float oldLength1 = oldLength0;
                oldLength1 += dist(i_1-1, i_1, linkCoordTourCpu[0]);
                //  cout << "2-opt oldLen = " << oldLength1 << endl;

                //judge 2-opt if exist break and continue next j_1
                float newLength_2 = dist(j_1-1, i_1-1, linkCoordTourCpu[0]) + dist(j_1, i_1, linkCoordTourCpu[0]);

                if(newLength_2 < oldLength1)
                {

                    unsigned int node1 = (unsigned int)linkCoordTourCpu[0][j_1-1].current;
                    unsigned int node3 = (unsigned int)linkCoordTourCpu[0][i_1-1].current;

                    unsigned long long result = 0;
                    result = result | node3;
                    float codekopt = 2;

                    this->optCandidateMap[0][node1] = result;
                    this->densityMap[0][node1] = codekopt;

                    //                    cout <<"search select 2-opt: " << j_1   << ", tour order  " << this->grayValueMap[0][node1] << ", " << this->grayValueMap[0][node3] << endl;
                    break;
                }

                bool foundOpt = 0;

                //begin 3-opt
                for(int row_1 = i_1+2; row_1 <N; row_1++)
                {
                    float oldLength2 = oldLength1;
                    oldLength2 += dist(row_1 -1, row_1, linkCoordTourCpu[0]);

                    //   cout << "3-opt oldLen = " << oldLength2 << endl;

                    //judge 3-opt if exist break and continue next j_1

                    double newLength_3[4];
                    newLength_3[0] = dist(j_1-1, i_1, linkCoordTourCpu[0]) + dist(row_1-1, i_1-1, linkCoordTourCpu[0]) + dist(row_1, j_1, linkCoordTourCpu[0]);
                    newLength_3[1] = dist(j_1-1, row_1-1,linkCoordTourCpu[0]) + dist(row_1, i_1-1, linkCoordTourCpu[0]) + dist(i_1,j_1,linkCoordTourCpu[0]);
                    newLength_3[2] = dist(j_1-1, i_1-1, linkCoordTourCpu[0]) + dist(row_1-1, j_1, linkCoordTourCpu[0]) + dist(row_1, i_1,linkCoordTourCpu[0]);
                    newLength_3[3] = dist(j_1-1, i_1, linkCoordTourCpu[0]) + dist(row_1-1, j_1,linkCoordTourCpu[0]) + dist(row_1,i_1-1,linkCoordTourCpu[0]);

                    int finalSelect = -1;
                    for(int i = 0; i < 4; i++)
                    {
                        float opti = oldLength2 - newLength_3[i];
                        if(opti > 0 )
                        {
                            finalSelect = i;
                            break;
                        }
                    }

                    if(finalSelect >= 0)
                    {

                        unsigned int node1 = (unsigned int)linkCoordTourCpu[0][j_1-1].current;
                        unsigned int node3 = (unsigned int)linkCoordTourCpu[0][i_1-1].current;
                        unsigned int node5 = (unsigned int)linkCoordTourCpu[0][row_1-1].current;

                        unsigned long long result = 0;
                        result = result | node3;
                        result = result << 16;
                        result = result | node5;

                        float codekopt = finalSelect; //WB.Q 2024 densityMap only mark the k value of k-opt, and mark which mode of k-exchange
                        codekopt = finalSelect * 100 + 3;

                        this->optCandidateMap[0][node1] = result;
                        this->densityMap[0][node1] = codekopt;

                        //                        cout << "search select 3-opt: " << j_1 << ", tour order " << this->grayValueMap[0][node1] << ", " << this->grayValueMap[0][node3]  << ", " << this->grayValueMap[0][node5]<< endl;


                        foundOpt = 1;
                        break;

                    }// end if foundOpt jump out of 3rd loop

                    //begin 4-opt
                    for(int j = row_1 +2; j<N; j++)
                    {

                        float oldLength3 = oldLength2;
                        oldLength3 += dist(j-1, j, linkCoordTourCpu[0]);
                        //  cout << "4-opt oldLen = " << oldLength3 << endl;


                        //judge 4-opt if exist break and continue next j_1

                        float newLength_4;//25 is fixed for 4-opt
                        int array[8];
                        array[0] = j_1-1;
                        array[1] = j_1;
                        array[2] = i_1-1;
                        array[3] = i_1;
                        array[4] = row_1-1;
                        array[5] = row_1;
                        array[6] = j-1;
                        array[7] = j;

                        int finalSelect = -1;

                        for(int opt = 0; opt < 200; opt +=8) //  4 edges 8 nodes
                        {
                            int nd1 = optPossibilitiesMap[0][opt] -1;
                            int nd2 = optPossibilitiesMap[0][opt+1] -1;
                            int nd3 = optPossibilitiesMap[0][opt+2] -1;
                            int nd4 = optPossibilitiesMap[0][opt+3] -1;
                            int nd5 = optPossibilitiesMap[0][opt+4] -1;
                            int nd6 = optPossibilitiesMap[0][opt+5] -1;
                            int nd7 = optPossibilitiesMap[0][opt+6] -1;
                            int nd8 = optPossibilitiesMap[0][opt+7] -1;

                            //  cout << ", nd1= " << nd1 << endl;

                            int optCandi = opt / 8;
                            //printf("GPU search nd1-8 %d, %d, %d, %d, %d, %d, %d, %d; optCandi=%d \n", nd1, nd2, nd3, nd4, nd5, nd6, nd7, nd8, optCandi);
                            newLength_4 = dist(array[nd1],array[nd2], linkCoordTourCpu[0]) + dist(array[nd3],array[nd4], linkCoordTourCpu[0]) + dist(array[nd5],array[nd6], linkCoordTourCpu[0])+ dist(array[nd7],array[nd8], linkCoordTourCpu[0]);

                            float opti = oldLength3 - newLength_4;
                            if(opti > 0)
                            {
                                finalSelect = optCandi;
                                break;

                            }
                        }

                        if(finalSelect >= 0)
                        {

                            unsigned int node1 = (unsigned int)linkCoordTourCpu[0][j_1-1].current;
                            unsigned int node3 = (unsigned int)linkCoordTourCpu[0][i_1-1].current;
                            unsigned int node5 = (unsigned int)linkCoordTourCpu[0][row_1-1].current;
                            unsigned int node7 = (unsigned int)linkCoordTourCpu[0][j-1].current;


                            unsigned long long result = 0;
                            result = result | node3;
                            result = result << 16;
                            result = result | node5;
                            result = result << 16;
                            result = result | node7;

                            float codekopt = finalSelect; //WB.Q 2024 densityMap only mark the k value of k-opt, and mark which mode of k-exchange
                            codekopt = finalSelect * 100 + 4;

                            this->optCandidateMap[0][node1] = result;
                            this->densityMap[0][node1] = codekopt;

                            //                            cout << "search select 4-opt: " << j_1 << ", tour order " << this->grayValueMap[0][node1] << ", " << this->grayValueMap[0][node3]
                            //                                    << ", " << this->grayValueMap[0][node5] << ", " << this->grayValueMap[0][node7] << endl;

                            foundOpt = 1;
                            break;
                        } // end 4-opt

                    }
                    if(foundOpt)
                        break;

                }

                if(foundOpt)
                    break;

            }

        }

    }//qiao end sequential 5-opt with first opt for one node



    //qiao add sequential 6-opt, using 6 loops of linkCoordTourCpu which is already ordered TSP tour
    void sequentialVariable5optFirst(Grid<doubleLinkedEdgeForTSP>& linkCoordTourCpu, Grid<GLint> optPossibilitiesMap, Grid<GLint> fiveOptPossibilitiesMap, Grid<GLint> sixOptPossibilitiesMap)
    {

        cout << "Enter sequential variable 5-opt first " << ", order: " <<  this->grayValueMap[0][(int)linkCoordTourCpu[0][0].current] << endl;

        int N =  this->adaptiveMap.width;
        for(int j_1 = 1; j_1 < N; j_1++)
        {
            float oldLength0 = dist(j_1-1, j_1, linkCoordTourCpu[0]);
            // cout << "oldLen0 = " << oldLength0 << endl;
            // unsigned int node3, node5, node7, node9, node11;

            //begin 2-opt
            for(int i_1 = j_1+2; i_1 < N; i_1++)
            {
                float oldLength1 = oldLength0;
                oldLength1 += dist(i_1-1, i_1, linkCoordTourCpu[0]);
                //  cout << "2-opt oldLen = " << oldLength1 << endl;

                //judge 2-opt if exist break and continue next j_1
                float newLength_2 = dist(j_1-1, i_1-1, linkCoordTourCpu[0]) + dist(j_1, i_1, linkCoordTourCpu[0]);

                if(newLength_2 < oldLength1)
                {

                    unsigned int node1 = (unsigned int)linkCoordTourCpu[0][j_1-1].current;
                    unsigned int node3 = (unsigned int)linkCoordTourCpu[0][i_1-1].current;


                    unsigned long long result = 0;
                    result = result | node3;
                    float codekopt = 2;

                    this->optCandidateMap[0][node1] = result;
                    this->densityMap[0][node1] = codekopt;

                    //  cout <<"search select 2-opt: " << j_1   << ", tour order  " << this->grayValueMap[0][node1] << ", " << this->grayValueMap[0][node3] << endl;
                    break;
                }

                bool foundOpt = 0;

                //begin 3-opt
                for(int row_1 = i_1+2; row_1 <N; row_1++)
                {
                    float oldLength2 = oldLength1;
                    oldLength2 += dist(row_1 -1, row_1, linkCoordTourCpu[0]);

                    //  cout << "3-opt oldLen = " << oldLength2 << endl;

                    //judge 3-opt if exist break and continue next j_1

                    double newLength_3[4];
                    newLength_3[0] = dist(j_1-1, i_1, linkCoordTourCpu[0]) + dist(row_1-1, i_1-1, linkCoordTourCpu[0]) + dist(row_1, j_1, linkCoordTourCpu[0]);
                    newLength_3[1] = dist(j_1-1, row_1-1,linkCoordTourCpu[0]) + dist(row_1, i_1-1, linkCoordTourCpu[0]) + dist(i_1,j_1,linkCoordTourCpu[0]);
                    newLength_3[2] = dist(j_1-1, i_1-1, linkCoordTourCpu[0]) + dist(row_1-1, j_1, linkCoordTourCpu[0]) + dist(row_1, i_1,linkCoordTourCpu[0]);
                    newLength_3[3] = dist(j_1-1, i_1, linkCoordTourCpu[0]) + dist(row_1-1, j_1,linkCoordTourCpu[0]) + dist(row_1,i_1-1,linkCoordTourCpu[0]);

                    int finalSelect = -1;
                    for(int i = 0; i < 4; i++)
                    {
                        float opti = oldLength2 - newLength_3[i];
                        if(opti > 0 )
                        {
                            finalSelect = i;
                            break;
                        }
                    }

                    if(finalSelect >= 0)
                    {

                        unsigned int node1 = (unsigned int)linkCoordTourCpu[0][j_1-1].current;
                        unsigned int node3 = (unsigned int)linkCoordTourCpu[0][i_1-1].current;
                        unsigned int node5 = (unsigned int)linkCoordTourCpu[0][row_1-1].current;

                        unsigned long long result = 0;
                        result = result | node3;
                        result = result << 16;
                        result = result | node5;

                        float codekopt = finalSelect; //WB.Q 2024 densityMap only mark the k value of k-opt, and mark which mode of k-exchange
                        codekopt = finalSelect * 100 + 3;

                        this->optCandidateMap[0][node1] = result;
                        this->densityMap[0][node1] = codekopt;

                        //                        cout << "search select 3-opt: " << j_1 << ", tour order " << this->grayValueMap[0][node1] << ", " << this->grayValueMap[0][node3]  << ", " << this->grayValueMap[0][node5]<< endl;


                        foundOpt = 1;
                        break;

                    }// end if foundOpt jump out of 3rd loop

                    //begin 4-opt
                    for(int j = row_1 +2; j<N; j++)
                    {

                        float oldLength3 = oldLength2;
                        oldLength3 += dist(j-1, j, linkCoordTourCpu[0]);
                        //  cout << "4-opt oldLen = " << oldLength3 << endl;


                        //judge 4-opt if exist break and continue next j_1

                        float newLength_4;//25 is fixed for 4-opt
                        int array[8];
                        array[0] = j_1-1;
                        array[1] = j_1;
                        array[2] = i_1-1;
                        array[3] = i_1;
                        array[4] = row_1-1;
                        array[5] = row_1;
                        array[6] = j-1;
                        array[7] = j;

                        int finalSelect = -1;

                        for(int opt = 0; opt < 200; opt +=8) //  4 edges 8 nodes
                        {
                            int nd1 = optPossibilitiesMap[0][opt] -1;
                            int nd2 = optPossibilitiesMap[0][opt+1] -1;
                            int nd3 = optPossibilitiesMap[0][opt+2] -1;
                            int nd4 = optPossibilitiesMap[0][opt+3] -1;
                            int nd5 = optPossibilitiesMap[0][opt+4] -1;
                            int nd6 = optPossibilitiesMap[0][opt+5] -1;
                            int nd7 = optPossibilitiesMap[0][opt+6] -1;
                            int nd8 = optPossibilitiesMap[0][opt+7] -1;

                            //  cout << ", nd1= " << nd1 << endl;

                            int optCandi = opt / 8;
                            //printf("GPU search nd1-8 %d, %d, %d, %d, %d, %d, %d, %d; optCandi=%d \n", nd1, nd2, nd3, nd4, nd5, nd6, nd7, nd8, optCandi);
                            newLength_4 = dist(array[nd1],array[nd2], linkCoordTourCpu[0]) + dist(array[nd3],array[nd4], linkCoordTourCpu[0]) + dist(array[nd5],array[nd6], linkCoordTourCpu[0])+ dist(array[nd7],array[nd8], linkCoordTourCpu[0]);

                            float opti = oldLength3 - newLength_4;
                            if(opti > 0)
                            {
                                finalSelect = optCandi;
                                break;

                            }
                        }

                        if(finalSelect >= 0)
                        {

                            unsigned int node1 = (unsigned int)linkCoordTourCpu[0][j_1-1].current;
                            unsigned int node3 = (unsigned int)linkCoordTourCpu[0][i_1-1].current;
                            unsigned int node5 = (unsigned int)linkCoordTourCpu[0][row_1-1].current;
                            unsigned int node7 = (unsigned int)linkCoordTourCpu[0][j-1].current;


                            unsigned long long result = 0;
                            result = result | node3;
                            result = result << 16;
                            result = result | node5;
                            result = result << 16;
                            result = result | node7;

                            float codekopt = finalSelect; //WB.Q 2024 densityMap only mark the k value of k-opt, and mark which mode of k-exchange
                            codekopt = finalSelect * 100 + 4;

                            this->optCandidateMap[0][node1] = result;
                            this->densityMap[0][node1] = codekopt;

                            //  cout << "search select 4-opt: " << j_1 << ", tour order " << this->grayValueMap[0][node1] << ", " << this->grayValueMap[0][node3]
                            //   << ", " << this->grayValueMap[0][node5] << ", " << this->grayValueMap[0][node7] << endl;

                            foundOpt = 1;
                            break;
                        } // end 4-opt


                        //begin 5-opt
                        for(int i = j+2; i < N; i++)
                        {
                            float oldLength4 = oldLength3;
                            oldLength4 += dist(i-1, i, linkCoordTourCpu[0]);

                            // cout << "5-opt oldLen = " << oldLength4 << endl;

                            //judge 5-opt if exist break and continue next j_1
                            float newLength_5;
                            int array[10];
                            array[0] = j_1-1;
                            array[1] = j_1;
                            array[2] = i_1-1;
                            array[3] = i;
                            array[4] = row_1-1;
                            array[5] = row_1;
                            array[6] = j-1;
                            array[7] = j;
                            array[8] = i-1;
                            array[9] = i;

                            int finalSelect = -1;

                            for(int opt = 0; opt < 2080; opt +=10) //  4 edges 8 nodes
                            {
                                int nd1 = fiveOptPossibilitiesMap[0][opt] -1;
                                int nd2 = fiveOptPossibilitiesMap[0][opt+1] -1;
                                int nd3 = fiveOptPossibilitiesMap[0][opt+2] -1;
                                int nd4 = fiveOptPossibilitiesMap[0][opt+3] -1;
                                int nd5 = fiveOptPossibilitiesMap[0][opt+4] -1;
                                int nd6 = fiveOptPossibilitiesMap[0][opt+5] -1;
                                int nd7 = fiveOptPossibilitiesMap[0][opt+6] -1;
                                int nd8 = fiveOptPossibilitiesMap[0][opt+7] -1;
                                int nd9 = fiveOptPossibilitiesMap[0][opt+8] -1;
                                int nd10 = fiveOptPossibilitiesMap[0][opt+9] -1;

                                int optCandi = opt / 10;
                                // printf("GPU search nd1-8 %d, %d, %d, %d, %d, %d, %d, %d; optCandi=%d \n", nd1, nd2, nd3, nd4, nd5, nd6, nd7, nd8, optCandi);
                                newLength_5 = dist(array[nd1],array[nd2], linkCoordTourCpu[0]) + dist(array[nd3],array[nd4], linkCoordTourCpu[0])
                                        + dist(array[nd5],array[nd6], linkCoordTourCpu[0])+ dist(array[nd7],array[nd8], linkCoordTourCpu[0]) + dist(array[nd9],array[nd10], linkCoordTourCpu[0]);

                                float opti = oldLength4 - newLength_5;
                                if(opti > 0 )
                                {
                                    finalSelect = optCandi;
                                    break;
                                }
                            }

                            if(finalSelect >= 0)
                            {

                                unsigned int node1 = (unsigned int)linkCoordTourCpu[0][j_1-1].current;
                                unsigned int node3 = (unsigned int)linkCoordTourCpu[0][i_1-1].current;
                                unsigned int node5 = (unsigned int)linkCoordTourCpu[0][row_1-1].current;
                                unsigned int node7 = (unsigned int)linkCoordTourCpu[0][j-1].current;
                                unsigned int node9 = (unsigned int)linkCoordTourCpu[0][i-1].current;

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

                                this->optCandidateMap[0][node1] = result;
                                this->densityMap[0][node1] = codekopt;

                                //                                cout << "search select 5-opt: " << j_1  << ", tour order " << this->grayValueMap[0][node1] << ", " << this->grayValueMap[0][node3]
                                //                                        << ", " << this->grayValueMap[0][node5] << ", "
                                //                                        << this->grayValueMap[0][node7] << ", " << this->grayValueMap[0][node9] << endl;

                                foundOpt = 1;
                                break;
                            }

                        }

                        if(foundOpt)
                            break;
                    }
                    if(foundOpt)
                        break;

                }

                if(foundOpt)
                    break;

            }

            //            if(foundOpt)
            //                continue;
        }

    }//qiao end sequential 5-opt with first opt for one node








    //qiao add sequential 6-opt, using 6 loops of linkCoordTourCpu which is already ordered TSP tour
    void sequentialVariable3optBest(Grid<doubleLinkedEdgeForTSP>& linkCoordTourCpu, Grid<GLint> optPossibilitiesMap, Grid<GLint> fiveOptPossibilitiesMap, Grid<GLint> sixOptPossibilitiesMap)
    {

        //        cout << "Enter sequential variable k-opt " << ", order: " <<  this->grayValueMap[0][(int)linkCoordTourCpu[0][0].current] << endl;

        int N =  this->adaptiveMap.width;
        for(int j_1 = 1; j_1 < N; j_1++)
        {
            float oldLength0 = dist(j_1-1, j_1, linkCoordTourCpu[0]);

            //begin 2-opt
            for(int i_1 = j_1+2; i_1 < N; i_1++)
            {
                float oldLength1 = oldLength0;
                oldLength1 += dist(i_1-1, i_1, linkCoordTourCpu[0]);
                //                cout << "2-opt oldLen = " << oldLength1 << endl;

                //judge 2-opt if exist break and continue next j_1
                float newLength_2 = dist(j_1-1, i_1-1, linkCoordTourCpu[0]) + dist(j_1, i_1, linkCoordTourCpu[0]);

                if(newLength_2 < oldLength1)
                {

                    float optimization = oldLength1 - newLength_2;

                    unsigned int node1 = (unsigned int)linkCoordTourCpu[0][j_1-1].current;
                    unsigned int node3 = (unsigned int)linkCoordTourCpu[0][i_1-1].current;

                    float localMinChange = this->minRadiusMap[0][node1];

                    if(optimization > localMinChange)
                    {
                        this->minRadiusMap[0][node1]= optimization;

                        unsigned long long result = 0;
                        result = result | node3;
                        float codekopt = 2;

                        this->optCandidateMap[0][node1] = result;
                        this->densityMap[0][node1] = codekopt;
                    }

                }


                //begin 3-opt
                for(int row_1 = i_1+2; row_1 <N; row_1++)
                {
                    float oldLength2 = oldLength1;
                    oldLength2 += dist(row_1 -1, row_1, linkCoordTourCpu[0]);
                    double newLength_3[4];
                    newLength_3[0] = dist(j_1-1, i_1, linkCoordTourCpu[0]) + dist(row_1-1, i_1-1, linkCoordTourCpu[0]) + dist(row_1, j_1, linkCoordTourCpu[0]);
                    newLength_3[1] = dist(j_1-1, row_1-1,linkCoordTourCpu[0]) + dist(row_1, i_1-1, linkCoordTourCpu[0]) + dist(i_1,j_1,linkCoordTourCpu[0]);
                    newLength_3[2] = dist(j_1-1, i_1-1, linkCoordTourCpu[0]) + dist(row_1-1, j_1, linkCoordTourCpu[0]) + dist(row_1, i_1,linkCoordTourCpu[0]);
                    newLength_3[3] = dist(j_1-1, i_1, linkCoordTourCpu[0]) + dist(row_1-1, j_1,linkCoordTourCpu[0]) + dist(row_1,i_1-1,linkCoordTourCpu[0]);

                    int finalSelect = -1;
                    float optimiz = -INFINITY;
                    unsigned int node1 = (int)linkCoordTourCpu[0][j_1-1].current;
                    float localMinChange = this->minRadiusMap[0][node1];

                    for(int i = 0; i < 4; i++)
                    {
                        float opti = oldLength2 - newLength_3[i];

                        if(opti > 0 && opti > localMinChange )
                        {
                            optimiz = opti;
                            finalSelect = i;
                        }
                    }

                    if(finalSelect >= 0)
                    {

                        unsigned int node1 = (unsigned int)linkCoordTourCpu[0][j_1-1].current;
                        unsigned int node3 = (unsigned int)linkCoordTourCpu[0][i_1-1].current;
                        unsigned int node5 = (unsigned int)linkCoordTourCpu[0][row_1-1].current;

                        unsigned long long result = 0;
                        result = result | node3;
                        result = result << 16;
                        result = result | node5;

                        float codekopt = finalSelect; //WB.Q 2024 densityMap only mark the k value of k-opt, and mark which mode of k-exchange
                        codekopt = finalSelect * 100 + 3;

                        this->optCandidateMap[0][node1] = result;
                        this->densityMap[0][node1] = codekopt;


                    }// end if foundOpt jump out of 3rd loop


                }

            }
        }

    }//qiao end sequential 3-opt with first opt for one node




    //qiao add sequential 6-opt, using 6 loops of linkCoordTourCpu which is already ordered TSP tour
    void sequentialVariable4optBest(Grid<doubleLinkedEdgeForTSP>& linkCoordTourCpu, Grid<GLint> optPossibilitiesMap, Grid<GLint> fiveOptPossibilitiesMap, Grid<GLint> sixOptPossibilitiesMap)
    {

        //        cout << "Enter sequential variable k-opt " << ", order: " <<  this->grayValueMap[0][(int)linkCoordTourCpu[0][0].current] << endl;

        int N =  this->adaptiveMap.width;
        for(int j_1 = 1; j_1 < N; j_1++)
        {
            float oldLength0 = dist(j_1-1, j_1, linkCoordTourCpu[0]);

            //begin 2-opt
            for(int i_1 = j_1+2; i_1 < N; i_1++)
            {
                float oldLength1 = oldLength0;
                oldLength1 += dist(i_1-1, i_1, linkCoordTourCpu[0]);
                //                cout << "2-opt oldLen = " << oldLength1 << endl;

                //judge 2-opt if exist break and continue next j_1
                float newLength_2 = dist(j_1-1, i_1-1, linkCoordTourCpu[0]) + dist(j_1, i_1, linkCoordTourCpu[0]);

                if(newLength_2 < oldLength1)
                {

                    float optimization = oldLength1 - newLength_2;

                    unsigned int node1 = (unsigned int)linkCoordTourCpu[0][j_1-1].current;
                    unsigned int node3 = (unsigned int)linkCoordTourCpu[0][i_1-1].current;

                    float localMinChange = this->minRadiusMap[0][node1];

                    if(optimization > localMinChange)
                    {
                        this->minRadiusMap[0][node1]= optimization;

                        unsigned long long result = 0;
                        result = result | node3;
                        float codekopt = 2;

                        this->optCandidateMap[0][node1] = result;
                        this->densityMap[0][node1] = codekopt;
                    }
                }

                //begin 3-opt
                for(int row_1 = i_1+2; row_1 <N; row_1++)
                {
                    float oldLength2 = oldLength1;
                    oldLength2 += dist(row_1 -1, row_1, linkCoordTourCpu[0]);

                    //   cout << "3-opt oldLen = " << oldLength2 << endl;

                    //judge 3-opt if exist break and continue next j_1

                    double newLength_3[4];
                    newLength_3[0] = dist(j_1-1, i_1, linkCoordTourCpu[0]) + dist(row_1-1, i_1-1, linkCoordTourCpu[0]) + dist(row_1, j_1, linkCoordTourCpu[0]);
                    newLength_3[1] = dist(j_1-1, row_1-1,linkCoordTourCpu[0]) + dist(row_1, i_1-1, linkCoordTourCpu[0]) + dist(i_1,j_1,linkCoordTourCpu[0]);
                    newLength_3[2] = dist(j_1-1, i_1-1, linkCoordTourCpu[0]) + dist(row_1-1, j_1, linkCoordTourCpu[0]) + dist(row_1, i_1,linkCoordTourCpu[0]);
                    newLength_3[3] = dist(j_1-1, i_1, linkCoordTourCpu[0]) + dist(row_1-1, j_1,linkCoordTourCpu[0]) + dist(row_1,i_1-1,linkCoordTourCpu[0]);

                    int finalSelect = -1;
                    float optimiz = -INFINITY;
                    unsigned int node1 = (int)linkCoordTourCpu[0][j_1-1].current;
                    float localMinChange = this->minRadiusMap[0][node1];

                    for(int i = 0; i < 4; i++)
                    {
                        float opti = oldLength2 - newLength_3[i];

                        if(opti > 0 && opti > localMinChange )
                        {
                            optimiz = opti;
                            finalSelect = i;
                        }
                    }

                    if(finalSelect >= 0)
                    {

                        unsigned int node1 = (unsigned int)linkCoordTourCpu[0][j_1-1].current;
                        unsigned int node3 = (unsigned int)linkCoordTourCpu[0][i_1-1].current;
                        unsigned int node5 = (unsigned int)linkCoordTourCpu[0][row_1-1].current;

                        unsigned long long result = 0;
                        result = result | node3;
                        result = result << 16;
                        result = result | node5;

                        float codekopt = finalSelect; //WB.Q 2024 densityMap only mark the k value of k-opt, and mark which mode of k-exchange
                        codekopt = finalSelect * 100 + 3;

                        this->optCandidateMap[0][node1] = result;
                        this->densityMap[0][node1] = codekopt;


                    }// end if foundOpt jump out of 3rd loop


                    //begin 4-opt
                    for(int j = row_1 +2; j<N; j++)
                    {

                        float oldLength3 = oldLength2;
                        oldLength3 += dist(j-1, j, linkCoordTourCpu[0]);
                        //  cout << "4-opt oldLen = " << oldLength3 << endl;


                        //judge 4-opt if exist break and continue next j_1

                        float newLength_4;//25 is fixed for 4-opt
                        int array[8];
                        array[0] = j_1-1;
                        array[1] = j_1;
                        array[2] = i_1-1;
                        array[3] = i_1;
                        array[4] = row_1-1;
                        array[5] = row_1;
                        array[6] = j-1;
                        array[7] = j;

                        int finalSelect = -1;
                        float optimiz = -INFINITY;
                        unsigned int node1 = (int)linkCoordTourCpu[0][j_1-1].current;
                        float localMinChange = this->minRadiusMap[0][node1];


                        for(int opt = 0; opt < 200; opt +=8) //  4 edges 8 nodes
                        {
                            int nd1 = optPossibilitiesMap[0][opt] -1;
                            int nd2 = optPossibilitiesMap[0][opt+1] -1;
                            int nd3 = optPossibilitiesMap[0][opt+2] -1;
                            int nd4 = optPossibilitiesMap[0][opt+3] -1;
                            int nd5 = optPossibilitiesMap[0][opt+4] -1;
                            int nd6 = optPossibilitiesMap[0][opt+5] -1;
                            int nd7 = optPossibilitiesMap[0][opt+6] -1;
                            int nd8 = optPossibilitiesMap[0][opt+7] -1;

                            //  cout << ", nd1= " << nd1 << endl;

                            int optCandi = opt / 8;
                            //printf("GPU search nd1-8 %d, %d, %d, %d, %d, %d, %d, %d; optCandi=%d \n", nd1, nd2, nd3, nd4, nd5, nd6, nd7, nd8, optCandi);
                            newLength_4 = dist(array[nd1],array[nd2], linkCoordTourCpu[0]) + dist(array[nd3],array[nd4], linkCoordTourCpu[0]) + dist(array[nd5],array[nd6], linkCoordTourCpu[0])+ dist(array[nd7],array[nd8], linkCoordTourCpu[0]);

                            float opti = oldLength3 - newLength_4;
                            if(opti > 0 && opti > localMinChange)
                            {
                                finalSelect = optCandi;
                                optimiz = opti;

                            }
                        }

                        if(finalSelect >= 0)
                        {

                            unsigned int node1 = (unsigned int)linkCoordTourCpu[0][j_1-1].current;
                            unsigned int node3 = (unsigned int)linkCoordTourCpu[0][i_1-1].current;
                            unsigned int node5 = (unsigned int)linkCoordTourCpu[0][row_1-1].current;
                            unsigned int node7 = (unsigned int)linkCoordTourCpu[0][j-1].current;


                            unsigned long long result = 0;
                            result = result | node3;
                            result = result << 16;
                            result = result | node5;
                            result = result << 16;
                            result = result | node7;

                            float codekopt = finalSelect; //WB.Q 2024 densityMap only mark the k value of k-opt, and mark which mode of k-exchange
                            codekopt = finalSelect * 100 + 4;

                            this->optCandidateMap[0][node1] = result;
                            this->densityMap[0][node1] = codekopt;


                        } // end 4-opt

                    }

                }

            }

        }

    }//qiao end sequential 5-opt with first opt for one node



    //qiao add sequential 6-opt, using 6 loops of linkCoordTourCpu which is already ordered TSP tour
    void sequentialVariable6optBest(Grid<doubleLinkedEdgeForTSP>& linkCoordTourCpu, Grid<GLint> optPossibilitiesMap, Grid<GLint> fiveOptPossibilitiesMap, Grid<GLint> sixOptPossibilitiesMap)
    {

        cout << "Enter sequential variable k-opt " << ", order: " <<  this->grayValueMap[0][(int)linkCoordTourCpu[0][0].current] << endl;

        int N =  this->adaptiveMap.width;
        for(int j_1 = 1; j_1 < N; j_1++)
        {
            float oldLength0 = dist(j_1-1, j_1, linkCoordTourCpu[0]);
            //             cout << "oldLen0 = " << oldLength0 << endl;
            // unsigned int node3, node5, node7, node9, node11;

            //begin 2-opt
            for(int i_1 = j_1+2; i_1 < N; i_1++)
            {
                float oldLength1 = oldLength0;
                oldLength1 += dist(i_1-1, i_1, linkCoordTourCpu[0]);
                //                cout << "2-opt oldLen = " << oldLength1 << endl;

                //judge 2-opt if exist break and continue next j_1
                float newLength_2 = dist(j_1-1, i_1-1, linkCoordTourCpu[0]) + dist(j_1, i_1, linkCoordTourCpu[0]);

                if(newLength_2 < oldLength1)
                {

                    unsigned int node1 = (unsigned int)linkCoordTourCpu[0][j_1-1].current;
                    unsigned int node3 = (unsigned int)linkCoordTourCpu[0][i_1-1].current;


                    unsigned long long result = 0;
                    result = result | node3;
                    float codekopt = 2;

                    this->optCandidateMap[0][node1] = result;
                    this->densityMap[0][node1] = codekopt;

                    //                    cout <<"search select 2-opt: " << j_1   << ", tour order  " << this->grayValueMap[0][node1] << ", " << this->grayValueMap[0][node3] << endl;
                    break;
                }

                bool foundOpt = 0;

                //begin 3-opt
                for(int row_1 = i_1+2; row_1 <N; row_1++)
                {
                    float oldLength2 = oldLength1;
                    oldLength2 += dist(row_1 -1, row_1, linkCoordTourCpu[0]);

                    //                    cout << "3-opt oldLen = " << oldLength2 << endl;

                    //judge 3-opt if exist break and continue next j_1

                    double newLength_3[4];
                    newLength_3[0] = dist(j_1-1, i_1, linkCoordTourCpu[0]) + dist(row_1-1, i_1-1, linkCoordTourCpu[0]) + dist(row_1, j_1, linkCoordTourCpu[0]);
                    newLength_3[1] = dist(j_1-1, row_1-1,linkCoordTourCpu[0]) + dist(row_1, i_1-1, linkCoordTourCpu[0]) + dist(i_1,j_1,linkCoordTourCpu[0]);
                    newLength_3[2] = dist(j_1-1, i_1-1, linkCoordTourCpu[0]) + dist(row_1-1, j_1, linkCoordTourCpu[0]) + dist(row_1, i_1,linkCoordTourCpu[0]);
                    newLength_3[3] = dist(j_1-1, i_1, linkCoordTourCpu[0]) + dist(row_1-1, j_1,linkCoordTourCpu[0]) + dist(row_1,i_1-1,linkCoordTourCpu[0]);

                    int finalSelect = -1;
                    for(int i = 0; i < 4; i++)
                    {
                        float opti = oldLength2 - newLength_3[i];
                        if(opti > 0 )
                        {
                            finalSelect = i;
                            break;
                        }
                    }

                    if(finalSelect >= 0)
                    {

                        unsigned int node1 = (unsigned int)linkCoordTourCpu[0][j_1-1].current;
                        unsigned int node3 = (unsigned int)linkCoordTourCpu[0][i_1-1].current;
                        unsigned int node5 = (unsigned int)linkCoordTourCpu[0][row_1-1].current;

                        unsigned long long result = 0;
                        result = result | node3;
                        result = result << 16;
                        result = result | node5;

                        float codekopt = finalSelect; //WB.Q 2024 densityMap only mark the k value of k-opt, and mark which mode of k-exchange
                        codekopt = finalSelect * 100 + 3;

                        this->optCandidateMap[0][node1] = result;
                        this->densityMap[0][node1] = codekopt;

                        //                        cout << "search select 3-opt: " << j_1 << ", tour order " << this->grayValueMap[0][node1] << ", " << this->grayValueMap[0][node3]  << ", " << this->grayValueMap[0][node5]<< endl;


                        foundOpt = 1;
                        break;

                    }// end if foundOpt jump out of 3rd loop

                    //begin 4-opt
                    for(int j = row_1 +2; j<N; j++)
                    {

                        float oldLength3 = oldLength2;
                        oldLength3 += dist(j-1, j, linkCoordTourCpu[0]);
                        //  cout << "4-opt oldLen = " << oldLength3 << endl;


                        //judge 4-opt if exist break and continue next j_1

                        float newLength_4;//25 is fixed for 4-opt
                        int array[8];
                        array[0] = j_1-1;
                        array[1] = j_1;
                        array[2] = i_1-1;
                        array[3] = i_1;
                        array[4] = row_1-1;
                        array[5] = row_1;
                        array[6] = j-1;
                        array[7] = j;

                        int finalSelect = -1;

                        for(int opt = 0; opt < 200; opt +=8) //  4 edges 8 nodes
                        {
                            int nd1 = optPossibilitiesMap[0][opt] -1;
                            int nd2 = optPossibilitiesMap[0][opt+1] -1;
                            int nd3 = optPossibilitiesMap[0][opt+2] -1;
                            int nd4 = optPossibilitiesMap[0][opt+3] -1;
                            int nd5 = optPossibilitiesMap[0][opt+4] -1;
                            int nd6 = optPossibilitiesMap[0][opt+5] -1;
                            int nd7 = optPossibilitiesMap[0][opt+6] -1;
                            int nd8 = optPossibilitiesMap[0][opt+7] -1;

                            //  cout << ", nd1= " << nd1 << endl;

                            int optCandi = opt / 8;
                            //printf("GPU search nd1-8 %d, %d, %d, %d, %d, %d, %d, %d; optCandi=%d \n", nd1, nd2, nd3, nd4, nd5, nd6, nd7, nd8, optCandi);
                            newLength_4 = dist(array[nd1],array[nd2], linkCoordTourCpu[0]) + dist(array[nd3],array[nd4], linkCoordTourCpu[0]) + dist(array[nd5],array[nd6], linkCoordTourCpu[0])+ dist(array[nd7],array[nd8], linkCoordTourCpu[0]);

                            float opti = oldLength3 - newLength_4;
                            if(opti > 0)
                            {
                                finalSelect = optCandi;
                                break;

                            }
                        }

                        if(finalSelect >= 0)
                        {

                            unsigned int node1 = (unsigned int)linkCoordTourCpu[0][j_1-1].current;
                            unsigned int node3 = (unsigned int)linkCoordTourCpu[0][i_1-1].current;
                            unsigned int node5 = (unsigned int)linkCoordTourCpu[0][row_1-1].current;
                            unsigned int node7 = (unsigned int)linkCoordTourCpu[0][j-1].current;


                            unsigned long long result = 0;
                            result = result | node3;
                            result = result << 16;
                            result = result | node5;
                            result = result << 16;
                            result = result | node7;

                            float codekopt = finalSelect; //WB.Q 2024 densityMap only mark the k value of k-opt, and mark which mode of k-exchange
                            codekopt = finalSelect * 100 + 4;

                            this->optCandidateMap[0][node1] = result;
                            this->densityMap[0][node1] = codekopt;

                            //                            cout << "search select 4-opt: " << j_1 << ", tour order " << this->grayValueMap[0][node1] << ", " << this->grayValueMap[0][node3]
                            //                                    << ", " << this->grayValueMap[0][node5] << ", " << this->grayValueMap[0][node7] << endl;

                            foundOpt = 1;
                            break;
                        } // end 4-opt


                        //begin 5-opt
                        for(int i = j+2; i < N; i++)
                        {
                            float oldLength4 = oldLength3;
                            oldLength4 += dist(i-1, i, linkCoordTourCpu[0]);

                            // cout << "5-opt oldLen = " << oldLength4 << endl;

                            //judge 5-opt if exist break and continue next j_1
                            float newLength_5;
                            int array[10];
                            array[0] = j_1-1;
                            array[1] = j_1;
                            array[2] = i_1-1;
                            array[3] = i;
                            array[4] = row_1-1;
                            array[5] = row_1;
                            array[6] = j-1;
                            array[7] = j;
                            array[8] = i-1;
                            array[9] = i;

                            int finalSelect = -1;

                            for(int opt = 0; opt < 2080; opt +=10) //  4 edges 8 nodes
                            {
                                int nd1 = fiveOptPossibilitiesMap[0][opt] -1;
                                int nd2 = fiveOptPossibilitiesMap[0][opt+1] -1;
                                int nd3 = fiveOptPossibilitiesMap[0][opt+2] -1;
                                int nd4 = fiveOptPossibilitiesMap[0][opt+3] -1;
                                int nd5 = fiveOptPossibilitiesMap[0][opt+4] -1;
                                int nd6 = fiveOptPossibilitiesMap[0][opt+5] -1;
                                int nd7 = fiveOptPossibilitiesMap[0][opt+6] -1;
                                int nd8 = fiveOptPossibilitiesMap[0][opt+7] -1;
                                int nd9 = fiveOptPossibilitiesMap[0][opt+8] -1;
                                int nd10 = fiveOptPossibilitiesMap[0][opt+9] -1;

                                int optCandi = opt / 10;
                                // printf("GPU search nd1-8 %d, %d, %d, %d, %d, %d, %d, %d; optCandi=%d \n", nd1, nd2, nd3, nd4, nd5, nd6, nd7, nd8, optCandi);
                                newLength_5 = dist(array[nd1],array[nd2], linkCoordTourCpu[0]) + dist(array[nd3],array[nd4], linkCoordTourCpu[0])
                                        + dist(array[nd5],array[nd6], linkCoordTourCpu[0])+ dist(array[nd7],array[nd8], linkCoordTourCpu[0]) + dist(array[nd9],array[nd10], linkCoordTourCpu[0]);

                                float opti = oldLength4 - newLength_5;
                                if(opti > 0 )
                                {
                                    finalSelect = optCandi;
                                    break;
                                }
                            }

                            if(finalSelect >= 0)
                            {

                                unsigned int node1 = (unsigned int)linkCoordTourCpu[0][j_1-1].current;
                                unsigned int node3 = (unsigned int)linkCoordTourCpu[0][i_1-1].current;
                                unsigned int node5 = (unsigned int)linkCoordTourCpu[0][row_1-1].current;
                                unsigned int node7 = (unsigned int)linkCoordTourCpu[0][j-1].current;
                                unsigned int node9 = (unsigned int)linkCoordTourCpu[0][i-1].current;

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

                                this->optCandidateMap[0][node1] = result;
                                this->densityMap[0][node1] = codekopt;

                                //                                cout << "search select 5-opt: " << j_1  << ", tour order " << this->grayValueMap[0][node1] << ", " << this->grayValueMap[0][node3]
                                //                                        << ", " << this->grayValueMap[0][node5] << ", "
                                //                                        << this->grayValueMap[0][node7] << ", " << this->grayValueMap[0][node9] << endl;

                                foundOpt = 1;
                                break;
                            }

                            //begin 6-opt
                            for(int row = i+2; row < N; row++)
                            {

                                //judge 6-opt if exist break and continue next j_1
                                float oldLength = oldLength4;
                                oldLength += dist(row-1, row, linkCoordTourCpu[0]);

                                // cout << "6-opt oldLen = " << oldLength << endl;
                                //                                 cout << "6-opt row = " << row << ", i= " << i << ", j= " << j << ", row_1" << row_1 << ", i_1=" << i_1 << ", j_1= " << j_1 << endl;

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

                                for(int opt = 0; opt < 23220; opt +=12) //  6 edges 12 nodes 1935 sets 1935*12=23220 nodes
                                {
                                    int nd1 = sixOptPossibilitiesMap[0][opt] -1;
                                    int nd2 = sixOptPossibilitiesMap[0][opt+1] -1;
                                    int nd3 = sixOptPossibilitiesMap[0][opt+2] -1;
                                    int nd4 = sixOptPossibilitiesMap[0][opt+3] -1;
                                    int nd5 = sixOptPossibilitiesMap[0][opt+4] -1;
                                    int nd6 = sixOptPossibilitiesMap[0][opt+5] -1;
                                    int nd7 = sixOptPossibilitiesMap[0][opt+6] -1;
                                    int nd8 = sixOptPossibilitiesMap[0][opt+7] -1;
                                    int nd9 = sixOptPossibilitiesMap[0][opt+8] -1;
                                    int nd10 = sixOptPossibilitiesMap[0][opt+9] -1;
                                    int nd11 = sixOptPossibilitiesMap[0][opt+10] -1;
                                    int nd12 = sixOptPossibilitiesMap[0][opt+11] -1;

                                    int optCandi = opt / 12;

                                    newLength = dist(array[nd1],array[nd2], linkCoordTourCpu[0]) + dist(array[nd3],array[nd4], linkCoordTourCpu[0])
                                            + dist(array[nd5],array[nd6], linkCoordTourCpu[0])+ dist(array[nd7],array[nd8], linkCoordTourCpu[0])
                                            + dist(array[nd9],array[nd10], linkCoordTourCpu[0]) + dist(array[nd11],array[nd12], linkCoordTourCpu[0] );

                                    float opti = oldLength - newLength;
                                    if(opti > 0)
                                    {
                                        finalSelect = optCandi;

                                        break;
                                    }

                                }
                                if(finalSelect >= 0)
                                {

                                    unsigned int node1 = (unsigned int)linkCoordTourCpu[0][j_1-1].current;
                                    unsigned int node3 = (unsigned int)linkCoordTourCpu[0][i_1-1].current;
                                    unsigned int node5 = (unsigned int)linkCoordTourCpu[0][row_1-1].current;
                                    unsigned int node7 = (unsigned int)linkCoordTourCpu[0][j-1].current;
                                    unsigned int node9 = (unsigned int)linkCoordTourCpu[0][i-1].current;
                                    unsigned int node11 = (unsigned int)linkCoordTourCpu[0][row-1].current;

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

                                    this->optCandidateMap[0][node1] = result;
                                    this->densityMap[0][node1] = codekopt;

                                    //                                    cout <<"search select 6-opt: " << j_1  << ", tour order " << this->grayValueMap[0][node1] << ", " << this->grayValueMap[0][node3]
                                    //                                           << ", " << this->grayValueMap[0][node5] << ", " << this->grayValueMap[0][node7]
                                    //                                              << ", " << this->grayValueMap[0][node9] << ", " << this->grayValueMap[0][node11] << endl;

                                    foundOpt = 1;
                                    break;

                                }

                            }

                            if(foundOpt)
                                break;

                        }

                        if(foundOpt)
                            break;
                    }
                    if(foundOpt)
                        break;

                }

                if(foundOpt)
                    break;

            }

            //            if(foundOpt)
            //                continue;
        }

    }//qiao end sequential 5-opt with first opt for one node




    //! QWB 07/16 add mark network sequentice as well reload doubly linked rout + coordinate
    void markNetLinkSequenceReloadRoutCoord(PointCoord ps, int direction,
                                            bool markActive, Grid<doubleLinkedEdgeForTSP>& linkCoordTourCpu){

        int N =  this->adaptiveMap.width;

        if(this->networkLinks[ps[1]][ps[0]].numLinks == 2)
        {

            this->grayValueMap[ps[1]][ps[0]] = 0;// grayValueMap stores the order position
            // reload double linked order tour
            linkCoordTourCpu[0][0].currentCoord[0] = this->adaptiveMap[ps[1]][ps[0]][0];
            linkCoordTourCpu[0][0].currentCoord[1] = this->adaptiveMap[ps[1]][ps[0]][1];

            //                        cout << "linkCoordTourCpu[0][0].currentCoord[0] = " << linkCoordTourCpu[0][0].currentCoord[0]  << endl;
            //                        cout << "linkCoordTourCpu[0][0].currentCoord[1] = " << linkCoordTourCpu[0][0].currentCoord[1]  << endl;
            //                        printf("pp1 (%f, %f)\n",this->adaptiveMap[ps[1]][ps[0]][0], this->adaptiveMap[ps[1]][ps[0]][1]);
            //                        cout << "this->grayValueMap[ps[1]][ps[0]= " << this->grayValueMap[ps[1]][ps[0]] << endl;


            PointCoord pLinkOfNode;
            this->networkLinks[ps[1]][ps[0]].get(direction, pLinkOfNode);
            PointCoord pco(-1, -1);
            pco[0] = (int)pLinkOfNode[0];
            pco[1] = (int)pLinkOfNode[1];
            this->grayValueMap[pco[1]][pco[0]] = 1;// qiao for test how many nodes being evaluated
            //                        printf("pco (%f, %f)\n",this->adaptiveMap[pco[1]][pco[0]][0], this->adaptiveMap[pco[1]][pco[0]][1]);
            //                        cout << "this->grayValueMap[pco[1]][pco[0]= " << this->grayValueMap[pco[1]][pco[0]] << endl;
            // reload double linked order tour
            linkCoordTourCpu[0][0].current = (int)ps[0];
            linkCoordTourCpu[0][1].currentCoord[0] = this->adaptiveMap[pco[1]][pco[0]][0];
            linkCoordTourCpu[0][1].currentCoord[1] = this->adaptiveMap[pco[1]][pco[0]][1]; //qiao 这里限定了2维坐标
            if(markActive)
                this->activeMap[pco[1]][pco[0]] = 1; // set these grayValue / 2 == 0 as searcher

            if(N == 2)
                return;
            else if(this->networkLinks[pco[1]][pco[0]].numLinks == 2)
            {
                PointCoord pLinkOfNode2;
                PointCoord pco2(-1, -1);
                this->networkLinks[pco[1]][pco[0]].get(0, pLinkOfNode2);
                pco2[0] = (int)pLinkOfNode2[0];
                pco2[1] = (int)pLinkOfNode2[1];
                if(pco2 == ps){
                    this->networkLinks[pco[1]][pco[0]].get(1, pLinkOfNode2);
                    pco2[0] = (int)pLinkOfNode2[0];
                    pco2[1] = (int)pLinkOfNode2[1];
                }

                // reload double linked order tour
                linkCoordTourCpu[0][1].current = pco[0];


                PointCoord pco2Avant;
                pco2Avant = pco;
                int k = 2; // grayValueMap

                for(int i = 2; i < N; i++){

                    PointCoord pLinkOfNode3;
                    PointCoord pco3(-1, -1);

                    this->networkLinks[pco2[1]][pco2[0]].get(0, pLinkOfNode3);
                    pco3[0] = (int)pLinkOfNode3[0];
                    pco3[1] = (int)pLinkOfNode3[1];
                    if(pco3 == pco2Avant){
                        this->networkLinks[pco2[1]][pco2[0]].get(1, pLinkOfNode3);
                        pco3[0] = (int)pLinkOfNode3[0];
                        pco3[1] = (int)pLinkOfNode3[1];
                    }


                    if(markActive)
                        this->activeMap[pco2[1]][pco2[0]] = k%2;// set these k % 2 == 0 as searcher

                    this->grayValueMap[pco2[1]][pco2[0]] = k ++;

                    //                        printf("pco (%f, %f)\n",this->adaptiveMap[pco2[1]][pco2[0]][0], this->adaptiveMap[pco2[1]][pco2[0]][1]);
                    //                        cout << "this->grayValueMap[pco2[1]][pco2[0]= " << this->grayValueMap[pco2[1]][pco2[0]] << endl;

                    //qiao 这里只适用于2维坐标点
                    linkCoordTourCpu[0][i].currentCoord[0] = this->adaptiveMap[pco2[1]][pco2[0]][0];
                    linkCoordTourCpu[0][i].currentCoord[1] = this->adaptiveMap[pco2[1]][pco2[0]][1];
                    linkCoordTourCpu[0][i].current = pco2[0];

                    pco2Avant = pco2;
                    pco2 = pco3;

                }
            }
        }
        else
            cout << "Error markNetWorkSequence: NO LINKS..." << endl;

    }// mark TSP sequence without recursive function

    //! QWB add evaluate weight for a single 2-connected network without recursive function
    //! 1D network
    template <class Distance>
    float evaluateWeightOfTSP(Distance dist, int& numTraversed){

        numTraversed = 0;
        int N =  this->adaptiveMap.width;

        this->fixedMap.resetValue(0); // to test how many nodes being evaluated
        double totalWeight = 0;

        PointCoord ps(0, 0);
        if(this->networkLinks[ps[1]][ps[0]].numLinks == 2){

            this->fixedMap[ps[1]][ps[0]] = 1;// qiao for test how many nodes being evaluated

            PointCoord pLinkOfNode;
            this->networkLinks[ps[1]][ps[0]].get(1, pLinkOfNode);
            PointCoord pco(-1, -1);
            pco[0] = (int)pLinkOfNode[0];
            pco[1] = (int)pLinkOfNode[1];
            this->fixedMap[pco[1]][pco[0]] = 1;// qiao for test how many nodes being evaluated

            //                        cout << "evaluate couple " << ps[0] << ", " << ps[1] << ", " << pco[0] << ", " << pco[1] << endl;
            //                        cout << "adaptivemap " << this->adaptiveMap[0][0][0] << ", " << this->adaptiveMap[0][0][1] << endl; //nn.adaptiveMap[0][i]
            //                        cout << "adaptivemap " << this->adaptiveMap[0][3][0] << ", " << this->adaptiveMap[0][3][1] << endl; //nn.adaptiveMap[0][i]
            totalWeight += dist(ps, pco, *this, *this);
            //                        cout << "dist = " << totalWeight << endl;
            numTraversed += 1;


            if(N == 2)
                return totalWeight;
            else if(this->networkLinks[pco[1]][pco[0]].numLinks == 2){

                PointCoord pLinkOfNode2;
                PointCoord pco2(-1, -1);
                this->networkLinks[pco[1]][pco[0]].get(0, pLinkOfNode2);
                pco2[0] = (int)pLinkOfNode2[0];
                pco2[1] = (int)pLinkOfNode2[1];
                if(pco2 == ps){
                    this->networkLinks[pco[1]][pco[0]].get(1, pLinkOfNode2);
                    pco2[0] = (int)pLinkOfNode2[0];
                    pco2[1] = (int)pLinkOfNode2[1];
                }
                // cout << "evaluate couple " << pco[0] << ", " << pco[1] << ", " << pco2[0] << ", " << pco2[1] << endl;
                // cout << "evaluate couple " << pco[0] << ", " << pco[1] << ", " << pco2[0] << ", " << pco2[1] << endl;
                // cout << "adaptivemap " << this->adaptiveMap[pco[1]][pco[0]] << ", " << this->adaptiveMap[pco2[1]][pco2[0]]<< endl;

                totalWeight += dist(pco, pco2, *this, *this);
                numTraversed += 1;

                PointCoord pco2Avant;
                pco2Avant = pco;

                //qiao add for stop
                int numWhile = 0;

                //                while(pco2 != ps)
                while(pco2 != ps && numWhile < N+10)
                {
                    numWhile += 1;
                    if (numWhile > N)
                        cout << "Error evaluate numWhile > N ............ " << endl;

                    //                    cout << "evaluate length pco2 " << pco2 << endl;

                    PointCoord pLinkOfNode3;
                    PointCoord pco3(-1, -1);
                    if(this->networkLinks[pco2[1]][pco2[0]].numLinks == 2){
                        this->networkLinks[pco2[1]][pco2[0]].get(0, pLinkOfNode3);
                        pco3[0] = (int)pLinkOfNode3[0];
                        pco3[1] = (int)pLinkOfNode3[1];
                        if(pco3 == pco2Avant){
                            this->networkLinks[pco2[1]][pco2[0]].get(1, pLinkOfNode3);
                            pco3[0] = (int)pLinkOfNode3[0];
                            pco3[1] = (int)pLinkOfNode3[1];
                        }
                        if(pco3 == pco2Avant){
                            cout << "open circle . ps: " << ps[0] << ", " <<  ps[1] << endl;
                            break;}

                        //                        cout << "evaluate couple " << pco2[0] << ", " << pco2[1] << ", " << pco3[0] << ", " << pco3[1] << endl;
                        //                        cout << "adaptivemap " << this->adaptiveMap[pco2[1]][pco2[0]] << ", " << this->adaptiveMap[pco3[1]][pco3[0]] << endl;

                        totalWeight += dist(pco2, pco3, *this, *this);
                        numTraversed += 1;
                        this->fixedMap[pco2[1]][pco2[0]] = 1;// qiao for test how many nodes being evaluated

                        pco2Avant = pco2;
                        pco2 = pco3;
                    }
                    else
                    {
                        cout << " Error evaluate distance, no links..." << endl;
                        break;
                    }

                }
            }
            else
                cout << "Error evaluate distance, no links..." << endl;
        }
        else
            cout << "Error evaluate distance, no links..." << endl;

        cout <<"Evaluation TSP length >> num of nodes evaluated inside: " << numTraversed << endl;
        if(numTraversed != N)
            cout << "Error evaluate TSP tour, tour cuted. " << endl;

        return totalWeight;
    }// evaluate total distance of tsp without recursive function


private:

    //!QWB: this function computes the total distance of a netLink, closed or open 2D ring, spanning tree
    //! every node should be connected at least once
    template <class Distance>
    void evaluateWeightOfTspRecursive(PointCoord psBegin, PointCoord psAvant, PointCoord ps, double& totalWeight)
    {

        this->fixedMap[ps[1]][ps[0]] += 1; // qiao for test how many nodes being evaluated

        Distance dist;
        Point2D pLinkOfNode;
        this->networkLinks[ps[1]][ps[0]].get(0, pLinkOfNode);
        PointCoord pco(-1);
        pco[0] = (int)pLinkOfNode[0];
        pco[1] = (int)pLinkOfNode[1];

        if(pco == psAvant){
            this->networkLinks[ps[1]][ps[0]].get(1, pLinkOfNode);
            pco[0] = (int)pLinkOfNode[0];
            pco[1] = (int)pLinkOfNode[1];
        }

        totalWeight += dist(ps, pco, *this, *this);

        if(pco == psBegin){
            return;
        }
        else
            evaluateWeightOfTspRecursive<Distance>(psBegin, ps, pco, totalWeight);

    }

    //!QWB: this function computes the total distance of a netLink, closed or open 2D ring, spanning tree
    //! every node should be connected at least once
    template <class Distance>
    void evaluateWeightOfSingleTreeRecursive(PointCoord ps, vector<PointCoord>& nodeAlreadyTraversed, double& totalWeight, Distance dist)
    {

        PointCoord pInitial(-1);

        //        this->grayValueMap[ps[1]][ps[0]] += 1;// for test if all node are traversed just once
        nodeAlreadyTraversed.push_back(ps);

        for (int i = 0; i < this->networkLinks[ps[1]][ps[0]].numLinks; i ++){

            PointCoord pLinkOfNode;
            this->networkLinks[ps[1]][ps[0]].get(i, pLinkOfNode);

            if(pLinkOfNode == pInitial)
                continue;
            else{

                PointCoord pco(0);
                pco[0] = (int)pLinkOfNode[0];
                pco[1] = (int)pLinkOfNode[1];

                // compare if the current pCoord is already be traversed
                bool traversed = 0;
                for (int k = 0; k < nodeAlreadyTraversed.size(); k ++){
                    PointCoord pLinkTemp(-1);
                    pLinkTemp = nodeAlreadyTraversed[k];
                    if (pco[0] == pLinkTemp[0] && pco[1] == pLinkTemp[1])
                        traversed = 1;
                }

                if(traversed)
                    continue;

                else{
                    totalWeight += dist(ps, pco, *this, *this);
                    nodeAlreadyTraversed.push_back(pco);
                    if(networkLinks[pco[1]][pco[0]].numLinks != 0){
                        evaluateWeightOfSingleTreeRecursive<Distance>(pco, nodeAlreadyTraversed, totalWeight, dist);
                    }
                }

            }
        }

    }
};


//wb.Q add class specifily for TSP nodes
template <class BufferLink, class Point>
class NeuralNetTSP : public NeuralNetEMST<Point, GLfloatP>{

public:
    typedef Point point_type;

    //wb.Q add christofids
    Grid<GLint> oddsNodeMap;

    //wb.Q 2022 adds for christofids odds's correpondencemap
    Grid<PointCoord> oddsCorrespondenceMap;
    Grid<GLfloatP> distanceMap;
    Grid<BufferLinkPcoChrist> networkLinks; //store odds nodes corres
    Grid<BufferLinkPcoEulerTour> eulerTourLinks; //store euler links
    Grid<GLint> disjointSetMap; // store small TSP result in the same O(N) size
    Grid<GLint> flagMap; // act as a flag to mark which node has been occupied before

public:
    DEVICE_HOST NeuralNetTSP() {}


    void resize(int w, int h){

        oddsNodeMap.resize(w, h);
        NeuralNetEMST<Point, GLfloatP>::resize(w, h);
    }

    void freeMem(){
        oddsNodeMap.freeMem();
        NeuralNetEMST<Point, GLfloatP>::freeMem();
    }


    void gpuResize(int w, int h){
        oddsNodeMap.gpuResize(w, h);
        NeuralNetEMST<Point, GLfloatP>::gpuResize(w, h);
    }

    void clone(NeuralNetTSP& nn) {
        oddsNodeMap.clone(nn.oddsNodeMap);
        NeuralNetEMST<Point, GLfloatP>::clone(nn);
    }

    void gpuClone(NeuralNetTSP& nn) {
        oddsNodeMap.gpuClone(nn.oddsNodeMap);
        NeuralNetEMST<Point, GLfloatP>::gpuClone(nn);
    }
    void setIdentical(NeuralNetTSP& nn) {
        oddsNodeMap.setIdentical(nn.oddsNodeMap);
        NeuralNetEMST<Point, GLfloatP>::setIdentical(nn);
    }

    void gpuSetIdentical(NeuralNetTSP& nn) {
        oddsNodeMap.gpuSetIdentical(nn.soddsNodeMap);
        NeuralNetEMST<Point, GLfloatP>::gpuSetIdentical(nn);
    }

    void gpuCopyHostToDevice(NeuralNetTSP & gpuNeuralNetLinks){

        this->oddsNodeMap.gpuCopyHostToDevice(gpuNeuralNetLinks.oddsNodeMap);
        this->NeuralNetEMST.gpuCopyHostToDevice(gpuNeuralNetLinks);
    }

    void gpuCopyDeviceToHost(NeuralNetTSP & gpuNeuralNetLinks){

        this->oddsNodeMap.gpuCopyDeviceToHost(gpuNeuralNetLinks.oddsNodeMap);
        this->NeuralNetEMST.gpuCopyDeviceToHost(gpuNeuralNetLinks);
    }

    void gpuCopyDeviceToDevice(NeuralNetTSP & gpuNeuralNetLinks){

        this->oddsNodeMap.gpuCopyDeviceToDevice(gpuNeuralNetLinks.oddsNodeMap);
        this->NeuralNetEMST.gpuCopyDeviceToDevice(gpuNeuralNetLinks);
    }

};

//! wenbao Qiao 060716 add for using static buffer links, Point2D has operations like ofstream
typedef NeuralNetEMST<BufferLinkPointCoord, Point2D> NetLinkPointCoord;
//typedef NetLinkPointCoord NNLinkPoint2D;
typedef NeuralNetEMST<BufferLinkPointCoord, Point3D> MatNetLinks;
typedef NeuralNetEMST<BufferLinkPointCoord, PointCoord> NetLinkPointInt;

}//namespace componentsEMST
#endif // NEURALNET_EMST_H
