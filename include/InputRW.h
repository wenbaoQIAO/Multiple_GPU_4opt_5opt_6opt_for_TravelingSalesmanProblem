#ifndef INPUTRW_H
#define INPUTRW_H
/*
 ***************************************************************************
 *
 * Author : Wenbao Qiao
 * Creation date : December. 2016
 *
 ***************************************************************************
 */
#include <iostream>
#include <fstream>

#include <QtGui/QImage>

#include "macros_cuda.h"
#include "GridOfNodes.h"
#include "Node.h"
#include "NeuralNet.h"

#define SFX_TSP_FILE  ".tsp"
#define SFX_2D_FILE ".2d"
#define SFX_3DImage_FILE  ".points"
#define SFX_3D_FILE  ".3d"
#define SFX_WORLD_TSP "world"
#define WORLD_RADIUS 63.0 // 6378388.0


//! qiao add for LR-check
#define SFX_STEREO_GT_MAP_L "disp0GT.pfm"
#define SFX_STEREO_GT_MAP_R "disp1GT.pfm"

//! wb.Q extern variables specialized for world.tsp
#define SIZE_WORLD_TSP 1904711
GLfloat Lattitude_World[SIZE_WORLD_TSP];
GLfloat Longitude_World[SIZE_WORLD_TSP];

using namespace std;
using namespace components;

namespace components
{

template <class Point,
          class Value>
class InputRW
{
public:

    //! wb.Q add March 2015
    int getPos(string str){

        std::size_t pos = str.find('.',  0);
        if (pos == std::string::npos)
            pos = str.length();

        //! no-matter aaa_xx.xx or aaa.xx or aaa_xx,  aaa.xx_xx
        //! we can always get the first three aaa and the same pos =  = 3
        std::size_t posTiret = str.find('_',  0);

        if(pos > posTiret)
            pos = posTiret; //! ensure to get the aaa in any name format aaa_xxx or aaa_xxx.lll

        return pos;
    }

    // wb.Q change lattitude and longitude to sphere coordinate, copy from www
    GLdouble toRadius(GLdouble deg){
        return deg / 180 * acos(-1);
    }
    void sphereCoordinate(GLfloat radius, GLfloat lat, GLfloat lng,
                          GLfloat &x, GLfloat &y, GLfloat &z){
        lat = toRadius(lat);
        lng = toRadius(lng);

        x = radius * cos(lat) * cos(lng);
        y = radius * cos(lat) * sin(lng);
        z = radius * sin(lat);
    }

    bool readWorldLLtitude(string file, NeuralNet<Point2D, Value>& nn, Point2D& pMin, Point2D& pMax ) {
        cout << "error read world.tsp " << endl;
        return readWorldLLtitude(file, nn, pMin[0], pMin[1], pMin[2] , pMax[0], pMax[1], pMax[2]);
    }

    /* wb.Q 08/June/2019 adds to chansfer world.tsp to world.3d
       use three arrays to finish this work:
       1st one holds the latitude of all cities; 2nd one holds the longitude of all cities, for compute travel cost between two cities
       3rd one holds the 3D points of all cities for build EMST, NN.adaptiveMap.
       read every line of world.tsp, change it to 3d point and store it into world.3d */
    bool readWorldLLtitude(string file, NeuralNet<Point3D, Value>& nn, Point3D& pMin, Point3D& pMax ) {
        return readWorldLLtitude(file, nn, pMin[0], pMin[1], pMin[2] , pMax[0], pMax[1], pMax[2]);
    }


    // wb.Q read lattitude and longitude to two arrays of size N
    // use two globe variables to store the lattitude and longitude and make least change of current codes
    bool readWorldLLtitude(string fileName, NeuralNet<Point, Value>& nn,
                           GLfloatP& min_x, GLfloatP& min_y, GLfloatP& min_z,
                           GLfloatP& max_x, GLfloatP& max_y, GLfloatP& max_z){

        cout << "read ll from " << fileName << endl;

        ifstream fi;
        // count total size of input points
        fi.open(fileName.c_str());
        if (!fi)
        {
            cout << "erreur ouverture file: " << fileName << endl;
            return false;
        }
        else
        {
            char buf[256];
            char *ptr;

            char str[256];
            char str_2[10];

            char name[50];
            int size = 0;

            cout << "ouvire correctly file " << fileName << endl;

            fi.getline(buf, 256);

            if ( (ptr = strstr(buf, "NAME")) != NULL )
            {
                sscanf(ptr, "%s%s%s", str, str_2, &name);
            }
            else
            {
                cout << "I am here qiao 1 " << endl;
                return(0);
            }

            for (int l = 1; l <= 4; l++)
            {
                fi.getline(buf, 256);
                if (!fi)
                    return(0);
            }

            if ( (ptr = strstr(buf, "DIMENSION")) != NULL )
            {
                sscanf(ptr, "%s%s%d", str, str_2, &size);
            }
            else
            {
                return(0);
            }

            if (size == SIZE_WORLD_TSP)
            {

                ofstream ofile;
                ofile.open("world.3d");
                ofile << size << endl;
                //                nn.resize(size, 1);

                for (int l = 1; l <= 2; l++)
                {
                    fi.getline(buf, 256);
                    if (!fi)
                        return(0);
                }

                int num = 0;
                float lat, lon;

                // Lecture de la matrice de pixels ligne par ligne
                for (int i = 0; i < size; i++)
                {

                    fi >> num;
                    fi >> lat;
                    fi >> lon;

                    Lattitude_World[i] = lat;
                    Longitude_World[i] = lon;

                    float cx, cy, cz;

                    max_x = -INFINITY;
                    max_y = -INFINITY;
                    max_z = -INFINITY;
                    min_x = +INFINITY;
                    min_y = +INFINITY;
                    min_z = +INFINITY;

                    sphereCoordinate(WORLD_RADIUS, lat, lon, cx, cy, cz);

                    Point city(cx, cy, cz);

                    //                    cout << cx << ", " << cy << ", " << cz << endl;
                    ofile << cx << " " << cy << " " << cz << endl;
                    //                    nn.adaptiveMap[0][i] = city;

                    if (cx >= max_x)
                        max_x = cx;
                    if (cy >= max_y)
                        max_y = cy;
                    if (cz >= max_z)
                        max_z = cz;
                    if (cx < min_x)
                        min_x = cx;
                    if (cy < min_y)
                        min_y = cy;
                    if (cz < min_z)
                        min_z = cz;
                }

                ofile.close();
            }
            else
            {
                cout << "error read world cities: num = 0." << endl;
                return 0;
            }

        }

        fi.close();
        return true;
    }

    //qiao add to read opt possibilites 2024
    bool readOptPossibilites(int functionModeChoice, Grid<GLint>& optPossibilitiesMap) {

        string fileName;
        ifstream fi;

        //wb.Q 2024 add read world opt possibilites
        if(functionModeChoice == 4){
            fileName = "4optPossibilities.txt";

            fi.open(fileName.c_str());
            if (!fi)
            {
                cout << "erreur ouverture file: " << fileName << endl;
                return false;
            }
            else
            {
                int size = 0; // nodes of 4 edges of 208 4 edges
                fi >> size;
                cout << "Read " << size << " 4-opt possibilites. " << endl;
                size = size * 8;
                optPossibilitiesMap.resize(size,1);
                for(int i = 0; i < size; i++){

                    int point = -1;
                    fi >> point;

                    optPossibilitiesMap[0][i] = point;
                }
                if(optPossibilitiesMap.width != size)
                    cout << "Error read opt possibilites " << endl;
            }

            //            //qiao only for test
            //            cout << "check read opt possibilites into grid >>>>>>>>>>>> " << endl;
            //            for(int i=0; i < optPossibilitiesMap.width; i+=8)
            //            {
            //                cout << optPossibilitiesMap[0][i] << ", " <<optPossibilitiesMap[0][i+1] << ", "
            //                    << optPossibilitiesMap[0][i+2] << ", " << optPossibilitiesMap[0][i+3] << ", "
            //                    << optPossibilitiesMap[0][i+4] << ", " << optPossibilitiesMap[0][i+5] << ", "
            //                    << optPossibilitiesMap[0][i+6] << ", " << optPossibilitiesMap[0][i+7] << ", "
            //                    << endl;
            //            }

        }
        //wb.Q 2024 add read world opt possibilites
        if(functionModeChoice == 5){

            fileName = "5optPossibilities.txt";

            // read opt possibilites
            fi.open(fileName.c_str());
            if (!fi)
            {
                cout << "erreur ouverture file: " << fileName << endl;
                return false;
            }
            else
            {
                int size = 0; //
                fi >> size;
                cout << "Read " << size  << " 5-opt possibilites. " << endl;
                size = size * 10;
                optPossibilitiesMap.resize(size,1);
                for(int i = 0; i < size; i++){

                    int point = -1;
                    fi >> point;

                    optPossibilitiesMap[0][i] = point;

                }
                if(optPossibilitiesMap.width != size)
                    cout << "Error read opt possibilites " << endl;

            }

            //            //qiao only for test
            //            cout << "check read opt possibilites into grid >>>>>>>>>>>> " << endl;
            //            for(int i=0; i < optPossibilitiesMap.width; i+=10)
            //            {
            //                cout << optPossibilitiesMap[0][i] << ", " <<optPossibilitiesMap[0][i+1] << ", "
            //                    << optPossibilitiesMap[0][i+2] << ", " << optPossibilitiesMap[0][i+3] << ", "
            //                    << optPossibilitiesMap[0][i+4] << ", " << optPossibilitiesMap[0][i+5] << ", "
            //                    << optPossibilitiesMap[0][i+6] << ", " << optPossibilitiesMap[0][i+7] << ", "
            //                    << optPossibilitiesMap[0][i+8] << ", " << optPossibilitiesMap[0][i+9] << ", "
            //                    << endl;
            //            }

        }
        //wb.Q 2024 add read world opt possibilites
        if(functionModeChoice == 6){

            fileName = "6optPossibilities.txt";

            fi.open(fileName.c_str());
            if (!fi)
            {
                cout << "erreur ouverture file: " << fileName << endl;
                return false;
            }
            else
            {
                int size = 0; //
                fi >> size;
                cout << "Read " << size<< " 6-opt possibilites. " << endl;
                size = size * 12;
                optPossibilitiesMap.resize(size,1);
                for(int i = 0; i < size; i++){

                    int point = -1;
                    fi >> point;

                    optPossibilitiesMap[0][i] = point;

                }

                if(optPossibilitiesMap.width != size)
                    cout << "Error read opt possibilites " << endl;
            }
            //            //qiao only for test
            //            cout << "check read opt possibilites into grid >>>>>>>>>>>> " << endl;
            //            for(int i=0; i < optPossibilitiesMap.width; i+=12)
            //            {
            //                cout << optPossibilitiesMap[0][i] << ", " <<optPossibilitiesMap[0][i+1] << ", "
            //                    << optPossibilitiesMap[0][i+2] << ", " << optPossibilitiesMap[0][i+3] << ", "
            //                    << optPossibilitiesMap[0][i+4] << ", " << optPossibilitiesMap[0][i+5] << ", "
            //                    << optPossibilitiesMap[0][i+6] << ", " << optPossibilitiesMap[0][i+7] << ", "
            //                    << optPossibilitiesMap[0][i+8] << ", " << optPossibilitiesMap[0][i+9] << ", "
            //                    << optPossibilitiesMap[0][i+10] << ", " << optPossibilitiesMap[0][i+11] << ", "
            //                    << endl;
            //            }

        }

        fi.close();

        return 0;
    }



    //qiao add to read opt possibilites 2024
    bool readOptPossibilites(int functionModeChoice, Grid<GLint>& optPossibilitiesMap, Grid<GLint>& fiveOptPossibilitiesMap, Grid<GLint>& sixOptPossibilitiesMap) {

        string fileName, fileName5, fileName6;
        ifstream fi, fi5, fi6;

        //wb.Q 2024 add read world opt possibilites
        fileName = "4optPossibilities.txt";

        fi.open(fileName.c_str());
        if (!fi)
        {
            cout << "erreur ouverture file: " << fileName << endl;
            return false;
        }
        else
        {
            int size = 0; // nodes of 4 edges of 208 4 edges
            fi >> size;
            cout << "Read " << size << " 4-opt possibilites. " << endl;
            size = size * 8;
            optPossibilitiesMap.resize(size,1);
            for(int i = 0; i < size; i++){

                int point = -1;
                fi >> point;

                optPossibilitiesMap[0][i] = point;
            }
            if(optPossibilitiesMap.width != size)
                cout << "Error read opt possibilites " << endl;
        }



        //wb.Q 2024 add read world opt possibilites

        fileName5 = "5optPossibilities.txt";
        // read opt possibilites
        fi5.open(fileName5.c_str());
        if (!fi5)
        {
            cout << "erreur ouverture file: " << fileName5 << endl;
            return false;
        }
        else
        {
            int size = 0; //
            fi5 >> size;
            cout << "Read " << size  << " 5-opt possibilites. " << endl;
            size = size * 10;
            fiveOptPossibilitiesMap.resize(size,1);
            for(int i = 0; i < size; i++){

                int point = -1;
                fi5 >> point;

                fiveOptPossibilitiesMap[0][i] = point;

            }
            if(fiveOptPossibilitiesMap.width != size)
                cout << "Error read opt possibilites " << endl;

        }


        //wb.Q 2024 add read world opt possibilites

        fileName6 = "6optPossibilities.txt";

        fi6.open(fileName6.c_str());
        if (!fi6)
        {
            cout << "erreur ouverture file: " << fileName6 << endl;
            return false;
        }
        else
        {
            int size = 0; //
            fi6 >> size;
            cout << "Read " << size<< " 6-opt possibilites. " << endl;
            size = size * 12;
            sixOptPossibilitiesMap.resize(size,1);
            for(int i = 0; i < size; i++){

                int point = -1;
                fi6 >> point;

                sixOptPossibilitiesMap[0][i] = point;

            }

            if(sixOptPossibilitiesMap.width != size)
                cout << "Error read opt possibilites " << endl;
        }

        fi.close();
        fi5.close();
        fi6.close();

        return 0;
    }


    bool readFile(string& file, NeuralNet<Point, Value>& nn, Point& pMin, Point& pMax ) {

        int pos = getPos(file);
        string file_sub, str_sub;

        file_sub = file.substr(0, pos);
        str_sub = file.substr(pos, (file.length() - pos));

//        cout << "Change file name " << file_sub << endl;
        file = file_sub;// change file name from xxx.tsp to xxx

        //wb.Q 26Jan2019 add read world  TSP
        if(file_sub.compare(SFX_WORLD_TSP) == 0){

            cout << ">>>>>Dealing with world.tsp input: " << endl;

            // read lattitude and longitude to 3D point cloud, write to world.3d file
            // to do a function read lattitude and longitude to 3D points
            string fileName = file_sub;
            fileName.append(SFX_3D_FILE);

            ifstream fi;
            // count total size of input points
            fi.open(fileName.c_str());
            if (!fi)
            {
                cout << "Does not exist world.3d file, read world.tsp and produce world.3d now " << endl;

                // read lattituede and longitude to two arrays and chansfer ll to cordinate, store it to nn.adaptiveMap, write into world.3d
                fileName = file_sub;
                fileName.append(SFX_TSP_FILE);
                readWorldLLtitude(fileName, nn, pMin, pMax);

                read3DUniform(file, nn, pMin, pMax);

            }
            else
            {
                cout << "world.3d file exists, read world.3d now " << endl;

                // read 3D point to 3D TSP
                read3DUniform(file, nn, pMin, pMax);
            }

            fi.close();

            return false;
        }
        else{

            if (str_sub.compare(SFX_3DImage_FILE) == 0)
                read3DImage(file, nn, pMin, pMax);
            //! wb.Q add to read uniform points
            else if (str_sub.compare(SFX_3D_FILE) == 0)
                read3DUniform(file, nn, pMin, pMax);
            else if (str_sub.compare(SFX_2D_FILE) == 0)
                read2DUniform(file, nn, pMin, pMax);
            else
                readTSP(file, nn, pMin, pMax);
        }
        return false;
    }

    bool readTSP(string file, NeuralNet<Point2D, Value>& nn, Point2D& pMin, Point2D& pMax ) {
        return readTSP(file, nn, pMin[0], pMin[1], pMax[0], pMax[1]);
    }

    bool readTSP(string file, NeuralNet<Point3D, Value>& nn, Point3D& pMin, Point3D& pMax ) {
        pMin[2] = 0;
        pMax[2] = 0;
        return readTSP(file, nn, pMin[0], pMin[1], pMax[0], pMax[1]);
    }

    bool read2DUniform(string file, NeuralNet<Point2D, Value>& nn, Point2D& pMin, Point2D& pMax ) {
        return readUniform(file, nn, pMin[0], pMin[1], pMax[0], pMax[1]);
    }

    bool read2DUniform(string file, NeuralNet<Point3D, Value>& nn, Point3D& pMin, Point3D& pMax) {
        pMin[2] = 0;
        pMax[2] = 0;
        return readUniform(file, nn, pMin[0], pMin[1], pMax[0], pMax[1]);
    }

    bool read3DUniform(string file, NeuralNet<Point3D, Value>& nn, Point3D& pMin, Point3D& pMax ) {
        return readUniform(file, nn, pMin[0], pMin[1], pMin[2] , pMax[0], pMax[1], pMax[2]);
    }
    bool read3DUniform(string file, NeuralNet<Point2D, Value>& nn, Point2D& pMin, Point2D& pMax ) {
        pMin[2] = 0;
        pMax[2] = 0;
        return readUniform(file, nn, pMin[0], pMin[1], pMax[0], pMax[1]);
    }

    bool read3DImage(string file, NeuralNet<Point2D, Value>& nn, Point2D& pMin, Point2D& pMax ) {
        read3DImage(file, nn, pMin[0], pMin[1], pMax[0], pMax[1]);
        return false;
    }

    bool read3DImage(string file, NeuralNet<Point3D, Value>& nn, Point3D& pMin, Point3D& pMax ) {
        read3DImage(file, nn, pMin[0], pMin[1], pMin[2], pMax[0], pMax[1], pMax[2]);
        return false;
    }

    //! wb.Q 10118 add to read 3D point from 3D image benchmarks
    //! 3D image points are stored in a 1D NN
    bool read3DImage(string str_file, NeuralNet<Point, Value>& nn,
                     GLfloatP& min_x, GLfloatP& min_y, GLfloatP& min_z,
                     GLfloatP& max_x, GLfloatP& max_y, GLfloatP& max_z)
    {
        ifstream fi;
        int pos = getPos(str_file);
        string str_sub;

        //! load xxx.tsp
        str_sub = str_file.substr(0, pos);
        str_sub.append(SFX_3DImage_FILE);

        cout << "read 3d points from: " << str_sub << endl;

        max_x = -INFINITY;
        max_y = -INFINITY;
        max_z = -INFINITY;
        min_x = +INFINITY;
        min_y = +INFINITY;
        min_z = +INFINITY;

        // count total size of input points
        fi.open(str_sub.c_str());
        if (!fi)
        {
            cout << "erreur ouverture 3D file: " << str_sub << endl;
            return false;
        }
        else
        {
            int size = 0, width = 0, height = 0;
            int comma;
            fi >> comma >> comma >> width;
            fi >> comma >> comma >> height;

            size = width * height;
            nn.resize(size, 1);

            for (int i = 0; i < size; i++)
            {
                float cx, cy, cz;
                fi >> cx >> cy >> cz;
                Point city(cx, cy, cz);
                nn.adaptiveMap[0][i] =  city;

                if (cx >= max_x)
                    max_x = cx;
                if (cy >= max_y)
                    max_y = cy;
                if (cz >= max_z)
                    max_z = cz;
                if (cx < min_x)
                    min_x = cx;
                if (cy < min_y)
                    min_y = cy;
                if (cz < min_z)
                    min_z = cz;
            }

        }

        fi.close();

        return true;
    }

    //! wb.Q 10118 add to read 3D point from 2D image benchmarks
    //! 2D image points are stored in a 1D NN
    bool read3DImage(string str_file, NeuralNet<Point, Value>& nn,
                     GLfloatP& min_x, GLfloatP& min_y,
                     GLfloatP& max_x, GLfloatP& max_y)
    {
        int pos = getPos(str_file);
        string str_sub;

        //! load xxx.tsp
        str_sub = str_file.substr(0, pos);
        str_sub.append(SFX_3DImage_FILE);

        cout << "read 3d points from: " << str_sub << endl;

        max_x = -INFINITY;
        max_y = -INFINITY;
        min_x = +INFINITY;
        min_y = +INFINITY;

        ifstream fi;
        // count total size of input points
        fi.open(str_sub.c_str());
        if (!fi)
        {
            cout << "erreur ouverture 3D file: " << str_sub << endl;
            return false;
        }
        else
        {
            int size = 0, width = 0, height = 0;
            int comma;
            fi >> comma >> comma >> width;
            fi >> comma >> comma >> height;

            size = width * height;
            nn.resize(size, 1);

            for (int i = 0; i < size; i++)
            {
                float cx, cy;
                fi >> cx >> cy;
                Point city(cx, cy);
                nn.adaptiveMap[0][i] =  city;

                if (cx >= max_x)
                    max_x = cx;
                if (cy >= max_y)
                    max_y = cy;
                if (cx < min_x)
                    min_x = cx;
                if (cy < min_y)
                    min_y = cy;
            }

        }

        fi.close();

        return true;
    }

    //! wb.Q 062018 add to read 3D point from Thermogauss benchmarks.
    bool read3DPointsThermogauss(string str_file, NeuralNet<Point, Value>& nn,
                                 GLfloatP& min_x, GLfloatP& min_y, GLfloatP& min_z,
                                 GLfloatP& max_x, GLfloatP& max_y, GLfloatP& max_z)
    {
        char buf[256];

        ifstream fi;
        int pos = getPos(str_file);
        string str_sub;

        //! load xxx.tsp
        str_sub = str_file.substr(0, pos);
        str_sub.append(SFX_3D_FILE);

        cout << "read 3d points from: " << str_sub << endl;

        // count total size of input points
        fi.open(str_sub.c_str());
        if (!fi)
        {
            cout << "erreur ouverture 3D file: " << str_sub << endl;
            return false;
        }
        else
        {
            int size = 0;
            while(fi.getline(buf, 256)){
                size ++;
            }
            fi.close();
            cout << "qiao size " << size << endl;

            if (size)
                nn.resize(size, 1);
            else
                cout << "error read cities: num = 0." << endl;

            fi.open(str_sub.c_str());

            float cx, cy, cz;
            float temp;

            max_x = -INFINITY;
            max_y = -INFINITY;
            max_z = -INFINITY;
            min_x = +INFINITY;
            min_y = +INFINITY;
            min_z = +INFINITY;

            // Lecture de la matrice de pixels ligne par ligne
            for (int i = 0; i < size; i++)
            {

                fi >> cx;
                fi >> cy;
                fi >> cz;
                fi >> temp;

                Point city(cx, cy, cz);

                nn.adaptiveMap[0][i] = city;

                if (cx >= max_x)
                    max_x = cx;
                if (cy >= max_y)
                    max_y = cy;
                if (cz >= max_z)
                    max_z = cz;
                if (cx < min_x)
                    min_x = cx;
                if (cy < min_y)
                    min_y = cy;
                if (cz < min_z)
                    min_z = cz;
            }

            fi.close();
        }


        return true;
    }

    //! wb.Q 062018 add to read 3D point from Thermogauss benchmarks.
    bool read3DPointsThermogauss(string str_file, NeuralNet<Point, Value>& nn,
                                 GLfloatP& min_x, GLfloatP& min_y,
                                 GLfloatP& max_x, GLfloatP& max_y)
    {
        char buf[256];

        ifstream fi;
        int pos = getPos(str_file);
        string str_sub;

        //! load xxx.tsp
        str_sub = str_file.substr(0, pos);
        str_sub.append(SFX_3D_FILE);

        cout << "read 3d points from: " << str_sub << endl;

        // count total size of input points
        fi.open(str_sub.c_str());
        if (!fi)
        {
            cout << "erreur ouverture 3D file: " << str_sub << endl;
            return false;
        }
        else
        {
            int size = 0;
            while(fi.getline(buf, 256)){
                size ++;
            }
            fi.close();
            cout << "qiao size " << size << endl;

            if (size)
                nn.resize(size, 1);
            else
                cout << "error read cities: num = 0." << endl;

            fi.open(str_sub.c_str());

            float cx, cy;
            float temp;

            max_x = -INFINITY;
            max_y = -INFINITY;
            min_x = +INFINITY;
            min_y = +INFINITY;

            // Lecture de la matrice de pixels ligne par ligne
            for (int i = 0; i < size; i++)
            {

                fi >> cx;
                fi >> cy;
                fi >> temp;
                fi >> temp;

                Point city(cx, cy);

                nn.adaptiveMap[0][i] = city;

                if (cx >= max_x)
                    max_x = cx;
                if (cy >= max_y)
                    max_y = cy;
                if (cx < min_x)
                    min_x = cx;
                if (cy < min_y)
                    min_y = cy;
            }

            fi.close();
        }

        return true;
    }

    //! wb.Q 200716 add to read tsp file into a (N * 1) NN (md).
    bool readTSP(string str_file, NeuralNet<Point, Value>& nn, GLfloatP& min_x, GLfloatP& min_y, GLfloatP& max_x, GLfloatP& max_y){

        char buf[256];
        char *ptr;

        char str[256];
        char str_2[10];

        char name[50];
        int size = 0;

        ifstream fi;
        int pos = getPos(str_file);
        string str_sub;

        //! load xxx.tsp
        str_sub = str_file.substr(0, pos);
        str_sub.append(SFX_TSP_FILE);

        cout << "read tsp from: " << str_sub << endl;

        fi.open(str_sub.c_str());
        if (!fi)
        {
            cout << "erreur ouverture tsp file: " << str_sub << endl;
            return false;
        }
        else
        {
            fi.getline(buf, 256);
            if (!fi)
                return(0);
            //                return(fi);// qiao: windows can not transfer correctly from ifstream to bool

            if ( (ptr = strstr(buf, "NAME")) != NULL )
            {
                sscanf(ptr, "%s%s%s", str, str_2, &name);
            }
            else
                return(0);
            //                return(fi);

            for (int l = 1; l <= 3; l++)
            {
                fi.getline(buf, 256);
                if (!fi)
                    return(0);
            }

            if ( (ptr = strstr(buf, "DIMENSION")) != NULL )
            {
                sscanf(ptr, "%s%s%d", str, str_2, &size);
            }
            else
                return(0);

            if (size)
                nn.resize(size, 1);
            else
                cout << "error read cities: num = 0." << endl;

            for (int l = 1; l <= 2; l++)
            {
                fi.getline(buf, 256);
                if (!fi)
                    return(0);
            }

            int num = 0;
            float cx, cy;

            max_x = -INFINITY;
            max_y = -INFINITY;
            min_x = +INFINITY;
            min_y = +INFINITY;


            // Lecture de la matrice de pixels ligne par ligne
            for (int i = 0; i < size; i++)
            {

                fi >> num;
                fi >> cx;
                fi >> cy;

                //                cout << "read tsp: cx,cy = " << cx << ", " << cy << endl;
                Point city(cx, cy);

                nn.adaptiveMap[0][i] = city;

                if (cx >= max_x)
                    max_x = cx;
                if (cy >= max_y)
                    max_y = cy;
                if (cx < min_x)
                    min_x = cx;
                if (cy < min_y)
                    min_y = cy;
            }

        }
        fi.close();

    }

    //! WBQ 200716 add to read tsp file into a (N * 1) NN (md).
    bool readUniform(string str_file, NeuralNet<Point, Value>& nn, GLfloatP& min_x, GLfloatP& min_y,
                     GLfloatP& max_x, GLfloatP& max_y){

        int pos = getPos(str_file);
        string str_sub;
        //! load xxx.txt
        str_sub = str_file.substr(0, pos);
        str_sub.append(SFX_2D_FILE);

        cout << "NN reads uniform data set from: " << str_sub << endl;

        max_x = -INFINITY;
        max_y = -INFINITY;
        min_x = +INFINITY;
        min_y = +INFINITY;

        ifstream fi;
        fi.open(str_sub.c_str());
        if (!fi)
        {
            cout << "erreur ouverture uniform file: " << str_sub << endl;
            return false;
        }
        else
        {
            int size;
            fi >> size;
            nn.resize(size, 1);

            for (int i = 0; i < size; i++) {
                float cx, cy;
                fi >> cx >> cy;
                Point city(cx, cy);
                nn.adaptiveMap[0][i] = city;

                if (cx >= max_x)
                    max_x = cx;
                if (cy >= max_y)
                    max_y = cy;
                if (cx < min_x)
                    min_x = cx;
                if (cy < min_y)
                    min_y = cy;
            }
        }

        fi.close();
        return 1;
    }// end 2D uniform

    //! WBQ 200716 add to read tsp file into a (N * 1) NN (md).
    bool readUniform(string str_file, NeuralNet<Point, Value>& nn, GLfloatP& min_x, GLfloatP& min_y, GLfloatP& min_z,
                     GLfloatP& max_x, GLfloatP& max_y, GLfloatP& max_z){

        int pos = getPos(str_file);
        string str_sub;
        //! load xxx.txt
        str_sub = str_file.substr(0, pos);
        str_sub.append(SFX_3D_FILE);

        cout << "NN reads input data set from: " << str_sub << endl;

        max_x = -INFINITY;
        max_y = -INFINITY;
        max_z = -INFINITY;
        min_x = +INFINITY;
        min_y = +INFINITY;
        min_z = +INFINITY;

        ifstream fi;
        fi.open(str_sub.c_str());
        if (!fi)
        {
            cout << "erreur ouverture input file: " << str_sub << endl;
            return false;
        }
        else
        {
            int size;
            fi >> size;
            nn.resize(size, 1);

            for (int i = 0; i < size; i++) {
                float cx, cy, cz;

                fi >> cx >> cy >> cz;
                Point city(cx, cy, cz);
                nn.adaptiveMap[0][i] = city;

                if (cx >= max_x)
                    max_x = cx;
                if (cy >= max_y)
                    max_y = cy;
                if (cz >= max_z)
                    max_z = cz;
                if (cx < min_x)
                    min_x = cx;
                if (cy < min_y)
                    min_y = cy;
                if (cz < min_z)
                    min_z = cz;
            }

        }

        fi.close();
        return 1;
    }// end 2D uniform

};

typedef InputRW<Point2D, GLfloat> EMSTRW;

} // namespace components

#endif // IMAGERW_H
