QT       += xml
QT       += gui
TEMPLATE = app
CONFIG += console
CONFIG += exceptions
#CONFIG += static



CONFIG(debug, debug|release) {
#    OUT_PATH=bin/imbricateur
    OUT_PATH=../../debug86/calculateur
} else {
#    OUT_PATH=../../release86/imbricateur
#Shadow Build au niveau qui precede "imbricateur"
    OUT_PATH=../../bin/calculateur
}

unix {
        CONFIG +=
#static
        DEFINES += QT_ARCH_ARMV6
        TARGET = $$OUT_PATH
}
win32 {
        TARGET = $$OUT_PATH
}

#DEFINES += POPIP_COALITION

SOURCES += ../src/main.cpp

#BOOST_PATH=C:/boost_1_66_0/

BOOST_PATH= C:\boost_1_83_0\

INCLUDEPATH  += ../include
INCLUDEPATH  += ../../basic_components/include
INCLUDEPATH  += ../../optimization_operators/include
INCLUDEPATH  += ../../coalition_framework/include
INCLUDEPATH  += $$BOOST_PATH


CONFIG(debug, debug|release) {
    LIBS += -L$$PWD/../bin/ -llibCalculateurEMST_KOPT
} else {
    LIBS += -L$$PWD/../../coalition_framework/bin/ -llibCoalition
    LIBS += -L$$PWD/../bin/ -llibCalculateurEMST_KOPT
}

win32:{

  #Do'nt use the full path.
  #Because it is include the space character,
  #use the short name of path, it may be NVIDIA~1 or NVIDIA~2 (C:/Progra~1/NVIDIA~1/CUDA/v5.0),
  #or use the mklink to create link in Windows 7 (mklink /d c:\cuda "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v5.0").
#  CUDA_DIR      = "C:/Progra~1/NVIDIA~2/CUDA/v9.1"
#  QTDIR = C:\Qt\Qt5.9.1\5.9.1\Qt5.9_static
#  BOOST_PATH=C:/boost_1_66_0/

#For wenbaoQiao hp computer configuration
  CUDA_DIR      = C:/Progra~1/NVIDIA~2/CUDA/v11.8# "C:/Progra~1/NVIDIA~2/CUDA/v9.0"
  QTDIR = E:\qt_static_5 #C:\QT5.10_static #C:\Qt\Qt5.9.1\5.9.1\Qt5.9_static
  BOOST_PATH= C:\boost_1_83_0\ #C:/boost_1_59_0

  QMAKE_LIBDIR += $$CUDA_DIR/lib/x64
  INCLUDEPATH  += ../include
  INCLUDEPATH  += ../src
  INCLUDEPATH  += $$BOOST_PATH
  INCLUDEPATH  += $$CUDA_DIR/include
  INCLUDEPATH  += $$QTDIR/include $$QTDIR/include/QtCore  $$QTDIR/include/QtXml
  INCLUDEPATH  += ../../basic_components/include
  INCLUDEPATH  += ../../optimization_operators/include
  INCLUDEPATH  += ../../coalition_framework/include
  INCLUDEPATH  += "C:\ProgramData\NVIDIA Corporation\CUDA Samples\common"
  LIBS         +=  -lcuda  -lcudart
}


unix:{
  CUDA_DIR      = /usr/local/cuda-11.2
  QMAKE_LIBDIR += $$CUDA_DIR/lib64
  INCLUDEPATH  += $$CUDA_DIR/include
  LIBS += -lcudart -lcuda

    #CUDA_DIR      = "C:/Progra~1/NVIDIA~2/CUDA/v11.2"
    CUDA_DIR      = /usr/local/cuda-11.2
    INCLUDEPATH  += $$CUDA_DIR/include
    INCLUDEPATH  += "C:\ProgramData\NVIDIA Corporation\CUDA Samples\v11.2\common\inc"
    QMAKE_LIBDIR += $$CUDA_DIR/lib/x64
    LIBS         +=  -lcuda  -lcudart

  LIBS  += -shared -Xcompiler -fPIC
  QMAKE_CXXFLAGS += -std=c++0x
  QTDIR = /home/qiao/qt5_static #C:\QT5.10_static #C:\Qt\Qt5.9.1\5.9.1\Qt5.9_static
#  QTDIR = /home/qiao/Qt5.12.12/5.12.12/gcc_64

#  BOOST_PATH= /home/qiao/boost_1_66_0

    INCLUDEPATH  += ../../basic_components/include
    INCLUDEPATH  += ../../optimization_operators/include
    INCLUDEPATH  += ../../coalition_framework/include
    INCLUDEPATH  += ../include
    INCLUDEPATH  += ../src
    INCLUDEPATH  += "/home/qiao/NVIDIA_CUDA-11.2_Samples/common/inc"
    INCLUDEPATH  += $$QTDIR/include $$QTDIR/include/QtCore  $$QTDIR/include/QtXml

}
