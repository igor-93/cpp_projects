cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(SudokuSolver VERSION 1.0)

# specify the C/C++ standard
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_C_STANDARD_REQUIRED True)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_compile_options(-Wall -Wextra)
add_link_options(-fopenmp)

# GSL lib
# find_package(GSL REQUIRED)
# include_directories(${GSL_INCLUDE_DIRS} ${GSLCBLAS_INCLUDE_DIRS})
# add_definitions(${GSL_DEFINITIONS})

# EIGEN
# include_directories( "/usr/include/eigen3/" )

# OMP
find_package(OpenMP)
# opencv
find_package(OpenCV 4.4.0 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
# LibTorch - c++ version of PyTorch
find_package(Torch REQUIRED PATHS ~/software/libtorch)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")



# project
set(SOURCE_FILES 
    src/main.cpp 
    src/Sudoku.cpp 
    src/MnistModel.cpp 
    src/ImgProc.cpp 
    include/Sudoku.hpp
    include/MnistModel.hpp
    include/ImgProc.hpp
)

# create a target
add_executable(SudokuSolver ${SOURCE_FILES})
target_include_directories(SudokuSolver PRIVATE include)


if(OpenMP_CXX_FOUND)
    target_link_libraries(SudokuSolver OpenMP::OpenMP_CXX)
endif()
target_link_libraries(SudokuSolver ${OpenCV_LIBS})
target_link_libraries(SudokuSolver "${TORCH_LIBRARIES}")
