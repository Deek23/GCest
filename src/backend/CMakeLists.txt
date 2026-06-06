cmake_minimum_required(VERSION 3.18)
project(BACKEND CXX)

# コンパイラ確認
include(cmake/find_toolchain.cmake)
# 外部プロジェクト利用宣言？
include(ExternalProject)

# Makefile の中身確認用
set(CMAKE_VERBOSE_MAKEFILE 1)

#ライブラリはオリジナルのモノを利用
set(CMAKE_BUILD_RPATH_USE_ORIGIN ON)

# Abseil requires C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# CUDA を利用するかどうかのオプション
# Ceres 本体が -DUSE_CUDA=ON でビルドされていることが前提
option(WITH_CUDA "Enable CUDA acceleration via Ceres dense linear algebra" OFF)

# これでいいのか？
set(Eigen3_DIR /usr/local/share/eigen3)

# glog を正しく組み込む方法
add_definitions(-DGLOG_USE_GLOG_EXPORT)

# boost は静的ライブラリを使用
set(Boost_USE_STATIC_LIBS ON)

# パッケージ確認
find_package(Ceres REQUIRED)
find_package(Eigen3 REQUIRED)
# boost用 動的（共有）ライブラリを優先するように明示
set(Boost_USE_STATIC_LIBS OFF)
# boost用 （コンポーネント名も指定）
find_package(Boost REQUIRED COMPONENTS timer)

# nlohmann JSON 用
find_package(nlohmann_json 3.11.2 REQUIRED)

# CUDA Toolkit 検出
if(WITH_CUDA)
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  # RTX 4000 Ada (sm_89) 想定。新しい世代を追加するなら "89;90" 等に拡張可能
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 89)
  endif()
  find_package(CUDAToolkit REQUIRED)
  message(STATUS "CUDA Toolkit found: ${CUDAToolkit_VERSION}")
  message(STATUS "CUDA arch: ${CMAKE_CUDA_ARCHITECTURES}")
  message(STATUS "CUDA include dirs: ${CUDAToolkit_INCLUDE_DIRS}")
  add_definitions(-DGCEST_USE_CUDA)
endif()

# ビルドタイプ
set(CMAKE_BUILD_TYPE "Release")

# このオプションが要る感じ
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# コンパイラオプション（g++）
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fopenmp -fPIC")

# リンカオプション
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}")

# ビルド（動的ライブラリ）
file(GLOB_RECURSE SOURCES "src/*.cpp")
# WITH_CUDA=ON のときのみ .cu を追加してコンパイル対象に含める
if(WITH_CUDA)
  file(GLOB_RECURSE CUDA_SOURCES "src/cuda/*.cu")
  list(APPEND SOURCES ${CUDA_SOURCES})
  message(STATUS "CUDA sources: ${CUDA_SOURCES}")
endif()
ADD_LIBRARY(BACKEND SHARED ${SOURCES})

# CUDA ビルド時のターゲットプロパティ
if(WITH_CUDA)
  set_target_properties(BACKEND PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    POSITION_INDEPENDENT_CODE ON
  )
endif()

# インクルードパス
target_include_directories(BACKEND
  PUBLIC 
    /usr/share/R/include
     ${CMAKE_CURRENT_SOURCE_DIR}/include
	PRIVATE
		${CMAKE_CURRENT_SOURCE_DIR}/src/internal
)

# リンクライブラリ（出力ファイル名は 'BACKEND' になるか）
target_link_libraries(BACKEND
  PRIVATE
    absl::time
    absl::log_initialize
    absl::flags_parse
    Ceres::ceres
    Eigen3::Eigen
    nlohmann_json::nlohmann_json
    Boost::timer
)

# CUDA リンク（Ceres が CUDA 有効でビルドされていれば内部で利用される）
if(WITH_CUDA)
  target_link_libraries(BACKEND
    PRIVATE
      CUDA::cudart
      CUDA::cublas
      CUDA::cusolver
  )
endif()

# some OS-specific library name conventions here; just leave as is as it would change the way Rcpp's Makefile needs to look.
if(WIN32)
  SET_TARGET_PROPERTIES(BACKEND PROPERTIES
    OUTPUT_NAME "BACKEND"
    PREFIX ""
  )
endif()
