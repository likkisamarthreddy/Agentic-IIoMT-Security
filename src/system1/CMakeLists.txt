cmake_minimum_required(VERSION 3.10)
project(EdgeAgent)

set(CMAKE_CXX_STANDARD 17)

add_executable(edge_agent edge_agent.cpp)

if(ONNXRUNTIME_DIR)
    target_include_directories(edge_agent PRIVATE ${ONNXRUNTIME_DIR}/include)
    target_link_directories(edge_agent PRIVATE ${ONNXRUNTIME_DIR}/lib)
endif()

target_link_libraries(edge_agent mosquitto onnxruntime)
