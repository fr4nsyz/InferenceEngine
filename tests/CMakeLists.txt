include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.tar.gz
)
FetchContent_MakeAvailable(googletest)

add_executable(test_tensor test_tensor.cc)
target_link_libraries(test_tensor PRIVATE inference_engine GTest::gtest_main)
add_test(NAME test_tensor COMMAND test_tensor)

add_executable(test_linear test_linear.cc)
target_link_libraries(test_linear PRIVATE inference_engine GTest::gtest_main)
add_test(NAME test_linear COMMAND test_linear)

add_executable(test_softmax test_softmax.cc)
target_link_libraries(test_softmax PRIVATE inference_engine GTest::gtest_main)
add_test(NAME test_softmax COMMAND test_softmax)

add_executable(test_relu test_relu.cc)
target_link_libraries(test_relu PRIVATE inference_engine GTest::gtest_main)
add_test(NAME test_relu COMMAND test_relu)

add_executable(test_add test_add.cc)
target_link_libraries(test_add PRIVATE inference_engine GTest::gtest_main)
add_test(NAME test_add COMMAND test_add)

add_executable(test_onnx_loader test_onnx_loader.cc)
target_link_libraries(test_onnx_loader PRIVATE inference_engine GTest::gtest_main)
add_test(NAME test_onnx_loader COMMAND test_onnx_loader)
set_tests_properties(test_onnx_loader PROPERTIES WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
