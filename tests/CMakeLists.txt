include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.tar.gz
)
FetchContent_MakeAvailable(googletest)

add_executable(test_matrix test_matrix.cc)
target_link_libraries(test_matrix PRIVATE inference_engine GTest::gtest_main)
add_test(NAME test_matrix COMMAND test_matrix)

add_executable(test_linear test_linear.cc)
target_link_libraries(test_linear PRIVATE inference_engine GTest::gtest_main)
add_test(NAME test_linear COMMAND test_linear)
