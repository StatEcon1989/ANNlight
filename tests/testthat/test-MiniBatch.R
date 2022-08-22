test_that("test Network, MiniBatch", {
  load("./test_data/test_network.rdata")
  layer_config <- list(
    list(number_of_nodes = 5, activation_fun = "relu", random_init = NULL),
    list(number_of_nodes = 2, activation_fun = "relu", random_init = NULL),
    list(number_of_nodes = 1, activation_fun = "sigmoid", random_init = NULL)
  )
  set.seed(270789)
  suppressMessages({
    ANN <- Network$new(cost_fun = "cross_entropy", layer_config = layer_config, input_data = input_data, output_data = output_data)
  })
  expect_snapshot(ANN$train(batch_size = 64L, epochs = 50L))
})