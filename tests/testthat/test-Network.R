test_that("test Network, regression", {
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
  cost <- list()
  cost[["cross_entropy"]] <- ANN$train(epochs = 1e2)
  suppressMessages({
    ANN <- Network$new(cost_fun = "quadratic", layer_config = layer_config, input_data = input_data, output_data = output_data)
  })
  cost[["quadratic"]] <- ANN$train(epochs = 1e2)

  expect_snapshot(cost)

  #Gradient check
  #TODO aside from relu and sigmoid, activation functions must be double checked
  ANN <- Network$new(cost_fun = "cross_entropy", layer_config = list(
    list(number_of_nodes = 5, activation_fun = "silu", random_init = NULL),
    list(number_of_nodes = 3, activation_fun = "relu", random_init = NULL),
    list(number_of_nodes = 3, activation_fun = "gelu", random_init = NULL),
    list(number_of_nodes = 3, activation_fun = "softplus", random_init = NULL),
    list(number_of_nodes = 1, activation_fun = "sigmoid", random_init = NULL)
  ), input_data = input_data, output_data = output_data)
  ANN$train(epochs = 5e2, learning_rate = 0.9)
  expect_snapshot(check_gradient(network = ANN, epsilon = 1e-7))
})