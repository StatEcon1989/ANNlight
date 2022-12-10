rm(list = ls()); devtools::load_all()
set.seed(270789)
input_data <- matrix(rnorm(9e4), ncol = 1e4, nrow = 9)
layer_config <- list(
  list(number_of_nodes = 5, activation_fun = "relu", bias = TRUE, random_init = NULL),
  list(number_of_nodes = 2, activation_fun = "relu", bias = TRUE, random_init = NULL),
  list(number_of_nodes = 1, activation_fun = "sigmoid", bias = TRUE, random_init = NULL)
)
ANN <- Network$new(cost_fun = "quadratic", layer_config = layer_config, input_data = input_data, output_data = NULL)
output_data <- ANN$calculate()
output_data <- (output_data>mean(output_data))*1

save(input_data, output_data, file = "./tests/testthat/test_data/test_network.rdata")
rm(list = ls())