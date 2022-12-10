test_that("test Layer", {
  # initialize
  ### INPUT LAYER FUNCTIONALITIES
  input <- matrix(1:8, nrow = 2, ncol = 4)
  input_layer <- Layer$new(number_of_nodes = 2, activation_fun = "identity", prior_layer = NULL, a_prev = input, random_init = list(weights = diag(2), bias = c(0, 0)))
  # should be null for input layer
  expect_equal(input_layer$get_params()$weight, expected = diag(2))
  # should be equal to the number of columns and not the provided input argument
  expect_equal(input_layer$get_number_of_nodes(), expected = 2L)
  # calculation handling
  expect_equal(input_layer$forward_pass(), input)
  input_layer$set_input(input[, -1, drop = FALSE])
  expect_equal(input_layer$forward_pass(), input[, -1])
  ### ALL OTHER LAYERS
  layer <- Layer$new(number_of_nodes = 3, activation_fun = "identity", prior_layer = NULL, a_prev = input, random_init = list(weights = matrix(1, 3, 2), bias = rep(1, 3)))
  expect_equal(layer$get_number_of_nodes(), expected = 3L)
  expect_equal(layer$get_params(), expected = list(weights = matrix(1L, nrow = 3, ncol = 2), bias = rep(1, 3)))
  layer$set_params(matrix(c(1, 0.5, 2), nrow = 3, ncol = 2))
  expect_equal(layer$get_params(), expected = list(weights = matrix(c(1, 0.5, 2), nrow = 3, ncol = 2), bias = rep(1, 3)))
  ## activation functions
  # identity
  expect_equal(layer$forward_pass(), expected = cbind(c(4, 2.5, 7), c(8, 4.5, 15), c(12, 6.5, 23), c(16, 8.5, 31)))
  layer$backward_pass()
  #expect_equal(layer$get_derivatives(), expected = list(derivative = matrix(c(-1, 1, 0, -1, 3), nrow = 3, ncol = 5, byrow = TRUE)))
  # other activation functions
  tol <- 1e-5
  act_fun <- c("sigmoid", "relu", "gelu", "silu", "softplus")
  act_fun_res <- rbind(c(0.26894, 0.5, 0.73106), c(0, 0, 1), c(-0.158655, 0, 0.84134), c(-0.26894, 0, 0.73106), c(0.31326, 0.69315, 1.31326))
  act_fun_deriv_res <- rbind(c(0.19661, 0.25, 0.19661), c(0, 0, 1), c(-0.083315, 0.5, 1.08331), c(0.07233, 0.5, 0.92767), c(0.26894, 0.5, 0.73106))
  for (fun in 1:length(act_fun)) {
    # set function
    suppressMessages({
      layer$.__enclos_env__$private$set_activation_fun(act_fun[fun])
    })
    # singular values
    expect_equal(layer$.__enclos_env__$private$activation_fun(-1), expected = act_fun_res[fun, 1], tolerance = tol)
    expect_equal(layer$.__enclos_env__$private$activation_fun(0), expected = act_fun_res[fun, 2], tolerance = tol)
    expect_equal(layer$.__enclos_env__$private$activation_fun(1), expected = act_fun_res[fun, 3], tolerance = tol)
    # vectors
    expect_equal(layer$.__enclos_env__$private$activation_fun(-1:1), expected = act_fun_res[fun,], tolerance = tol)
    # matrices
    res <- diag(act_fun_res[fun,]); res[res == 0] <- act_fun_res[fun, 2]
    expect_equal(layer$.__enclos_env__$private$activation_fun(diag(-1:1)), expected = res, tolerance = tol)
    # derivative
    expect_equal(layer$
                   .__enclos_env__$
                   private$
                   activation_fun_deriv(-1), expected = act_fun_deriv_res[fun, 1], tolerance = tol)
    expect_equal(layer$
                   .__enclos_env__$
                   private$
                   activation_fun_deriv(0), expected = act_fun_deriv_res[fun, 2], tolerance = tol)
    expect_equal(layer$
                   .__enclos_env__$
                   private$
                   activation_fun_deriv(1), expected = act_fun_deriv_res[fun, 3], tolerance = tol)
    # vectors
    expect_equal(layer$
                   .__enclos_env__$
                   private$
                   activation_fun_deriv(-1:1), expected = act_fun_deriv_res[fun,], tolerance = tol)
    # matrices
    res <- diag(act_fun_deriv_res[fun,]); res[res == 0] <- act_fun_deriv_res[fun, 2]
    expect_equal(layer$
                   .__enclos_env__$
                   private$
                   activation_fun_deriv(diag(-1:1)), expected = res, tolerance = tol)
  }

})
