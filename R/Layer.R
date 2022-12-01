#' @title Layer of a neural net
#'
#' @description The Layer class contains all functionalities needed (e.g. calculate the layer, calculate the derivative
#' of its weight matrix, etc.) for its use in the [Network] class.
#' One layer is not able to function on its own (aside from the input layer), the class is intentionally not exported.
Layer <- R6::R6Class(
  classname = "Layer",
  private = list(
    # `matrix<numeric>` the cached activations of dimension 'nodes in layer' x 'number of observations'
    a = NULL,
    # `matrix<numeric>` if input layer, the data is stored here.
    a_prev = NULL,
    # `matrix<numeric>` the cached score of dimension 'nodes in layer' x 'number of observations'
    z = NULL,
    # `integer` number of nodes in the layer (what did you expect?)
    number_of_nodes = NULL,
    # `Layer` the complete, previous layer
    prior_layer = NULL,
    # `matrix<numeric>` the weighting matrix of dimension 'nodes in layer' x 'nodes in previous layer'
    weights = NULL,
    # `matrix<numeric>` the bias vector/matrix of dimenstion 'nodes in layer' x 1
    bias = NULL,
    # `matrix<numeric>` the derivative w.r.t. the weighting matrix of dimension 'nodes in layer' x 'nodes in previous layer'
    dw = NULL,
    # `matrix<numeric>` the derivative w.r.t. the bias vector/matrix of dimenstion 'nodes in layer' x 1
    db = NULL,
    # `function` with input argument `z`
    activation_fun = NULL,
    # `function` with input argument `z`
    activation_fun_deriv = NULL,
    # FUNCTIONS
    # depending on the character variable `activation_fun`, sets the activation function and its derivative. Is called during
    # initialization of the class.
    set_activation_fun = function(activation_fun) {
      switch(activation_fun,
             relu = {

               private$activation_fun <- function(z) {
                 z[z < 0] <- 0
                 return(z)
               }

               message("Assuming value of derivative of 0 at 0 for relu activation function.")
               private$activation_fun_deriv <- function(z) (z > 0) * 1
             },
             identity = {
               private$activation_fun <- function(z) z
               private$activation_fun_deriv <- function(z) replace(z, value = 1)
             },
             sigmoid = {
               private$activation_fun <- function(z) 1 / (1 + exp(-z))
               private$activation_fun_deriv <- function(z) private$activation_fun(z) * (1 - private$activation_fun(z))
             },
             gelu = {
               private$activation_fun <- function(z) z * pnorm(q = z)
               private$activation_fun_deriv <- function(z) pnorm(q = z) + z * dnorm(x = z)
             },
             silu = {
               private$activation_fun <- function(z) z / (1 + exp(-z))
               private$activation_fun_deriv <- function(z) (1 + exp(-z) + z * exp(-z)) / (1 + exp(-z))^2
             },
             softplus = {
               private$activation_fun <- function(z) log(1 + exp(z))
               private$activation_fun_deriv <- function(z) 1 / (1 + exp(-z))
             },
             stop("Activation function ", activation_fun, "not supported!"))
    }
  ),
  public = list(
    # @formatter:off
    #' @description Initialize a new layer
    #'
    #' @param number_of_nodes `integer` that specifies the number of nodes in the layer.
    #' @param activation_fun `character` that contains the activation function to be used. See details.
    #' @param prior_layer `Layer` in case of a hidden or output layer, the previous layer.
    #' @param a_prev `matrix<numeric>` in case of the layer being the input layer, a`matrix` with numeric entries that contains the input_data.
    #' @param random_init `list<matrix/vector>`: If `NULL` initializesthe weights with random N(0, 0.01) values and the bias with zeros.
    #' If a list with elements `weights` and `bias` (correct dimensions) is given, uses those values for initialization.
    #'
    #' @details Denote `a`the output of the activation function and `z` its input. Currently, the following activation functions are supported:
    #' * `relu`: \eqn{a = max(0,z)}
    #' * `identity`: \eqn{a=z}
    #' * `sigmoid`: \eqn{a = \frac{1}{1+\exp(-z)}}
    #' * `gelu`: \eqn{z \cdot \Phi(z)}
    #' * `silu`: \eqn{a = \frac{z}{1+\exp(-z)}}
    #' * `softplus`: \eqn{ a = \log(1+\exp{z})}
    #'
    #' @return `Layer`: The class instance. Invisibly, for chaining.
    # @formatter:on
    initialize = function(number_of_nodes, activation_fun, prior_layer = NULL, a_prev = NULL, random_init = NULL) {

      if (!is.null(prior_layer)) {
        private$prior_layer <- prior_layer
        nodes_prior_layer <- prior_layer$get_number_of_nodes()
      } else {
        stopifnot("Must provide either 'prior_layer' or 'a_prev'!" = !is.null(a_prev))
        private$a_prev <- a_prev
        nodes_prior_layer <- nrow(a_prev)
      }
      stopifnot("Must contain at least 1 node!" = number_of_nodes >= 1)
      private$number_of_nodes <- as.integer(number_of_nodes)
      if (is.null(random_init)) {
        private$weights <- matrix(
          data = rnorm(private$number_of_nodes * nodes_prior_layer),
          nrow = private$number_of_nodes, ncol = nodes_prior_layer
        ) * 0.01
        private$bias <- rep(0, private$number_of_nodes)
      } else {
        if (!all.equal(dim(random_init$weights), c(private$number_of_nodes, nodes_prior_layer)) | length(random_init$bias) != private$number_of_nodes) {
          stop("Wrong dimensions in initialization values!")
        }
        private$weights <- random_init$weights
        private$bias <- random_init$bias
      }
      private$set_activation_fun(tolower(activation_fun))
      return(invisible(self))
    },

    # @formatter:off
    #' @description Calculate the output of the current layer
    #'
    #' @return `matrix<numeric>` Either the input- (for an input layer) or the output of the activation function.
    # @formatter:on
    forward_pass = function() {
      if (!is.null(private$prior_layer)) {
        private$a_prev <- private$prior_layer$forward_pass()
      }
      private$z <- private$weights %*% private$a_prev + matrix(private$bias, nrow = length(private$bias), ncol = ncol(private$a_prev))
      private$a <- private$activation_fun(private$z)
      return(private$a)
    },
    # @formatter:off
    #' @description Calculate the derivative of the current layer wrt to its weights.
    #'
    #' @param da `matrix<numeric>`: derivative of the cost function wrt the activations of the next layer..
    #'
    #' @return `list<matrix<numeric>>` A list with two elements:
    #' * `derivative`: The derivative for the weighting matrix.
    #' * `w_times_delta`: A matrix product needed for the calculation of the previous layer.
    # @formatter:on
    backward_pass = function(da = 1) {
      dz <- da * private$activation_fun_deriv(private$z)
      private$dw <- dz %*% t(private$a_prev)
      private$db <- rowSums(dz)
      if (!is.null(private$prior_layer)) {
        da <- t(private$weights) %*% dz
        private$prior_layer$backward_pass(da = da)
      }
    },

    # @formatter:off
    #' @description Get all cached derivatives.
    #'
    #' @return `list` of derivatives.
    # @formatter:on
    get_derivatives = function() {
      return(list(dw = private$dw, db = private$db))
    },

    # @formatter:off
    #' @description Get the number of nodes of the layer.
    #'
    #' @return `integer` The number of nodes.
    # @formatter:on
    get_number_of_nodes = function() {
      return(private$number_of_nodes)
    },

    # @formatter:off
    #' @description Get the weight matrix and bias of the layer.
    #'
    #' @return `matrix<numeric>` The weights of the current layer.
    # @formatter:on
    get_params = function() {
      return(list(weights = private$weights, bias = private$bias))
    },

    # @formatter:off
    #' @description Set/overwrite the weight matrix of the layer.
    #' @param weight_mat `matrix` with numeric entries.
    #' @param bias_vec `vector` with numeric entries.
    # @formatter:on
    set_params = function(weight_mat, bias_vec = NULL) {
      private$weights <- weight_mat
      if (!is.null(bias_vec)) {
        private$bias <- bias_vec
      }
    },

    # @formatter:off
    #' @description Set/overwrite the input data of the layer.
    #' @param a_prev `matrix` with numeric entries.
    # @formatter:on
    set_input = function(a_prev) {
      private$a_prev <- a_prev
    },

    # @formatter:off
    #' @description Update weights and biases after backward pass.
    #'
    #' @param learning_rate `numeric`: value between 0 and 1 that governs the sensitivity towards the derivatives.
    # @formatter:on
    update_params = function(learning_rate) {
      private$weights <- private$weights - learning_rate * private$dw
      private$bias <- private$bias - learning_rate * private$db
    }
  )
)