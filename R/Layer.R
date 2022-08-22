#' @title Layer of a neural net
#'
#' @description The Layer class contains all functionalities needed (e.g. calculate the layer, calculate the derivative
#' of its weight matrix, etc.) for its use in the [Network] class.
#' One layer is not able to function on its own (aside from the input layer), the class is intentionally not exported.
Layer <- R6::R6Class(
  classname = "Layer",
  private = list(
    # `logical` for including constant. For now, will always be set to true during initialization.
    bias = NULL,
    # `matrix<numeric>` if input layer, the data is stored here.
    input_data = NULL,
    # `logical` that stores the information whether the layer is an input layer or not.
    input_layer = FALSE,
    # `integer` number of nodes in the layer (what did you expect?)
    number_of_nodes = NULL,
    # `Layer` the complete, previous layer
    prior_layer = NULL,
    # `matrix<numeric>` the weighting matrix, (optionally) including a column for the bias
    weights = NULL,
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
    #' @param input_data `matrix<numeric>` in case of the layer being the input layer, a`matrix` with numeric entries that contains the input_data.
    #' @param random_init `integer`: Initializes all weights and biases with this value. If `NULL` assigns random values between -0.1 and 0.1.
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
    initialize = function(number_of_nodes, activation_fun, prior_layer = NULL, input_data = NULL, random_init = NULL) {
      if (is.null(input_data)) {
        private$prior_layer <- prior_layer
        stopifnot("Must contain at least 1 node!" = number_of_nodes>=1)
        private$number_of_nodes <- as.integer(number_of_nodes)
        private$bias <- TRUE
        if (is.null(random_init)) {
          private$weights <- matrix(
            runif(private$number_of_nodes * private$prior_layer$get_number_of_nodes(), min = -1, max = 1),
            nrow = private$number_of_nodes, ncol = private$prior_layer$get_number_of_nodes()
          )
          if (private$bias) private$weights <- cbind(runif(private$number_of_nodes, min = -1e-1, max = 1e-1), private$weights)
        } else {
          private$weights <- matrix(data = random_init, nrow = private$number_of_nodes, ncol = private$prior_layer$get_number_of_nodes() + private$bias)
        }
        private$set_activation_fun(tolower(activation_fun))
      } else {
        # number of nodes is determined by number of features
        private$number_of_nodes <- ncol(input_data)
        private$input_data <- input_data
        # identify input layer
        private$input_layer <- TRUE
      }
      return(invisible(self))
    },

    # @formatter:off
    #' @description Calculate the output of the current layer
    #'
    #' @return `matrix<numeric>` Either the input- (for an input layer) or the output of the activation function.
    # @formatter:on
    calc_layer = function() {
      if (!private$input_layer) {
        x <- private$prior_layer$calc_layer()
        private$input_data <- x
        if (private$bias) x <- rbind(x, 1)
        result <- private$activation_fun(private$weights %*% x)
        return(result)
      } else {
        # only return input data
        return(t(private$input_data))
      }
    },
    # @formatter:off
    #' @description Calculate the derivative of the current layer wrt to its weights.
    #'
    #' @param w_times_delta `matrix<numeric>` either an output of the `calc_derivative()` method of the next layer, or the
    #' gradient of the cost function (if the current layer is the output layer).
    #'
    #' @return `list<matrix<numeric>>` A list with two elements:
    #' * `derivative`: The derivative for the weighting matrix.
    #' * `w_times_delta`: A matrix product needed for the calculation of the previous layer.
    # @formatter:on
    calc_derivative = function(w_times_delta = 1) {
      if (!private$input_layer) {
        x <- private$input_data
        if (private$bias) x <- rbind(x, 1)
        delta <- w_times_delta * private$activation_fun_deriv(private$weights %*% x)
        derivative <- delta %*% t(x)
        w_times_delta <- t(private$weights[, 1:private$prior_layer$get_number_of_nodes(), drop = FALSE]) %*% delta
        return(list(w_times_delta = w_times_delta, derivative = derivative))
      } else {
        # no derivative for input layer
        return(NULL)
      }
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
    #' @description Get the weight matrix of the layer.
    #'
    #' @return `matrix<numeric>` The weights of the current layer.
    # @formatter:on
    get_weights = function() {
      if (!private$input_layer) {
        return(private$weights)
      } else {
        return(NULL)
      }
    },

    # @formatter:off
    #' @description Set/overwrite the weight matrix of the layer.
    #' @param weight_mat `matrix` with numeric entries.
    # @formatter:on
    set_weights = function(weight_mat) {
      private$weights <- weight_mat
    },

    # @formatter:off
    #' @description Set/overwrite the input data of the layer.
    #' @param input_data `matrix` with numeric entries.
    # @formatter:on
    set_input = function(input_data) {
      private$input_data <- input_data
    }
  )
)