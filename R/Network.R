#' @title Neural Net
#'
#' @description The user interface for setting up, training and generating forecasts from a simple neural net.
#'
#' @details The training algorithm is speed-up by making use of the matrices instead of loops.
#'
#'@export
Network <- R6::R6Class(
  classname = "Network",
  private = list(
    # the cost function
    cost_fun = NULL,
    # the derivative of the cost function
    cost_fun_deriv = NULL,
    # input data, usually a matrix
    input_data = NULL,
    # the network, stored as a list
    layers = list(),
    # number of layers including input layer
    number_of_layers = integer(),
    # output data, usually a matrix
    output_data = NULL,
    # predicted output, stored for efficiency
    y_hat = NULL,
    # function that calculates the output
    forward_pass = function() {
      return(private$layers[[private$number_of_layers]]$forward_pass())
    },
    # function to set both the cost function and its derivative
    set_cost_fun = function(cost_fun) {
      switch(cost_fun,
             quadratic = {

               private$cost_fun <- function(output_data) {
                 private$y_hat <- private$forward_pass()
                 mean((private$y_hat - output_data)^2) / 2
               }

               private$cost_fun_deriv <- function(output_data) (private$y_hat - output_data)
             },
             cross_entropy = {

               private$cost_fun <- function(output_data) {
                 private$y_hat <- private$forward_pass()
                 -mean(output_data * log(private$y_hat) + (1 - output_data) * log(1 - private$y_hat))
               }

               private$cost_fun_deriv <- function(output_data) {
                 (1 - output_data) / (1 - private$y_hat) - output_data / private$y_hat
               }

             },
             stop("Cost function ", cost_fun, "not supported!"))
    }
  ),
  public = list(
    # @formatter:off
    #' @description Initialize a new class instance.
    #'
    #' @param cost_fun `character` Declare the cost function to be used. Currently implemented: `quadratic` and `cross_entropy` (only for classification).
    #' @param layer_config `list` Defines the parametrization of each desired layer, except for the input_layer (which does not need to be parameterized).
    #' Therefore it is of length `L-1` with `L` being the number of all layers. Each list itself is a list, containing the input arguments of [Layer$new()][Layer]. See the example for more details.
    #' @param input_data `matrix<numeric>` The input data to be used for classification/regression. Each row corresponds to a different feature and each column corresponds to a different observation.
    #' @param output_data `matrix<numeric>` The labels (encoded as numerics) or targets to be used for classification/regression. Each column corresponds to a different observation.
    #'
    #' @examples
    #' layer_config <- list(
    #' list(number_of_nodes = 3, activation_fun = "relu", random_init = NULL),
    #' list(number_of_nodes = 1, activation_fun = "sigmoid", random_init = NULL)
    #' )
    #' ANN <- Network$new(cost_fun = "cross_entropy", layer_config = layer_config,
    #'                    input_data = matrix(runif(24), ncol = 4, nrow = 6),
    #'                    output_data = as.matrix(c(1,0,0,1), nrow = 1))
    #'
    #' @return `Network`: The class instance. Invisibly, for chaining.
    # @formatter:on
    initialize = function(cost_fun = "quadratic", layer_config, input_data, output_data) {
      stopifnot("Unequal number of observations in 'input_data' and 'output_data'!" = ncol(input_data) == ncol(output_data))
      # set cost function
      private$set_cost_fun(tolower(cost_fun))
      # create the layers
      stopifnot('input_data must be supplied to set the input layer!' = !missing(input_data))
      private$input_data <- input_data
      private$layers[[1]] <- Layer$new(a_prev = input_data, number_of_nodes = layer_config[[1]]$number_of_nodes,
                                       activation_fun = layer_config[[1]]$activation_fun, prior_layer = NULL, random_init = layer_config[[1]]$random_init)
      for (i in 2:length(layer_config)) {
        private$layers[[i]] <- Layer$new(number_of_nodes = layer_config[[i]]$number_of_nodes,
                                         activation_fun = layer_config[[i]]$activation_fun, prior_layer = private$layers[[i - 1]], random_init = layer_config[[i]]$random_init)
      }
      private$number_of_layers <- length(private$layers)
      if (!is.null(output_data)) {
        private$output_data <- self$set_output_data(output_data)
      }
      return(invisible(self))
    },

    # @formatter:off
    #' @description Calculate the output of the network for the complete input_data
    #'
    #' @returns `matrix<numeric>` A matrix with predictions from the network for each datapoint in input_data
    # @formatter:on
    calculate = function() {
      private$layers[[1]]$set_input(private$input_data)
      result <- private$forward_pass()
      return(result)
    },

    # @formatter:off
    #' @description Calculate the loss for the complete input_data
    #'
    #' @returns `numeric` The loss over the complete input_data
    # @formatter:on
    calculate_loss = function() {
      return(private$cost_fun(private$output_data))
    },

    # @formatter:off
    #' @description Get the parameters of all layers.
    #'
    #' @param as_vector `logical` Should the result be written as one long vector?
    #'
    #' @return A list or vector, containing the parameters of all layers
    # @formatter:on
    get_all_params = function(as_vector = FALSE) {
      params_list <- lapply(private$layers, FUN = function(x) x$get_params())
      names(params_list) <- 1:private$number_of_layers
      if (as_vector) {
        return(unlist(params_list))
      } else {
        return(params_list)
      }
    },

    # @formatter:off
    #' @description Get the derivatives of all layers.
    #'
    #' @param as_vector `logical` Should the result be written as one long vector?
    #'
    #' @return A list or vector, containing the parameters of all layers
    # @formatter:on
    get_all_derivatives = function(as_vector = FALSE) {
      params_list <- lapply(private$layers, FUN = function(x) x$get_derivatives())
      names(params_list) <- 1:private$number_of_layers
      if (as_vector) {
        return(unlist(params_list))
      } else {
        return(params_list)
      }
    },

    # @formatter:off
    #' @description Set/overwrite the weights and biases for each layer
    #'
    #' @param param_list `list<matrix<numeric>>` A list consisting of lists (weights = ..., bias = ...) for each layer.
    # @formatter:on
    set_all_params = function(param_list) {
      for(i in 1: private$number_of_layers){
        private$layers[[i]]$set_params(weight_mat = param_list[[i]]$weights, bias_vec = param_list[[i]]$bias)
      }
    },

    # @formatter:off
    #' @description Set/overwrite the input_data of the layer
    #'
    #' @param input_data `matrix<numeric>`.
    # @formatter:on
    set_input_data = function(input_data) {
      private$input_data <- input_data
    },

    # @formatter:off
    #' @description Set/overwrite the output_data of the network
    #'
    #' @param output_data `matrix<numeric>`
    # @formatter:on
    set_output_data = function(output_data) {
      private$output_data <- output_data
    },

    # @formatter:off
    #' @description Train the neural net using backpropagation.
    #'
    #' @param batch_size `integer` The size of the batch used for updating the weights per epoch.
    #' If `batch_size = NULL` or `batch_size = n` (`n`: sample size), then batch gradient descend will be performed and
    #' the update of the weights will take place AFTER ALL observations are processed (once per epoch).
    #' On the other hand, if `batch_size = 1`, then stochastic gradient descend will be performed and the weights will be
    #' updated after each observation (`n` times per epoch).
    #' For `1 < batch_size < n` mini batch gradient descend will be performed and the update of the weights will take place
    #' after a set of `batch_size` observations is processed.
    #' @param epochs `integer` The number of epochs used for training.
    #' @param learning_rate `numeric` defines, how much the weight parameters should be adjusted conditional on the value of the derivative.
    #'
    #' @return `vector<numeric>` The value of the cost function BEFORE each epoch, so the first element corresponds to the
    #' randomly initiolized model.
    # @formatter:on
    train = function(batch_size = NULL, epochs = 1e4, learning_rate = 0.1) {
      m <- ncol(private$input_data)
      if (is.null(batch_size)) {
        batch_size <- m
        batch_list <- list(1:m)
      }else {
        batch_list <- split(1:m, ceiling(seq_along(1:m) / batch_size))
      }
      loss <- rep(0, epochs)
      # iterate fpr each epoch
      for (e in 1:epochs) {
        # run the batch
        for (batch in batch_list) {
          # setting the input
          private$layers[[1]]$set_input(private$input_data[, batch, drop = FALSE])
          # calculating cost and forward pass simultaneously
          loss[e] <- loss[e] + private$cost_fun(private$output_data[, batch, drop = FALSE])
          # calculate the backward pass recursively
          private$layers[[private$number_of_layers]]$backward_pass(da = private$cost_fun_deriv(output_data = private$output_data[, batch, drop = FALSE])/batch_size)
          # update the weight matrix
          for (l in length(private$layers):1) {
            private$layers[[l]]$update_params(learning_rate = learning_rate)
          }
        }
      }
      return(loss)
    },

    # @formatter:off
    #' @description performs one backward pass that updates the derivatives in each layer.
    # @formatter:on
    backward_pass = function(){
      private$layers[[private$number_of_layers]]$backward_pass(da = private$cost_fun_deriv(output_data = private$output_data)/ncol(private$input_data))
    }
  )

)