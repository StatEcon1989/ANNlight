#' @title Gradient checking
#' 
#' @description Checks if the calculated derivatives are approximately equal to the approximated derivatives. If difference
#' is close to zero (smaller than `1e-6`), the backward pass is likely to be implemented correctly. Larger values may indicate
#' errors in the implementation.
#' 
#' @param network `Network`: The network to be checked.
#' @param epsilon `numeric`: The (small) value used for calculating the approximated derivatives by stressing each parameter
#' up and down.
#'
#' @return `numeric`: The normalized L2-distance between the calculated/approximated derivatives.
#' @export
check_gradient <- function(network, epsilon = 1e-7) {
  params <- ANN$get_all_params(FALSE)
  list <- utils::as.relistable(params)
  params <- unlist(params)
  # calculate one forward and one backward pass to ensure matching parameters and derivatives
  network$calculate_loss()
  network$backward_pass()
  deriv <- network$get_all_derivatives(TRUE)
  loss_up <- rep(0, length(params))
  loss_dn <- loss_up
  # stress each parameter by epsilon and calculate loss
  for (i in 1:length(params)) {
    params_up <- params
    params_dn <- params
    params_up[i] <- params_up[i] + epsilon
    params_dn[i] <- params_dn[i] - epsilon
    network$set_all_params(utils::relist(params_up, skeleton = list))
    loss_up[i] <- ANN$calculate_loss()
    network$set_all_params(utils::relist(params_dn, skeleton = list))
    loss_dn[i] <- network$calculate_loss()
  }
  deriv_approx <- (loss_up - loss_dn) / (2 * epsilon)
  norm_l2_distance <- sqrt(sum((deriv_approx - deriv)^2)) / (sqrt(sum(deriv^2)) + sqrt(sum(deriv_approx^2)))
  if (abs(norm_l2_distance) > 1e-4) {
    warning("The normalized distance between the approximated and calculated derivatives is quite large, indicating a wrong implementation!")
  }
  return(norm_l2_distance)
}