//
// This Stan program defines a simple model, with a
// vector of values 'y' modeled as normally distributed
// with mean 'mu' and standard deviation 'sigma'.
//
// Learn more about model development with Stan at:
//
//    http://mc-stan.org/users/interfaces/rstan.html
//    https://github.com/stan-dev/rstan/wiki/RStan-Getting-Started
//

// The input data is a vector 'y' of length 'N'.
data {
  int<lower=1> n;
  array[n] int<lower=0, upper=1> h;
}

// The parameters accepted by the model.
// WSLS would require 2 parameters or 1 symmetrical param (win_stay?)
parameters {
  real<lower=0, upper=0.5> win_stay_chance;
}

// The model to be estimated. We model the output
// 'y' to be normally distributed with mean 'mu'
// and standard deviation 'sigma'.
model {
  target += beta_lpdf(win_stay_chance | 1,1);
  target += bernoulli_lpmf(h | win_stay_chance);
}

#the last bit, for getting the posteriors etc
generated quantities {
  real<lower=0, upper=1> ws_prior;
  real<lower=0, upper=1> ws_posterior;
  
  int<lower=0, upper=n> prior_preds;
  int<lower=0, upper=n> posterior_preds;
  
  ws_prior= inv_logit(normal_rng(0,1));
  ws_posterior= inv_logit(win_stay_chance)
  
  prior_preds= binomial_rng(n, ws_prior); // n is the n of trials
  posterior_preds= binomial_rng(n, inv_logit(win_stay_chance));
}


