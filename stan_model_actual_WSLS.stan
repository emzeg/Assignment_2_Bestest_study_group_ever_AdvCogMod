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
  array[n] int<lower=0, upper=1> h;        // choices (1=right, 0=left)
  array[n] int<lower=0, upper=1> feedback;  // outcomes (1=win, 0=lose)
}

// The parameters accepted by the model.
// WSLS would require 2 parameters
parameters {
  real<lower=0, upper=1> win_stay;   // prob of staying after a win
  real<lower=0, upper=1> lose_shift; // prob of shifting after a loss
}

// The model to be estimated. We model the output
// 'y' to be normally distributed with mean 'mu'
// and standard deviation 'sigma'.
model {
  // Priors
  target += beta_lpdf(win_stay   | 1, 1);
  target += beta_lpdf(lose_shift | 1, 1);
  // uninformative: we can end up with very high or low win_stay, and very high or low lose_shift

  // Likelihood: start at trial 2 (no previous outcome for trial 1)
  for (t in 2:n) {
    real p;
    if (feedback[t-1] == 1)
      p = win_stay;
    else
      p = 1 - lose_shift;

    target += bernoulli_lpmf(h[t] == h[t-1] | p);
  }
}
//    the last bit, for getting the posteriors etc
generated quantities {
  real<lower=0, upper=1> ws_prior = beta_rng(1, 1);
  real<lower=0, upper=1> ls_prior = beta_rng(1, 1);
  
  array[n] int prior_preds;
  array[n] int posterior_preds;
  
  prior_preds[1]     = bernoulli_rng(0.5);
  posterior_preds[1] = bernoulli_rng(0.5);
  
  for (t in 2:n) {
    int stay_prior    = bernoulli_rng(feedback[t-1] ? ws_prior    : 1 - ls_prior);
    int stay_post     = bernoulli_rng(feedback[t-1] ? win_stay    : 1 - lose_shift);
    
    prior_preds[t]     = stay_prior ? prior_preds[t-1]     : 1 - prior_preds[t-1];
    posterior_preds[t] = stay_post  ? posterior_preds[t-1] : 1 - posterior_preds[t-1];
  }
}


